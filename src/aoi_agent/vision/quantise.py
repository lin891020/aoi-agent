"""Export the re-verifier to ONNX and quantise it to INT8.

Why this exists, and what it is honestly for. The re-verifier already costs
2.50ms per candidate on CPU (docs/benchmarks.md, 2026-08-23), so nothing here
is chasing a latency problem -- there is not one. What INT8 buys is a 42.7MB
float32 checkpoint becoming something a constrained box can hold, and a
runtime that does not need torch installed to serve it. Whether it buys that
without moving the operating point is the question `scripts/quantisation_report.py`
asks, and the answer is allowed to be "not worth it".

Three artefacts, in order, each built from the one before:

* **FP32 ONNX** -- `torch.onnx.export` of the trained checkpoint.
* **INT8 dynamic** -- weights quantised at conversion time, activations
  quantised per-inference from their own observed range. No calibration data,
  so nothing about it can leak the test split.
* **INT8 static** -- weights *and* activations quantised against ranges
  measured on a calibration set. That set is drawn from the **trainval**
  patches, never from test: calibrating on the split the operating point is
  then reported on is the same error as tuning a threshold against it, and it
  would flatter every number in the report.

The export uses the TorchScript exporter (`dynamo=False`) deliberately. The
dynamo exporter is torch 2.13's default and works, but it writes the weights
to a sidecar `.onnx.data` file, which makes "model size on disk" a two-file
answer and gives onnxruntime's quantiser a second thing to lose. A
self-contained graph is worth a deprecation warning here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# onnxruntime's wheel starts Microsoft's 1DS telemetry uploader at import and
# nothing joins it at exit; a response landing during static destruction aborts
# the process after every test has passed. `aoi_agent.store.standards` carries
# the full account. The variable is read while onnxruntime builds its Env, so
# it has to be set above the import, in every module that can be the first one
# to trigger it -- this is now the second such door.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

#: Named so the ONNX graph reads like the thing it serves rather than like
#: `input.1`. Anything loading these files should use these names.
ONNX_INPUT = "patch"
ONNX_OUTPUT = "logits"

#: Opset 17 is what onnxruntime 1.29's quantiser is happiest with, and it is
#: old enough that a runtime on an edge box is unlikely to be behind it.
EXPORT_OPSET = 17

#: How many trainval patches the static quantiser sees. Enough for the
#: activation ranges to settle, small enough that the calibration pass is
#: seconds rather than minutes -- the ranges are percentile-ish statistics over
#: a distribution, not a training signal.
CALIBRATION_SAMPLES = 512

#: Fixed, so two runs of the report calibrate on the same patches and any
#: difference between them is the model, not the draw.
CALIBRATION_SEED = 20260823


# --------------------------------------------------------------------------
# pure -- no onnx, no onnxruntime, no torch
# --------------------------------------------------------------------------


def calibration_indices(
    total: int, sample: int = CALIBRATION_SAMPLES, seed: int = CALIBRATION_SEED
) -> np.ndarray:
    """Which trainval patches the static quantiser sees.

    Deterministic and without replacement, and it never asks for more than
    there are. Sorted on the way out only so a run is reproducible to read.
    """
    if total <= 0:
        raise ValueError("no patches to calibrate on")
    take = min(sample, total)
    drawn = np.random.default_rng(seed).choice(total, size=take, replace=False)
    return np.sort(drawn)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, matching what `ReVerifier` does in torch.

    Here rather than borrowed from scipy because the ONNX graph stops at the
    logits: the export deliberately ends where the trained module ends, so the
    quantiser is not asked to reason about a softmax it cannot help.
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class ShapeContract:
    """What any converted model must still do to a stack of patches."""

    batch: int
    channels: int
    size: int
    classes: int


def shape_violations(probabilities: np.ndarray, contract: ShapeContract) -> list[str]:
    """Everything wrong with one converted model's output, named.

    A conversion that silently drops the batch axis, transposes the classes or
    stops summing to one produces numbers that still look like probabilities.
    The operating-point sweep would then read them without complaint and the
    report would publish a curve for a broken model. This is the assertion that
    has to run before any of that.
    """
    problems: list[str] = []
    if probabilities.ndim != 2:
        problems.append(f"expected a 2-D output, got {probabilities.ndim}-D")
        return problems
    rows, columns = probabilities.shape
    if rows != contract.batch:
        problems.append(f"expected {contract.batch} rows, got {rows}")
    if columns != contract.classes:
        problems.append(f"expected {contract.classes} classes, got {columns}")
    if not np.all(np.isfinite(probabilities)):
        problems.append("output contains non-finite values")
        return problems
    if probabilities.min() < -1e-6 or probabilities.max() > 1 + 1e-6:
        problems.append("probabilities outside [0, 1]")
    sums = probabilities.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-4):
        problems.append(f"rows do not sum to 1 (worst {abs(sums - 1).max():.2e})")
    return problems


def agreement(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """How far the quantised model drifted from the float one, per candidate.

    Two numbers, because they answer different questions. The class agreement
    is what an operator would notice. The mean absolute difference in
    P(false_call) is what the *threshold* notices, and the threshold is what
    this project reports on -- a model can agree on every class and still
    move every dismissal decision by sitting a hair the other side of 0.915.
    """
    if reference.shape != candidate.shape:
        raise ValueError(
            f"cannot compare {reference.shape} against {candidate.shape}"
        )
    same_class = reference.argmax(1) == candidate.argmax(1)
    return {
        "n": int(len(reference)),
        "class_agreement": float(same_class.mean()),
        "disagreements": int((~same_class).sum()),
        "max_probability_delta": float(np.abs(reference - candidate).max()),
        "mean_probability_delta": float(np.abs(reference - candidate).mean()),
    }


# --------------------------------------------------------------------------
# conversion -- needs onnx and onnxruntime
# --------------------------------------------------------------------------


def export_module(
    model,
    destination: Path | str,
    size: int | None = None,
    opset: int = EXPORT_OPSET,
) -> Path:
    """Any eval-mode module to a self-contained FP32 ONNX graph.

    Split out from `export_onnx` so the conversion can be exercised on a model
    small enough to build in a test. The trained checkpoint is gitignored and
    rebuilt by `scripts/train.py`, so a test that needed it would be a test
    that never ran on a clean checkout -- and the export's contract, that the
    batch axis stays dynamic and the graph stays one file, is not a property of
    ResNet-18.

    The batch axis is dynamic because the pipeline's batch is the board's
    candidate count and varies board to board. A fixed-batch export would make
    `classify_batch` a lie.
    """
    import torch

    from aoi_agent.vision.patches import PATCH_SIZE

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    example = torch.zeros(1, 3, size or PATCH_SIZE, size or PATCH_SIZE)
    torch.onnx.export(
        model,
        (example,),
        str(destination),
        input_names=[ONNX_INPUT],
        output_names=[ONNX_OUTPUT],
        dynamic_axes={ONNX_INPUT: {0: "batch"}, ONNX_OUTPUT: {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    return destination


def export_onnx(
    checkpoint: Path | str,
    destination: Path | str,
    opset: int = EXPORT_OPSET,
) -> Path:
    """Trained checkpoint to a self-contained FP32 ONNX graph."""
    from aoi_agent.vision.inference import ReVerifier

    verifier = ReVerifier(checkpoint, device="cpu")
    return export_module(verifier.model, destination, opset=opset)


def preprocess_onnx(source: Path | str, destination: Path | str) -> Path:
    """Shape inference and constant folding, before either quantiser runs.

    onnxruntime warns about skipping this and it is not decoration: the
    quantiser places its scale/zero-point nodes from the inferred shapes, and
    an unfolded graph gives it less to work with. Done once, and both INT8
    conversions start from the result, so dynamic and static are compared on
    the same graph rather than on two.
    """
    from onnxruntime.quantization.shape_inference import quant_pre_process

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    quant_pre_process(Path(source), destination, skip_symbolic_shape=False)
    return destination


def quantise_dynamic(source: Path | str, destination: Path | str) -> Path:
    """INT8 weights, activations quantised per-inference from their own range.

    No calibration set, so there is nothing here that could have seen the test
    split. This is the conversion to reach for first: it is one call, it cannot
    be got wrong in a way that leaks, and if it holds the operating point there
    is no reason to run the other one.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        Path(source), destination, weight_type=QuantType.QInt8
    )
    return destination


def quantise_static(
    source: Path | str,
    destination: Path | str,
    calibration_patches: np.ndarray,
) -> Path:
    """INT8 weights and activations, against ranges measured on trainval.

    `calibration_patches` must come from the trainval split. Nothing in this
    function can check that -- it is handed an array -- so the caller carries
    the obligation, and `scripts/quantisation_report.py` loads
    `data/patches/trainval.npz` by name and says so in the report it writes.
    """
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    class _PatchReader(CalibrationDataReader):
        """Feeds the calibration patches once, one at a time.

        One at a time because the export's batch axis is dynamic and the
        calibrator only needs the activation ranges, not throughput; a large
        batch here would change nothing but the peak memory of the conversion.
        """

        def __init__(self, patches: np.ndarray):
            self._batches = iter(
                {ONNX_INPUT: patch[None].astype(np.float32) / 255.0}
                for patch in patches
            )

        def get_next(self):
            return next(self._batches, None)

    quantize_static(
        Path(source),
        destination,
        _PatchReader(calibration_patches),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    return destination


class OnnxReVerifier:
    """Runs a converted graph over patches, the way `ReVerifier` runs the torch one.

    Deliberately the same shape as `ReVerifier.classify_batch`'s inner path --
    uint8 patches to float, divide by 255, forward, softmax -- so that the
    latency figures in the quantisation report and the ones in the FP32 report
    are measuring the same work with a different engine underneath. The
    softmax is here rather than in the graph because the export ends at the
    logits; see the module docstring.
    """

    def __init__(self, model_path: Path | str, threads: int | None = None):
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads is not None:
            options.intra_op_num_threads = threads
        self.path = Path(model_path)
        self.session = ort.InferenceSession(
            str(self.path),
            options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def logits(self, patches: np.ndarray) -> np.ndarray:
        batch = patches.astype(np.float32) / 255.0
        return self.session.run(None, {self.input_name: batch})[0]

    def probabilities(self, patches: np.ndarray) -> np.ndarray:
        return softmax(self.logits(patches))

    @property
    def size_mb(self) -> float:
        return self.path.stat().st_size / (1024 * 1024)
