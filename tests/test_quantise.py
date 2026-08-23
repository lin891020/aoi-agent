"""The conversion's contracts, on parts that do not need the trained weights.

`models/reverifier.pt` is gitignored and rebuilt by `scripts/train.py`, so a
test that loaded it would be a test that never runs on a clean checkout. What
is guarded here instead is everything the report reads *through*: the
calibration draw, the softmax the export stops short of, the shape contract
that has to hold before any converted model's numbers reach an operating-point
sweep, and the drift measurement the verdict is decided on.

The one test that really converts a model builds a three-layer net in the test
itself. The export's contract -- one self-contained file, a dynamic batch axis,
a distribution per candidate on the other side -- is not a property of
ResNet-18, and checking it on something that fits in a second is the difference
between a guard that runs in CI and one that does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from aoi_agent.vision.quantise import (
    CALIBRATION_SEED,
    ShapeContract,
    agreement,
    calibration_indices,
    softmax,
)


# --------------------------------------------------------------------------
# the calibration draw
# --------------------------------------------------------------------------


def test_calibration_draw_is_the_same_every_run():
    # Two runs of the report have to calibrate on the same patches, or a
    # difference between them is the draw and not the model.
    first = calibration_indices(5000, sample=64)
    second = calibration_indices(5000, sample=64)
    assert np.array_equal(first, second)


def test_calibration_draw_takes_no_patch_twice():
    drawn = calibration_indices(1000, sample=256)
    assert len(np.unique(drawn)) == 256


def test_calibration_draw_stays_inside_the_split():
    drawn = calibration_indices(37, sample=10)
    assert drawn.min() >= 0
    assert drawn.max() < 37


def test_calibration_draw_never_asks_for_more_than_there_is():
    # A small trainval split is a smaller calibration set, not an error.
    assert len(calibration_indices(12, sample=512)) == 12


def test_calibration_draw_refuses_an_empty_split():
    with pytest.raises(ValueError):
        calibration_indices(0)


def test_calibration_draw_moves_with_the_seed():
    fixed = calibration_indices(5000, sample=64)
    other = calibration_indices(5000, sample=64, seed=CALIBRATION_SEED + 1)
    assert not np.array_equal(fixed, other)


# --------------------------------------------------------------------------
# the softmax the graph stops short of
# --------------------------------------------------------------------------


def test_softmax_rows_sum_to_one():
    rows = softmax(np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]]))
    assert np.allclose(rows.sum(axis=1), 1.0)


def test_softmax_keeps_the_ordering_of_the_logits():
    rows = softmax(np.array([[0.5, 3.0, -1.0]]))
    assert rows.argmax(1)[0] == 1


def test_softmax_survives_logits_large_enough_to_overflow():
    # The shift by the row max is the only reason this does not become nan,
    # and an INT8 graph can hand back larger logits than a float one.
    rows = softmax(np.array([[1000.0, 999.0, 1.0]]))
    assert np.all(np.isfinite(rows))
    assert np.allclose(rows.sum(axis=1), 1.0)


# --------------------------------------------------------------------------
# the shape contract
# --------------------------------------------------------------------------


CONTRACT = ShapeContract(batch=4, channels=3, size=64, classes=7)


def valid_output(batch: int = 4, classes: int = 7) -> np.ndarray:
    raw = np.linspace(0.1, 1.0, batch * classes).reshape(batch, classes)
    return raw / raw.sum(axis=1, keepdims=True)


def test_shape_contract_passes_a_real_distribution():
    assert shape_problems(valid_output()) == []


def shape_problems(output: np.ndarray) -> list[str]:
    from aoi_agent.vision.quantise import shape_violations

    return shape_violations(output, CONTRACT)


def test_shape_contract_catches_a_dropped_batch_axis():
    # The failure that looks most like success: a conversion that collapses the
    # batch still returns numbers, and the sweep would read them.
    problems = shape_problems(valid_output()[0])
    assert problems and "2-D" in problems[0]


def test_shape_contract_catches_the_wrong_number_of_candidates():
    problems = shape_problems(valid_output(batch=3))
    assert any("expected 4 rows" in problem for problem in problems)


def test_shape_contract_catches_the_wrong_number_of_classes():
    problems = shape_problems(valid_output(classes=6))
    assert any("expected 7 classes" in problem for problem in problems)


def test_shape_contract_catches_rows_that_are_not_a_distribution():
    logits = np.linspace(-3, 3, 28).reshape(4, 7)   # never normalised
    assert any("sum to 1" in problem for problem in shape_problems(logits))


def test_shape_contract_catches_probabilities_out_of_range():
    output = valid_output()
    output[0] = [-0.5, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    assert any("outside [0, 1]" in problem for problem in shape_problems(output))


def test_shape_contract_catches_a_graph_that_produced_nan():
    output = valid_output()
    output[2, 3] = np.nan
    assert any("non-finite" in problem for problem in shape_problems(output))


# --------------------------------------------------------------------------
# drift from the float model
# --------------------------------------------------------------------------


def test_agreement_with_itself_is_total():
    reference = valid_output()
    drift = agreement(reference, reference.copy())
    assert drift["class_agreement"] == 1.0
    assert drift["disagreements"] == 0
    assert drift["max_probability_delta"] == 0.0


def test_agreement_counts_the_candidates_that_changed_class():
    reference = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]])
    candidate = np.array([[0.4, 0.6], [0.4, 0.6], [0.2, 0.8]])
    drift = agreement(reference, candidate)
    assert drift["disagreements"] == 1
    assert drift["class_agreement"] == pytest.approx(2 / 3)


def test_agreement_reports_the_probability_move_the_threshold_would_see():
    # The point of the second measure: no class changed, and every dismissal
    # decision at a 0.915 cut did.
    reference = np.array([[0.92, 0.08]])
    candidate = np.array([[0.91, 0.09]])
    drift = agreement(reference, candidate)
    assert drift["class_agreement"] == 1.0
    assert drift["max_probability_delta"] == pytest.approx(0.01)


def test_agreement_refuses_two_things_that_cannot_be_compared():
    with pytest.raises(ValueError):
        agreement(valid_output(batch=4), valid_output(batch=3))


# --------------------------------------------------------------------------
# the conversion itself, on a model small enough to build here
# --------------------------------------------------------------------------


@pytest.fixture
def tiny_model():
    torch = pytest.importorskip("torch")
    from torch import nn

    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, CONTRACT.classes),
    )


def test_the_export_is_one_self_contained_file(tmp_path, tiny_model):
    pytest.importorskip("onnx")
    from aoi_agent.vision.quantise import export_module

    destination = export_module(tiny_model, tmp_path / "tiny.onnx", size=16)
    assert destination.exists()
    # The dynamo exporter writes the weights to a sidecar `.onnx.data`. This
    # project exports with `dynamo=False` precisely so "model size on disk" is
    # one number, and so the quantiser has one file to read.
    assert list(tmp_path.iterdir()) == [destination]


def test_the_exported_graph_keeps_the_batch_axis_dynamic(tmp_path, tiny_model):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from aoi_agent.vision.quantise import OnnxReVerifier, export_module, shape_violations

    export_module(tiny_model, tmp_path / "tiny.onnx", size=16)
    runner = OnnxReVerifier(tmp_path / "tiny.onnx")
    assert isinstance(runner.session, ort.InferenceSession)

    # The pipeline's batch is the board's candidate count, which is different
    # on every board. Two batches, one export.
    for batch in (1, CONTRACT.batch):
        patches = np.random.default_rng(0).integers(
            0, 256, size=(batch, 3, 16, 16), dtype=np.uint8
        )
        probabilities = runner.probabilities(patches)
        contract = ShapeContract(batch=batch, channels=3, size=16,
                                 classes=CONTRACT.classes)
        assert shape_violations(probabilities, contract) == []


def test_dynamic_quantisation_shrinks_the_file_and_holds_the_contract(
    tmp_path, tiny_model
):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from aoi_agent.vision.quantise import (
        OnnxReVerifier,
        export_module,
        preprocess_onnx,
        quantise_dynamic,
        shape_violations,
    )

    export_module(tiny_model, tmp_path / "fp32.onnx", size=16)
    preprocess_onnx(tmp_path / "fp32.onnx", tmp_path / "pre.onnx")
    quantised = quantise_dynamic(tmp_path / "pre.onnx", tmp_path / "int8.onnx")

    patches = np.random.default_rng(1).integers(
        0, 256, size=(CONTRACT.batch, 3, 16, 16), dtype=np.uint8
    )
    probabilities = OnnxReVerifier(quantised).probabilities(patches)
    contract = ShapeContract(batch=CONTRACT.batch, channels=3, size=16,
                             classes=CONTRACT.classes)
    assert shape_violations(probabilities, contract) == []


def test_static_quantisation_calibrates_and_holds_the_contract(tmp_path, tiny_model):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from aoi_agent.vision.quantise import (
        OnnxReVerifier,
        export_module,
        preprocess_onnx,
        quantise_static,
        shape_violations,
    )

    export_module(tiny_model, tmp_path / "fp32.onnx", size=16)
    preprocess_onnx(tmp_path / "fp32.onnx", tmp_path / "pre.onnx")
    calibration = np.random.default_rng(2).integers(
        0, 256, size=(24, 3, 16, 16), dtype=np.uint8
    )
    quantised = quantise_static(
        tmp_path / "pre.onnx", tmp_path / "int8_static.onnx", calibration
    )

    patches = np.random.default_rng(3).integers(
        0, 256, size=(2, 3, 16, 16), dtype=np.uint8
    )
    probabilities = OnnxReVerifier(quantised).probabilities(patches)
    contract = ShapeContract(batch=2, channels=3, size=16, classes=CONTRACT.classes)
    assert shape_violations(probabilities, contract) == []


def test_serving_a_quantised_model_does_not_need_the_conversion_dependency(
    tmp_path, tiny_model
):
    """The runner must load with `onnxruntime` alone.

    `onnx` is in the dev group, not the runtime dependencies, and the argument
    for that is exactly this: onnxruntime is already in the shipped image
    through Chroma, so a station serving a quantised model needs nothing new
    and the 2.43GB image does not grow. That argument is only true while
    `OnnxReVerifier` never reaches for `onnx`, which is a property of the
    import graph and would break silently.
    """
    import importlib.abc
    import sys

    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from aoi_agent.vision.quantise import (
        OnnxReVerifier,
        export_module,
        preprocess_onnx,
        quantise_dynamic,
    )

    export_module(tiny_model, tmp_path / "fp32.onnx", size=16)
    preprocess_onnx(tmp_path / "fp32.onnx", tmp_path / "pre.onnx")
    quantised = quantise_dynamic(tmp_path / "pre.onnx", tmp_path / "int8.onnx")

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "onnx" or name.startswith("onnx."):
                raise ImportError(f"{name} is not in the shipped image")
            return None

    already_imported = {
        name: module for name, module in sys.modules.items()
        if name == "onnx" or name.startswith("onnx.")
    }
    for name in already_imported:
        del sys.modules[name]
    blocker = Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        probabilities = OnnxReVerifier(quantised).probabilities(
            np.zeros((2, 3, 16, 16), dtype=np.uint8)
        )
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(already_imported)

    contract = ShapeContract(batch=2, channels=3, size=16, classes=CONTRACT.classes)
    from aoi_agent.vision.quantise import shape_violations

    assert shape_violations(probabilities, contract) == []
