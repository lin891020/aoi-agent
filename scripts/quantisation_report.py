"""What does quantising the re-verifier cost, in the terms this project reports in?

The open INT8/ONNX item was scoped by `scripts/reverifier_latency.py` rather
than by intuition: the re-verifier already costs 2.50ms per candidate on CPU,
so there is no latency problem here to solve. That makes this script's job the
opposite of the usual one. It is not looking for a speed-up. It is asking what
a 4x smaller model costs at the escape budget, and it is allowed to come back
with "nothing worth having, do not ship it".

So the headline here is the same headline the rest of the project uses --
**manual review removed at an escape budget** -- with a column per engine
instead of one. A conversion that halves the latency and moves the ≤0.5%
budget's review reduction from 56.2% to 40% is a bad trade, and this is the
only framing that shows it. Latency, size and memory sit underneath, where
they belong.

Four engines, all on CPU, all in one process:

* **FP32 torch** -- the deployed path, re-measured here rather than quoted from
  the earlier run, so every number in the comparison comes off the same machine
  in the same minutes.
* **FP32 ONNX** -- the export, unquantised. Its job is to separate "ONNX changed
  the answer" from "INT8 changed the answer", which are different findings and
  would otherwise be one column.
* **INT8 dynamic** -- weights quantised at conversion, activations per-inference.
  No calibration data at all.
* **INT8 static** -- weights and activations, calibrated on patches drawn from
  the **trainval** split. Never test: calibrating on the split the curve is
  then reported on is threshold-tuning against the test set wearing a different
  hat.

Contention is checked the same two ways `reverifier_latency.py` checks it, by
importing that check rather than restating it -- and that check learned about
`ffmpeg` on the day this script was written, because it found four transcodes
holding 800% CPU while both of its checks came back clean. A CPU contender was
survivable when the interesting number was an MPS one. INT8 is a CPU story end
to end, so it is not survivable here.

    uv run python scripts/quantisation_report.py
    uv run python scripts/quantisation_report.py --dry-run --soak 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aoi_agent.vision.inference import DEFAULT_CHECKPOINT, ReVerifier  # noqa: E402
from aoi_agent.vision.operating_point import (  # noqa: E402
    best_at_escape_budget,
    sweep,
)
from aoi_agent.vision.patches import PATCH_SIZE, PatchSet  # noqa: E402
from aoi_agent.vision.quantise import (  # noqa: E402
    CALIBRATION_SAMPLES,
    CALIBRATION_SEED,
    OnnxReVerifier,
    ShapeContract,
    agreement,
    calibration_indices,
    export_onnx,
    preprocess_onnx,
    quantise_dynamic,
    quantise_static,
    shape_violations,
)
from reverifier_latency import (  # noqa: E402
    WARMUP_ITERATIONS,
    competing_processes,
    format_ms,
    ollama_ps,
    peak_rss_mb,
    process_table,
    resident_models,
    summarise,
    thermal_split,
    throttle_verdict,
)

#: The same budgets `scripts/report.py` publishes. Changing them here would
#: make the comparison unreadable against the table it is meant to sit beside.
BUDGETS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]

#: The budget the project leads on and the one the deployed threshold comes
#: from. Every verdict in this report is decided at this row.
HEADLINE_BUDGET = 0.005

#: Smaller than the FP32 report's sweep: past batch 8 the CPU per-candidate
#: cost is already known to get worse, and this report is about the engines
#: rather than about re-deriving that cliff.
SWEEP_BATCHES = [1, 2, 4, 8, 16]

#: How much review reduction the project is willing to lose to save the disk.
#: Below this the trade is refused in the report rather than left to the
#: reader, because the reader is the one who would otherwise ship it.
REVIEW_REDUCTION_TOLERANCE = 0.01


# --------------------------------------------------------------------------
# pure -- no torch, no onnx, no device
# --------------------------------------------------------------------------


def operating_row(point, budget: float) -> dict:
    """One engine's answer at one escape budget, flattened for the table."""
    if point is None:
        return {"budget": budget, "reachable": False}
    return {
        "budget": budget,
        "reachable": True,
        "threshold": point.threshold,
        "escape_rate": point.escape_rate,
        "review_reduction": point.review_reduction,
        "escapes": point.escapes,
        "defects_total": point.defects_total,
    }


def review_reduction_delta(baseline: dict, candidate: dict) -> float | None:
    """Percentage points of review reduction gained or lost, or None.

    None when either engine cannot reach the budget at all, which is a real
    answer rather than a zero -- a model that cannot be deployed at a tolerance
    has not "lost 0 points", it has no operating point there.
    """
    if not baseline.get("reachable") or not candidate.get("reachable"):
        return None
    return candidate["review_reduction"] - baseline["review_reduction"]


def trade_verdict(
    baseline_row: dict,
    candidate_row: dict,
    candidate_label: str,
    baseline_p50: float,
    candidate_p50: float,
    baseline_mb: float,
    candidate_mb: float,
    tolerance: float = REVIEW_REDUCTION_TOLERANCE,
) -> str:
    """Is this conversion worth shipping? Decided on the curve, not the clock.

    The order of the sentence is the argument: what it costs at the escape
    budget first, what it buys second. A quantisation report that leads with
    milliseconds has already answered the wrong question, and this project
    would then have two headline metrics that disagree.
    """
    delta = review_reduction_delta(baseline_row, candidate_row)
    size_ratio = baseline_mb / candidate_mb if candidate_mb else float("nan")
    speed_ratio = baseline_p50 / candidate_p50 if candidate_p50 else float("nan")
    speed = (
        f"{speed_ratio:.2f}x the speed"
        if speed_ratio >= 1
        else f"{1 / speed_ratio:.2f}x *slower*"
    )
    bought = f"{size_ratio:.1f}x smaller on disk and {speed}"

    if delta is None:
        return (
            f"**{candidate_label}: refused.** It has no operating point at the "
            f"≤{baseline_row['budget']:.2%} budget, so what it buys -- {bought} "
            f"-- is not purchasable at a price this line accepts."
        )
    if delta < -tolerance:
        return (
            f"**{candidate_label}: not worth taking.** It gives up "
            f"{-delta * 100:.1f} points of review reduction at the "
            f"≤{baseline_row['budget']:.2%} budget "
            f"({baseline_row['review_reduction']:.1%} to "
            f"{candidate_row['review_reduction']:.1%}) to buy {bought}. The lost "
            f"reduction is operators back in front of regions the FP32 model was "
            f"willing to dismiss, every shift, for a saving on a disk that was "
            f"not full."
        )
    held = "unchanged" if abs(delta) < 1e-9 else f"{delta * 100:+.1f} points"
    return (
        f"**{candidate_label}: the curve holds.** Review reduction at the "
        f"≤{baseline_row['budget']:.2%} budget is {held} "
        f"({baseline_row['review_reduction']:.1%} against "
        f"{candidate_row['review_reduction']:.1%}), and it buys {bought}."
    )


def line_rate_implication(
    candidates_per_board: float,
    boards: int,
    p50_ms: float,
    best_int8_p50_ms: float,
    seconds_per_board: float = 10.0,
) -> str:
    """Was inference ever the constraint? Answered in boards, not milliseconds.

    `seconds_per_board` is a stated assumption, not a measurement -- this
    project has no line to time. It is deliberately generous to the *fast*
    side, so the conclusion survives a faster line than the one anyone would
    build.
    """
    per_board_ms = candidates_per_board * p50_ms
    int8_per_board_ms = candidates_per_board * best_int8_p50_ms
    budget_ms = seconds_per_board * 1000
    share = per_board_ms / budget_ms
    saved_ms = per_board_ms - int8_per_board_ms
    return (
        f"**Was inference ever the constraint?** No, and the arithmetic is not "
        f"close. The test split is {boards} boards carrying "
        f"{candidates_per_board:.1f} candidates each on average, so one board's "
        f"re-verification is {per_board_ms:.0f}ms of FP32 inference. Against a "
        f"board every {seconds_per_board:.0f} seconds -- fast for a line that has "
        f"an AOI stage and a conveyor in front of it -- that is {share:.2%} of the "
        f"cycle. The best INT8 engine here takes it to {int8_per_board_ms:.0f}ms, "
        f"a saving of {saved_ms:.0f}ms per board on a budget of {budget_ms:.0f}ms. "
        f"There is no queue to drain and no operator waiting on it. Latency is "
        f"not what this conversion is for, and a report that sold it as one "
        f"would be selling a rounding error."
    )


def render(results: dict) -> list[str]:
    """The markdown section, from primitives only.

    Free of torch, onnx and the measurement loop, so the report's shape is
    testable on a machine with none of them -- the same reason
    `reverifier_latency.render` is its own function.
    """
    baseline = results["baseline_key"]
    engines = results["engines"]
    by_key = {engine["key"]: engine for engine in engines}

    lines = [
        "### Quantisation — what INT8 costs at the escape budget",
        "",
        f"ResNet-18 re-verifier, {by_key[baseline]['size_mb']:.1f}MB float32 "
        f"checkpoint, {results['parameters'] / 1e6:.1f}M parameters, "
        f"3x{PATCH_SIZE}x{PATCH_SIZE} input, exported to ONNX and quantised to "
        f"INT8 two ways. Every engine below is scored on all "
        f"{results['patch_count']} candidates of the official DeepPCB test split "
        f"and timed on the same machine in the same run, on CPU, at "
        f"{results['cpu_threads']} threads.",
        "",
        "The static quantiser's calibration set is "
        f"{results['calibration_samples']} patches drawn from "
        f"`data/patches/trainval.npz` with seed {results['calibration_seed']} -- "
        "the **training** split, never test. Calibrating activation ranges on the "
        "split the operating point is then reported on is threshold-tuning "
        "against the test set with an extra step in front of it.",
        "",
        "Contention was checked the way `reverifier_latency.py` checks it, before "
        "and after. That check gained its CPU claimants on the day this ran: it "
        "found four `ffmpeg` transcodes from a neighbouring project holding "
        "roughly 800% CPU while `ollama ps` and the process sweep both came back "
        "clean, because the sweep's list had been written for a GPU benchmark. "
        "INT8 is a CPU result end to end, so that hole had to close before any "
        "number here was worth writing down.",
        "",
        "```",
        "ollama ps before the run",
        results["ps_before"],
        "",
        "busy processes before the run",
        "\n".join(results["busy_before"]) or "(none)",
        "",
        "ollama ps after the run",
        results["ps_after"],
        "",
        "busy processes after the run",
        "\n".join(results["busy_after"]) or "(none)",
        "```",
        "",
        "#### Manual review removed at an escape budget — FP32 against INT8",
        "",
        "This is the comparison. Everything below it is detail.",
        "",
        "| escape budget | " + " | ".join(e["label"] for e in engines) + " |",
        "|---" * (len(engines) + 1) + "|",
    ]

    for budget in results["budgets"]:
        cells = []
        for engine in engines:
            row = results["operating_points"][engine["key"]][str(budget)]
            if not row["reachable"]:
                cells.append("not reachable")
                continue
            cells.append(
                f"**{row['review_reduction']:.1%}** "
                f"({row['escape_rate']:.2%}, {row['escapes']}/{row['defects_total']})"
            )
        lines.append(f"| ≤{budget:.2%} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Each cell is the review reduction, with the escape rate it achieved and "
        "the escapes behind it in brackets. The thresholds differ between "
        "columns: each engine is given the best threshold *it* can reach inside "
        "the budget, which is the fairest reading and the one that makes a "
        "column's loss unambiguous.",
        "",
        "**Read the tightest budgets with the escape count beside them.** At "
        "≤0.10% the whole column is decided by two escapes out of "
        f"{results['defects_total']} defects, so a swing of several points there "
        "is a handful of candidates landing the other side of a cut, not an "
        "engine being better. The ≤0.50% row is the one the deployed threshold "
        "comes from and the one every verdict below is decided on.",
        "",
    ]

    lines += results["verdicts"]
    lines += [
        "",
        "#### How far each engine drifted from the float model",
        "",
        "Class agreement is what an operator would notice. The probability "
        "deltas are what the *threshold* notices, and the threshold is what this "
        "project reports on -- a model can agree on every class and still move "
        "every dismissal by sitting a hair the other side of the cut.",
        "",
        "| engine | class agreement | disagreements | mean Δp | max Δp |",
        "|---|---|---|---|---|",
    ]
    for engine in engines:
        if engine["key"] == baseline:
            continue
        drift = results["agreement"][engine["key"]]
        lines.append(
            f"| {engine['label']} | {drift['class_agreement']:.3%} | "
            f"{drift['disagreements']}/{drift['n']} | "
            f"{drift['mean_probability_delta']:.2e} | "
            f"{drift['max_probability_delta']:.2e} |"
        )

    lines += [
        "",
        "#### Single candidate, warm — CPU",
        "",
        "| engine | calls | p50 | p90 | p99 | max | mean |",
        "|---|---|---|---|---|---|---|",
    ]
    for engine in engines:
        s = engine["single"]
        lines.append(
            f"| {engine['label']} | {s['n']} | {format_ms(s['p50'])} | "
            f"{format_ms(s['p90'])} | {format_ms(s['p99'])} | "
            f"{format_ms(s['max'])} | {format_ms(s['mean'])} |"
        )

    lines += [
        "",
        results["latency_note"],
        "",
        "#### Batched throughput — CPU",
        "",
        "| batch | " + " | ".join(e["label"] for e in engines) + " |",
        "|---" * (len(engines) + 1) + "|",
    ]
    for size in results["batch_sizes"]:
        cells = []
        for engine in engines:
            entry = engine["batches"].get(size)
            cells.append(
                "—" if entry is None
                else f"{entry['per_candidate_ms']:.2f}ms/cand, "
                     f"{entry['throughput']:,.0f}/s"
            )
        marker = " ←" if size == results["pipeline_batch_bucket"] else ""
        lines.append(f"| {size}{marker} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "The arrow marks the bucket nearest the pipeline's real batch "
        f"({results['pipeline_batch']} candidates on the median board).",
        "",
        "#### Cold against warm",
        "",
        "A station restarting mid-shift pays the load and the first inference.",
        "",
        "| engine | load | first inference | warm p50 | cold penalty |",
        "|---|---|---|---|---|",
    ]
    for engine in engines:
        warm = engine["single"]["p50"]
        lines.append(
            f"| {engine['label']} | {format_ms(engine['load_ms'])} | "
            f"{format_ms(engine['cold_ms'])} | {format_ms(warm)} | "
            f"{engine['cold_ms'] / warm:.0f}x |"
        )

    lines += [
        "",
        f"#### Thermal — first {int(results['thermal_split_s'])}s against steady state",
        "",
        f"{results['soak_s']:.0f}s of sustained batched inference per engine on a "
        f"fanless M5 Air.",
        "",
        "| engine | first 60s (n, p50) | steady state (n, p50) |",
        "|---|---|---|",
    ]
    for engine in engines:
        early, late = engine["soak_early"], engine["soak_late"]
        lines.append(
            f"| {engine['label']} | "
            + (f"{early['n']}, {format_ms(early['p50'])}" if early else "—")
            + " | "
            + (f"{late['n']}, {format_ms(late['p50'])}" if late else "—")
            + " |"
        )

    lines.append("")
    for engine in engines:
        lines.append(f"- {engine['label']}: {engine['throttle']}")

    lines += [
        "",
        "#### Footprint",
        "",
        "| engine | on disk | of FP32 | peak RSS while serving |",
        "|---|---|---|---|",
    ]
    for engine in engines:
        lines.append(
            f"| {engine['label']} | {engine['size_mb']:.1f}MB | "
            f"{engine['size_mb'] / by_key[baseline]['size_mb']:.0%} | "
            f"{engine['rss_mb']:.0f}MB |"
        )

    lines += [
        "",
        "Peak RSS is measured as a high-water mark over the whole process, so "
        "each engine's figure includes everything loaded before it. The column "
        "is a floor on the footprint, not an isolated measurement of one engine, "
        "and it is stated rather than corrected for.",
        "",
        results["line_rate"],
        "",
        results["conclusion"],
        "",
    ]
    return lines


def conclusion(recommended: str | None, saved_mb: float, baseline_mb: float) -> str:
    """What the project should actually do with this, in one paragraph."""
    if recommended is None:
        return (
            "**What this changes: nothing, and that is the result.** No INT8 "
            "engine measured here holds the operating point the deployed "
            "threshold was swept for, so the station keeps the float32 "
            "checkpoint and the torch path. The conversion is written, tested "
            "and reproducible -- `scripts/quantisation_report.py` rebuilds every "
            "artefact from the checkpoint -- so that the day this project meets "
            "a box that genuinely cannot hold "
            f"{baseline_mb:.0f}MB, the trade is one already priced rather than "
            "one to be discovered under deadline."
        )
    return (
        f"**What this changes.** {recommended} is the conversion that survives "
        f"the curve, and it takes the model off the disk from "
        f"{baseline_mb:.1f}MB to {baseline_mb - saved_mb:.1f}MB. It is not "
        f"deployed here, because this station is a laptop with a 43MB checkpoint "
        f"and no memory problem; it is measured so that a box which does have "
        f"one can be given a number rather than a hope. The deployed threshold "
        f"stays with the float32 model it was swept for -- an engine change is a "
        f"model change, and `DEFAULT_DISMISS_THRESHOLD` follows the model that "
        f"produced it."
    )


def latency_note(engines: list[dict], baseline_key: str) -> str:
    """Whether the conversion was even faster, said plainly either way."""
    by_key = {engine["key"]: engine for engine in engines}
    base = by_key[baseline_key]["single"]["p50"]
    parts = []
    for engine in engines:
        if engine["key"] == baseline_key:
            continue
        ratio = base / engine["single"]["p50"]
        direction = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        parts.append(f"{engine['label']} {format_ms(engine['single']['p50'])} "
                     f"({direction})")
    return (
        f"Against FP32 torch at {format_ms(base)}: " + "; ".join(parts) + ". "
        "Read these as the answer to \"does INT8 cost latency\" rather than as a "
        "reason to ship it: the FP32 figure was already 0.03% of a board's cycle "
        "and nothing downstream was waiting on it."
    )


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


def onnx_timed_series(runner, patches: np.ndarray, batch: int, repeats: int) -> list[float]:
    """`repeats` timed inferences of `batch` patches, mirroring `timed_series`.

    The timed path matches `reverifier_latency.classify_once` deliberately:
    uint8 to float, divide by 255, run, softmax. If one changes the other has
    to, or the two reports stop being comparable, which is the only reason
    either of them is worth writing.
    """
    total = len(patches)
    samples = []
    for index in range(repeats):
        start = (index * batch) % max(1, total - batch)
        chunk = np.ascontiguousarray(patches[start:start + batch])
        began = time.perf_counter()
        runner.probabilities(chunk)
        samples.append((time.perf_counter() - began) * 1000)
    return samples


def onnx_soak(runner, patches: np.ndarray, batch: int, seconds: float) -> list[tuple[float, float]]:
    """Sustained load, keeping each sample's offset so it can be split at 60s."""
    samples: list[tuple[float, float]] = []
    started = time.perf_counter()
    index = 0
    total = len(patches)
    while (time.perf_counter() - started) < seconds:
        start = (index * batch) % max(1, total - batch)
        chunk = np.ascontiguousarray(patches[start:start + batch])
        offset = time.perf_counter() - started
        began = time.perf_counter()
        runner.probabilities(chunk)
        samples.append((offset, (time.perf_counter() - began) * 1000))
        index += 1
    return samples


def score_split(runner, patches: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Every candidate on the split through one engine, in order."""
    return np.concatenate(
        [runner.probabilities(patches[start:start + chunk])
         for start in range(0, len(patches), chunk)]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--patches", type=Path, default=Path("data/patches/test.npz"))
    parser.add_argument("--trainval", type=Path,
                        default=Path("data/patches/trainval.npz"))
    parser.add_argument("--artefacts", type=Path, default=Path("models/onnx"))
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--batch-repeats", type=int, default=30)
    parser.add_argument("--soak", type=float, default=150.0)
    parser.add_argument("--soak-batch", type=int, default=8)
    parser.add_argument("--calibration-samples", type=int, default=CALIBRATION_SAMPLES)
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument("--dry-run", action="store_true", help="print, do not append")
    parser.add_argument("--allow-contended", action="store_true",
                        help="append even though something else was computing")
    args = parser.parse_args()

    import torch

    for path in (args.patches, args.trainval):
        if not path.exists():
            print(f"{path} not found; run scripts/build_patches.py", file=sys.stderr)
            return 1

    test_set = PatchSet.load(args.patches)
    patches, labels = test_set.patches, test_set.labels
    label_names = test_set.label_names
    false_call_index = label_names.index("false_call")
    print(f"{len(patches)} test-split patches, {patches.shape[1:]} each")

    per_board = np.bincount(test_set.image_index)
    per_board = per_board[per_board > 0]
    pipeline_batch = int(np.median(per_board))
    boards = int(len(per_board))

    own = os.getpid()
    ps_before = ollama_ps()
    busy_before = competing_processes(process_table(), own)
    print(f"\nollama ps before:\n{ps_before}\n")
    print("busy processes before:\n  "
          + ("\n  ".join(busy_before) if busy_before else "(none)") + "\n")
    contended = resident_models(ps_before) + busy_before
    if contended:
        print("WARNING: something else is computing on this machine:\n  "
              + "\n  ".join(contended), file=sys.stderr)

    # --- build the artefacts -------------------------------------------------
    args.artefacts.mkdir(parents=True, exist_ok=True)
    fp32_onnx = args.artefacts / "reverifier_fp32.onnx"
    preprocessed = args.artefacts / "reverifier_fp32_preprocessed.onnx"
    int8_dynamic = args.artefacts / "reverifier_int8_dynamic.onnx"
    int8_static = args.artefacts / "reverifier_int8_static.onnx"

    print("\nexporting to ONNX ...")
    export_onnx(args.checkpoint, fp32_onnx)
    preprocess_onnx(fp32_onnx, preprocessed)
    print("quantising, dynamic ...")
    quantise_dynamic(preprocessed, int8_dynamic)

    trainval = PatchSet.load(args.trainval)
    indices = calibration_indices(len(trainval), args.calibration_samples)
    print(f"quantising, static, on {len(indices)} trainval patches ...")
    quantise_static(preprocessed, int8_static, trainval.patches[indices])

    # --- the shape contract, before anything reads the numbers ---------------
    contract = ShapeContract(
        batch=4, channels=3, size=PATCH_SIZE, classes=len(label_names)
    )
    for path in (fp32_onnx, int8_dynamic, int8_static):
        probe = OnnxReVerifier(path).probabilities(patches[:contract.batch])
        problems = shape_violations(probe, contract)
        if problems:
            print(f"{path.name} fails the shape contract: {problems}", file=sys.stderr)
            return 1
    print("shape contract: all engines return a valid distribution per candidate")

    # --- score the whole split, every engine ---------------------------------
    verifier = ReVerifier(args.checkpoint, device="cpu")

    def torch_probabilities(chunk: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            batch = torch.from_numpy(chunk).float().div_(255.0)
            return torch.softmax(verifier.model(batch), dim=1).numpy()

    class _TorchRunner:
        def probabilities(self, chunk):
            return torch_probabilities(chunk)

    runners = {
        "fp32_torch": (_TorchRunner(), args.checkpoint.stat().st_size / (1024 * 1024)),
        "fp32_onnx": (OnnxReVerifier(fp32_onnx, threads=torch.get_num_threads()),
                      fp32_onnx.stat().st_size / (1024 * 1024)),
        "int8_dynamic": (OnnxReVerifier(int8_dynamic, threads=torch.get_num_threads()),
                         int8_dynamic.stat().st_size / (1024 * 1024)),
        "int8_static": (OnnxReVerifier(int8_static, threads=torch.get_num_threads()),
                        int8_static.stat().st_size / (1024 * 1024)),
    }
    labels_for = {
        "fp32_torch": "FP32 torch",
        "fp32_onnx": "FP32 ONNX",
        "int8_dynamic": "INT8 dynamic",
        "int8_static": "INT8 static",
    }

    print("\nscoring the test split ...")
    probabilities = {}
    for key, (runner, _) in runners.items():
        began = time.perf_counter()
        probabilities[key] = score_split(runner, patches)
        print(f"  {labels_for[key]:<13} {time.perf_counter() - began:>6.1f}s")

    points = {
        key: sweep(value[:, false_call_index], labels, false_call_index)
        for key, value in probabilities.items()
    }
    operating_points = {
        key: {
            str(budget): operating_row(best_at_escape_budget(value, budget), budget)
            for budget in BUDGETS
        }
        for key, value in points.items()
    }
    drift = {
        key: agreement(probabilities["fp32_torch"], value)
        for key, value in probabilities.items()
        if key != "fp32_torch"
    }

    print("\nreview removed at the ≤0.5% budget:")
    for key in runners:
        row = operating_points[key][str(HEADLINE_BUDGET)]
        print(f"  {labels_for[key]:<13} "
              + (f"{row['review_reduction']:.1%} at {row['escape_rate']:.2%} "
                 f"(threshold {row['threshold']:.3f})"
                 if row["reachable"] else "not reachable"))

    # --- latency, same loop for every engine ---------------------------------
    engines = []
    for key, (runner, size_mb) in runners.items():
        print(f"\n=== {labels_for[key]} ===")
        began = time.perf_counter()
        cold_runner = (
            _TorchRunner() if key == "fp32_torch"
            else OnnxReVerifier(
                {"fp32_onnx": fp32_onnx, "int8_dynamic": int8_dynamic,
                 "int8_static": int8_static}[key],
                threads=torch.get_num_threads(),
            )
        )
        load_ms = (time.perf_counter() - began) * 1000
        began = time.perf_counter()
        cold_runner.probabilities(np.ascontiguousarray(patches[:1]))
        cold_ms = (time.perf_counter() - began) * 1000
        print(f"  load {load_ms:.0f}ms   first inference {cold_ms:.2f}ms")

        onnx_timed_series(runner, patches, 1, WARMUP_ITERATIONS)
        single = summarise(onnx_timed_series(runner, patches, 1, args.iterations))
        print(f"  single p50 {single['p50']:.2f}ms  p90 {single['p90']:.2f}ms")

        batches = {}
        for size in SWEEP_BATCHES:
            stats = summarise(
                onnx_timed_series(runner, patches, size, max(5, args.batch_repeats))
            )
            per_candidate = stats["p50"] / size
            batches[size] = {
                "p50": stats["p50"],
                "per_candidate_ms": per_candidate,
                "throughput": 1000.0 / per_candidate,
            }
            print(f"  batch {size:>3}  {per_candidate:.3f}ms/cand")

        print(f"  soaking {args.soak:.0f}s at batch {args.soak_batch} ...")
        samples = onnx_soak(runner, patches, args.soak_batch, args.soak)
        early, late = thermal_split(samples)

        engines.append({
            "key": key,
            "label": labels_for[key],
            "size_mb": size_mb,
            "load_ms": load_ms,
            "cold_ms": cold_ms,
            "single": single,
            "batches": batches,
            "soak_early": summarise(early) if early else None,
            "soak_late": summarise(late) if late else None,
            "throttle": throttle_verdict(early, late),
            "rss_mb": peak_rss_mb(),
        })

    ps_after = ollama_ps()
    busy_after = competing_processes(process_table(), own)
    print(f"\nollama ps after:\n{ps_after}")
    print("busy processes after:\n  "
          + ("\n  ".join(busy_after) if busy_after else "(none)"))
    contended = contended or resident_models(ps_after) + busy_after

    payload = torch.load(args.checkpoint, map_location="cpu")
    parameters = sum(int(np.prod(v.shape)) for v in payload["state_dict"].values())

    by_key = {engine["key"]: engine for engine in engines}
    baseline_row = operating_points["fp32_torch"][str(HEADLINE_BUDGET)]
    verdicts = [
        trade_verdict(
            baseline_row,
            operating_points[key][str(HEADLINE_BUDGET)],
            labels_for[key],
            by_key["fp32_torch"]["single"]["p50"],
            by_key[key]["single"]["p50"],
            by_key["fp32_torch"]["size_mb"],
            by_key[key]["size_mb"],
        )
        for key in ("fp32_onnx", "int8_dynamic", "int8_static")
    ]

    int8_keys = ["int8_dynamic", "int8_static"]
    survivors = [
        key for key in int8_keys
        if (delta := review_reduction_delta(
            baseline_row, operating_points[key][str(HEADLINE_BUDGET)]
        )) is not None and delta >= -REVIEW_REDUCTION_TOLERANCE
    ]
    recommended = (
        max(survivors, key=lambda k: by_key["fp32_torch"]["size_mb"] - by_key[k]["size_mb"])
        if survivors else None
    )
    fastest_int8 = min(by_key[k]["single"]["p50"] for k in int8_keys)

    results = {
        "baseline_key": "fp32_torch",
        "engines": engines,
        "budgets": BUDGETS,
        "batch_sizes": SWEEP_BATCHES,
        "operating_points": operating_points,
        "agreement": drift,
        "parameters": parameters,
        "patch_count": len(patches),
        "cpu_threads": torch.get_num_threads(),
        "calibration_samples": int(len(indices)),
        "calibration_seed": CALIBRATION_SEED,
        "defects_total": int((labels != false_call_index).sum()),
        "pipeline_batch": pipeline_batch,
        "pipeline_batch_bucket": min(SWEEP_BATCHES,
                                     key=lambda b: abs(b - pipeline_batch)),
        "soak_s": args.soak,
        "thermal_split_s": 60.0,
        "ps_before": ps_before,
        "ps_after": ps_after,
        "busy_before": busy_before,
        "busy_after": busy_after,
        "verdicts": verdicts,
        "latency_note": latency_note(engines, "fp32_torch"),
        "line_rate": line_rate_implication(
            float(per_board.mean()), boards,
            by_key["fp32_torch"]["single"]["p50"], fastest_int8,
        ),
        "conclusion": conclusion(
            labels_for[recommended] if recommended else None,
            (by_key["fp32_torch"]["size_mb"] - by_key[recommended]["size_mb"])
            if recommended else 0.0,
            by_key["fp32_torch"]["size_mb"],
        ),
    }

    report = "\n".join(
        [f"## {date.today().isoformat()} · commit {git_commit()}", ""] + render(results)
    )
    print("\n" + report)

    if args.dry_run:
        return 0
    if contended and not args.allow_contended:
        print(f"\nNOT appended: {', '.join(contended)} was computing during the run. "
              f"A contended run is discarded, not published.", file=sys.stderr)
        return 1

    existing = args.out.read_text() if args.out.exists() else "# Benchmarks\n"
    args.out.write_text(existing.rstrip() + "\n\n" + report + "\n")
    print(f"\nappended to {args.out}")
    return 0


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "uncommitted"


if __name__ == "__main__":
    raise SystemExit(main())
