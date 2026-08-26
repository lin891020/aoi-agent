"""What does the difference image alone buy, with no model at all?

The headline claim is that the re-verifier removes 50.2% of the manual review
queue at a 0.25% escape budget. A number with no floor under it is not
evidence: an interviewer's first question is what the queue would look like if
you skipped the network entirely and thresholded the difference image, which is
what the AOI stage has already computed by the time a candidate exists.

So this is the floor. No weights, no training, no calibration set. For each
candidate it reduces channel 2 -- ``|test - template|``, built in
``vision/patches.build_patch`` -- to one scalar, maps it monotonically into
[0, 1] so that a *small* difference reads as a high P(false_call), and hands it
to the same ``operating_point.sweep`` the model is scored with. Same split,
same labels, same metric definitions, same escape budgets.

The model is then re-scored on the identical array, because two numbers from
two bases cannot be subtracted -- and when this was written (2026-08-26) the
README still quoted a superseded 8,143-candidate run while ``test.npz`` held
7,322. The basis is reconciled now; the re-scoring stays, because the next
drift will be as silent as that one was.

Run:
    uv run python scripts/model_free_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.vision.operating_point import (  # noqa: E402
    best_at_escape_budget,
    sweep,
)

BUDGETS = (0.0025, 0.0050, 0.0100)
ROOT = Path(__file__).resolve().parents[1]


def to_false_call_score(statistic: np.ndarray) -> np.ndarray:
    """Map a defect-ness statistic to a P(false_call)-shaped score.

    Monotone decreasing and min-max scaled, so ties survive and every
    achievable operating point stays reachable by the sweep.
    """
    lo, hi = float(statistic.min()), float(statistic.max())
    if hi == lo:
        return np.zeros_like(statistic, dtype=float)
    return (hi - statistic) / (hi - lo)


def statistics(diff: np.ndarray) -> dict[str, np.ndarray]:
    """Every plain reduction of the difference patch worth trying."""
    flat = diff.reshape(len(diff), -1).astype(np.float32)
    sorted_desc = np.sort(flat, axis=1)[:, ::-1]
    stats = {
        "mean |test-template|": flat.mean(axis=1),
        "max |test-template|": flat.max(axis=1),
        "sum of top 32 pixels": sorted_desc[:, :32].sum(axis=1),
        "sum of top 64 pixels": sorted_desc[:, :64].sum(axis=1),
        "sum of top 128 pixels": sorted_desc[:, :128].sum(axis=1),
    }
    for tau in (32, 64, 96, 128):
        stats[f"count of pixels > {tau}"] = (flat > tau).sum(axis=1).astype(np.float32)
    return stats


def row(name: str, points, labels) -> dict:
    out = {"name": name}
    for budget in BUDGETS:
        best = best_at_escape_budget(points, budget)
        out[budget] = best
    return out


def main() -> int:
    data = np.load(ROOT / "data/patches/test.npz", allow_pickle=False)
    patches, labels = data["patches"], data["labels"]
    names = [str(n) for n in data["label_names"]]
    false_call_index = names.index("false_call")
    boards = len(np.unique(data["image_index"]))

    n = len(labels)
    defects = int((labels != false_call_index).sum())
    print(f"Split: {n} candidates from {boards} boards "
          f"({defects} real defects, {n - defects} false calls)")
    print(f"Label names: {names}\n")

    results = []

    # --- the floor: no model ---
    for name, statistic in statistics(patches[:, 2]).items():
        points = sweep(to_false_call_score(statistic), labels, false_call_index)
        results.append(row(name, points, labels))

    # --- a second floor: dismiss nothing / dismiss everything are trivial, but
    #     "predict the majority class" is the other thing an interviewer names ---
    prior = np.full(n, 0.5)
    results.append(row("constant score (no signal)",
                       sweep(prior, labels, false_call_index), labels))

    # --- the model, re-scored on this exact array ---
    from aoi_agent.vision.quantise import OnnxReVerifier  # noqa: E402

    onnx = ROOT / "models/onnx/reverifier_fp32.onnx"
    if onnx.exists():
        runner = OnnxReVerifier(onnx)
        probs = np.concatenate(
            [runner.probabilities(patches[i:i + 512]) for i in range(0, n, 512)]
        )
        points = sweep(probs[:, false_call_index], labels, false_call_index)
        results.append(row("ResNet-18 re-verifier (ONNX fp32)", points, labels))
    else:
        print(f"!! {onnx} missing -- model row skipped\n")

    header = f"{'':<34}" + "".join(f"{b:>26.2%}" for b in BUDGETS)
    print(header)
    print(f"{'':<34}" + "".join(f"{'review removed (escape)':>26}" for _ in BUDGETS))
    print("-" * len(header))
    for r in results:
        line = f"{r['name']:<34}"
        for budget in BUDGETS:
            best = r[budget]
            cell = ("--" if best is None
                    else f"{best.review_reduction:.1%} ({best.escape_rate:.2%}, "
                         f"{best.escapes}/{best.defects_total})")
            line += f"{cell:>26}"
        print(line)

    print("\nRead the columns, not the row maxima: every value here is the best "
          "threshold *on this split*, which is the same overfit the model's own "
          "number carries. The comparison is fair because both sides inherit it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
