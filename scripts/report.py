"""Operating-point report on the held-out test split.

Produces the numbers that go in the README: how much of the manual review
queue the model removes at each escape budget the line might accept.

Two readings, and they are not interchangeable:

* the **sweep table** gives each budget the best threshold this split can
  reach. That is a fair comparison between engines -- every engine gets its
  own oracle -- and it is what this file has always printed.
* the **deployment row** reports what one already-chosen threshold does here.
  Pass `--threshold`, or let it read `models/cv_threshold.json`, which
  `scripts/threshold_cv.py` writes from out-of-fold predictions that never
  touched this split. A threshold swept on the test predictions has seen the
  answers, and until 2026-08-31 that was the number this project shipped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.stats import wilson  # noqa: E402
from aoi_agent.vision.operating_point import (  # noqa: E402
    best_at_escape_budget,
    sweep,
)

BUDGETS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "uncommitted"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, default=Path("models/test_predictions.npz")
    )
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument("--threshold", type=float, default=None,
                        help="the deployed threshold; default reads models/cv_threshold.json")
    parser.add_argument("--cv", type=Path, default=Path("models/cv_threshold.json"))
    args = parser.parse_args()

    data = np.load(args.predictions, allow_pickle=False)
    probabilities = data["probabilities"]
    labels = data["labels"]
    label_names = [str(n) for n in data["label_names"]]
    false_call_index = label_names.index("false_call")

    p_false_call = probabilities[:, false_call_index]
    points = sweep(p_false_call, labels, false_call_index)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    accuracy = float((probabilities.argmax(1) == labels).mean())
    is_defect = labels != false_call_index

    emit(f"## {date.today().isoformat()} · commit {git_commit()}")
    emit()
    emit("Model: ResNet-18, 3x64x64 (template / test / difference), 10 epochs")
    emit("Hardware: MacBook Air M5, 32GB, MPS")
    emit(f"Test split: {len(labels)} AOI candidates from 499 unseen boards "
         f"({int(is_defect.sum())} real defects, {int((~is_defect).sum())} false calls)")
    emit()
    emit("### Operating points")
    emit()
    emit("Every candidate goes to a human today. The model dismisses the ones it is")
    emit("confident are false calls; the rest still go to a human.")
    emit()
    emit("| escape budget | achieved escape rate | manual review removed | escapes | false calls dismissed |")
    emit("|---|---|---|---|---|")

    for budget in BUDGETS:
        best = best_at_escape_budget(points, budget)
        if best is None:
            emit(f"| ≤{budget:.2%} | — | not reachable | — | — |")
            continue
        emit(
            f"| ≤{budget:.2%} | {best.escape_rate:.2%} | "
            f"**{best.review_reduction:.1%}** | "
            f"{best.escapes}/{best.defects_total} | "
            f"{best.false_calls_dismissed}/{best.false_calls_total} "
            f"({best.false_call_recall:.1%}) |"
        )

    headline = best_at_escape_budget(points, 0.005)

    deployed, source = args.threshold, "given on the command line"
    if deployed is None and args.cv.exists():
        import json

        record = json.loads(args.cv.read_text())
        chosen = record.get("upper_bound") or record.get("point_estimate")
        if chosen:
            deployed = chosen["threshold"]
            source = (f"out-of-fold over trainval, {record['folds']} folds, chosen on the "
                      f"95% interval's upper bound at ≤{record['budget']:.2%} "
                      f"({chosen['escapes']}/{chosen['defects_total']} out-of-fold escapes)")

    if deployed is not None:
        point = sweep(p_false_call, labels, false_call_index,
                      thresholds=np.array([deployed]))[0]
        low, high = wilson(point.escapes, point.defects_total)
        emit()
        emit("### The deployed threshold, read on this split")
        emit()
        emit("This is the row the README quotes. The threshold was chosen without")
        emit("seeing these labels; the table above gives each budget the best")
        emit("threshold *this* split can reach, which is an oracle and is not")
        emit("deployable. Both are printed because the gap between them is the")
        emit("price of choosing honestly.")
        emit()
        emit("| | threshold | escape rate | 95% interval | escapes | manual review removed |")
        emit("|---|---|---|---|---|---|")
        emit(f"| **deployed** ({source}) | {deployed:.4f} | {point.escape_rate:.3%} | "
             f"{low:.2%}–{high:.2%} | {point.escapes}/{point.defects_total} | "
             f"**{point.review_reduction:.2%}** |")
        if headline:
            oracle_low, oracle_high = wilson(headline.escapes, headline.defects_total)
            emit(f"| oracle on this split (not deployable) | {headline.threshold:.4f} | "
                 f"{headline.escape_rate:.3%} | {oracle_low:.2%}–{oracle_high:.2%} | "
                 f"{headline.escapes}/{headline.defects_total} | "
                 f"{headline.review_reduction:.2%} |")
        emit()

    emit()
    emit(f"Overall classification accuracy: {accuracy:.1%} "
         f"(reported for reference only — it weighs an escape the same as a false call)")
    emit()

    if headline:
        emit("### Whole-line escape rate")
        emit()
        emit("Not computed here. This script reads `test_predictions.npz`, which")
        emit("holds one row per *candidate* and excludes every candidate labelled")
        emit("`fragment` -- so it cannot see whether anything was flagged on a")
        emit("given defect, which is exactly the question a line escape rate asks.")
        emit("Composing one from an AOI miss rate handed in on the command line is")
        emit("what produced the 5.4% this project published until 2026-08-23, and")
        emit("that number was wrong by an order of magnitude.")
        emit()
        emit("Run `scripts/escape_accounting.py`. It accounts per defect rather")
        emit("than per box, with the model in the loop, and reports the two figures")
        emit("separately: what the dismissal threshold governs, and what nothing")
        emit("recovers.")
        emit()

        emit("### Where the escapes are")
        emit()
        dismissed = p_false_call >= headline.threshold
        emit(f"At the ≤0.5% budget (threshold {headline.threshold:.3f}):")
        emit()
        emit("| defect class | in test set | escaped | escape rate |")
        emit("|---|---|---|---|")
        for index, name in enumerate(label_names):
            if index == false_call_index:
                continue
            mask = labels == index
            escaped = int((dismissed & mask).sum())
            count = int(mask.sum())
            emit(f"| {name} | {count} | {escaped} | {escaped / count:.2%} |")
        emit()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = args.out.read_text() if args.out.exists() else "# Benchmarks\n"
    args.out.write_text(existing.rstrip() + "\n\n" + "\n".join(lines) + "\n")
    print(f"\nappended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
