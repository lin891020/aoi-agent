"""Operating-point report on the held-out test split.

Produces the numbers that go in the README: how much of the manual review
queue the model removes at each escape budget the line might accept.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.vision.operating_point import (  # noqa: E402
    best_at_escape_budget,
    sweep,
    system_escape_rate,
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
    parser.add_argument(
        "--aoi-escape-rate",
        type=float,
        default=0.050,
        help="share of defects the AOI simulator never flagged on the test split",
    )
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
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
    emit()
    emit(f"Overall classification accuracy: {accuracy:.1%} "
         f"(reported for reference only — it weighs an escape the same as a false call)")
    emit()

    if headline:
        total = system_escape_rate(headline.escape_rate, args.aoi_escape_rate)
        emit("### Whole-line escape rate")
        emit()
        emit("The model only sees what the AOI stage flagged. Defects the AOI never")
        emit("caught are already gone and no threshold recovers them.")
        emit()
        emit(f"- AOI stage misses: **{args.aoi_escape_rate:.1%}** of defects")
        emit(f"- re-verification adds: {headline.escape_rate:.2%} of what reached it")
        emit(f"- **whole line: {total:.1%}**")
        emit()
        emit("The AOI stage dominates. Tuning the model past this point buys nothing")
        emit("until the detector's recall improves.")
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
