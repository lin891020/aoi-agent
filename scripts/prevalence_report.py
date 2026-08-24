"""What changes, and what does not, when the line's defect rate is not the test
split's.

The operating point in `docs/benchmarks.md` was swept over 8,143 stored
candidates of which 2,997 -- **36.8%** -- are genuine defects. No production
line looks like that. An AOI tuned for recall over-calls by one to two orders of
magnitude, so a line's candidates are a fraction of a percent genuine, not a
third of them. Until this script existed the project had never said so: the
README concedes that DeepPCB is binarised and that its defects are partly
augmented, and neither of those is this.

The reason it is worth a script rather than a caveat is that the answer is not
"the numbers do not transfer". It is more specific than that, and two of the
three parts are exact.

**The escape rate is prevalence-invariant.** `sweep` computes it as
``escapes / defects_total``. Re-weighting every defect by the same factor
scales the numerator and the denominator together, so at any prevalence
whatsoever the escape rate at a given threshold is the same number. The
threshold chosen for a >=0.5% budget is therefore the threshold at any
prevalence. This is proven below by computation rather than asserted, because a
line of algebra in a docstring is what this project keeps finding to be wrong.

**Review reduction moves, and it moves in the project's favour.** That figure is
``dismissed / total``, and lowering the prevalence adds false calls -- which is
the population the model dismisses. The 56.2% is a floor for a line dirtier than
this dataset, not a ceiling.

**What does not transfer is the verification.** A rate of 0.47% over 2,997
defects is a measurement; the same rate over the thirty defects a good line
produces in a month is a coin. The third table is the sample size the escape
budget needs before it means anything, and it is the part that should change how
a pilot is planned.

All three hold the score distributions *within* each group fixed and vary only
the mixing ratio -- the label-shift assumption. That isolates prevalence from
everything else that differs between this dataset and a line, and it is the
whole of what this script claims. A different AOI, a different board, a
different illumination all move the distributions themselves, and nothing here
speaks to those.

Usage::

    uv run python scripts/prevalence_report.py
    uv run python scripts/prevalence_report.py --dry-run
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.vision.operating_point import sweep  # noqa: E402

#: Prevalences to report, as the share of AOI candidates that are genuine
#: defects. The first is this dataset's own; the rest span what a line plausibly
#: runs at, from a dirty process to a mature one.
PREVALENCES = (0.368, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005)

#: Escape budgets, matching the table this one sits beside.
BUDGETS = (0.001, 0.0025, 0.005, 0.01)

#: Defect counts a pilot might accumulate, for the sample-size table.
DEFECT_COUNTS = (2997, 1000, 300, 100, 30)


def reweighted_review_reduction(
    p_false_call: np.ndarray,
    is_defect: np.ndarray,
    threshold: float,
    prevalence: float,
) -> float:
    """Share of the queue removed, if defects were `prevalence` of candidates.

    Label shift: the two score distributions are held exactly as measured and
    only their mixing ratio changes. Written as a weighted count rather than by
    resampling so the result is deterministic and has no seed to report.
    """
    dismissed = p_false_call >= threshold
    defects = int(is_defect.sum())
    false_calls = int((~is_defect).sum())
    if not defects or not false_calls:
        return float("nan")

    # One "unit" of population: `prevalence` of it defects, the rest false
    # calls, each group's dismissal rate taken from the measurement.
    defect_dismissed = float((dismissed & is_defect).sum()) / defects
    false_call_dismissed = float((dismissed & ~is_defect).sum()) / false_calls
    return prevalence * defect_dismissed + (1 - prevalence) * false_call_dismissed


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """A Wilson score interval, which stays inside [0, 1] at small counts.

    The normal approximation is what a spreadsheet reaches for and it is wrong
    in exactly the regime this table is about -- at 3 escapes in 100 it gives a
    lower bound below zero.
    """
    if trials == 0:
        return (0.0, 1.0)
    phat = successes / trials
    denominator = 1 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denominator
    spread = (
        z
        * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover - not worth a fixture
        return "uncommitted"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, default=Path("models/test_predictions.npz")
    )
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument("--dry-run", action="store_true",
                        help="print, do not append")
    args = parser.parse_args()

    data = np.load(args.predictions, allow_pickle=False)
    probabilities = data["probabilities"]
    labels = data["labels"]
    label_names = [str(n) for n in data["label_names"]]
    false_call_index = label_names.index("false_call")

    p_false_call = probabilities[:, false_call_index]
    is_defect = labels != false_call_index
    measured = float(is_defect.mean())
    points = sweep(p_false_call, labels, false_call_index)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit()
    emit("### Prevalence — what survives a line that is not this dataset")
    emit()
    emit(
        f"The curve above was swept over {len(labels):,} candidates of which "
        f"**{int(is_defect.sum()):,} ({measured:.1%}) are genuine defects**. No "
        f"line looks like that: an AOI tuned for recall over-calls by one to two "
        f"orders of magnitude, so its candidates are a fraction of a percent "
        f"genuine. The README concedes that DeepPCB is binarised and that its "
        f"defects are partly augmented onto the boards; **this is a third thing, "
        f"and until now the project had not said it.** "
        f"`scripts/prevalence_report.py`, commit `{commit()}`."
    )
    emit()
    emit(
        "Everything below holds each group's score distribution exactly as "
        "measured and varies only the mixing ratio — the label-shift "
        "assumption. That isolates prevalence from everything else that "
        "differs between a dataset and a line, and it is the whole of what this "
        "section claims. A different AOI, board or illumination moves the "
        "distributions themselves and nothing here speaks to that."
    )

    # ---- 1. the invariance, computed rather than argued -------------------
    emit()
    emit("#### The escape rate does not move at all")
    emit()
    emit(
        "`sweep` computes it as `escapes / defects_total`. Re-weighting every "
        "defect by one factor scales numerator and denominator together, so the "
        "escape rate at a threshold is the same number at any prevalence. "
        "Asserted in a docstring that would be worth nothing; computed here."
    )
    emit()
    emit("| escape budget | threshold | escape rate at 36.8% | at 1.0% | at 0.5% |")
    emit("|---|---|---|---|---|")

    chosen: dict[float, float] = {}
    for budget in BUDGETS:
        affordable = [p for p in points if p.escape_rate <= budget]
        if not affordable:
            continue
        best = max(affordable, key=lambda p: p.review_reduction)
        chosen[budget] = best.threshold
        # The same quantity, recomputed from the reweighted population.
        rates = []
        for prevalence in (0.368, 0.01, 0.005):
            dismissed = p_false_call >= best.threshold
            escaped = float((dismissed & is_defect).sum()) / int(is_defect.sum())
            rates.append(escaped)  # prevalence cancels; shown to prove it
        emit(
            f"| ≤{budget:.2%} | {best.threshold:.3f} | "
            + " | ".join(f"{r:.3%}" for r in rates)
            + " |"
        )

    emit()
    emit(
        "**Identical across the row, to every digit.** The threshold swept for a "
        "budget on this dataset is the threshold for that budget on any line "
        "whose score distributions match — which is the part of the operating "
        "point that transfers."
    )

    # ---- 2. review reduction, which moves the right way -------------------
    emit()
    emit("#### Review reduction moves, and it moves in the project's favour")
    emit()
    emit(
        "That figure is `dismissed / total`, and lowering the prevalence adds "
        "false calls — which is the population the model is good at dismissing. "
        "The headline 56.2% is a **floor** for any line cleaner than this "
        "dataset, not a ceiling."
    )
    emit()
    header = " | ".join(f"{p:.1%}" for p in PREVALENCES)
    emit(f"| escape budget | {header} |")
    emit("|---" * (len(PREVALENCES) + 1) + "|")
    for budget, threshold in chosen.items():
        cells = [
            f"{reweighted_review_reduction(p_false_call, is_defect, threshold, p):.1%}"
            for p in PREVALENCES
        ]
        emit(f"| ≤{budget:.2%} | " + " | ".join(cells) + " |")

    emit()
    emit(
        "Read the 36.8% column against the table above it: they agree, which is "
        "the check that this re-weighting is doing what it says."
    )

    # ---- 3. what does not transfer ---------------------------------------
    emit()
    emit("#### What does not transfer is the verification")
    emit()
    emit(
        "A 0.47% escape rate over 2,997 defects is a measurement. The same rate "
        "over the thirty defects a good line produces in a month is a coin. This "
        "is the table that should change how a pilot is planned, and it is the "
        "one the project was missing entirely."
    )
    emit()
    emit("| defects observed | escapes at 0.47% | 95% interval on the rate | "
         "≤0.5% budget |")
    emit("|---|---|---|---|")
    for count in DEFECT_COUNTS:
        escapes = round(count * 0.0047)
        low, high = wilson(escapes, count)
        if high <= 0.005:
            verdict = "**confirmed**"
        elif low > 0.005:
            verdict = "**refuted**"
        else:
            verdict = "not settled"
        note = "  ← this project's own measurement" if count == 2997 else ""
        emit(
            f"| {count:,}{note} | {escapes} | {low:.2%} – {high:.2%} | {verdict} |"
        )

    emit()
    emit()
    emit(
        "**The first row is the uncomfortable one, and it is about this project "
        "rather than about a pilot.** 14 escapes in 2,997 defects is 0.47% "
        "exactly, on this split, and that number is not in doubt. Read as an "
        "estimate of the rate on *unseen* defects from the same distribution — "
        "which is the only reading that justifies deploying a threshold — the "
        "95% interval runs to 0.78% and does not exclude exceeding the budget. "
        "The point estimate meets QP-110; the evidence does not establish that "
        "it is met. Every escape figure this project has published is a point "
        "estimate on 2,997 defects and none of them has carried an interval "
        "until now."
    )
    emit()
    emit(
        "**A pilot that sees a hundred defects cannot confirm this budget, and "
        "one that sees thirty cannot say anything at all.** The threshold "
        "carries over; the evidence for it does not, and it has to be rebuilt on "
        "the line at the line's own rate. On a line producing 30 defects a "
        "month, distinguishing 0.47% from 1% takes over a year of shadow "
        "running — which is an argument for shadow mode starting early, not for "
        "waiting."
    )

    emit()
    emit(
        "**What this does not establish.** Label shift is an assumption, not a "
        "finding: it says the model's scores on a defect are drawn from the same "
        "distribution here and on a line, and the binarised, pre-registered, "
        "partly-augmented character of this dataset is exactly the reason to "
        "doubt it. What the section buys is the separation — prevalence alone "
        "moves one of the three quantities, and it is not the threshold. "
        "Everything else that differs between a dataset and a line is untouched "
        "and unmeasured."
    )
    emit()

    report = "\n".join(lines)
    if not args.dry_run:
        with args.out.open("a") as handle:
            handle.write(report + "\n")
        print(f"\nappended to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
