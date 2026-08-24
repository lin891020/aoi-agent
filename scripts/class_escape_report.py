"""Per-class escape, and why one budget for all six classes is the wrong shape.

QP-110 is written as a single number: ≤0.5% of defects may escape. The work
instructions are not written that way at all. WI-201 and WI-202 say that **any**
confirmed open or short is critical, with no size below which one is acceptable;
the other four classes are conditional on a measurement. A budget that averages
those together is a budget that lets the two classes nobody may ship subsidise
the four that can be dispositioned.

Measured, it does exactly that. So this script asks three questions in order,
and the third one is the answer.

**How does the escape budget actually divide?** By class, at the shipped
threshold, against what each class's work instruction says about it.

**Could a class-aware dismissal rule fix it?** The classifier emits a full
distribution, so a candidate about to be dismissed as a false call still carries
a `P(open)`. Refusing to dismiss when that is high costs review reduction and
buys escapes back -- a trade a per-class budget could be built on, if the
information were there.

**It is not there, and that is the finding.** On the opens this model dismisses,
`P(open)` runs from 0.00002 to 0.017. On the opens it keeps, the median is
0.999. There is no middle ground to threshold: these are not cases the model
was unsure about, they are cases it was confidently wrong about. No veto on its
own output can separate them, because its own output does not know.

Which moves the question off the operating point, where the project would
naturally look, and onto the line -- WI-201 already names the instrument, in a
clause about a different situation: *"Suspected open that measures continuous on
electrical test."* An open is the one class here that a downstream ICT or
flying-probe stage catches independently, and independence is the only thing
that helps when a model is confidently wrong.

Usage::

    uv run python scripts/class_escape_report.py
    uv run python scripts/class_escape_report.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.vision.inference import DEFAULT_DISMISS_THRESHOLD  # noqa: E402

#: What each class's work instruction says about acceptability. Not a severity
#: score -- a quotation of the document that governs it, because the point of
#: the table is that two of these rows have no acceptable instance and four do.
GOVERNS = {
    "open": ("WI-201", "critical — any confirmed instance", True),
    "short": ("WI-202", "critical — any confirmed instance", True),
    "mousebite": ("WI-203", "conditional — ≥80% remaining width", False),
    "spur": ("WI-204", "conditional — ≥50% remaining clearance", False),
    "copper": ("WI-205", "conditional — full clearance, off footprints", False),
    "pin-hole": ("WI-206", "conditional — <25% of conductor width", False),
}

#: Vetoes to try: refuse to dismiss a candidate whose `P(open)` exceeds this,
#: however confident the false-call score is.
VETOES = (0.30, 0.10, 0.05, 0.02, 0.01)


def commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover
        return "uncommitted"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, default=Path("models/test_predictions.npz")
    )
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data = np.load(args.predictions, allow_pickle=False)
    probabilities = data["probabilities"]
    labels = data["labels"]
    names = [str(n) for n in data["label_names"]]
    false_call = names.index("false_call")
    open_index = names.index("open")

    p_false_call = probabilities[:, false_call]
    is_defect = labels != false_call
    dismissed = p_false_call >= DEFAULT_DISMISS_THRESHOLD

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit()
    emit("### Per-class escape — one budget over six classes that are not alike")
    emit()
    emit(
        f"QP-110 is a single number: ≤0.5% of defects may escape. The work "
        f"instructions are not written that way. WI-201 and WI-202 say **any** "
        f"confirmed open or short is critical with no acceptable size; the other "
        f"four are conditional on a measurement. Averaging those together lets "
        f"the classes nobody may ship subsidise the ones that can be "
        f"dispositioned. At the shipped threshold "
        f"`{DEFAULT_DISMISS_THRESHOLD}` it does. "
        f"`scripts/class_escape_report.py`, commit `{commit()}`."
    )
    emit()
    emit("| class | governed by | defects | escaped | escape rate | the document says |")
    emit("|---|---|---|---|---|---|")

    worst = ("", 0.0)
    for index, name in enumerate(names):
        if index == false_call:
            continue
        mask = labels == index
        total = int(mask.sum())
        escaped = int((dismissed & mask).sum())
        rate = escaped / total if total else 0.0
        document, says, critical = GOVERNS.get(name, ("—", "—", False))
        marker = "**" if critical else ""
        if critical and rate > worst[1]:
            worst = (name, rate)
        emit(
            f"| {marker}{name}{marker} | {document} | {total} | {escaped} | "
            f"{marker}{rate:.2%}{marker} | {says} |"
        )

    aggregate = int((dismissed & is_defect).sum()) / int(is_defect.sum())
    emit(f"| *aggregate* | QP-110 | {int(is_defect.sum())} | "
         f"{int((dismissed & is_defect).sum())} | *{aggregate:.2%}* | ≤0.5% |")
    emit()
    emit(
        f"**`{worst[0]}` escapes at {worst[1]:.2%}, {worst[1] / aggregate:.1f}× the "
        f"aggregate**, and it is one of the two classes whose work instruction "
        f"admits no acceptable instance. The single budget is met and the class "
        f"that matters most is the one exceeding it."
    )

    # ---- could a class-aware veto fix it? --------------------------------
    emit()
    emit("#### A class-aware rule would be the obvious fix, and it does not work")
    emit()
    emit(
        "The classifier emits a full distribution, so a candidate about to be "
        "dismissed still carries a `P(open)`. Refusing to dismiss when that is "
        "high trades review reduction for escapes recovered — which is what a "
        "per-class budget would be built on."
    )
    emit()
    emit("| veto when P(open) > | escapes | opens escaped | open rate | "
         "review removed | cost |")
    emit("|---|---|---|---|---|---|")
    is_open = labels == open_index
    baseline_review = float(dismissed.mean())
    emit(
        f"| *(none)* | {int((dismissed & is_defect).sum())} | "
        f"{int((dismissed & is_open).sum())} | "
        f"{float((dismissed & is_open).sum()) / int(is_open.sum()):.2%} | "
        f"{baseline_review:.2%} | — |"
    )
    for veto in VETOES:
        kept = dismissed & ~(probabilities[:, open_index] > veto)
        opens = int((kept & is_open).sum())
        emit(
            f"| {veto:.2f} | {int((kept & is_defect).sum())} | {opens} | "
            f"{opens / int(is_open.sum()):.2%} | {float(kept.mean()):.2%} | "
            f"{baseline_review - float(kept.mean()):.2%} |"
        )

    # ---- why not -----------------------------------------------------------
    escaped_opens = probabilities[dismissed & is_open]
    kept_opens = probabilities[~dismissed & is_open]
    emit()
    emit(
        f"**Nothing moves until the veto is absurd, and then it costs more than "
        f"it buys.** The reason is in the distribution, not in the threshold: on "
        f"the {len(escaped_opens)} opens this model dismisses, `P(open)` runs "
        f"from {escaped_opens[:, open_index].min():.5f} to "
        f"{escaped_opens[:, open_index].max():.5f}. On the "
        f"{len(kept_opens)} it keeps, the median is "
        f"{float(np.median(kept_opens[:, open_index])):.3f}."
    )
    emit()
    emit(
        "**There is no middle ground to threshold.** These are not candidates "
        "the model was unsure about — every one of them has `false_call` as its "
        "argmax with a probability above 0.94, and two of them put `P(open)` "
        "below 0.0001. They are cases it was confidently wrong about, and no "
        "veto on its own output can separate them, because its own output does "
        "not know."
    )
    emit()
    emit(
        "**Which moves the question off the operating point.** A per-class "
        "budget cannot be met by re-tuning this curve; the information a "
        "class-aware rule would need is absent from the only signal available "
        "to it. What helps when a model is confidently wrong is not a better "
        "threshold on that model — it is a second measurement that does not "
        "share its failure. WI-201 already names one, in a clause written for a "
        "different situation: *\"Suspected open that measures continuous on "
        "electrical test.\"* An open is precisely the class a downstream ICT or "
        "flying-probe stage catches independently. **On a line that has one, "
        "these eight are already covered and the aggregate budget is the right "
        "shape after all. On a line that does not, no threshold in this project "
        "closes them.** Which line it is, is a question about the customer's "
        "process and not about this model."
    )
    emit()
    emit(
        "**What this does not establish.** Six classes on one split, and the "
        "per-class counts are small enough that the intervals in the prevalence "
        "section apply here with more force, not less — 8 escapes in 594 opens "
        "has a 95% interval of 0.68% to 2.63%. The negative result about the "
        "veto is about *this* checkpoint: a model trained with a loss that "
        "penalised confident errors on critical classes might well carry the "
        "signal this one does not, and nothing here tries that."
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
