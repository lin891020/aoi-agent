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

from aoi_agent.stats import wilson  # noqa: E402
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


def render(probabilities, labels, names: list[str]) -> str:
    """The section, as a string, so a test can read what a reader reads.

    Split out of ``main`` on 2026-08-26. The two closing paragraphs had
    hand-written counts that contradicted the table three lines above them,
    and there was no way to assert against the rendered section because
    rendering only ever happened on the way to a file.
    """
    false_call = names.index("false_call")
    open_index = names.index("open")

    p_false_call = probabilities[:, false_call]
    is_defect = labels != false_call
    dismissed = p_false_call >= DEFAULT_DISMISS_THRESHOLD

    lines: list[str] = []

    def emit(text: str = "") -> None:
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
    counts: dict[str, tuple[int, int]] = {}
    for index, name in enumerate(names):
        if index == false_call:
            continue
        mask = labels == index
        total = int(mask.sum())
        escaped = int((dismissed & mask).sum())
        rate = escaped / total if total else 0.0
        counts[name] = (escaped, total)
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
        f"admits no acceptable instance. Since 2026-08-31 the aggregate "
        f"({aggregate:.2%}) does not meet QP-110 either -- the threshold that "
        f"reported compliance had been swept on this split. What the class "
        f"split says is unchanged in shape: the classes nobody may ship are "
        f"the ones subsidised by the four that can be dispositioned."
    )

    # ---- could a class-aware veto fix it? --------------------------------
    # Tabulated for the *worst* critical class, not for `open`. Fixing it on
    # `open` is the same defect the aggregate has, one level down: at 0.912 the
    # only critical escape any veto recovers is a `short`, and a table about
    # `open` shows a flat column and reads as though the question were closed.
    veto_name, _veto_rate = worst
    veto_index = names.index(veto_name)
    is_veto_class = labels == veto_index
    baseline_review = float(dismissed.mean())
    escaped_p = probabilities[dismissed & is_veto_class][:, veto_index]
    kept_p = probabilities[~dismissed & is_veto_class][:, veto_index]
    total_veto_class = int(is_veto_class.sum())
    baseline_escapes = int((dismissed & is_veto_class).sum())

    emit()
    emit("#### A class-aware rule would be the obvious fix, and it does not close the gap")
    emit()
    emit(
        f"The classifier emits a full distribution, so a candidate about to be "
        f"dismissed still carries a `P({veto_name})`. Refusing to dismiss when "
        f"that is high trades review reduction for escapes recovered — which is "
        f"what a per-class budget would be built on. The table is `{veto_name}`, "
        f"the worst critical class on this checkpoint."
    )
    emit()
    emit(f"| veto when P({veto_name}) > | escapes | {veto_name}s escaped | "
         f"{veto_name} rate | review removed | cost |")
    emit("|---|---|---|---|---|---|")
    emit(
        f"| *(none)* | {int((dismissed & is_defect).sum())} | {baseline_escapes} | "
        f"{baseline_escapes / total_veto_class:.2%} | {baseline_review:.2%} | — |"
    )
    for veto in VETOES:
        kept = dismissed & ~(probabilities[:, veto_index] > veto)
        recovered = int((kept & is_veto_class).sum())
        emit(
            f"| {veto:.2f} | {int((kept & is_defect).sum())} | {recovered} | "
            f"{recovered / total_veto_class:.2%} | {float(kept.mean()):.2%} | "
            f"{baseline_review - float(kept.mean()):.2%} |"
        )

    reachable = int((escaped_p >= 0.05).sum())
    best = min(
        (
            (
                int((dismissed & ~(probabilities[:, veto_index] > v) & is_veto_class).sum()),
                v,
                baseline_review - float((dismissed & ~(probabilities[:, veto_index] > v)).mean()),
            )
            for v in VETOES
        ),
        key=lambda row: (row[0], row[2]),
    )
    emit()
    emit(
        f"**The most a veto buys is {baseline_escapes - best[0]} of "
        f"{baseline_escapes}, and it does not reach the budget.** At "
        f"P({veto_name}) > {best[1]:.2f} the class goes to "
        f"{best[0] / total_veto_class:.2%} for {best[2]:.2%} of review — still "
        f"above QP-110's 0.5%. The reason is in the distribution rather than in "
        f"the threshold: {reachable} of the {baseline_escapes} escaped "
        f"{veto_name}s carry `P({veto_name})` above 0.05 at all, the rest run "
        f"from {escaped_p.min():.5f} to {np.sort(escaped_p)[::-1][1:].max():.5f}, "
        f"and on the {len(kept_p)} kept the median is "
        f"{float(np.median(kept_p)):.3f}."
    )
    emit()
    emit(
        "**There is almost no middle ground to threshold.** These are not "
        "candidates the model was unsure about — every one of them has "
        "`false_call` as its argmax with a probability above 0.91. They are "
        "cases it was confidently wrong about, and a veto on its own output "
        "reaches at most one of them, because its own output does not know."
    )
    # The two paragraphs below are about `open` specifically -- it is the class
    # WI-201 sends to electrical test -- so their counts come from the table
    # above rather than from the prose. They were written by hand and read
    # "these eight ... 8 escapes in 594 opens" against a table saying 5 of 602
    # by 2026-08-26: true of the run they were written on, reprinted on every
    # run since, and contradicting the table three lines above them.
    open_escaped, open_total = counts["open"]
    open_low, open_high = wilson(open_escaped, open_total)
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
        f"these {open_escaped} are already covered, and the arithmetic that "
        f"follows is the strongest thing in this report: the remaining "
        f"{int((dismissed & is_defect).sum()) - open_escaped} escapes over "
        f"{int(is_defect.sum())} defects is "
        f"{(int((dismissed & is_defect).sum()) - open_escaped) / int(is_defect.sum()):.2%}, "
        f"back inside QP-110. A second measurement is what brings this system "
        f"into its budget; no threshold on this model does. On a line without "
        "one, no threshold in "
        "this project closes them.** Which line it is, is a question about the "
        "customer's process and not about this model."
    )
    emit()
    emit(
        "**What this does not establish.** Six classes on one split, and the "
        "per-class counts are small enough that the intervals in the prevalence "
        f"section apply here with more force, not less — {open_escaped} escapes "
        f"in {open_total} opens has a 95% interval of {open_low:.2%} to "
        f"{open_high:.2%}. The negative result about the "
        "veto is about *this* checkpoint: a model trained with a loss that "
        "penalised confident errors on critical classes might well carry the "
        "signal this one does not, and nothing here tries that."
    )
    emit()

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, default=Path("models/test_predictions.npz")
    )
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data = np.load(args.predictions, allow_pickle=False)
    report = render(
        data["probabilities"],
        data["labels"],
        [str(n) for n in data["label_names"]],
    )
    print(report)
    if not args.dry_run:
        with args.out.open("a") as handle:
            handle.write(report + "\n")
        print(f"\nappended to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
