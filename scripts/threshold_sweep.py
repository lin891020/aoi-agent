"""What the two graph thresholds cost and what they buy.

``DEFAULT_DISMISS_THRESHOLD`` has had a sweep behind it since the first
benchmark. The two constants in ``graph/flow.py`` never did. ``ESCALATE_BELOW``
was documented as "the lowest threshold adding no escape to the budget" and
``CONFIDENT`` as "WI-300 decision authority", and neither claim was ever
measured: no script swept them and WI-300 states no such number. This is that
script.

Both are swept over the stored predictions for the official DeepPCB test split,
which is where ``routing_report.py`` reads from too, so it needs no GPU and no
model call. ``fragment`` ground truth is held out, as in training and in the
operating-point report.

Read as curves, not as points. A sweep that returns one number is the same
mistake in a smaller box: the value that wins on this split by three decimal
places is a value fitted to this split.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import select  # noqa: E402

from aoi_agent.graph.flow import CONFIDENT, ESCALATE_BELOW  # noqa: E402
from aoi_agent.store.boards import session_factory  # noqa: E402
from aoi_agent.store.models import Board, CandidateRecord  # noqa: E402
from aoi_agent.vision.inference import DEFAULT_DISMISS_THRESHOLD  # noqa: E402

ESCALATE_GRID = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.86, 0.87, 0.875,
                 0.88, 0.89, 0.90, 0.91, 0.915, 0.92, 0.95]
CONFIDENT_GRID = [0.70, 0.80, 0.85, 0.90, 0.915, 0.92, 0.95, 0.97, 0.99, 0.999]

#: Ground truths that are neither a defect nor a false call. A fragment is a
#: candidate that clips a real defect without covering it; it is held out of
#: training and it is held out here for the same reason -- scoring it either way
#: is a claim the label does not support.
NOT_SCORED = ("fragment",)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "uncommitted"


def is_defect(ground_truth: str | None) -> bool:
    return ground_truth is not None and ground_truth != "false_call"


def outcome(
    predicted: str, confidence: float, false_call: float, confident: float, escalate: float
) -> tuple[str, str, str | None]:
    """Mirror of the flow's two routers plus its four terminal nodes.

    Returns ``(route, disposition, verdict)``. ``confirm`` and ``decide`` both
    disposition on ``model_class`` -- that identity is what the ``CONFIDENT``
    sweep below turns on, so it is written out here rather than assumed.
    """
    if false_call >= DEFAULT_DISMISS_THRESHOLD:
        return "dismiss", "dismissed", "false_call"
    if confidence >= confident and predicted not in ("open", "false_call"):
        return "confirm", "defect_confirmed", predicted
    if confidence < escalate:
        return "escalate", "pending_operator", None
    return (
        "decide",
        "dismissed" if predicted == "false_call" else "defect_confirmed",
        predicted,
    )


def load_rows() -> list[tuple[str, float, float, str | None]]:
    with session_factory()() as session:
        rows = session.execute(
            select(
                CandidateRecord.predicted_class,
                CandidateRecord.confidence,
                CandidateRecord.false_call_probability,
                CandidateRecord.ground_truth,
            ).join(Board)
        ).all()
    return [tuple(row) for row in rows if row[3] not in NOT_SCORED]


def tally(rows, confident: float, escalate: float) -> dict[str, int]:
    counts = {
        "dismiss": 0, "confirm": 0, "decide": 0, "escalate": 0,
        "agent_dismissed": 0, "agent_escapes": 0, "model_escapes": 0,
        "decided_right": 0, "confirmed_right": 0,
    }
    for predicted, confidence, false_call, truth in rows:
        route, disposition, verdict = outcome(
            predicted, confidence, false_call, confident, escalate
        )
        counts[route] += 1
        if route == "dismiss" and is_defect(truth):
            counts["model_escapes"] += 1
        if route == "confirm" and verdict == truth:
            counts["confirmed_right"] += 1
        if route == "decide":
            if verdict == truth:
                counts["decided_right"] += 1
            if disposition == "dismissed":
                counts["agent_dismissed"] += 1
                if is_defect(truth):
                    counts["agent_escapes"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    rows = load_rows()
    if not rows:
        print("the store is empty; run scripts/seed_store.py", file=sys.stderr)
        return 1

    total = len(rows)
    defects = sum(1 for row in rows if is_defect(row[3]))
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit(f"### Threshold sweep — `ESCALATE_BELOW` and `CONFIDENT` "
         f"({date.today().isoformat()} · commit {git_commit()})")
    emit()
    emit(f"{total} stored candidates from the official DeepPCB test split "
         f"({defects} real defects), `fragment` held out. No GPU: the sweep reads "
         "the predictions already in the store, the same source "
         "`routing_report.py` uses.")
    emit()
    emit(f"Held fixed: `DEFAULT_DISMISS_THRESHOLD` = {DEFAULT_DISMISS_THRESHOLD}, "
         f"which by itself dismisses "
         f"{tally(rows, CONFIDENT, ESCALATE_BELOW)['model_escapes']} real defects "
         f"— the whole of QP-110's ≤0.5% escape budget. There is no room left in "
         "the budget for a second dismissing branch, so the criterion for "
         "`ESCALATE_BELOW` is zero *added* escapes, not a share of one.")
    emit()

    emit("#### `ESCALATE_BELOW` — the confidence at which a region goes to a person")
    emit()
    emit("The only way this branch can add an escape is `decide_node` dismissing: "
         "the classifier's class is `false_call`, so `confidence` *is* "
         "`P(false call)`, and the region sits in the band "
         "[`ESCALATE_BELOW`, `DEFAULT_DISMISS_THRESHOLD`). Everything else the "
         "branch does is confirm a defect or hand it over, and neither ships a board.")
    emit()
    emit("| `ESCALATE_BELOW` | escalated | decided | of those, dismissed | escapes added | line escape rate | decided class right |")
    emit("|---|---|---|---|---|---|---|")
    for value in ESCALATE_GRID:
        counts = tally(rows, CONFIDENT, value)
        line_escapes = counts["model_escapes"] + counts["agent_escapes"]
        emit(
            f"| {value:.3f} | {counts['escalate']} ({counts['escalate'] / total:.1%}) | "
            f"{counts['decide']} | {counts['agent_dismissed']} | "
            f"**{counts['agent_escapes']}** | {line_escapes / defects:.3%} | "
            f"{counts['decided_right'] / max(counts['decide'], 1):.1%} |"
        )
    emit()

    worst = max(
        (row[1] for row in rows
         if row[0] == "false_call"
         and row[2] < DEFAULT_DISMISS_THRESHOLD
         and is_defect(row[3])),
        default=0.0,
    )
    lowest = min(
        (v for v in ESCALATE_GRID if tally(rows, CONFIDENT, v)["agent_escapes"] == 0),
        default=None,
    )
    emit(f"The highest-confidence real defect this branch would dismiss carries "
         f"**{worst:.4f}**. So on this grid the lowest threshold adding no escape "
         f"is **{lowest:.3f}** — not 0.90, which the citation in "
         "`docs/architecture.md` claimed until this run and which clears the "
         f"same bar with {0.90 - worst:.3f} to spare.")
    emit()
    emit(f"Neither is the value to ship. {lowest:.3f} sits {lowest - worst:.4f} above the "
         "worst miss on this split: that is a threshold read off the test set at "
         "three decimal places, and the next lot's tail lands on top of it. And "
         "0.90 is a round number that happened to be conservative — it was never "
         "derived from anything, which is the finding, not the fix.")
    emit()
    emit(f"The value that needs no split at all is `DEFAULT_DISMISS_THRESHOLD` "
         f"({DEFAULT_DISMISS_THRESHOLD}). At or above it the band is empty by "
         "construction: a region the classifier calls `false_call` above that "
         "confidence was already dismissed upstream, so it never reaches "
         "`decide_node`. The agent branch may confirm a defect; it cannot dismiss "
         "one. That holds for any model and survives a retrain, where a swept "
         "number would have to be swept again and silently would not be.")
    emit()

    emit("#### `CONFIDENT` — the confidence at which the classifier's class skips the LLM")
    emit()
    emit("`confirm_node` and `decide_node` write the same verdict: `model_class`. "
         "So above `ESCALATE_BELOW` this threshold moves candidates between two "
         "paths that disposition them identically. It is a cost gate, not a "
         "decision gate — the sweep below is of LLM calls, and the "
         "\"dispositions changed\" column is what makes that claim checkable.")
    emit()
    emit("| `CONFIDENT` | confirmed without the LLM | that class right | reaching the LLM | escalated | escapes added | dispositions changed vs "
         f"{CONFIDENT} |")
    emit("|---|---|---|---|---|---|---|")
    baseline = [outcome(*row[:3], CONFIDENT, ESCALATE_BELOW)[1:] for row in rows]
    for value in CONFIDENT_GRID:
        counts = tally(rows, value, ESCALATE_BELOW)
        changed = sum(
            1 for row, was in zip(rows, baseline, strict=True)
            if outcome(*row[:3], value, ESCALATE_BELOW)[1:] != was
        )
        reaching = counts["decide"] + counts["escalate"]
        emit(
            f"| {value:.3f} | {counts['confirm']} | "
            f"{counts['confirmed_right'] / max(counts['confirm'], 1):.1%} | "
            f"{reaching} | {counts['escalate']} | {counts['agent_escapes']} | "
            f"**{changed}** |"
        )
    emit()
    emit("Zero dispositions change anywhere at or above `ESCALATE_BELOW`, and the "
         "escape column never moves. Below it the threshold stops being free: it "
         "starts confirming, unreviewed, regions the flow would have handed to a "
         "person. That is the one thing `CONFIDENT` must not do, and it is a "
         "constraint the code can hold rather than a number a sweep can pick — "
         "`CONFIDENT` must be at least `ESCALATE_BELOW`.")
    emit()
    emit("Within that constraint the choice buys an operator a written rationale "
         "on the record, at one 20B-model call each. It is a cost dial and the "
         "citation should say so; it is not a decision authority and WI-300 never "
         "gave it one.")
    emit()

    emit("#### What the constants are set to now")
    emit()
    counts = tally(rows, CONFIDENT, ESCALATE_BELOW)
    emit(f"| constant | value | escalated | reaching the LLM | escapes added |")
    emit("|---|---|---|---|---|")
    emit(f"| `ESCALATE_BELOW` | {ESCALATE_BELOW} | {counts['escalate']} "
         f"({counts['escalate'] / total:.1%}) | "
         f"{counts['decide'] + counts['escalate']} | {counts['agent_escapes']} |")
    emit(f"| `CONFIDENT` | {CONFIDENT} | — | — | — |")
    emit()
    if counts["agent_escapes"]:
        emit(f"**The configured `ESCALATE_BELOW` adds {counts['agent_escapes']} "
             "escapes on this split.** That is the thing this sweep exists to "
             "catch.")
        emit()

    report = "\n".join(lines)
    if args.no_write:
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = args.out.read_text() if args.out.exists() else "# Benchmarks\n"
    args.out.write_text(existing.rstrip() + "\n\n" + report + "\n")
    print(f"\nappended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
