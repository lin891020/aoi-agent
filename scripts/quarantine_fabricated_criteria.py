"""Mark the explanations written while the criteria retrieval was unscoped.

Between 2026-08-21 and 2026-08-22 the retrieval handed the reasoning model
WI-206's pin-hole limit -- "Within limits and outside pads: release. Inside a
pad: reject." -- alongside WI-201 whenever it asked about an `open`. The model
fused the two and told operators the criteria require establishing whether the
open is inside a pad "(critical) or outside". WI-201 says the opposite: any
confirmed open is a critical defect and there is no size or location below
which it is acceptable. The sentence points at releasing a critical defect.

Eight rows carry it: five escalation reasons and three agent rationales.

They are **marked, not rewritten and not deleted**.

- Not deleted, because this project has already learned what hand-deleting
  rows costs: five human decisions were removed from `review_decisions` and
  their `escalations` rows were left resolved, so the two tables still
  disagree about what happened to those regions. A quality record with a hole
  in it is worse than one with a correction on it.
- Not regenerated, because the `reason` on an escalation is the record of what
  the operator was shown when they answered. Replacing it with what today's
  retrieval would produce makes the record claim they read something nobody
  wrote until the day after they decided. The fix belongs in front of the next
  operator, not behind the last one.

So the original text is kept verbatim, under a banner that says what is wrong
with it. The banner is a fixed token, so the marking is detectable, countable
and idempotent -- running this twice marks nothing twice.

Re-runnable, needs no LLM and no GPU:

    uv run python scripts/quarantine_fabricated_criteria.py --dry-run
    uv run python scripts/quarantine_fabricated_criteria.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import select  # noqa: E402

from aoi_agent.store.boards import session_factory  # noqa: E402
from aoi_agent.store.models import (  # noqa: E402
    Board,
    CandidateRecord,
    Escalation,
    ReviewDecision,
)

#: The banner's first token. Fixed, because "is this row already marked" has to
#: be answerable without matching the prose, which is a model's and varies.
MARKER = "[QUARANTINED"

BANNER = (
    "[QUARANTINED 2026-08-23] The explanation below cites an acceptance rule "
    "that does not exist. WI-201: any confirmed open is a critical defect, "
    "with no size or location below which it is acceptable. The "
    "inside-a-pad-or-outside rule is WI-206's and governs pin holes only. It "
    "reached this text because the criteria retrieval was not scoped to the "
    "class it was asked about -- see docs/benchmarks.md. Kept verbatim below, "
    "because it is what the operator was shown.\n\n"
)

#: The claim itself. Both halves are required: a pin-hole explanation may talk
#: about pads correctly, and an open explanation may mention a pad in passing.
#: What is wrong is an open dispositioned *by* the pad.
_PAD = re.compile(r"\bpads?\b", re.IGNORECASE)
_LOCATION = re.compile(r"\b(inside|outside|within)\b", re.IGNORECASE)

#: The one class the pad rule genuinely belongs to. A pin-hole rationale citing
#: it is correct and must not be marked.
GOVERNED_BY_THE_PAD_RULE = "pin-hole"


def is_fabricated(text: str | None, defect_class: str | None) -> bool:
    """Does this explanation disposition a non-pin-hole defect by a pad?"""
    if not text or defect_class == GOVERNED_BY_THE_PAD_RULE:
        return False
    if text.startswith(MARKER):
        return False
    return bool(_PAD.search(text) and _LOCATION.search(text))


def mark(text: str) -> str:
    """Prepend the banner, once."""
    if text.startswith(MARKER):
        return text
    return BANNER + text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be marked and change nothing")
    args = parser.parse_args()

    marked = 0
    with session_factory()() as session:
        escalation_rows = session.execute(
            select(Escalation, CandidateRecord, Board)
            .join(CandidateRecord, Escalation.candidate_id == CandidateRecord.id)
            .join(Board, CandidateRecord.board_id == Board.id)
        ).all()
        for escalation, candidate, board in escalation_rows:
            if not is_fabricated(escalation.reason, candidate.predicted_class):
                continue
            print(f"escalations #{escalation.id} {board.stem}#"
                  f"{candidate.index_on_board} ({candidate.predicted_class})")
            marked += 1
            if not args.dry_run:
                escalation.reason = mark(escalation.reason)

        decision_rows = session.execute(
            select(ReviewDecision, CandidateRecord, Board)
            .join(CandidateRecord, ReviewDecision.candidate_id == CandidateRecord.id)
            .join(Board, CandidateRecord.board_id == Board.id)
        ).all()
        for decision, candidate, board in decision_rows:
            if not is_fabricated(decision.rationale, candidate.predicted_class):
                continue
            print(f"review_decisions #{decision.id} {board.stem}#"
                  f"{candidate.index_on_board} ({decision.source}, "
                  f"{candidate.predicted_class})")
            marked += 1
            if not args.dry_run:
                decision.rationale = mark(decision.rationale)

        if not args.dry_run:
            session.commit()

    print(f"\n{marked} row(s) {'would be ' if args.dry_run else ''}marked.")
    if marked and not args.dry_run:
        print("The checkpointer still holds each run's own state verbatim, "
              "which is what it is for: it records what the run did, and this "
              "records what was wrong with it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
