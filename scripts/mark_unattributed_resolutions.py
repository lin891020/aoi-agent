"""Mark the five queue entries closed by a deletion rather than by an answer.

``escalations`` and ``review_decisions`` are meant to agree. ``resume_review``
writes the operator's verdict *first* and closes the queue entry second, so a
crash between the two leaves the region on the queue and it gets looked at
again -- the other order drops a verdict silently. Five rows in this store are
closed with no human decision anywhere beneath them.

They are not a bug in that ordering. They are what is left of an incident this
project already has a record of: five regions were clicked through by someone
without the domain knowledge to judge them, four of the five labels were wrong,
and the labels were removed from ``review_decisions`` by hand. The queue rows
were not touched, so since 2026-08-22 the two tables have disagreed about what
happened to those five regions, and every count taken from either one has been
answering a slightly different question.

They are **marked, not repaired**, on the precedent this repository set the day
before with ``quarantine_fabricated_criteria.py``:

- Not deleted. Deleting rows is what caused this. A quality record with a hole
  in it is worse than one with a correction on it, and a second hand-deletion
  to tidy away the first is the same mistake with more confidence.
- Not re-resolved into ``pending``. That would put five regions back in front
  of an operator as though they had never been looked at, three days after the
  boards they belong to left the line, and it would quietly assert that the
  first review never happened. It did happen; what it did not do is leave an
  attributable answer.
- Not back-filled with a synthetic human decision. Inventing the label that was
  deleted is the one thing worse than losing it: it would be indistinguishable
  from an operator's judgement and it would feed the next training round.

So the status becomes ``resolved_unattributed`` -- a third state, which the
station cannot write and which no query treats as pending -- and the reason
carries a banner saying what is wrong with the row. The banner is a fixed
token, so the marking is detectable, countable and idempotent.

Re-runnable, needs no LLM and no GPU:

    uv run python scripts/mark_unattributed_resolutions.py --dry-run
    uv run python scripts/mark_unattributed_resolutions.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import func, select  # noqa: E402

from aoi_agent.store.boards import session_factory  # noqa: E402
from aoi_agent.store.escalations import (  # noqa: E402
    RESOLVED,
    RESOLVED_UNATTRIBUTED,
)
from aoi_agent.store.models import (  # noqa: E402
    Board,
    CandidateRecord,
    Escalation,
    ReviewDecision,
)

#: The banner's first token. Fixed, so "is this row already marked" is
#: answerable without matching prose.
MARKER = "[UNATTRIBUTED"

BANNER = (
    "[UNATTRIBUTED 2026-08-23] This queue entry is closed, and no human "
    "decision exists for the region beneath it. The station cannot produce "
    "that combination -- it writes the verdict before it closes the entry -- "
    "so this row is the residue of five operator labels deleted by hand on "
    "2026-08-22, after five regions were clicked through without the domain "
    "knowledge to judge them and four of the five were wrong. The deletion was "
    "right; leaving these rows closed was not. The region was reviewed and the "
    "review is not attributable to anyone. Kept verbatim below, because it is "
    "what the operator was shown.\n\n"
)


def is_unattributed(status: str, human_decisions: int) -> bool:
    """Is this a closed entry that no human decision backs?

    Both halves are required. A closed entry with a decision beneath it is the
    ordinary case; a *pending* entry with no decision is the ordinary case too,
    and marking either would teach a reader to ignore the banner.
    """
    return status in (RESOLVED, RESOLVED_UNATTRIBUTED) and human_decisions == 0


def needs_marking(status: str, reason: str | None, human_decisions: int) -> bool:
    """The same question, with "have I already done this" folded in."""
    if not is_unattributed(status, human_decisions):
        return False
    return not (reason or "").startswith(MARKER)


def mark(text: str) -> str:
    """Prepend the banner, once."""
    if text.startswith(MARKER):
        return text
    return BANNER + text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be marked and change nothing")
    # Parsed from an argument rather than from ``sys.argv``, so the suite can
    # drive both modes without reaching into the interpreter's state.
    args = parser.parse_args(argv)

    marked = 0
    already = 0
    with session_factory()() as session:
        rows = session.execute(
            select(Escalation, CandidateRecord, Board)
            .join(CandidateRecord, Escalation.candidate_id == CandidateRecord.id)
            .join(Board, CandidateRecord.board_id == Board.id)
            .order_by(Escalation.id)
        ).all()

        for escalation, candidate, board in rows:
            human = session.execute(
                select(func.count(ReviewDecision.id)).where(
                    ReviewDecision.candidate_id == candidate.id,
                    ReviewDecision.source == "human",
                )
            ).scalar()
            if not is_unattributed(escalation.status, human):
                continue
            reference = f"{board.stem}#{candidate.index_on_board}"
            if not needs_marking(escalation.status, escalation.reason, human):
                already += 1
                print(f"escalations #{escalation.id} {reference} -- already marked")
                continue

            print(f"escalations #{escalation.id} {reference} "
                  f"({candidate.predicted_class}, resolved "
                  f"{escalation.resolved_at}) -- no human decision")
            marked += 1
            if not args.dry_run:
                escalation.status = RESOLVED_UNATTRIBUTED
                escalation.reason = mark(escalation.reason)

        if not args.dry_run:
            session.commit()

    print(f"\n{marked} row(s) {'would be ' if args.dry_run else ''}marked"
          f"{f', {already} already marked' if already else ''}.")
    if marked and not args.dry_run:
        print("The regions were not re-queued and no decision was invented. "
              "The record now says a review happened that cannot be attributed "
              "to anyone, which is what happened.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
