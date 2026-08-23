"""The review station's work queue.

An escalation is a graph run that stopped at ``interrupt`` and is waiting for a
person. The checkpointer holds the run; these rows hold the fact that someone
still owes it an answer, which is the one question the checkpointer cannot be
asked.

Rows are written by the review service when a run comes back suspended, not by
the graph itself. ``interrupt`` re-runs its node on resume, so a write inside
``escalate_node`` would fire twice for one escalation.
"""

from __future__ import annotations

from sqlalchemy import func, select

from aoi_agent.store.boards import session_factory
from aoi_agent.store.models import Board, CandidateRecord, Escalation


#: The queue's two ordinary states, and the one that is a mark on the record.
PENDING = "pending"
RESOLVED = "resolved"

#: A queue entry closed as answered whose candidate carries no human decision.
#:
#: Not a state the station can reach: ``resume_review`` writes the decision
#: before it closes the entry, precisely so a crash between the two leaves the
#: region on the queue rather than losing the verdict. Five rows in this store
#: are in it anyway -- the residue of five operator labels deleted by hand in
#: August 2026, which left ``escalations`` and ``review_decisions`` permanently
#: disagreeing about what happened to those regions. Marked rather than
#: repaired; see ``scripts/mark_unattributed_resolutions.py``.
RESOLVED_UNATTRIBUTED = "resolved_unattributed"


def _as_dict(escalation: Escalation, candidate: CandidateRecord, board: Board) -> dict:
    return {
        "reference": f"{board.stem}#{candidate.index_on_board}",
        "board_stem": board.stem,
        "index": candidate.index_on_board,
        "thread_id": escalation.thread_id,
        "reason": escalation.reason,
        "agent_verdict": escalation.agent_verdict,
        "explanation_status": escalation.explanation_status,
        "status": escalation.status,
        "raised_at": escalation.raised_at.isoformat() if escalation.raised_at else None,
        "model_class": candidate.predicted_class,
        "model_confidence": candidate.confidence,
        "false_call_probability": candidate.false_call_probability,
        "lot_id": board.lot_id,
        "line_id": board.line_id,
        "machine_id": board.machine_id,
        "shift": board.shift,
    }
    # Deliberately absent: ``candidate.ground_truth``. The operator's answer is
    # the next training round's label, so showing them the answer first would
    # not collect a judgement, it would collect an echo.


def _resolve(session, reference: str) -> CandidateRecord | None:
    if "#" not in reference:
        return None
    stem, _, index = reference.partition("#")
    if not index.isdigit():
        return None
    return session.execute(
        select(CandidateRecord)
        .join(Board)
        .where(Board.stem == stem, CandidateRecord.index_on_board == int(index))
    ).scalar()


def raise_escalation(
    reference: str,
    thread_id: str,
    reason: str,
    agent_verdict: str | None = None,
    explanation_status: str | None = None,
) -> bool:
    """Put a suspended run on the queue, or refresh one already there.

    Re-running a candidate that is still pending updates the reason in place
    rather than queueing it twice: it is one region and it needs one answer.
    """
    with session_factory()() as session:
        candidate = _resolve(session, reference)
        if candidate is None:
            return False

        existing = session.execute(
            select(Escalation).where(Escalation.thread_id == thread_id)
        ).scalar()
        if existing is not None:
            existing.reason = reason
            existing.agent_verdict = agent_verdict
            existing.explanation_status = explanation_status
            existing.status = PENDING
            existing.resolved_at = None
        else:
            session.add(
                Escalation(
                    candidate_id=candidate.id,
                    thread_id=thread_id,
                    reason=reason,
                    agent_verdict=agent_verdict,
                    explanation_status=explanation_status,
                    status=PENDING,
                )
            )
        session.commit()
    return True


def resolve_escalation(thread_id: str) -> bool:
    """Mark one escalation answered. The verdict itself goes to
    ``review_decisions`` -- this only closes the queue entry."""
    with session_factory()() as session:
        row = session.execute(
            select(Escalation).where(Escalation.thread_id == thread_id)
        ).scalar()
        if row is None:
            return False
        row.status = RESOLVED
        # ``func.now()``, not ``datetime.now()``. Every other timestamp in the
        # schema is stamped by the database, which for SQLite is UTC; a Python
        # local-time value here would put two columns of one row eight hours
        # apart on this machine and read as an escalation resolved before it was
        # raised. These are quality records -- the clock has to be one clock.
        row.resolved_at = func.now()
        session.commit()
    return True


def pending(limit: int = 200) -> list[dict]:
    """The queue, oldest first: whoever has waited longest goes next."""
    with session_factory()() as session:
        rows = session.execute(
            select(Escalation, CandidateRecord, Board)
            .join(CandidateRecord, Escalation.candidate_id == CandidateRecord.id)
            .join(Board, CandidateRecord.board_id == Board.id)
            .where(Escalation.status == PENDING)
            .order_by(Escalation.raised_at.asc(), Escalation.id.asc())
            .limit(limit)
        ).all()
        return [_as_dict(*row) for row in rows]


def get(thread_id: str) -> dict | None:
    with session_factory()() as session:
        row = session.execute(
            select(Escalation, CandidateRecord, Board)
            .join(CandidateRecord, Escalation.candidate_id == CandidateRecord.id)
            .join(Board, CandidateRecord.board_id == Board.id)
            .where(Escalation.thread_id == thread_id)
        ).first()
        return _as_dict(*row) if row else None


def unattributed_resolutions() -> list[dict]:
    """Queue entries closed as answered that no human decision backs.

    The two tables are meant to agree: an entry is closed only after the
    verdict that closed it is written. Where they do not, the disagreement is
    the finding, and this is how it is asked about rather than discovered.
    """
    from aoi_agent.store.models import ReviewDecision

    with session_factory()() as session:
        rows = session.execute(
            select(Escalation, CandidateRecord, Board)
            .join(CandidateRecord, Escalation.candidate_id == CandidateRecord.id)
            .join(Board, CandidateRecord.board_id == Board.id)
            .where(Escalation.status.in_((RESOLVED, RESOLVED_UNATTRIBUTED)))
            .order_by(Escalation.id)
        ).all()

        found = []
        for escalation, candidate, board in rows:
            human = session.execute(
                select(func.count(ReviewDecision.id)).where(
                    ReviewDecision.candidate_id == candidate.id,
                    ReviewDecision.source == "human",
                )
            ).scalar()
            if human:
                continue
            found.append(_as_dict(escalation, candidate, board))
        return found


def counts() -> dict[str, int]:
    with session_factory()() as session:
        rows = session.execute(
            select(Escalation.status, func.count(Escalation.id))
            .group_by(Escalation.status)
        ).all()
        return {status: count for status, count in rows}


def explanation_counts() -> dict[str, int]:
    """How many pending regions carry an explanation, and how many do not.

    WI-300's rationale-deadline clause requires this to be answerable: "a
    station shall be able to report how many of its dispositions carry no
    written rationale, and for what reason". It was not answerable before
    2026-08-23 -- an unexplained region and an explained one looked the same in
    every table, and the only trace of the difference was a ``ReadTimeout``
    inside a string an operator was invited to read as a rationale.

    Keyed by ``explanation_status``; ``unknown`` covers rows written before the
    column existed.
    """
    with session_factory()() as session:
        rows = session.execute(
            select(Escalation.explanation_status, func.count(Escalation.id))
            .where(Escalation.status == PENDING)
            .group_by(Escalation.explanation_status)
        ).all()
        return {status or "unknown": count for status, count in rows}
