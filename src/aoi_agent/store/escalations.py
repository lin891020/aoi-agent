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

import json

from sqlalchemy import func, select

from aoi_agent.store.boards import session_factory
from aoi_agent.store.models import Board, CandidateRecord, Deferral, Escalation


#: The queue's two ordinary states, and the one that is a mark on the record.
PENDING = "pending"
RESOLVED = "resolved"

#: A person looked at it and could not judge it.
#:
#: Not a closure and not a verdict. The suspended run stays suspended, because
#: the region still needs an answer; what changes is that it leaves the flow of
#: the operator who could not answer it. Before this state existed the only way
#: to act on "I don't know" was to navigate away, and a region navigated away
#: from is indistinguishable from one nobody has reached -- so the next person
#: met it in the same place and declined it again, and nothing anywhere counted
#: how often that happened.
DEFERRED = "deferred"

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
        "rationale_flags": json.loads(escalation.rationale_flags or "[]"),
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


def _flags_column(flags: list[str] | None) -> str | None:
    """``None`` stays ``NULL`` -- no rationale was checked -- and a list is stored as JSON."""
    return None if flags is None else json.dumps(list(flags), ensure_ascii=False)


def raise_escalation(
    reference: str,
    thread_id: str,
    reason: str,
    agent_verdict: str | None = None,
    explanation_status: str | None = None,
    rationale_flags: list[str] | None = None,
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
            existing.rationale_flags = _flags_column(rationale_flags)
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
                    rationale_flags=_flags_column(rationale_flags),
                    status=PENDING,
                )
            )
        session.commit()
    return True


def defer(thread_id: str, operator: str, attribution: str,
          note: str | None = None) -> bool:
    """Record that a person could not judge this region, and take it out of
    their flow.

    Three things this deliberately does not do, each of which would turn "I
    don't know" into something it is not:

    It does not resume the graph. The interrupt is what makes the region still
    answerable, and consuming it would close a run nobody answered.

    It does not write a ``ReviewDecision``. That table is the next training
    round's labels; a deferral belongs in it about as much as a blank answer
    sheet belongs in an answer key.

    It does not resolve the escalation. The region is still waiting on a
    person, just not on this one.

    Refuses an operator with no attribution behind the name, which is the same
    rule ``boards.record_decision`` applies to a human verdict. A deferral is
    weaker evidence than a verdict but it is still evidence -- three declines
    on one region is how a genuinely hard case announces itself -- and evidence
    that names nobody cannot be weighed.
    """
    if not operator or not attribution:
        raise ValueError(
            f"{thread_id} cannot be deferred by an unattributable operator "
            f"(operator={operator!r}, attribution={attribution!r})"
        )
    with session_factory()() as session:
        row = session.execute(
            select(Escalation).where(Escalation.thread_id == thread_id)
        ).scalar()
        if row is None:
            return False
        if row.status in (RESOLVED, RESOLVED_UNATTRIBUTED):
            # Answered regions are not deferrable. Without this a stale tab
            # could reopen a closed queue entry, and the store would hold a
            # decision and a later "nobody could decide" about the same region.
            return False
        session.add(
            Deferral(
                escalation_id=row.id, operator=operator,
                attribution=attribution, note=note or None,
            )
        )
        row.status = DEFERRED
        session.commit()
    return True


def deferred(limit: int = 200) -> list[dict]:
    """Regions someone declined, hardest-looking first.

    Ordered by how many people have declined each one, because that is the only
    ranking here that carries information: a region three operators could not
    judge is a different object from one that a single tired person skipped.
    """
    with session_factory()() as session:
        counts = (
            select(Deferral.escalation_id, func.count().label("declines"))
            .group_by(Deferral.escalation_id)
            .subquery()
        )
        rows = session.execute(
            select(Escalation, CandidateRecord, Board, counts.c.declines)
            .join(CandidateRecord, Escalation.candidate_id == CandidateRecord.id)
            .join(Board, CandidateRecord.board_id == Board.id)
            .join(counts, counts.c.escalation_id == Escalation.id)
            .where(Escalation.status == DEFERRED)
            .order_by(counts.c.declines.desc(), Escalation.raised_at.asc())
            .limit(limit)
        ).all()
        return [
            {**_as_dict(escalation, candidate, board), "declines": int(declines)}
            for escalation, candidate, board, declines in rows
        ]


def deferred_count() -> int:
    """How many regions are waiting because nobody could judge them.

    A ``COUNT(*)``, for the reason ``pending_count`` is one.
    """
    with session_factory()() as session:
        return int(
            session.execute(
                select(func.count())
                .select_from(Escalation)
                .where(Escalation.status == DEFERRED)
            ).scalar_one()
        )


def declines_for(thread_id: str) -> list[dict]:
    """Who has already declined this region, and what they said.

    Shown to the next person who opens it. Someone who declined it yesterday
    should not spend five minutes rediscovering that, and a note saying "cannot
    tell the notch from the plating" is the closest thing this system has to
    handing over a case.
    """
    with session_factory()() as session:
        rows = session.execute(
            select(Deferral)
            .join(Escalation, Deferral.escalation_id == Escalation.id)
            .where(Escalation.thread_id == thread_id)
            .order_by(Deferral.deferred_at.asc(), Deferral.id.asc())
        ).scalars().all()
        return [
            {
                "operator": row.operator,
                "attribution": row.attribution,
                "note": row.note,
                "deferred_at": row.deferred_at.isoformat() if row.deferred_at else None,
            }
            for row in rows
        ]


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


def pending_count() -> int:
    """How many regions are actually waiting on a person.

    Separate from ``pending`` on purpose, and the separation is the fix for a
    defect rather than a tidiness. The queue page used to count with
    ``len(pending())``, which is a ``LIMIT``-ed list: at 250 waiting it rendered
    "200 waiting" and said nothing about the other 50. A number that is quietly
    wrong and looks completely normal, on a quality record -- which is the exact
    shape this project's invariants exist to catch, committed by the project.

    So a count is a ``COUNT(*)`` and never the length of a page of rows. The two
    are different questions and the code now has to ask the one it means.
    """
    with session_factory()() as session:
        return int(
            session.execute(
                select(func.count())
                .select_from(Escalation)
                .where(Escalation.status == PENDING)
            ).scalar_one()
        )


def pending_unexplained_count() -> int:
    """Of those, how many carry no written rationale.

    Counted over the whole queue for the same reason as above: the point of the
    figure is to make a shift in which the model wrote nothing visible, and a
    version of it computed over the first screenful cannot see one.
    """
    with session_factory()() as session:
        return int(
            session.execute(
                select(func.count())
                .select_from(Escalation)
                .where(Escalation.status == PENDING)
                .where(Escalation.explanation_status.is_not(None))
                .where(Escalation.explanation_status != "ok")
            ).scalar_one()
        )


def pending(limit: int = 200) -> list[dict]:
    """The queue, oldest first: whoever has waited longest goes next.

    This returns at most ``limit`` rows and is not a census. Anything that wants
    to know *how many* are waiting calls ``pending_count``; anything that
    displays these rows has to say how many it is not displaying.
    """
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
