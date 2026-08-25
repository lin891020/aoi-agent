"""Whether a board was released or held, and on what basis.

Everything else in this store records a judgement about a region. A line ships
boards, and "was this board released, by whom, when, and under which model" was
unanswerable here for every row: the candidate-level decisions held the
material, but the rule that turns them into a board's fate lived in nobody's
code and in nobody's head the same way twice.

The rule is here and it is deliberately dull:

* a region's fate is its **latest** decision, because decisions accumulate --
  an operator's correction is a second row, not an edit of the first
* a region with no decision, or one still on the queue, is **pending**
* a region whose latest verdict is anything but ``false_call`` is **confirmed**
* a board with any confirmed or any pending region is **held**; otherwise it is
  **released**

Both reasons to hold collapse into one state because the line does the same
thing with them -- the board does not move -- and the counts on the row say
which it was. A board is not released because nobody has looked at it yet.

Rows accumulate the way decisions do. A board held on Monday and released on
Tuesday after an operator answered its queue is two rows, and the pair is the
record an auditor is entitled to; the newest row is the current disposition.
"""

from __future__ import annotations

from sqlalchemy import func, select

from aoi_agent.provenance import UNAVAILABLE, DecisionProvenance, code_version
from aoi_agent.store.boards import session_factory
from aoi_agent.store.models import (
    Board,
    BoardDisposition,
    CandidateRecord,
    Escalation,
    ReviewDecision,
)

RELEASED = "released"
HELD = "held"

#: What the board-level ``model_digest`` says when the decisions beneath it do
#: not agree on one. A board part-decided before a retrain and part after was
#: not released under a model, it was released under two, and naming one of
#: them would be the more comfortable lie.
MIXED = "mixed"

#: Escalation states that still owe an answer. ``resolved_unattributed`` is not
#: one of them: those rows were closed, badly, and are marked as such -- see
#: ``scripts/mark_unattributed_resolutions.py``. Counting them as pending would
#: put five boards back on a queue no one is working.
#:
#: ``deferred`` **is** one of them, and leaving it out was a defect for the
#: length of one commit -- the worst-directed defect this file could have had.
#: The sequence: the model dispositions a region `false_call`, the flow
#: escalates it anyway, a person opens it and says they cannot judge it. With
#: only ``pending`` here, that person's answer moved the escalation out of the
#: open set while the model's row stayed the standing decision -- so the board
#: went from **held to released**, and the act of an operator refusing to guess
#: is what shipped it. A feature built to stop uncertain regions being answered
#: as certain would have been the thing certifying them. Held by
#: ``tests/test_deferral.py::test_a_deferral_cannot_release_a_board``.
OPEN_STATUSES = ("pending", "deferred")


def _latest_decisions(session, board: Board) -> dict[int, ReviewDecision]:
    """The decision that currently stands for each candidate on this board."""
    rows = session.execute(
        select(ReviewDecision)
        .join(CandidateRecord, ReviewDecision.candidate_id == CandidateRecord.id)
        .where(CandidateRecord.board_id == board.id)
        .order_by(ReviewDecision.decided_at.asc(), ReviewDecision.id.asc())
    ).scalars().all()
    latest: dict[int, ReviewDecision] = {}
    for decision in rows:
        latest[decision.candidate_id] = decision
    return latest


def _board(session, stem: str) -> Board | None:
    return session.execute(select(Board).where(Board.stem == stem)).scalar()


def assess(stem: str) -> dict | None:
    """What this board's regions currently say, without writing anything.

    Separated from the write so the rule can be tested, and read, on its own.
    """
    with session_factory()() as session:
        board = _board(session, stem)
        if board is None:
            return None

        candidates = session.execute(
            select(CandidateRecord).where(CandidateRecord.board_id == board.id)
        ).scalars().all()
        latest = _latest_decisions(session, board)

        queued = set(
            session.execute(
                select(Escalation.candidate_id)
                .join(CandidateRecord, Escalation.candidate_id == CandidateRecord.id)
                .where(
                    CandidateRecord.board_id == board.id,
                    Escalation.status.in_(OPEN_STATUSES),
                )
            ).scalars().all()
        )

        confirmed = pending = 0
        digests: set[str] = set()
        for candidate in candidates:
            decision = latest.get(candidate.id)
            if decision is None or candidate.id in queued:
                pending += 1
                continue
            if decision.verdict != "false_call":
                confirmed += 1
            digests.add(decision.model_digest or UNAVAILABLE)

        disposition = HELD if (confirmed or pending) else RELEASED
        return {
            "board_stem": stem,
            "disposition": disposition,
            "candidate_count": len(candidates),
            "confirmed_count": confirmed,
            "pending_count": pending,
            "dismissed_count": len(candidates) - confirmed - pending,
            "model_digests": sorted(digests),
            "basis": (
                f"{len(candidates)} flagged regions: {confirmed} confirmed as "
                f"defects, {pending} still waiting on a person, "
                f"{len(candidates) - confirmed - pending} dismissed"
            ),
        }


def record(stem: str, decided_by: str = "automated") -> dict | None:
    """Write one board-level disposition, and return what it says.

    Called when a board reaches a settled state: at the end of a board run, and
    again when an operator answers the last region on it. Not on every region,
    which would fill the table with a row per candidate saying the board is
    still held.
    """
    assessment = assess(stem)
    if assessment is None:
        return None

    digests = [d for d in assessment["model_digests"] if d != UNAVAILABLE]
    digest = (
        digests[0] if len(set(digests)) == 1
        else MIXED if digests
        else UNAVAILABLE
    )
    provenance = DecisionProvenance(
        model_digest=digest,
        thresholds=_thresholds(),
        code_version=code_version(),
    )

    with session_factory()() as session:
        board = _board(session, stem)
        if board is None:
            return None
        session.add(
            BoardDisposition(
                board_id=board.id,
                disposition=assessment["disposition"],
                decided_by=decided_by,
                basis=assessment["basis"][:512],
                candidate_count=assessment["candidate_count"],
                confirmed_count=assessment["confirmed_count"],
                pending_count=assessment["pending_count"],
                **provenance.columns(),
            )
        )
        session.commit()

    return {**assessment, "decided_by": decided_by, **provenance.columns()}


def _thresholds() -> dict[str, float]:
    """The operating point in force, read from the flow rather than restated.

    Imported here rather than at module scope: the graph imports the store, and
    a store that imports the graph at load closes the circle.
    """
    from aoi_agent.graph.flow import CONFIDENT, ESCALATE_BELOW
    from aoi_agent.vision.inference import DEFAULT_DISMISS_THRESHOLD

    return {
        "dismiss": DEFAULT_DISMISS_THRESHOLD,
        "escalate_below": ESCALATE_BELOW,
        "confident": CONFIDENT,
    }


def _as_dict(row: BoardDisposition, stem: str) -> dict:
    return {
        "board_stem": stem,
        "disposition": row.disposition,
        "decided_by": row.decided_by,
        "basis": row.basis,
        "candidate_count": row.candidate_count,
        "confirmed_count": row.confirmed_count,
        "pending_count": row.pending_count,
        "model_digest": row.model_digest,
        "thresholds_json": row.thresholds_json,
        "code_version": row.code_version,
        "decided_at": row.decided_at.isoformat() if row.decided_at else None,
    }


def history(stem: str, limit: int = 20) -> list[dict]:
    """Every disposition this board has had, newest first."""
    with session_factory()() as session:
        board = _board(session, stem)
        if board is None:
            return []
        rows = session.execute(
            select(BoardDisposition)
            .where(BoardDisposition.board_id == board.id)
            .order_by(BoardDisposition.decided_at.desc(), BoardDisposition.id.desc())
            .limit(limit)
        ).scalars().all()
        return [_as_dict(row, stem) for row in rows]


def latest(stem: str) -> dict | None:
    rows = history(stem, limit=1)
    return rows[0] if rows else None


def _standing_ids():
    """The id of each board's standing disposition, as a subquery.

    Rows accumulate, so "this board's disposition" is a *rule* and not a
    column, and the rule is the one ``history`` already uses: newest
    ``decided_at``, ties broken by ``id``. Written once here because a second
    definition of latest is a second answer to the only question this table
    exists to answer, and the two would disagree precisely on the boards an
    auditor asks about -- the ones dispositioned twice in a day.

    ``test_the_index_and_the_board_page_agree_on_what_stands`` holds the two
    together rather than a refactor: ``latest()`` reads one board and this
    reads all of them, and the guard is that they never differ.
    """
    ranked = select(
        BoardDisposition.id.label("id"),
        func.row_number()
        .over(
            partition_by=BoardDisposition.board_id,
            order_by=(
                BoardDisposition.decided_at.desc(),
                BoardDisposition.id.desc(),
            ),
        )
        .label("rank"),
    ).subquery()
    return select(ranked.c.id).where(ranked.c.rank == 1)


def board_counts() -> dict[str, int]:
    """How many boards stand held and how many released.

    A ``COUNT(*)`` over the standing rows, deliberately not ``len(recent())``.
    The queue badge counted the length of a capped list for a while and was the
    same wrong number in five places at once; an aggregate that silently agrees
    with its own page size is worse than no aggregate, because the page looks
    consistent.

    Boards with no disposition row are in neither count and in no total here.
    That is not an oversight: a board nobody has run is not a released board,
    and the fleet size lives in ``boards``, not in this table.
    """
    with session_factory()() as session:
        rows = session.execute(
            select(BoardDisposition.disposition, func.count(BoardDisposition.id))
            .where(BoardDisposition.id.in_(_standing_ids()))
            .group_by(BoardDisposition.disposition)
        ).all()
    counts = {HELD: 0, RELEASED: 0}
    counts.update({disposition: count for disposition, count in rows})
    counts["total"] = sum(count for _disposition, count in rows)
    return counts


def recent(limit: int = 50, status: str | None = None) -> list[dict]:
    """The boards this system has dispositioned, newest first.

    The station could show the queue and one board reached by a link from it,
    which meant the 82% the agent settled had no page at all -- the only thing
    visible was what it could not settle. A reviewer reading only the queue is
    reading the failures and calling it the system.

    Read-only, and no ``ground_truth``: ``_as_dict`` carries none, which is the
    same dict boundary the queue and the analysis page use rather than a rule
    about templates.
    """
    with session_factory()() as session:
        query = (
            select(BoardDisposition, Board.stem)
            .join(Board, BoardDisposition.board_id == Board.id)
            .where(BoardDisposition.id.in_(_standing_ids()))
        )
        if status is not None:
            query = query.where(BoardDisposition.disposition == status)
        rows = session.execute(
            query.order_by(
                BoardDisposition.decided_at.desc(), BoardDisposition.id.desc()
            ).limit(limit)
        ).all()
        return [_as_dict(row, stem) for row, stem in rows]


def decision_provenance(stem: str) -> list[dict]:
    """Every region on a board with the decision that stands and what made it.

    This is the auditor's row: one line per region, naming the verdict, who or
    what reached it, the weights behind it, the operating point it was reached
    at, and the code that ran. No ``ground_truth`` -- the same boundary the
    queue keeps, for the same reason.
    """
    with session_factory()() as session:
        board = _board(session, stem)
        if board is None:
            return []
        candidates = session.execute(
            select(CandidateRecord)
            .where(CandidateRecord.board_id == board.id)
            .order_by(CandidateRecord.index_on_board)
        ).scalars().all()
        latest_by_candidate = _latest_decisions(session, board)
        queued = {
            candidate_id: status
            for candidate_id, status in session.execute(
                select(Escalation.candidate_id, Escalation.status)
                .join(CandidateRecord, Escalation.candidate_id == CandidateRecord.id)
                .where(CandidateRecord.board_id == board.id)
            ).all()
        }

        rows = []
        for candidate in candidates:
            decision = latest_by_candidate.get(candidate.id)
            rows.append(
                {
                    "reference": f"{stem}#{candidate.index_on_board}",
                    "index": candidate.index_on_board,
                    "model_class": candidate.predicted_class,
                    "model_confidence": candidate.confidence,
                    "verdict": decision.verdict if decision else None,
                    "source": decision.source if decision else None,
                    "reviewer": decision.reviewer if decision else None,
                    "reviewer_auth": decision.reviewer_auth if decision else None,
                    "model_digest": decision.model_digest if decision else None,
                    "thresholds_json": decision.thresholds_json if decision else None,
                    "code_version": decision.code_version if decision else None,
                    "decided_at": (
                        decision.decided_at.isoformat()
                        if decision and decision.decided_at else None
                    ),
                    "queue_status": queued.get(candidate.id),
                }
            )
        return rows


def unattributable() -> dict[str, int]:
    """How many automated decisions cannot be attributed to a set of weights.

    ``unrecorded`` predates the columns; ``unavailable`` was written after they
    existed and still could not name the model. Both are absences and they are
    different absences, which is the whole reason they are two words.
    """
    with session_factory()() as session:
        rows = session.execute(
            select(ReviewDecision.model_digest, func.count(ReviewDecision.id))
            .group_by(ReviewDecision.model_digest)
        ).all()
        counts: dict[str, int] = {}
        for digest, count in rows:
            key = digest or "null"
            counts[key] = counts.get(key, 0) + count
        return counts
