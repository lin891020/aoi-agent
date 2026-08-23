"""Persistence for boards, AOI candidates and review decisions.

SQLite by default so the prototype runs with nothing installed; the URL comes
from ``AOI_AGENT_DATABASE_URL`` so swapping in PostgreSQL is a config change
rather than a rewrite.

The ``ReviewDecision`` table is the feedback loop's landing point. Every
verdict -- whether the model produced it or a human did -- is recorded from day
one, so retraining on operator corrections later needs no schema change.

``Escalation`` is the review station's work queue: which suspended runs are
still waiting on a person, and which checkpointer thread each one resumes from.
"""

from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from aoi_agent.provenance import UNRECORDED

DATABASE_URL = os.getenv("AOI_AGENT_DATABASE_URL", "sqlite:///data/aoi_agent.db")


class Base(DeclarativeBase):
    pass


class Board(Base):
    """One inspected panel.

    The image and its defects come from DeepPCB. The production context -- lot,
    line, machine, shift, timestamp -- is **simulated**: real datasets do not
    ship factory metadata. The generator plants a deterministic pattern so the
    context queries have something to find; see ``store.seed``.
    """

    __tablename__ = "boards"

    id: Mapped[int] = mapped_column(primary_key=True)
    stem: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    split: Mapped[str] = mapped_column(String(16), index=True)

    lot_id: Mapped[str] = mapped_column(String(24), index=True)
    line_id: Mapped[str] = mapped_column(String(16), index=True)
    machine_id: Mapped[str] = mapped_column(String(16), index=True)
    shift: Mapped[str] = mapped_column(String(8), index=True)
    inspected_at: Mapped[datetime] = mapped_column(DateTime, index=True)

    candidates: Mapped[list["CandidateRecord"]] = relationship(
        back_populates="board", cascade="all, delete-orphan"
    )


class CandidateRecord(Base):
    """A region the AOI flagged, with what the model made of it."""

    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(primary_key=True)
    board_id: Mapped[int] = mapped_column(ForeignKey("boards.id"), index=True)
    index_on_board: Mapped[int] = mapped_column(Integer)

    x1: Mapped[int] = mapped_column(Integer)
    y1: Mapped[int] = mapped_column(Integer)
    x2: Mapped[int] = mapped_column(Integer)
    y2: Mapped[int] = mapped_column(Integer)
    area: Mapped[int] = mapped_column(Integer)

    predicted_class: Mapped[str] = mapped_column(String(16), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    false_call_probability: Mapped[float] = mapped_column(Float)

    ground_truth: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    """Held for evaluation only. The agent must never read this."""

    board: Mapped[Board] = relationship(back_populates="candidates")
    decisions: Mapped[list["ReviewDecision"]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan"
    )

    @property
    def reference(self) -> str:
        return f"{self.board.stem}#{self.index_on_board}"


class ReviewDecision(Base):
    """A verdict on one candidate, from the model or from a person.

    This is what the feedback loop consumes: operator corrections accumulate
    here and become the next training set.
    """

    __tablename__ = "review_decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"), index=True)

    verdict: Mapped[str] = mapped_column(String(16))
    """A defect class name, or ``false_call``."""

    source: Mapped[str] = mapped_column(String(16), index=True)
    """``model``, ``agent`` or ``human``."""

    reviewer: Mapped[str | None] = mapped_column(String(64), nullable=True)
    """Who answered, where a person did. ``NULL`` on an automated row, which is
    the truthful value there and is read together with ``reviewer_auth``."""

    reviewer_auth: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    """What stands behind that name.

    ``mike`` typed into a text box and ``mike`` read off a signed session are
    the same four characters and are not the same claim, and until 2026-08-23
    this table could only hold the first. The corrections it feeds are the next
    training round's labels, so the distinction is not administrative: an
    expert's judgement and an anonymous click were indistinguishable rows, and
    the only remedy anyone had was to delete five of them by hand.

    ``signed_in`` the station authenticated the operator and took the name from
    the session; ``host_account`` the CLI attributed it to the OS account;
    ``automated`` no person was involved and the provenance columns answer
    instead; ``unrecorded`` the row predates this column. Indexed because
    selecting a training set by it is the point of having it.

    Nullable in the schema for the reason ``model_digest`` is -- SQLite cannot
    add a NOT NULL column to a populated table. What holds it is
    ``store.boards.record_decision``, which will not write a ``human`` row
    whose identity is not attributable, and the migration, which leaves no
    ``NULL`` behind."""

    rationale: Mapped[str | None] = mapped_column(String(2048), nullable=True)

    explanation_status: Mapped[str | None] = mapped_column(
        String(16), nullable=True, index=True
    )
    """``ok``, or why this decision carries no written rationale.

    Indexed because WI-300 requires the absence to be countable: "a station
    shall be able to report how many of its dispositions carry no written
    rationale, and for what reason". A count is what makes the failure visible
    -- the queue held one escalation reading ``the model did not answer
    (ReadTimeout)`` and no way to ask how many more there had been. ``None`` on
    a human decision, which was never going to have one."""

    model_digest: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    """Which weights produced this, by the SHA-256 of the checkpoint file.

    The auditor's question -- "under which model was this decided" -- had no
    answer here for 9,140 rows. A ``model_version`` string would have been
    worse than nothing: somebody has to bump it, one day nobody does, and from
    then on it is wrong and looks right. This is derived from the artefact, so
    there is nothing to remember. ``unrecorded`` on rows that predate the
    column, ``unavailable`` where it genuinely could not be determined; see
    ``aoi_agent.provenance``.

    Nullable in the schema because a column added to a table that already has
    rows has to be, and because SQLite cannot add a NOT NULL column without a
    default. What holds it is ``store.boards.record_decision``, which refuses
    an automated decision that does not carry one, and the migration, which
    leaves no NULL behind."""

    thresholds_json: Mapped[str | None] = mapped_column(String(256), nullable=True)
    """The operating point in force when this was decided, as JSON.

    The same weights disposition differently at a different threshold, and the
    threshold is the half that moves most often -- ``ESCALATE_BELOW`` moved on
    2026-08-23. A decision recorded without it can be reproduced only by
    guessing which sweep was current."""

    code_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    """The commit the deciding process was running, ``+dirty`` where the tree
    had uncommitted changes, ``unknown`` where it could not be read."""

    decided_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    candidate: Mapped[CandidateRecord] = relationship(back_populates="decisions")


class Escalation(Base):
    """One candidate waiting for a person, and the thread it is suspended on.

    The graph's checkpointer already holds everything needed to resume a
    suspended run, but it is keyed by thread id and has no notion of "who is
    still waiting". This table is that index and nothing more: the queue the
    review station renders, and the pointer back into the checkpointer. Graph
    state stays in the checkpointer -- duplicating it here would give two
    answers to the same question.
    """

    __tablename__ = "escalations"

    id: Mapped[int] = mapped_column(primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"), index=True)

    thread_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    """The checkpointer key. Resuming means ``Command(resume=...)`` on this."""

    reason: Mapped[str] = mapped_column(String(2048))
    """Why the agent handed over -- shown to the operator as the question."""

    agent_verdict: Mapped[str | None] = mapped_column(String(16), nullable=True)
    """What the agent leaned towards. A suggestion, never pre-selected."""

    explanation_status: Mapped[str | None] = mapped_column(
        String(16), nullable=True, index=True
    )
    """``ok``, or why this region reached the queue with no explanation.

    The queue renders it as a notice rather than as a rationale, and
    ``escalations.explanation_counts`` reports it."""

    status: Mapped[str] = mapped_column(String(16), index=True, default="pending")
    """``pending``, ``resolved``, or ``resolved_unattributed``.

    The third is not a state the station can write. It is a mark applied by
    ``scripts/mark_unattributed_resolutions.py`` to rows this store already
    held: five escalations closed as answered, whose candidates carry no human
    decision at all, left behind when five operator labels were deleted by hand
    in August 2026. Closing without an attributable answer is not something the
    code does, so it is recorded as a distinct state rather than repaired into
    one of the two the code produces."""

    raised_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    candidate: Mapped[CandidateRecord] = relationship()


class BoardDisposition(Base):
    """What happened to one board, and on what basis.

    Every other table in this store records a judgement about a *region*. A
    line ships *boards*, and an auditor asking "who released this board, when,
    and under what" was asking a question nothing here could answer: the
    candidate-level rows had to be re-aggregated by whoever was asked, by a
    rule that lived in no code.

    So the rule lives in ``store.dispositions`` and its output lives here. One
    row per time a board reached a settled state, appended rather than
    overwritten -- a board held on Monday and released on Tuesday after an
    operator answered its queue is two rows, and the pair is the record. The
    newest row is the current disposition.

    Deliberately small. It holds the four things the question asks for --
    released or held, when, on whose authority, under which model -- plus the
    counts that make the verdict checkable against the regions beneath it. It
    does not duplicate the regions: they are one join away, and two answers to
    one question is what the escalations table was carefully built to avoid.
    """

    __tablename__ = "board_dispositions"

    id: Mapped[int] = mapped_column(primary_key=True)
    board_id: Mapped[int] = mapped_column(ForeignKey("boards.id"), index=True)

    disposition: Mapped[str] = mapped_column(String(16), index=True)
    """``released`` or ``held``.

    ``held`` covers both reasons a board is not going anywhere -- a confirmed
    defect on it, or a region still waiting on a person. They are one state
    because the line does the same thing with them, and the counts below say
    which it was."""

    decided_by: Mapped[str] = mapped_column(String(64), index=True)
    """``automated``, or the reviewer whose answer settled the last region.

    In the second case it is the operator's signed-in name, carried up from
    the region-level decision that settled the board rather than typed
    anywhere. What established that name is recorded one table down, in
    ``ReviewDecision.reviewer_auth``; this row does not duplicate it, for the
    same reason it does not duplicate the regions."""

    basis: Mapped[str] = mapped_column(String(512))
    """One sentence naming the regions this verdict was computed from."""

    candidate_count: Mapped[int] = mapped_column(Integer)
    confirmed_count: Mapped[int] = mapped_column(Integer)
    """Regions whose latest verdict is a defect class rather than ``false_call``."""

    pending_count: Mapped[int] = mapped_column(Integer)
    """Regions with no decision yet, or still on the queue."""

    model_digest: Mapped[str] = mapped_column(String(80), index=True)
    """The weights behind the decisions beneath this row.

    ``mixed`` where they do not agree -- a board part-decided before a retrain
    and part after has no single model behind it, and saying so is the honest
    answer to "under which model was this released"."""

    thresholds_json: Mapped[str] = mapped_column(String(256))
    code_version: Mapped[str] = mapped_column(String(64))

    decided_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    board: Mapped[Board] = relationship()


class AnalysisRun(Base):
    """One natural-language question, and everything needed to redraw it.

    The plan, the raw results and the chart specification are kept rather than
    an image, so a run recorded this quarter renders next quarter without asking
    a model to reproduce a plan it would not reproduce. It is also what the
    evaluation script reads, and what a live view of a running graph would
    replay.
    """

    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(String(1024))
    plan_json: Mapped[str | None] = mapped_column(String, nullable=True)
    results_json: Mapped[str] = mapped_column(String, default="[]")
    chart_json: Mapped[str | None] = mapped_column(String, nullable=True)
    answer: Mapped[str] = mapped_column(String, default="")
    timings_json: Mapped[str] = mapped_column(String, default="{}")
    refused: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    asked_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    """The signed-in operator who asked, off the session rather than off the
    form. It was free text until 2026-08-23, and this page is what turned that
    from a backlog item into a precondition: a query interface exposes
    plant-wide statistics where the queue exposed a queue.

    No ``reviewer_auth`` twin here, and deliberately: nothing on this page
    dispositions a board or writes a label, so what this field is for is
    knowing who asked, not weighing what they produced. Rows written before
    the sign-in existed carry whatever was typed."""

    asked_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


def make_engine(url: str | None = None):
    return create_engine(url or DATABASE_URL, future=True)


def make_session_factory(url: str | None = None):
    return sessionmaker(bind=make_engine(url), expire_on_commit=False, future=True)


#: Nullable columns added to tables that already had rows.
#:
#: ``CREATE TABLE IF NOT EXISTS`` does nothing to a table that already exists,
#: so a store holding a queue and a season of operator corrections would have
#: had to be deleted to gain a column -- and the corrections are the next
#: training round's labels. This is not a migration framework and must not
#: become one: additive, nullable, and never a change to a column that exists.
#: Anything beyond that wants a real tool.
ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "review_decisions": {
        "explanation_status": "VARCHAR(16)",
        "model_digest": "VARCHAR(80)",
        "thresholds_json": "VARCHAR(256)",
        "code_version": "VARCHAR(64)",
        "reviewer_auth": "VARCHAR(16)",
    },
    "escalations": {"explanation_status": "VARCHAR(16)"},
}

#: Columns whose ``NULL`` would mean two different things, and the value that
#: takes one of those meanings away.
#:
#: A column added to a populated table starts life ``NULL`` on every existing
#: row. For ``explanation_status`` that was tolerable: the reader folds ``NULL``
#: into ``unknown`` and the count is honest. For provenance it is not, because
#: ``NULL`` would have to mean both "this decision predates the columns" and
#: "this decision was written without provenance", and the second is the thing
#: the guard exists to make impossible. So the migration stamps the first
#: meaning explicitly and the guard forbids the second, which leaves ``NULL``
#: meaning nothing at all -- and a ``NULL`` appearing later is therefore a bug
#: with a test on it.
#:
#: Backfilled only on the pass that adds the column. A row inserted afterwards
#: is a row the running code wrote, and stamping it ``unrecorded`` would hide
#: exactly the failure this is for.
BACKFILL_ON_ADD: dict[str, dict[str, str]] = {
    "review_decisions": {
        "model_digest": UNRECORDED,
        "thresholds_json": UNRECORDED,
        "code_version": UNRECORDED,
        # The same argument, one column over. Every one of the 9,140 rows this
        # store held on 2026-08-23 has ``reviewer`` NULL, and NULL would have
        # to mean both "written before anyone recorded how a reviewer was
        # identified" and "no reviewer, because no person was involved" -- and
        # the second is a fact about a model decision worth stating. So the
        # migration stamps the first meaning and ``record_decision`` writes
        # ``automated`` for the second.
        "reviewer_auth": UNRECORDED,
    },
}


def create_all(url: str | None = None) -> None:
    engine = make_engine(url)
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)


def _add_missing_columns(engine) -> list[str]:
    """Bring an existing store up to the columns declared above.

    Returns what it added, so a caller -- or a test -- can see that it ran.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    added = []
    with engine.begin() as connection:
        for table, columns in ADDED_COLUMNS.items():
            if table not in existing_tables:
                continue
            present = {c["name"] for c in inspector.get_columns(table)}
            for name, sql_type in columns.items():
                if name in present:
                    continue
                connection.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
                )
                backfill = BACKFILL_ON_ADD.get(table, {}).get(name)
                if backfill is not None:
                    # Every row in the table at this instant predates the
                    # column, so this is the one moment at which "written
                    # before provenance existed" is knowable without guessing.
                    connection.execute(
                        text(f"UPDATE {table} SET {name} = :value "
                             f"WHERE {name} IS NULL"),
                        {"value": backfill},
                    )
                added.append(f"{table}.{name}")
    return added
