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
    """``pending`` or ``resolved``."""

    raised_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    candidate: Mapped[CandidateRecord] = relationship()


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
    """Free text until the station has authentication -- same gap as
    ``ReviewDecision.reviewer``, and this page widens it, because a query
    interface exposes plant-wide statistics where the queue exposed a queue."""

    asked_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


def make_engine(url: str | None = None):
    return create_engine(url or DATABASE_URL, future=True)


def make_session_factory(url: str | None = None):
    return sessionmaker(bind=make_engine(url), expire_on_commit=False, future=True)


def create_all(url: str | None = None) -> None:
    Base.metadata.create_all(make_engine(url))
