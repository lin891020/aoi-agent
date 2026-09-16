"""Re-running a board leaves a region someone handed back exactly as they left it.

``start_review`` skipped a region waiting on a person only when its queue
status was ``pending``. A region an operator had declined -- ``deferred``, the
senior's list -- was re-run like any other, and that did three things the
deferral exists to prevent. If the re-run escalated again, ``raise_escalation``
rewrote the reason and the flags in place, so the operators' declines now
pointed at a paragraph that no longer existed. If it did not -- new weights, a
moved threshold -- the thread finished, the interrupt was gone, and a senior's
later answer was recorded against the re-run's state rather than the one the
region was handed back under. And the CLI, without ``--queue``, then prompted
at the terminal for a verdict: the senior-only rule for handed-back regions
lives in the station's route, so the host account answered it.

The board was never released -- ``assess`` counts every open status first --
which is why nothing at the board level saw it. The shape is the one the
deferral paragraph in CLAUDE.md names: a new queue state has to be checked
against every rule that reads queue state, and this was one more.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from conftest import BELOW_ESCALATION
from langgraph.checkpoint.memory import InMemorySaver
from test_graph import StubClient, stub_tools  # noqa: F401  (fixture)

from aoi_agent import cli
from aoi_agent.graph import flow
from aoi_agent.station import service
from aoi_agent.store import boards, escalations
from aoi_agent.store.models import (
    Board,
    CandidateRecord,
    ReviewDecision,
    create_all,
    make_session_factory,
)

STEM = "20085293"
REFERENCE = f"{STEM}#0"
DISMISS = {
    "predicted_class": "false_call", "confidence": 0.99,
    "false_call_probability": 0.99, "recommendation": "dismiss",
}


@pytest.fixture
def store(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'test.db'}"
    create_all(url)
    factory = make_session_factory(url)
    monkeypatch.setattr(boards, "_session_factory", factory)
    with factory() as session:
        board = Board(
            stem=STEM, split="test", lot_id="LOT-2201", line_id="L2",
            machine_id="M22", shift="A", inspected_at=datetime(2026, 8, 20, 9, 0, tzinfo=UTC),
        )
        session.add(board)
        session.flush()
        session.add(CandidateRecord(
            board_id=board.id, index_on_board=0,
            x1=100, y1=120, x2=140, y2=155, area=1400,
            predicted_class="mousebite", confidence=0.55,
            false_call_probability=0.40, ground_truth="mousebite",
        ))
        session.commit()
    return factory


def _escalated(stub_tools):  # noqa: F811
    """A graph, and the region on it suspended and queued."""
    stub_tools["classify"]["confidence"] = BELOW_ESCALATION
    graph = flow.build_graph(StubClient(rationale="the paragraph they declined"), InMemorySaver())
    state = service.start_review(graph, REFERENCE)
    assert "__interrupt__" in state
    return graph


def _decisions(store) -> list[ReviewDecision]:
    with store() as session:
        return session.query(ReviewDecision).all()


def test_a_handed_back_region_is_not_re_run(store, stub_tools):  # noqa: F811
    graph = _escalated(stub_tools)
    before = escalations.get(service.thread_for(REFERENCE))
    assert escalations.defer(service.thread_for(REFERENCE), "mike", "signed_in", "notch or plating")

    # The weights moved, so a re-run would now dismiss it outright.
    stub_tools["classify"].update(DISMISS)
    state = service.start_review(graph, REFERENCE)

    assert "already_deferred" in state
    after = escalations.get(service.thread_for(REFERENCE))
    assert after["status"] == escalations.DEFERRED
    assert after["reason"] == before["reason"]
    assert _decisions(store) == []
    # The interrupt is still there for the senior to answer.
    assert graph.get_state(service._config(REFERENCE)).next


def test_a_resolved_region_is_still_re_reviewed(store, stub_tools):  # noqa: F811
    """The fix must not freeze every region that was ever queued."""
    graph = _escalated(stub_tools)
    escalations.resolve_escalation(service.thread_for(REFERENCE))

    stub_tools["classify"].update(DISMISS)
    state = service.start_review(graph, REFERENCE)

    assert "already_deferred" not in state and "already_pending" not in state
    assert state["decided_by"] == "model"
    assert [row.source for row in _decisions(store)] == ["model"]


def test_the_cli_says_handed_back_and_how_many_declined(store, stub_tools, capsys):  # noqa: F811
    graph = _escalated(stub_tools)
    thread = service.thread_for(REFERENCE)
    escalations.defer(thread, "mike", "signed_in")
    escalations.defer(thread, "ana", "signed_in")

    state = cli._run_one(graph, REFERENCE, auto_answer="open", to_queue=False)
    cli._print_result(REFERENCE, state)

    printed = capsys.readouterr().out
    assert "HANDED BACK" in printed
    assert "2 declines" in printed
    assert "QUEUED" not in printed
    # No terminal prompt was answered on the senior's behalf.
    assert _decisions(store) == []
