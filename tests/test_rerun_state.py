"""A region re-run on a fresh path does not inherit the last run's rationale.

The checkpointer is keyed by region, so the thread a region ran on in August
still holds everything the reason node wrote then. ``start_review`` reset only
``trace`` and ``timings_ms`` -- the two channels that append -- and left the
reason node's channels to the last value the thread held. A region that had
escalated at the old dismissal threshold and is dismissed by the classifier at
the new one therefore came back with the old rationale on the new state, and
``record_decision`` stored it: on 2026-09-13 the demo board's region 29 held
31 ``model`` decisions since 09-05, every one carrying a rationale quoting a
0.961 threshold no line of the current prompt contains, for a decision no
model had been asked to explain. Found by running the demo's CLI step before
the presentation; the CLI printed the paragraph under ``classify -> dismiss``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from conftest import IN_THE_EXPLANATION_BAND
from langgraph.checkpoint.memory import InMemorySaver
from test_graph import StubClient, stub_tools  # noqa: F401  (fixture)

from aoi_agent.graph import flow
from aoi_agent.station import service
from aoi_agent.store import boards
from aoi_agent.store.models import (
    Board,
    CandidateRecord,
    ReviewDecision,
    create_all,
    make_session_factory,
)

STEM = "20085293"
REFERENCE = f"{STEM}#0"


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
            predicted_class="open", confidence=0.55,
            false_call_probability=0.45, ground_truth="mousebite",
        ))
        session.commit()
    return factory


def test_a_rerun_that_skips_the_reason_node_records_no_rationale(store, stub_tools):  # noqa: F811
    saver = InMemorySaver()
    stub_tools["classify"]["confidence"] = IN_THE_EXPLANATION_BAND
    graph = flow.build_graph(StubClient(rationale="written at the old threshold"), saver)

    first = service.start_review(graph, REFERENCE)
    assert first["decided_by"] == "agent"
    assert first["agent_rationale"] == "written at the old threshold"

    # The threshold moved. The same region, on the same thread, is now the
    # classifier's to dismiss, and the reason node is never entered.
    stub_tools["classify"].update({
        "predicted_class": "false_call", "confidence": 0.99,
        "false_call_probability": 0.99, "recommendation": "dismiss",
    })
    second = service.start_review(graph, REFERENCE)
    assert second["trace"] == ["classify", "dismiss"]
    assert second["decided_by"] == "model"
    assert not second.get("agent_rationale")
    assert not second.get("explanation_status")
    assert not second.get("rationale_flags")

    with store() as session:
        rows = session.query(ReviewDecision).order_by(ReviewDecision.id).all()
    assert [row.source for row in rows] == ["agent", "model"]
    assert rows[0].rationale == "written at the old threshold"
    assert rows[1].rationale is None
    assert rows[1].explanation_status is None
    assert rows[1].rationale_flags is None
