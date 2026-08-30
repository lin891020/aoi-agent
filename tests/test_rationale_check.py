"""A rationale may not cite a figure the model was never shown.

The first Chinese rationale measured on 2026-08-29 explained a region against
"a 0.85 threshold". No work instruction names that number, no threshold table
holds it and no line of the prompt contained it; it read exactly like the
confidence beside it. The analysis page had flagged that shape of failure
since the same morning and the disposition path had nothing.

Three properties:

1. **The ground is the prompt.** A figure that renders from the prompt --
   the confidence at fewer places, a share as a percentage, the lot number
   inside its id -- is grounded. Anything else is flagged, and the check
   never waves a figure through because it looks plausible.
2. **The flag travels with the rationale.** It is computed where the prompt
   exists, stored on the queue row and on the decision, and rendered where
   the rationale is read. The disposition does not move.
3. **Small counts are not figures.** "2 passages" is not a fabrication and
   flagging it would train operators to ignore the chip.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from conftest import IN_THE_EXPLANATION_BAND, read_in, sign_in
from aoi_agent.graph import flow
from aoi_agent.graph.rationale_check import unsourced_figures
from aoi_agent.station import app as station_app
from aoi_agent.station import service
from aoi_agent.store import boards, escalations
from aoi_agent.store.models import (
    Board,
    CandidateRecord,
    ReviewDecision,
    create_all,
    make_session_factory,
)
from test_graph import StubClient, stub_tools  # noqa: F401  (fixture)

PROMPT = """Region: 20085293#0

Vision model reading:
  class      open
  confidence 0.550
  P(false call) 0.450

Production context:
  lot LOT-2201 on line L2, machine M22, shift A
  this lot averages 3.4 defects per board
  This board's machine (L2-M22) runs 37.5% of its defects as open, against a fleet average of 21.0%.

Applicable acceptance criteria:
[WI-201 / 4.2]
Any confirmed open is critical."""


# ---- 1. the ground is the prompt -------------------------------------------


@pytest.mark.parametrize(
    "rationale",
    [
        "Confidence 0.55 sits below the band.",
        "The model is 55% sure this is an open.",
        "Lot 2201 averages 3.4 defects per board, above M22's 37.5% share.",
        "信心 0.550，本機台 open 佔 37.5%，全廠 21%。",
    ],
)
def test_a_figure_that_renders_from_the_prompt_is_grounded(rationale):
    assert unsourced_figures(rationale, PROMPT) == []


@pytest.mark.parametrize(
    "rationale,flagged",
    [
        ("Confidence 0.55 is below the 0.85 threshold WI-201 sets.", ["0.85"]),
        ("信心 0.55 低於 0.85 的門檻。", ["0.85"]),
        ("The lot runs 12.5% opens against a 37.5% share.", ["12.5"]),
        ("0.85 here and 0.85 again, plus 0.9.", ["0.85", "0.9"]),
    ],
)
def test_a_figure_from_nowhere_is_flagged_as_written(rationale, flagged):
    assert unsourced_figures(rationale, PROMPT) == flagged


def test_a_stricter_rendering_than_the_prompt_holds_is_not_grounded():
    """`0.550` in the prompt does not license `0.5504` in the rationale."""
    assert unsourced_figures("confidence 0.5504", PROMPT) == ["0.5504"]


# ---- 3. small counts are not figures ---------------------------------------


def test_small_bare_integers_are_counts_not_figures():
    assert unsourced_figures("Both of the 2 passages agree; see item 3.", PROMPT) == []


def test_a_large_bare_integer_is_still_a_figure():
    assert unsourced_figures("about 150 defects an hour", PROMPT) == ["150"]


# ---- 2. the flag travels with the rationale --------------------------------

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
            machine_id="M22", shift="A", inspected_at=datetime(2026, 8, 20, 9, 0),
        )
        session.add(board)
        session.flush()
        session.add(
            CandidateRecord(
                board_id=board.id, index_on_board=0,
                x1=100, y1=120, x2=140, y2=155, area=1400,
                predicted_class="open", confidence=0.55,
                false_call_probability=0.45, ground_truth="mousebite",
            )
        )
        session.commit()
    return factory


def _run(client, confidence):
    graph = flow.build_graph(client, InMemorySaver())
    return graph.invoke(
        {"candidate_ref": REFERENCE, "trace": [], "timings_ms": {}},
        config={"configurable": {"thread_id": REFERENCE}},
    )


def test_the_reason_node_checks_against_the_prompt_it_sent(stub_tools):  # noqa: F811
    stub_tools["classify"]["confidence"] = IN_THE_EXPLANATION_BAND
    client = StubClient(rationale="Confidence is below the 0.85 threshold.")
    state = _run(client, IN_THE_EXPLANATION_BAND)
    assert state["rationale_flags"] == ["0.85"]
    # And the confidence the model was actually shown is not flagged.
    shown = f"{IN_THE_EXPLANATION_BAND:.3f}"
    client = StubClient(rationale=f"Confidence {shown} is in the band.")
    assert _run(client, IN_THE_EXPLANATION_BAND)["rationale_flags"] == []


def test_the_flag_reaches_the_queue_row_and_the_page(stub_tools, store, monkeypatch,  # noqa: F811
                                                    operators):
    """Stored on the escalation, rendered as a chip on the queue and a note on the region."""
    stub_tools["classify"]["confidence"] = flow.ESCALATE_BELOW - 0.05
    graph = flow.build_graph(
        StubClient(rationale="Below the 0.85 threshold."), InMemorySaver()
    )
    state = service.start_review(graph, REFERENCE)
    assert "__interrupt__" in state
    row = escalations.get(service.thread_for(REFERENCE))
    assert row["rationale_flags"] == ["0.85"]

    monkeypatch.setattr(station_app, "_graph", graph)
    client = read_in(sign_in(TestClient(station_app.app)), "en")
    queue = client.get("/").text
    assert "1 figure(s) not in the evidence" in queue
    region = client.get(f"/c/{STEM}/0").text
    assert "0.85" in region and "never shown" in region


def test_a_decision_written_by_the_agent_carries_its_check(stub_tools, store):  # noqa: F811
    stub_tools["classify"]["confidence"] = IN_THE_EXPLANATION_BAND
    graph = flow.build_graph(StubClient(rationale="Above the 0.85 line."), InMemorySaver())
    state = service.start_review(graph, REFERENCE)
    assert "__interrupt__" not in state
    with store() as session:
        row = session.query(ReviewDecision).one()
    assert row.rationale_flags == '["0.85"]'


def test_a_row_with_no_rationale_has_no_check(store):
    """NULL means nothing was checked -- a human answer has no prompt behind it."""
    from aoi_agent.provenance import ReviewerIdentity
    boards.record_decision(REFERENCE, "open", "human", ReviewerIdentity.signed_in("mike"))
    with store() as session:
        row = session.query(ReviewDecision).one()
    assert row.rationale_flags is None
