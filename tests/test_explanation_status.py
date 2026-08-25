"""What the station does when the model does not write an explanation.

Three ways that can happen -- the deadline expires, the model is unreachable,
the response will not parse -- and until 2026-08-23 all three ended the same
way: the string ``the model did not answer (ReadTimeout)`` in the panel where
the operator reads why a region is in front of them, and nothing anywhere
counting how often it happened. The queue held one such entry. Nobody could say
whether it was one of one or one of fifty.

Three properties are held here, and they are the whole fix:

1. **The disposition does not move.** The classifier decided before the reason
   node was entered, so a missing explanation costs an explanation.
2. **The absence is shown as an absence.** Not as a rationale, and not as an
   exception class name. WI-300 forbids filling the gap by any other means, so
   the notice is rendered from the status rather than written into the record.
3. **The absence is counted.** WI-300 requires a station to be able to report
   how many of its dispositions carry no written rationale and for what reason.
"""

from __future__ import annotations

from datetime import datetime

import httpx
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from conftest import IN_THE_EXPLANATION_BAND, read_in, sign_in
from aoi_agent.graph import flow
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
from aoi_agent.provenance import DecisionProvenance, ReviewerIdentity
from test_graph import STUB_DIGEST, StubClient, stub_tools  # noqa: F401  (fixture)

#: An automated decision written straight into the store still has to say
#: what produced it -- these tests are about the explanation column, not a
#: licence to write an unattributable row.
PROVENANCE = DecisionProvenance(
    model_digest=STUB_DIGEST, thresholds={"dismiss": 0.915}, code_version="test"
)

STEM = "20085293"
REFERENCE = f"{STEM}#0"


class SlowClient:
    """A model that never answers inside the deadline."""

    def chat(self, messages, **kwargs):
        raise httpx.ReadTimeout("deadline expired")


class DeadClient:
    """A model that is not there at all."""

    def chat(self, messages, **kwargs):
        raise httpx.ConnectError("connection refused")


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
        for index, (klass, conf) in enumerate([("open", 0.55), ("short", 0.99)]):
            session.add(
                CandidateRecord(
                    board_id=board.id, index_on_board=index,
                    x1=100, y1=120, x2=140, y2=155, area=1400,
                    predicted_class=klass, confidence=conf,
                    false_call_probability=round(1 - conf, 3),
                    ground_truth="mousebite",
                )
            )
        session.commit()
    return factory


# ---- the three statuses -------------------------------------------------


@pytest.mark.parametrize(
    "client,expected",
    [(SlowClient(), "timed_out"), (DeadClient(), "unreachable")],
)
def test_the_two_operational_failures_are_told_apart(stub_tools, client, expected):  # noqa: F811
    """A model that is slow and a model that is down want different actions.

    One says the deadline or the model is wrong for this line; the other says
    Ollama is not running. Collapsed into one error string they were the same
    event, and neither could be counted.
    """
    stub_tools["classify"]["confidence"] = IN_THE_EXPLANATION_BAND
    graph = flow.build_graph(client, InMemorySaver())
    state = graph.invoke(
        {"candidate_ref": REFERENCE, "trace": [], "timings_ms": {}},
        config={"configurable": {"thread_id": REFERENCE}},
    )
    assert state["explanation_status"] == expected


def test_an_unparseable_response_is_a_third_status(stub_tools):  # noqa: F811
    """The model answered. What it said is unusable, which is not an outage."""
    stub_tools["classify"]["confidence"] = IN_THE_EXPLANATION_BAND
    graph = flow.build_graph(StubClient(raw_text="sure, looks fine"), InMemorySaver())
    state = graph.invoke(
        {"candidate_ref": REFERENCE, "trace": [], "timings_ms": {}},
        config={"configurable": {"thread_id": REFERENCE}},
    )
    assert state["explanation_status"] == "unparsed"


def test_a_written_explanation_is_marked_as_one(stub_tools):  # noqa: F811
    """``ok`` has to be recorded too. Without it the only way to read the count
    is "rows that failed", and a station that stopped calling the model at all
    would report a perfect record."""
    stub_tools["classify"]["confidence"] = IN_THE_EXPLANATION_BAND
    graph = flow.build_graph(StubClient(), InMemorySaver())
    state = graph.invoke(
        {"candidate_ref": REFERENCE, "trace": [], "timings_ms": {}},
        config={"configurable": {"thread_id": REFERENCE}},
    )
    assert state["explanation_status"] == flow.EXPLAINED
    assert state["agent_rationale"] == "stub"


# ---- the disposition is untouched ---------------------------------------


def test_the_disposition_survives_every_way_the_explanation_can_fail(stub_tools):  # noqa: F811
    """The point of the whole arrangement.

    ``decide_node`` reads ``model_class`` and ``route_after_reason`` reads
    ``model_confidence``. Both exist before the reason node is entered, so the
    verdict is the same whatever the LLM does or fails to do -- which is why a
    60s explanation deadline can sit inside a 10s response budget without
    contradiction.
    """
    stub_tools["classify"]["confidence"] = IN_THE_EXPLANATION_BAND
    verdicts = set()
    for index, client in enumerate([StubClient(), SlowClient(), DeadClient(),
                                    StubClient(raw_text="nope")]):
        graph = flow.build_graph(client, InMemorySaver())
        state = graph.invoke(
            {"candidate_ref": REFERENCE, "trace": [], "timings_ms": {}},
            config={"configurable": {"thread_id": f"t{index}"}},
        )
        assert "__interrupt__" not in state
        verdicts.add((state["verdict"], state["decided_by"], state["disposition"]))
    assert verdicts == {("open", "agent", "defect_confirmed")}


# ---- what the operator is shown -----------------------------------------


def test_the_notice_says_what_is_missing_and_that_the_verdict_is_not():
    """The operator's first question on seeing a blank explanation is whether
    the verdict beside it is blank too. The notice answers it before they ask."""
    notice = " ".join(flow.explanation_notice("timed_out").split())
    assert "No explanation was written" in notice
    assert "60-second explanation deadline" in notice
    assert "was not affected" in notice


def test_no_exception_class_name_reaches_the_operator(stub_tools):  # noqa: F811
    """``ReadTimeout`` is a fact about httpx. It is not a fact about the board."""
    for status in ("timed_out", "unreachable", "unparsed"):
        notice = flow.explanation_notice(status)
        assert "Timeout" not in notice
        assert "Error" not in notice
        assert "(" not in notice


def test_an_explained_region_gets_no_notice():
    assert flow.explanation_notice(flow.EXPLAINED) == ""
    assert flow.explanation_notice("") == ""


def test_the_handover_reason_is_the_confidence_not_the_outage(store, stub_tools):  # noqa: F811
    """A region is on the queue because the classifier was unsure. It has never
    been on the queue because the LLM failed -- ``route_after_reason`` does not
    read the LLM at all -- so the reason shown to the operator says so."""
    graph = flow.build_graph(SlowClient(), InMemorySaver())
    service.start_review(graph, REFERENCE)

    row = escalations.get(REFERENCE)
    assert "0.550" in row["reason"]
    assert "escalation threshold" in row["reason"]
    assert row["explanation_status"] == "timed_out"


def test_the_queue_page_flags_the_absence_and_counts_it(store, stub_tools, monkeypatch,  # noqa: F811
                                                        operators):
    graph = flow.build_graph(SlowClient(), InMemorySaver())
    service.start_review(graph, REFERENCE)
    monkeypatch.setattr(station_app, "_graph", graph)

    body = read_in(sign_in(TestClient(station_app.app)), "en").get("/").text
    assert "no explanation" in body
    assert "1 of 1 carry no written explanation" in body
    assert "ReadTimeout" not in body


def test_the_region_page_shows_an_absence_rather_than_a_rationale(
    store, stub_tools, monkeypatch, operators  # noqa: F811
):
    graph = flow.build_graph(SlowClient(), InMemorySaver())
    service.start_review(graph, REFERENCE)
    monkeypatch.setattr(station_app, "_graph", graph)

    body = read_in(sign_in(TestClient(station_app.app)), "en").get(f"/c/{STEM}/0").text
    assert "no written explanation" in body
    assert "No explanation was written" in body
    assert "ReadTimeout" not in body


# ---- counting -----------------------------------------------------------


def test_a_decision_records_whether_it_carries_an_explanation(store, stub_tools):  # noqa: F811
    """The rationale column stays empty. WI-300: an absent rationale is absent,
    and the gap is not to be filled by any other means -- including by a notice
    that would then read back as something the model wrote."""
    stub_tools["classify"]["confidence"] = IN_THE_EXPLANATION_BAND
    service.start_review(flow.build_graph(SlowClient(), InMemorySaver()), REFERENCE)

    with store() as session:
        row = session.query(ReviewDecision).one()
    assert row.explanation_status == "timed_out"
    assert row.rationale is None


def test_the_counts_answer_the_question_wi_300_asks(store, stub_tools):  # noqa: F811
    """Two candidates, two outcomes, one number a person can read off."""
    stub_tools["classify"]["confidence"] = IN_THE_EXPLANATION_BAND
    service.start_review(flow.build_graph(StubClient(), InMemorySaver()), REFERENCE)
    service.start_review(
        flow.build_graph(SlowClient(), InMemorySaver()), f"{STEM}#1"
    )

    counts = boards.explanation_status_counts()
    assert counts == {"ok": 1, "timed_out": 1}


def test_an_unrecorded_explanation_state_is_not_counted_as_explained(store):
    """Rows written before the column existed are ``unknown``, not ``ok``.

    Folding them into ``ok`` would report a station that never recorded the
    state as a station that never failed, which is the shape of the defect this
    file exists for.
    """
    boards.record_decision(REFERENCE, "open", "agent", provenance=PROVENANCE)
    assert boards.explanation_status_counts() == {"unknown": 1}


def test_a_human_decision_is_left_out_of_the_count(store):
    """An operator's verdict was never going to carry a model's rationale.

    Counting it as a miss dilutes the rate that measures the model, which is
    the only thing this count is for.
    """
    boards.record_decision(
        REFERENCE, "open", "agent", explanation_status="ok", provenance=PROVENANCE
    )
    boards.record_decision(REFERENCE, "short", "human",
                           identity=ReviewerIdentity.signed_in("mike"))
    assert boards.explanation_status_counts() == {"ok": 1}


def test_the_queue_count_is_scoped_to_what_is_still_waiting(store, stub_tools):  # noqa: F811
    graph = flow.build_graph(SlowClient(), InMemorySaver())
    service.start_review(graph, REFERENCE)
    assert escalations.explanation_counts() == {"timed_out": 1}

    escalations.resolve_escalation(REFERENCE)
    assert escalations.explanation_counts() == {}


# ---- the column reaching a store that already has rows -------------------


def test_the_column_is_added_to_a_store_that_predates_it(tmp_path):
    """A schema change with no path forward is a schema change that eats the
    corrections.

    The queue and `review_decisions` are quality records -- the corrections in
    them are the next training round's labels -- so gaining a nullable column
    must not mean rebuilding the store. `create_all` adds it in place, and this
    builds the old schema explicitly rather than trusting that it would.
    """
    from sqlalchemy import create_engine, inspect, text
    from aoi_agent.store.models import create_all

    url = f"sqlite:///{tmp_path / 'old.db'}"
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE review_decisions ("
            " id INTEGER PRIMARY KEY, candidate_id INTEGER, verdict VARCHAR(16),"
            " source VARCHAR(16), reviewer VARCHAR(64), rationale VARCHAR(2048),"
            " decided_at DATETIME)"
        ))
        connection.execute(text(
            "INSERT INTO review_decisions (verdict, source) VALUES ('open', 'agent')"
        ))

    create_all(url)

    columns = {c["name"] for c in inspect(engine).get_columns("review_decisions")}
    assert "explanation_status" in columns
    with engine.begin() as connection:
        rows = connection.execute(text("SELECT count(*) FROM review_decisions")).scalar()
    assert rows == 1, "the migration must not touch a row"


def test_migrating_twice_changes_nothing(tmp_path):
    """`seed_store.py --migrate-only` is something a person runs when unsure."""
    from sqlalchemy import create_engine, inspect
    from aoi_agent.store.models import _add_missing_columns, create_all

    url = f"sqlite:///{tmp_path / 'fresh.db'}"
    create_all(url)
    engine = create_engine(url)
    assert _add_missing_columns(engine) == []
    assert "explanation_status" in {
        c["name"] for c in inspect(engine).get_columns("escalations")
    }
