"""A decision that cannot be attributed to a model is not a record.

The finding this file answers came from someone who has argued with auditors: a
customer returns a batch, and the questions are "who decided this board was
fine, when, and on what basis". This store could answer none of them. It held
9,140 decisions, every one of them naming a verdict and a source and nothing
about the weights, the operating point or the code that produced it, and no row
anywhere about a *board* -- which is the thing a line actually ships.

Three properties are held here.

1. **An automated decision cannot be written without its provenance.** Not
   "should carry"; cannot. The columns are storage and this is the mechanism --
   a schema with three nullable columns and no guard is the same store with
   more places for a NULL.
2. **The identity is derived, never declared.** A ``model_version`` string
   somebody has to bump is worse than no field at all, because the release it
   is forgotten on is the release where it is wrong and looks right. The digest
   comes from the checkpoint's bytes and the code version from git, so there is
   nothing to remember and nothing to forget.
3. **The absences are told apart.** ``unrecorded`` predates the columns,
   ``unavailable`` was written after them and still could not name the model,
   ``unknown`` is a code version that could not be read. None of them is NULL,
   and none of them is a digest.

No Ollama, no GPU, no real checkpoint: the model reading is stubbed the same
way the rest of the suite stubs it.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import create_engine, inspect, text

from conftest import sign_in
from aoi_agent.graph import flow
from aoi_agent.provenance import (
    UNAVAILABLE,
    UNKNOWN,
    UNRECORDED,
    DecisionProvenance,
    ReviewerIdentity,
    checkpoint_digest,
    code_version,
)
from aoi_agent.station import app as station_app
from aoi_agent.station import service
from aoi_agent.store import boards, dispositions, escalations
from aoi_agent.store.models import (
    Board,
    BoardDisposition,
    CandidateRecord,
    ReviewDecision,
    create_all,
    make_session_factory,
)
from test_graph import STUB_DIGEST, StubClient, stub_tools  # noqa: F401  (fixture)

STEM = "20085293"
REFERENCE = f"{STEM}#0"

MIKE = ReviewerIdentity.signed_in("mike")

ANOTHER = DecisionProvenance(
    model_digest="sha256:ffffffffffffffff",
    thresholds={"dismiss": 0.915},
    code_version="deadbeef",
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """One board, two flagged regions, no decisions yet."""
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


def stub_provenance(digest: str = STUB_DIGEST) -> DecisionProvenance:
    return DecisionProvenance(
        model_digest=digest, thresholds={"dismiss": 0.915}, code_version="test"
    )


# ---- what identifies a set of weights -----------------------------------


def test_the_digest_follows_the_bytes_and_not_the_path(tmp_path):
    """``models/reverifier.pt`` is a slot, not an identity.

    Every training run writes that same path. A decision recorded against the
    filename is recorded against whatever happens to be there when someone next
    looks, which is the failure mode a ``model_version`` field has as well and
    a hash does not.
    """
    checkpoint = tmp_path / "reverifier.pt"
    checkpoint.write_bytes(b"weights of the first training round")
    first = checkpoint_digest(checkpoint)

    checkpoint.write_bytes(b"weights of the second training round")
    second = checkpoint_digest(checkpoint)

    assert first != second, "the same path held two models and said one thing"
    assert first.startswith("sha256:")


def test_the_same_weights_give_the_same_digest_from_anywhere(tmp_path):
    """A checkpoint copied to a second machine is the same checkpoint."""
    (tmp_path / "a.pt").write_bytes(b"identical weights")
    (tmp_path / "b.pt").write_bytes(b"identical weights")
    assert checkpoint_digest(tmp_path / "a.pt") == checkpoint_digest(tmp_path / "b.pt")


def test_the_code_version_is_read_from_the_environment(monkeypatch):
    """Read, not declared. A constant in the source is the bump-me field again,
    and a container with no git says ``unknown`` rather than something stale."""
    import aoi_agent.provenance as provenance

    monkeypatch.setattr(provenance, "_code_version", None)
    monkeypatch.setenv("AOI_AGENT_CODE_VERSION", "build-4711")
    assert provenance.code_version() == "build-4711"

    monkeypatch.setattr(provenance, "_code_version", None)
    monkeypatch.delenv("AOI_AGENT_CODE_VERSION")
    monkeypatch.setattr(provenance, "_from_git", lambda: None)
    assert provenance.code_version() == UNKNOWN


def test_this_tree_can_say_what_code_it_is(monkeypatch):
    import aoi_agent.provenance as provenance

    monkeypatch.setattr(provenance, "_code_version", None)
    monkeypatch.delenv("AOI_AGENT_CODE_VERSION", raising=False)
    assert code_version() != ""


# ---- the guard ----------------------------------------------------------


@pytest.mark.parametrize("source", ["model", "agent"])
def test_an_automated_decision_cannot_be_written_without_provenance(store, source):
    """The mechanism. Everything else on this page is storage."""
    with pytest.raises(ValueError, match="without provenance"):
        boards.record_decision(REFERENCE, "open", source)

    with store() as session:
        assert session.query(ReviewDecision).count() == 0


@pytest.mark.parametrize("absence", [UNRECORDED, UNAVAILABLE, ""])
def test_a_statement_of_absence_is_not_provenance(store, absence):
    """``unrecorded`` is what the migration writes on rows that predate the
    columns. Accepting it from a live caller would let a decision launder
    itself as a historical one."""
    with pytest.raises(ValueError, match="without provenance"):
        boards.record_decision(
            REFERENCE, "open", "agent", provenance=stub_provenance(absence)
        )


def test_an_automated_decision_carries_the_weights_the_thresholds_and_the_code(
    store, stub_tools  # noqa: F811
):
    """End to end, through the flow the station actually runs."""
    stub_tools["classify"]["confidence"] = 0.93
    graph = flow.build_graph(StubClient(verdict="short"), InMemorySaver())
    service.start_review(graph, REFERENCE)

    with store() as session:
        row = session.query(ReviewDecision).one()
    assert row.source == "agent"
    assert row.model_digest == STUB_DIGEST
    assert row.code_version not in (None, "", UNRECORDED)
    assert '"escalate_below"' in row.thresholds_json
    assert str(flow.ESCALATE_BELOW) in row.thresholds_json


def test_the_thresholds_on_the_record_are_the_ones_that_routed_it(store, stub_tools):  # noqa: F811
    """The operating point moves -- ``ESCALATE_BELOW`` moved on 2026-08-23 --
    and the same weights disposition differently either side of the move. A row
    naming the model and not the threshold names half of what decided it."""
    import json

    stub_tools["classify"]["confidence"] = 0.93
    service.start_review(
        flow.build_graph(StubClient(verdict="short"), InMemorySaver()), REFERENCE
    )
    with store() as session:
        recorded = json.loads(session.query(ReviewDecision).one().thresholds_json)

    assert recorded == {
        "confident": flow.CONFIDENT,
        "dismiss": flow.DEFAULT_DISMISS_THRESHOLD,
        "escalate_below": flow.ESCALATE_BELOW,
    }


# ---- the operator's half ------------------------------------------------


def test_an_operators_answer_is_attributed_to_the_model_they_were_shown(
    store, stub_tools  # noqa: F811
):
    """Not to whatever checkpoint is on disk when they get to the queue.

    The provenance rides in the graph state, so it comes back through the
    checkpointer with everything else the operator was shown. An escalation
    answered a week after a retrain still names the weights that raised it.
    """
    graph = flow.build_graph(StubClient(confident=False), InMemorySaver())
    service.start_review(graph, REFERENCE)
    service.resume_review(graph, REFERENCE, "copper", MIKE)

    with store() as session:
        human = (
            session.query(ReviewDecision)
            .filter(ReviewDecision.source == "human")
            .one()
        )
    assert human.model_digest == STUB_DIGEST


def test_an_operators_answer_is_never_refused_for_want_of_provenance(store):
    """A judgement is worth more than a field.

    A run checkpointed before provenance existed resumes carrying none. Turning
    that into an exception would throw away an operator's label -- the scarcest
    thing in the system and the next training round's ground truth -- to protect
    a column. The gap is named instead.
    """
    assert boards.record_decision(REFERENCE, "short", "human", identity=MIKE)

    with store() as session:
        row = session.query(ReviewDecision).one()
    assert row.model_digest == UNAVAILABLE
    assert row.model_digest != UNRECORDED, "two different absences, two words"
    assert row.code_version not in (None, "")


# ---- migrating a store that already holds corrections -------------------


def _old_schema(url: str) -> None:
    """The table as it stood before provenance, with a row in it."""
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE review_decisions ("
            " id INTEGER PRIMARY KEY, candidate_id INTEGER, verdict VARCHAR(16),"
            " source VARCHAR(16), reviewer VARCHAR(64), rationale VARCHAR(2048),"
            " explanation_status VARCHAR(16), decided_at DATETIME)"
        ))
        connection.execute(text(
            "INSERT INTO review_decisions (candidate_id, verdict, source, reviewer)"
            " VALUES (1, 'open', 'human', 'an operator whose label still counts')"
        ))


def test_the_columns_reach_a_store_that_already_holds_corrections(tmp_path):
    """The corrections are the next training round's labels. Gaining a column
    must not mean rebuilding the store, and this builds the old schema
    explicitly rather than trusting that it would."""
    url = f"sqlite:///{tmp_path / 'old.db'}"
    _old_schema(url)

    create_all(url)

    engine = create_engine(url)
    columns = {c["name"] for c in inspect(engine).get_columns("review_decisions")}
    assert {"model_digest", "thresholds_json", "code_version"} <= columns
    with engine.begin() as connection:
        row = connection.execute(text(
            "SELECT reviewer, model_digest FROM review_decisions"
        )).one()
    assert row[0] == "an operator whose label still counts"


def test_a_row_written_before_the_columns_is_not_a_row_with_no_value(tmp_path):
    """``NULL`` would have to mean both "predates the column" and "written
    without provenance". The second is the thing the guard exists to prevent,
    so the first is stamped and NULL is left meaning nothing at all."""
    url = f"sqlite:///{tmp_path / 'old.db'}"
    _old_schema(url)
    create_all(url)

    engine = create_engine(url)
    with engine.begin() as connection:
        digest, thresholds, version = connection.execute(text(
            "SELECT model_digest, thresholds_json, code_version FROM review_decisions"
        )).one()
        nulls = connection.execute(text(
            "SELECT count(*) FROM review_decisions WHERE model_digest IS NULL"
        )).scalar()

    assert (digest, thresholds, version) == (UNRECORDED, UNRECORDED, UNRECORDED)
    assert nulls == 0


def test_the_backfill_does_not_reach_rows_written_afterwards(tmp_path, monkeypatch):
    """Stamping a new row ``unrecorded`` would hide the failure this is for."""
    url = f"sqlite:///{tmp_path / 'old.db'}"
    _old_schema(url)
    create_all(url)

    factory = make_session_factory(url)
    monkeypatch.setattr(boards, "_session_factory", factory)
    with factory() as session:
        board = Board(
            stem=STEM, split="test", lot_id="L", line_id="L2", machine_id="M22",
            shift="A", inspected_at=datetime(2026, 8, 20, 9, 0),
        )
        session.add(board)
        session.flush()
        session.add(CandidateRecord(
            board_id=board.id, index_on_board=0, x1=1, y1=1, x2=2, y2=2, area=1,
            predicted_class="open", confidence=0.9, false_call_probability=0.1,
        ))
        session.commit()

    boards.record_decision(REFERENCE, "open", "agent", provenance=stub_provenance())
    create_all(url)  # a second migration pass, as `--migrate-only` would run

    with factory() as session:
        digests = {row.model_digest for row in session.query(ReviewDecision).all()}
    assert digests == {UNRECORDED, STUB_DIGEST}


# ---- the board-level record ---------------------------------------------


def _decide(reference: str, verdict: str, provenance=None) -> None:
    boards.record_decision(
        reference, verdict, "agent", provenance=provenance or stub_provenance()
    )


def test_a_board_whose_regions_were_all_dismissed_is_released(store):
    _decide(REFERENCE, "false_call")
    _decide(f"{STEM}#1", "false_call")

    record = dispositions.record(STEM)
    assert record["disposition"] == "released"
    assert record["confirmed_count"] == 0


def test_a_board_carrying_a_confirmed_defect_is_held(store):
    _decide(REFERENCE, "false_call")
    _decide(f"{STEM}#1", "short")

    assert dispositions.record(STEM)["disposition"] == "held"


def test_a_board_with_a_region_still_on_the_queue_is_not_released(store, stub_tools):  # noqa: F811
    """"Nobody has looked at it yet" is not a release.

    This is the reading that ships a board: every region that has been decided
    is a dismissal, so an aggregate over decisions alone comes back clean while
    a person is still holding the one that matters.
    """
    _decide(REFERENCE, "false_call")
    graph = flow.build_graph(StubClient(confident=False), InMemorySaver())
    service.start_review(graph, f"{STEM}#1")

    assessment = dispositions.assess(STEM)
    assert assessment["pending_count"] == 1
    assert assessment["disposition"] == "held"


def test_the_board_record_names_the_model_the_thresholds_and_the_code(store):
    _decide(REFERENCE, "false_call")
    _decide(f"{STEM}#1", "false_call")

    row = dispositions.record(STEM)
    assert row["model_digest"] == STUB_DIGEST
    assert row["code_version"] not in (None, "", UNRECORDED)
    assert "escalate_below" in row["thresholds_json"]


def test_two_models_beneath_one_board_are_not_reported_as_one(store):
    """A board part-decided before a retrain and part after was not released
    under a model. Naming either one of them is the comfortable lie."""
    _decide(REFERENCE, "false_call")
    _decide(f"{STEM}#1", "false_call", provenance=ANOTHER)

    assert dispositions.record(STEM)["model_digest"] == dispositions.MIXED


def test_board_dispositions_accumulate_rather_than_overwrite(store):
    """Held on Monday, released on Tuesday after an operator answered: two
    rows, and the pair is the record. An overwritten disposition is a board
    that was never held."""
    _decide(REFERENCE, "short")
    _decide(f"{STEM}#1", "false_call")
    dispositions.record(STEM)

    boards.record_decision(REFERENCE, "false_call", "human", identity=MIKE)
    dispositions.record(STEM, decided_by="mike")

    history = dispositions.history(STEM)
    assert [row["disposition"] for row in history] == ["released", "held"]
    assert history[0]["decided_by"] == "mike"


def test_the_operator_who_answered_the_last_region_is_the_authority(
    store, stub_tools  # noqa: F811
):
    """"On whose authority" is the half of the auditor's question that a
    timestamp cannot answer."""
    _decide(REFERENCE, "false_call")
    graph = flow.build_graph(StubClient(confident=False), InMemorySaver())
    service.start_review(graph, f"{STEM}#1")
    service.resume_review(graph, f"{STEM}#1", "false_call", MIKE)

    latest = dispositions.latest(STEM)
    assert latest["decided_by"] == "mike"
    assert latest["disposition"] == "released"


def test_a_board_still_holding_a_question_gets_no_disposition_row(
    store, stub_tools  # noqa: F811
):
    """A row per unanswered region would bury the one that means something."""
    _decide(REFERENCE, "false_call")
    graph = flow.build_graph(StubClient(confident=False), InMemorySaver())
    service.start_review(graph, f"{STEM}#1")

    assert service.settle_board(REFERENCE) is None
    with store() as session:
        assert session.query(BoardDisposition).count() == 0


def test_the_board_record_never_carries_the_ground_truth(store):
    """The same boundary the queue keeps, enforced at the same place: the dict.

    This page is one link from a region an operator is judging, and their
    answer is the next training round's label.
    """
    _decide(REFERENCE, "false_call")
    rows = dispositions.decision_provenance(STEM)

    assert rows
    for row in rows:
        assert "ground_truth" not in row
        assert "mousebite" not in str(row.values())


# ---- where it answers the question --------------------------------------


def test_the_board_page_shows_what_decided_each_region(store, monkeypatch, operators):
    monkeypatch.setattr(station_app, "_graph", object())
    _decide(REFERENCE, "false_call")
    dispositions.record(STEM)

    body = sign_in(TestClient(station_app.app)).get(f"/board/{STEM}").text
    assert STUB_DIGEST in body
    assert "released" in body or "held" in body
    assert "mousebite" not in body, "the answer key is not on this page either"


def test_the_board_page_is_404_for_a_board_the_store_does_not_hold(store, monkeypatch,
                                                                  operators):
    monkeypatch.setattr(station_app, "_graph", object())
    client = sign_in(TestClient(station_app.app))
    assert client.get("/board/99999999").status_code == 404


def test_the_cli_answers_the_auditors_question(store, capsys):
    from aoi_agent.cli import main

    _decide(REFERENCE, "false_call")
    _decide(f"{STEM}#1", "short")
    dispositions.record(STEM)

    assert main(["provenance", STEM]) == 0
    out = capsys.readouterr().out
    assert "HELD" in out
    assert STUB_DIGEST in out
    assert "escalate_below" in out


def test_the_escalation_queue_is_untouched_by_any_of_this(store, stub_tools):  # noqa: F811
    """A guard that changes who reaches a person would be a worse defect than
    the one it fixes."""
    graph = flow.build_graph(StubClient(confident=False), InMemorySaver())
    service.start_review(graph, REFERENCE)
    assert [row["reference"] for row in escalations.pending()] == [REFERENCE]
