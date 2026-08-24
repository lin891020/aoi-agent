"""The review station: the queue, the hand-off, and what the operator is shown.

These run against a temporary SQLite store and a stubbed LLM, so they need
neither Ollama nor a GPU. The image endpoints need DeepPCB and are marked.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from aoi_agent.graph import flow
from aoi_agent.provenance import ReviewerIdentity
from aoi_agent.station import app as station_app
from aoi_agent.station import service
from aoi_agent.store import boards, escalations
from aoi_agent.store.models import Board, CandidateRecord, create_all, make_session_factory
from conftest import read_in, sign_in
from test_graph import StubClient, stub_tools  # noqa: F401  (fixture)

STEM = "20085293"
REFERENCE = f"{STEM}#0"

#: The operator the suite signs in as, as the store records them.
MIKE = ReviewerIdentity.signed_in("mike")


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A store holding one board with two flagged regions."""
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
                    # The answer key. Nothing the operator sees may contain it.
                    ground_truth="mousebite",
                )
            )
        session.commit()
    return factory


@pytest.fixture
def graph(stub_tools):  # noqa: F811
    """The real flow with a model that always declines to decide."""
    return flow.build_graph(StubClient(confident=False), InMemorySaver())


@pytest.fixture
def client(store, graph, monkeypatch, operators):
    """Signed in, because every page is behind a session now.

    The sign-in is a real form post against the real credential file, not a
    fixture that reaches past the check -- see `conftest.sign_in`. What an
    unauthenticated client gets instead is `tests/test_attribution.py`.
    """
    monkeypatch.setattr(station_app, "_graph", graph)
    return sign_in(TestClient(station_app.app))


# ---- the queue ----------------------------------------------------------


def test_an_escalated_run_lands_on_the_queue(store, graph):
    state = service.start_review(graph, REFERENCE)

    assert "__interrupt__" in state
    queue = escalations.pending()
    assert [row["reference"] for row in queue] == [REFERENCE]
    assert queue[0]["reason"] == "stub"


def test_a_settled_run_never_reaches_the_queue(store, stub_tools):  # noqa: F811
    """The queue is for what the agent could not settle, not for everything."""
    stub_tools["classify"]["confidence"] = 0.93
    settled = flow.build_graph(StubClient(confident=True, verdict="short"), InMemorySaver())
    state = service.start_review(settled, REFERENCE)

    assert state["decided_by"] == "agent"
    assert escalations.pending() == []


def test_a_region_already_waiting_is_not_re_run(store, graph):
    """Re-running would spend another LLM call and could hand the operator a
    different rationale than the one already on their screen."""
    service.start_review(graph, REFERENCE)
    again = service.start_review(graph, REFERENCE)

    assert "already_pending" in again
    assert len(escalations.pending()) == 1


def test_answering_resolves_the_queue_entry_and_records_the_decision(store, graph):
    service.start_review(graph, REFERENCE)
    state = service.resume_review(graph, REFERENCE, "mousebite", MIKE)

    assert state["decided_by"] == "human"
    assert escalations.pending() == []
    assert escalations.get(REFERENCE)["status"] == "resolved"

    recorded = boards.corrections()
    assert recorded[0]["human_said"] == "mousebite"
    assert recorded[0]["overruled"] is True


def test_a_verdict_outside_the_class_list_is_refused(store, graph):
    service.start_review(graph, REFERENCE)
    with pytest.raises(ValueError):
        service.resume_review(graph, REFERENCE, "looks bad", MIKE)


# ---- the pages ----------------------------------------------------------


def test_the_queue_page_lists_what_is_waiting(client, graph):
    service.start_review(graph, REFERENCE)
    page = client.get("/").text

    assert REFERENCE in page
    assert "open" in page


def test_the_queue_page_says_so_when_it_is_empty(client):
    assert "Nothing waiting" in read_in(client, "en").get("/").text


def test_the_station_shows_the_evidence_the_agent_had(client, graph):
    service.start_review(graph, REFERENCE)
    page = client.get(f"/c/{STEM}/0").text

    assert "0.550" in page, "the model's confidence"
    assert "M22" in page, "the production context"
    assert "critical defect" in page, "the retrieved acceptance criteria"
    assert "stub" in page, "why the agent handed over"


def test_the_ground_truth_never_leaves_the_store(client, graph):
    """The operator's answer is the next training round's label. Showing them
    the answer first collects an echo, not a judgement.

    Checked at the boundary rather than by searching the HTML: the ground truth
    is always one of the seven class names, and those are on the page anyway as
    the verdict buttons, so grepping the HTML for a class name proves nothing.
    What must hold is that no dict the templates are handed carries the field.
    """
    service.start_review(graph, REFERENCE)

    assert "ground_truth" not in boards.resolve_candidate(REFERENCE)
    assert "ground_truth" not in escalations.get(REFERENCE)
    assert "ground_truth" not in escalations.pending()[0]


def test_submitting_a_verdict_resumes_the_run(client, graph):
    service.start_review(graph, REFERENCE)
    response = client.post(
        f"/c/{STEM}/0/verdict",
        data={"verdict": "mousebite"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert escalations.pending() == []
    assert boards.corrections()[0]["reviewer"] == "mike"


def test_a_second_verdict_on_the_same_region_is_ignored(client, graph):
    """Two operators on one region, or a back button. The first answer stands:
    two labels on one region silently corrupts the training set."""
    service.start_review(graph, REFERENCE)
    client.post(f"/c/{STEM}/0/verdict", data={"verdict": "mousebite"},
                follow_redirects=False)
    client.post(f"/c/{STEM}/0/verdict", data={"verdict": "short"},
                follow_redirects=False)

    recorded = boards.corrections()
    assert len(recorded) == 1
    assert recorded[0]["human_said"] == "mousebite"


def test_an_invented_verdict_is_rejected_at_the_edge(client, graph):
    service.start_review(graph, REFERENCE)
    response = client.post(f"/c/{STEM}/0/verdict", data={"verdict": "probably_fine"})

    assert response.status_code == 400
    assert escalations.pending(), "the region stays on the queue"


def test_next_walks_the_queue_and_ends_at_the_queue_page(client, graph):
    service.start_review(graph, REFERENCE)
    service.start_review(graph, f"{STEM}#1")

    first = client.get("/next", follow_redirects=False)
    assert first.headers["location"] == f"/c/{STEM}/0"

    skipped = client.get(f"/next/{STEM}/0", follow_redirects=False)
    assert skipped.headers["location"] == f"/c/{STEM}/1"

    client.post(f"/c/{STEM}/0/verdict", data={"verdict": "open"}, follow_redirects=False)
    client.post(f"/c/{STEM}/1/verdict", data={"verdict": "short"}, follow_redirects=False)
    assert client.get("/next", follow_redirects=False).headers["location"] == "/"


def test_an_unknown_region_is_a_404(client):
    assert client.get(f"/c/{STEM}/99").status_code == 404
    assert client.get("/c/nosuchboard/0").status_code == 404


def test_the_queue_count_endpoint_tracks_the_queue(client, graph):
    assert client.get("/queue-count").text == "0"
    service.start_review(graph, REFERENCE)
    assert client.get("/queue-count").text == "1"


# ---- the images ---------------------------------------------------------


@pytest.mark.dataset
def test_the_triptych_renders_three_panels(client, graph):
    from aoi_agent.station.images import CONTEXT_SIZE, PANEL_GAP, SCALE

    response = client.get(f"/c/{STEM}/0/triptych.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    import io

    from PIL import Image

    image = Image.open(io.BytesIO(response.content))
    assert image.size == (CONTEXT_SIZE * SCALE * 3 + PANEL_GAP * 2, CONTEXT_SIZE * SCALE)


@pytest.mark.dataset
def test_the_model_patch_is_the_window_the_model_saw(client):
    from aoi_agent.station.images import PANEL_GAP, SCALE
    from aoi_agent.vision.patches import PATCH_SIZE

    import io

    from PIL import Image

    response = client.get(f"/c/{STEM}/0/patch.png")
    image = Image.open(io.BytesIO(response.content))
    side = PATCH_SIZE * SCALE * 2
    assert image.size == (side * 3 + PANEL_GAP * 2, side)


# ---- corrections -------------------------------------------------------


def _answer(client, index: int, verdict: str) -> None:
    """Answer one region as whoever the client signed in as."""
    client.post(f"/c/{STEM}/{index}/verdict", data={"verdict": verdict},
                follow_redirects=False)


def test_the_corrections_page_says_so_when_nothing_is_recorded(client):
    body = read_in(client, "en").get("/corrections").text
    assert "No human decisions recorded yet" in body


def test_an_answered_region_appears_in_corrections(client, graph):
    service.start_review(graph, REFERENCE)
    _answer(client, 0, "copper")

    page = read_in(client, "en").get("/corrections").text
    assert REFERENCE in page
    assert "overruled" in page
    assert "mike" in page


def test_agreeing_with_the_model_is_recorded_but_not_counted_as_a_correction(
    client, graph
):
    """Agreement is still a label, and still worth keeping -- it just is not a
    correction, and conflating the two inflates the overrule rate."""
    service.start_review(graph, REFERENCE)
    _answer(client, 0, "open")  # the model also said `open`

    summary = boards.correction_summary()
    assert summary["total"] == 1
    assert summary["overruled"] == 0


def test_the_summary_groups_by_what_the_model_said(client, graph):
    service.start_review(graph, REFERENCE)
    _answer(client, 0, "copper")
    service.start_review(graph, f"{STEM}#1")
    _answer(client, 1, "spur")

    summary = boards.correction_summary()
    assert summary["total"] == 2
    assert summary["overruled"] == 2

    by_class = {entry["model_said"]: entry for entry in summary["by_model_class"]}
    assert by_class["open"]["corrected_to"] == [{"verdict": "copper", "count": 1}]
    assert by_class["short"]["corrected_to"] == [{"verdict": "spur", "count": 1}]
    assert by_class["open"]["overruled_share"] == 1.0


def test_the_summary_ranks_the_worst_class_first(store):
    """The point of the aggregate is triage, so the class that keeps getting
    overturned has to be the one at the top."""
    from aoi_agent.store.boards import record_decision

    record_decision(REFERENCE, "copper", "human", MIKE)
    record_decision(REFERENCE, "spur", "human", MIKE)
    record_decision(f"{STEM}#1", "short", "human", MIKE)  # agrees with the model

    summary = boards.correction_summary()
    assert summary["by_model_class"][0]["model_said"] == "open"
    assert summary["by_model_class"][0]["overruled"] == 2


def test_the_corrections_page_does_not_serve_the_ground_truth(client, graph):
    """Engineering-facing, but one link from the queue an operator is using."""
    service.start_review(graph, REFERENCE)
    _answer(client, 0, "copper")

    for row in boards.corrections():
        assert "ground_truth" not in row
    assert "ground_truth" not in boards.correction_summary()


def test_raised_and_resolved_are_stamped_by_the_same_clock(store, graph):
    """One row, two timestamps, and they must be comparable.

    The database stamps `raised_at`; if Python stamped `resolved_at` the two
    would sit in different time zones and a resolved escalation could read as
    having been answered before it was raised.
    """
    from sqlalchemy import select

    from aoi_agent.store.models import Escalation

    service.start_review(graph, REFERENCE)
    service.resume_review(graph, REFERENCE, "copper", MIKE)

    with boards.session_factory()() as session:
        row = session.execute(select(Escalation)).scalar()

    assert row.resolved_at is not None
    assert row.resolved_at >= row.raised_at


# ---------------------------------------------------------------------------
# The acceptance ruler
# ---------------------------------------------------------------------------

def _last_human_measurement():
    """The reading on the most recent operator decision, through the store's
    own session factory -- the fixture redirects that, so this follows it."""
    from sqlalchemy import select

    from aoi_agent.store.boards import session_factory
    from aoi_agent.store.models import ReviewDecision

    with session_factory()() as session:
        return session.execute(
            select(ReviewDecision.measurement)
            .where(ReviewDecision.source == "human")
            .order_by(ReviewDecision.id.desc())
            .limit(1)
        ).scalar()


def test_the_ruler_is_offered_only_where_a_ratio_decides(client, graph):
    """`open` and `short` admit no acceptable instance under WI-201 and WI-202.
    A ruler under one of those implies a limit that no document contains, which
    is the same failure as quoting the wrong class's criteria -- one surface
    over."""
    service.start_review(graph, REFERENCE)
    page = client.get(f"/c/{STEM}/0").text

    assert 'id="ruler"' in page, "the panel is templated for every class"
    assert "measure.js" in page, "and the geometry is vendored, not inlined"
    # ...and the script returns before showing it when the class has no
    # criterion, which is `measure.js`'s decision and is tested under node.
    assert "criterionFor" in page


def test_a_measurement_reaches_the_record_with_the_verdict(client, graph):
    """A measurement nobody stores is a measurement that never happened --
    WI-300's argument about corrections, one level down. The next reviewer of
    this region has to be able to tell a measured judgement from a guess."""
    service.start_review(graph, REFERENCE)
    reading = "nominal conductor width -> remaining width = 84.2% (>=80%, within)"
    client.post(f"/c/{STEM}/0/verdict",
                data={"verdict": "mousebite", "measurement": reading},
                follow_redirects=False)

    assert _last_human_measurement() == reading


def test_answering_without_measuring_stores_nothing_rather_than_something(
    client, graph
):
    """`NULL` here means what it says and is allowed to. Unlike provenance this
    is not two absences wearing one value: nobody measured, and that is a fact
    about the decision worth keeping."""
    service.start_review(graph, REFERENCE)
    client.post(f"/c/{STEM}/0/verdict", data={"verdict": "open"},
                follow_redirects=False)

    assert _last_human_measurement() is None
