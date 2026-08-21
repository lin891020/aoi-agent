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
from aoi_agent.station import app as station_app
from aoi_agent.station import service
from aoi_agent.store import boards, escalations
from aoi_agent.store.models import Board, CandidateRecord, create_all, make_session_factory
from test_graph import StubClient, stub_tools  # noqa: F401  (fixture)

STEM = "20085293"
REFERENCE = f"{STEM}#0"


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
def client(store, graph, monkeypatch):
    monkeypatch.setattr(station_app, "_graph", graph)
    return TestClient(station_app.app)


# ---- the queue ----------------------------------------------------------


def test_an_escalated_run_lands_on_the_queue(store, graph):
    state = service.start_review(graph, REFERENCE)

    assert "__interrupt__" in state
    queue = escalations.pending()
    assert [row["reference"] for row in queue] == [REFERENCE]
    assert queue[0]["reason"] == "stub"


def test_a_settled_run_never_reaches_the_queue(store, stub_tools):  # noqa: F811
    """The queue is for what the agent could not settle, not for everything."""
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
    state = service.resume_review(graph, REFERENCE, "mousebite", "mike")

    assert state["decided_by"] == "human"
    assert escalations.pending() == []
    assert escalations.get(REFERENCE)["status"] == "resolved"

    recorded = boards.corrections()
    assert recorded[0]["human_said"] == "mousebite"
    assert recorded[0]["overruled"] is True


def test_a_verdict_outside_the_class_list_is_refused(store, graph):
    service.start_review(graph, REFERENCE)
    with pytest.raises(ValueError):
        service.resume_review(graph, REFERENCE, "looks bad", "mike")


# ---- the pages ----------------------------------------------------------


def test_the_queue_page_lists_what_is_waiting(client, graph):
    service.start_review(graph, REFERENCE)
    page = client.get("/").text

    assert REFERENCE in page
    assert "open" in page


def test_the_queue_page_says_so_when_it_is_empty(client):
    assert "Nothing waiting" in client.get("/").text


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
        data={"verdict": "mousebite", "reviewer": "mike"},
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
