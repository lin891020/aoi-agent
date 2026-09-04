"""The front door shows the denominator, not the failures.

Until 2026-09-05 `/` was the queue: the regions the agent could *not* settle.
`/boards` gave the settled boards an index, but the door still opened on a
list of failures, and a reviewer took them for the system. Now `/` is one
sentence, six figures and three doors; the queue lives at `/queue`.

Three properties.

1. **The figures are aggregates over their tables**, never the length of a
   page -- the queue badge's own defect, written down once more.
2. **Nothing region-level is rendered on the door.** No reference, no class,
   and therefore no `ground_truth` can leak through it.
3. **The door does not stand between an operator and their work.** `/next`
   with nothing waiting lands on the queue, which says so, not on the cover.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from conftest import TEST_OPERATOR, TEST_SECRET, read_in, sign_in
from fastapi.testclient import TestClient

from aoi_agent.provenance import DecisionProvenance
from aoi_agent.station import app as station_app
from aoi_agent.store import boards, dispositions, escalations
from aoi_agent.store.models import (
    Board,
    CandidateRecord,
    Escalation,
    create_all,
    make_session_factory,
)

PROVENANCE = DecisionProvenance(
    model_digest="sha256:" + "a" * 16, thresholds={"dismiss": 0.915}, code_version="test"
)

#: Two released, one held, one waiting on a person, and one region nobody
#: could judge -- so every figure on the door is different from every other.
STEMS = ["20085290", "20085291", "20085292", "20085293"]


@pytest.fixture
def store(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'test.db'}"
    create_all(url)
    factory = make_session_factory(url)
    monkeypatch.setattr(boards, "_session_factory", factory)

    with factory() as session:
        candidate_ids: dict[str, list[int]] = {}
        for offset, stem in enumerate(STEMS):
            board = Board(
                stem=stem, split="test", lot_id="LOT-2201", line_id="L2",
                machine_id="M22", shift="A",
                inspected_at=datetime(2026, 8, 20, 9, 0, tzinfo=UTC) + timedelta(hours=offset),
            )
            session.add(board)
            session.flush()
            candidate_ids[stem] = []
            for index in range(2):
                candidate = CandidateRecord(
                    board_id=board.id, index_on_board=index,
                    x1=100, y1=120, x2=140, y2=155, area=1400,
                    predicted_class="open", confidence=0.99,
                    false_call_probability=0.01,
                    # The answer key. It must not reach the door either.
                    ground_truth="mousebite",
                )
                session.add(candidate)
                session.flush()
                candidate_ids[stem].append(candidate.id)
        # The fourth board waits on a person: one region on the queue and one
        # handed back, so both counts on the door are non-zero and distinct
        # from each other.
        session.add(Escalation(candidate_id=candidate_ids[STEMS[3]][0],
                               thread_id="t-waiting", reason="stub", status="pending"))
        session.add(Escalation(candidate_id=candidate_ids[STEMS[3]][1],
                               thread_id="t-deferred", reason="stub", status="deferred"))
        session.commit()

    def decide(stem: str, verdicts: tuple[str, str]) -> None:
        for index, verdict in enumerate(verdicts):
            boards.record_decision(
                f"{stem}#{index}", verdict, "agent", provenance=PROVENANCE
            )

    for stem in STEMS[:2]:
        decide(stem, ("false_call", "false_call"))
    decide(STEMS[2], ("false_call", "open"))
    for stem in STEMS[:3]:
        dispositions.record(stem)
    return factory


@pytest.fixture
def client(store, operators):
    return sign_in(TestClient(station_app.app))


def test_the_front_door_shows_the_denominator(client):
    counts = dispositions.board_counts()
    assert (counts["released"], counts["held"], counts["waiting"]) == (2, 1, 1)

    page = read_in(client, "en").get("/").text

    assert "The line right now" in page
    for figure in (counts["total"], counts["held"], counts["released"], counts["waiting"]):
        assert f"<b>{figure}</b>" in page or f'<b class="warn">{figure}</b>' in page
    assert f"<b>{escalations.pending_count()}</b>" in page
    assert f"<b>{escalations.deferred_count()}</b>" in page
    for door in ("/queue", "/boards", "/deferred", "/ask"):
        assert f'href="{door}"' in page, f"no door to {door}"


def test_the_figures_are_taken_over_the_table_and_not_over_a_page(client, monkeypatch):
    """The badge's defect, once more: a figure that agrees with its own page
    size is worse than none. The door renders whatever the aggregate says even
    when the page-sized listing is empty."""
    monkeypatch.setattr(station_app.escalations, "pending", lambda: [])
    monkeypatch.setattr(station_app.escalations, "pending_count", lambda: 250)

    assert "<b>250</b>" in client.get("/").text


def test_nothing_region_level_reaches_the_door(client):
    page = client.get("/").text

    assert "mousebite" not in page, "the answer key"
    for stem in STEMS:
        assert f"{stem}#" not in page, "a region reference on the cover"
        assert f"/c/{stem}/" not in page


def test_the_queue_lives_one_click_behind_the_door(client):
    door = client.get("/").text
    queue = client.get("/queue").text

    assert f"{STEMS[3]}#0" in queue
    assert f"{STEMS[3]}#0" not in door


def test_next_with_nothing_waiting_lands_on_the_queue_not_the_door(client, monkeypatch):
    """The queue says "nothing waiting"; the door would leave the operator one
    click further from finding that out."""
    monkeypatch.setattr(station_app.escalations, "pending", lambda: [])

    response = client.get("/next", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/queue"


def test_signing_in_lands_on_the_door(operators):
    """The default `next` is the front page, not the queue: the first thing a
    new session sees is the shape of the line."""
    fresh = TestClient(station_app.app)
    response = fresh.post(
        "/login", data={"name": TEST_OPERATOR, "secret": TEST_SECRET}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/"
