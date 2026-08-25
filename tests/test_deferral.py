"""The operator who does not know.

This system's whole doctrine is that an uncertain region goes to a person. Until
2026-08-25 what the screen offered that person was seven certain answers, and
nothing else -- so an operator who could not judge a region had two options,
guess or walk away, and walking away left the region indistinguishable from one
nobody had reached yet.

That absence has a price already paid: five regions were clicked through without
the domain knowledge to judge them, four of the five wrong, and because nothing
could tell those labels from an expert's they all had to be deleted by hand.

These tests are about what a deferral must *not* become. It is not a verdict, it
is not a closure, and it is not a resumption -- and each of those is a way the
feature could be built that would quietly undo the reason for building it.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aoi_agent.provenance import ReviewerIdentity
from aoi_agent.station import service
from aoi_agent.store import boards, escalations
from aoi_agent.store.models import (
    Board,
    CandidateRecord,
    Escalation,
    ReviewDecision,
    create_all,
    make_session_factory,
)

STEM = "20085294"
REFERENCE = f"{STEM}#0"
THREAD = REFERENCE

MIKE = ReviewerIdentity.signed_in("mike")
SANDY = ReviewerIdentity.signed_in("sandy")


@pytest.fixture
def queued(tmp_path, monkeypatch):
    """One region on the queue, waiting for a person."""
    url = f"sqlite:///{tmp_path / 'defer.db'}"
    create_all(url)
    factory = make_session_factory(url)
    monkeypatch.setattr(boards, "_session_factory", factory)

    with factory() as session:
        board = Board(
            stem=STEM, split="test", lot_id="LOT-1", line_id="L2",
            machine_id="M22", shift="A", inspected_at=datetime(2026, 8, 25, 8, 0),
        )
        session.add(board)
        session.flush()
        candidate = CandidateRecord(
            board_id=board.id, index_on_board=0,
            x1=10, y1=10, x2=50, y2=50, area=1600,
            predicted_class="mousebite", confidence=0.62,
            false_call_probability=0.38, ground_truth="mousebite",
        )
        session.add(candidate)
        session.flush()
        session.add(
            Escalation(
                candidate_id=candidate.id, thread_id=THREAD,
                reason="below the escalation threshold", status="pending",
                raised_at=datetime(2026, 8, 25, 8, 0),
            )
        )
        session.commit()
    return factory


# ---- what a deferral is not ---------------------------------------------


def test_a_deferral_writes_no_decision(queued):
    """The one that matters most. ``review_decisions`` is the next training
    round's labels, and "I don't know" is not a label -- putting it there would
    feed the training set the exact thing this exists to keep out of it."""
    service.defer_review(REFERENCE, MIKE, note="cannot tell the notch from plating")

    with queued() as session:
        assert session.query(ReviewDecision).count() == 0


def test_a_deferral_does_not_close_the_queue_entry(queued):
    """The region still needs an answer. A deferral that resolved it would be a
    delete button wearing a humbler label."""
    service.defer_review(REFERENCE, MIKE)

    row = escalations.get(THREAD)
    assert row["status"] == escalations.DEFERRED
    assert row["status"] != escalations.RESOLVED


def test_a_deferred_region_leaves_the_flow_but_not_the_store(queued):
    """It has to go somewhere the next operator is not sent, or they meet it and
    decline it again -- and it has to stay visible, or deferring is deleting."""
    service.defer_review(REFERENCE, MIKE)

    assert escalations.pending_count() == 0
    assert escalations.deferred_count() == 1
    assert [row["reference"] for row in escalations.deferred()] == [REFERENCE]


def test_a_deferral_still_names_who_made_it(queued):
    """Weaker evidence than a verdict is still evidence: the decline count is
    what ranks the deferred queue. One that named nobody could not be weighed,
    and could not tell one operator declining twice from two declining once."""
    with pytest.raises(ValueError, match="unattributable"):
        service.defer_review(REFERENCE, ReviewerIdentity.automated())

    assert escalations.deferred_count() == 0, "the refusal must write nothing"


def test_an_answered_region_cannot_then_be_deferred(queued):
    """A stale tab must not be able to reopen a closed entry, or the store holds
    a decision and a later "nobody could decide" about the same region."""
    escalations.resolve_escalation(THREAD)

    assert service.defer_review(REFERENCE, MIKE) is False
    assert escalations.deferred_count() == 0


# ---- what it is ----------------------------------------------------------


def test_declines_accumulate_rather_than_overwrite(queued):
    """A region three people could not judge is a different object from one a
    single tired person skipped, and that ordering is information no model in
    this system produces."""
    service.defer_review(REFERENCE, MIKE, note="cannot tell the notch from plating")
    service.defer_review(REFERENCE, SANDY, note="same, and the template looks off")

    declines = escalations.declines_for(THREAD)
    assert [row["operator"] for row in declines] == ["mike", "sandy"]
    assert declines[0]["note"] == "cannot tell the notch from plating"
    assert escalations.deferred()[0]["declines"] == 2


def test_the_same_operator_may_decline_twice(queued):
    """Coming back to a region later and still not knowing is a true thing about
    the region. Deduplicating would hide that the queue is stuck."""
    service.defer_review(REFERENCE, MIKE)
    service.defer_review(REFERENCE, MIKE)

    assert len(escalations.declines_for(THREAD)) == 2


def test_declining_without_saying_why_is_allowed(queued):
    """A required box gets filled with a full stop. ``None`` means they declined
    without saying, which is real and common."""
    service.defer_review(REFERENCE, MIKE)

    assert escalations.declines_for(THREAD)[0]["note"] is None


def test_the_deferred_queue_is_ordered_by_how_many_could_not_judge(queued):
    """Hardest-looking first, which is the only ranking here that carries
    information."""
    with queued() as session:
        candidate = CandidateRecord(
            board_id=1, index_on_board=1,
            x1=60, y1=60, x2=90, y2=90, area=900,
            predicted_class="spur", confidence=0.58,
            false_call_probability=0.42, ground_truth="spur",
        )
        session.add(candidate)
        session.flush()
        session.add(
            Escalation(
                candidate_id=candidate.id, thread_id=f"{STEM}#1",
                reason="below the escalation threshold", status="pending",
                raised_at=datetime(2026, 8, 25, 8, 1),
            )
        )
        session.commit()

    service.defer_review(f"{STEM}#1", MIKE)
    service.defer_review(REFERENCE, MIKE)
    service.defer_review(REFERENCE, SANDY)

    assert [row["declines"] for row in escalations.deferred()] == [2, 1]
    assert escalations.deferred()[0]["reference"] == REFERENCE


def test_the_deferred_queue_still_hides_the_answer_key(queued):
    """Same boundary as everywhere else. A region nobody could judge is exactly
    the one where showing the answer would collect an echo."""
    service.defer_review(REFERENCE, MIKE)

    assert "ground_truth" not in escalations.deferred()[0]


def test_a_deferred_region_can_still_be_answered(queued):
    """Deferring must not strand the region. The interrupt was never consumed,
    so the ordinary answer path still works on it -- which is the whole reason
    this does not touch the graph."""
    service.defer_review(REFERENCE, MIKE)

    assert escalations.resolve_escalation(THREAD) is True
    assert escalations.get(THREAD)["status"] == escalations.RESOLVED


# ---- through the station -------------------------------------------------
#
# The store-level tests above cannot see a defect in the routes, and there was
# one: `submit_verdict` tested `status != "pending"`, so the moment a region was
# deferred it became permanently unanswerable through the station -- silently,
# by redirect, with the run still suspended and the region still on a list
# telling somebody to go and answer it. A deferral is supposed to move a region
# to another person, and a bug that turns it into a bin is the one failure this
# whole feature cannot survive.

from types import SimpleNamespace  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from aoi_agent.station import app as station_app  # noqa: E402
from conftest import read_in, sign_in  # noqa: E402


@pytest.fixture
def station(queued, operators, monkeypatch):
    class DeadGraph:
        """The verdict path is exercised in test_station.py; here the point is
        which requests reach it at all, so this records rather than runs."""

        def __init__(self):
            self.resumed = []

        def invoke(self, command, config=None):
            self.resumed.append(config)
            return {"trace": ["classify"], "provenance": None}

        def get_state(self, config):
            # The region page renders the evidence out of the checkpointer.
            # Empty is a valid state here -- these tests are about the queue
            # entry and the decline notes, not about the agent's findings.
            return SimpleNamespace(values={}, next=())

    graph = DeadGraph()
    monkeypatch.setattr(station_app, "_graph", graph)
    monkeypatch.setattr(station_app, "graph", lambda: graph)
    return sign_in(TestClient(station_app.app)), graph


def test_the_button_defers_and_writes_no_decision(station, queued):
    client, graph = station

    client.post(f"/c/{STEM}/0/defer", data={"note": "cannot tell"},
                follow_redirects=False)

    assert escalations.get(THREAD)["status"] == escalations.DEFERRED
    assert escalations.declines_for(THREAD)[0]["note"] == "cannot tell"
    assert graph.resumed == [], "deferring must not resume the suspended run"
    with queued() as session:
        assert session.query(ReviewDecision).count() == 0


def test_a_deferred_region_is_still_answerable_by_a_senior(station, senior):
    """The defect this section exists for: deferring must move a region to
    somebody, not strand it. `senior` promotes the same operator by rewriting
    the credential file, so the role comes through the real parse."""
    client, graph = station
    client.post(f"/c/{STEM}/0/defer", follow_redirects=False)

    client.post(f"/c/{STEM}/0/verdict", data={"verdict": "mousebite"},
                follow_redirects=False)

    assert graph.resumed, (
        "the verdict was redirected away: a deferred region cannot be answered"
    )
    assert escalations.get(THREAD)["status"] == escalations.RESOLVED


def test_the_deferring_operator_is_the_session_and_never_the_form(station):
    """Same rule as a verdict. A name a browser can post names whoever it
    likes, and the decline count is only worth having if it counts people."""
    client, _ = station

    client.post(f"/c/{STEM}/0/defer",
                data={"note": "x", "operator": "somebody-else"},
                follow_redirects=False)

    assert escalations.declines_for(THREAD)[0]["operator"] == "mike"


def test_a_visitor_who_has_not_signed_in_cannot_defer(queued, operators):
    """A deferral names a person, so it is behind the session like everything
    else that does."""
    anonymous = TestClient(station_app.app)

    response = anonymous.post(f"/c/{STEM}/0/defer", follow_redirects=False)

    assert response.status_code in (303, 401, 403)
    assert escalations.deferred_count() == 0


def test_the_region_page_shows_who_already_declined_it(station):
    """The handover. Rediscovering yesterday's dead end is the cost of not
    showing it."""
    client, _ = station
    client.post(f"/c/{STEM}/0/defer",
                data={"note": "cannot separate notch from plating"},
                follow_redirects=False)

    body = read_in(client, "en").get(f"/c/{STEM}/0").text

    assert "cannot separate notch from plating" in body
    assert "could not judge it" in body


def test_the_queue_links_to_what_nobody_could_judge(station):
    """Off the queue on purpose, but a thing off the queue and unlinked is a
    thing that is gone."""
    client, _ = station
    client.post(f"/c/{STEM}/0/defer", follow_redirects=False)

    assert '/deferred' in read_in(client, "en").get("/").text


def test_the_deferred_page_says_it_assigns_nothing_to_nobody(station):
    """The station has no seniority, so this list is not a work assignment. A
    list that looks like one and is not will be read as one, and the first
    person to find out is whoever assumed somebody else had it."""
    client, _ = station
    client.post(f"/c/{STEM}/0/defer", follow_redirects=False)

    body = read_in(client, "en").get("/deferred").text

    assert REFERENCE in body
    assert "not an assignment" in body


# ---- who may pick a handed-back region up --------------------------------


def test_an_operator_is_refused_rather_than_redirected(station):
    """Refused with a 403, not sent to `/next`.

    A redirect here is indistinguishable from "somebody else answered it
    first", so an operator who is not allowed would conclude the queue moved
    under them and try the next one, and the next one. The rule has to be
    legible at the moment it applies.
    """
    client, graph = station
    client.post(f"/c/{STEM}/0/defer", follow_redirects=False)

    response = client.post(f"/c/{STEM}/0/verdict", data={"verdict": "mousebite"},
                           follow_redirects=False)

    assert response.status_code == 403
    assert graph.resumed == []
    assert escalations.get(THREAD)["status"] == escalations.DEFERRED


def test_an_ordinary_region_is_answerable_by_an_ordinary_operator(station):
    """The role gates one thing. If it gated the whole queue the station would
    stop working for everyone the moment roles arrived, and this is the test
    that would have caught that."""
    client, graph = station

    response = client.post(f"/c/{STEM}/0/verdict", data={"verdict": "mousebite"},
                           follow_redirects=False)

    assert response.status_code == 303
    assert graph.resumed


def test_an_operator_may_still_hand_a_region_back(station, senior):
    """A senior who cannot read it either must be able to say so, and the
    decline count is the more useful for it."""
    client, _ = station

    client.post(f"/c/{STEM}/0/defer", follow_redirects=False)

    assert escalations.declines_for(THREAD)[0]["operator"] == "mike"


def test_the_page_tells_an_operator_they_cannot_answer_these(station, senior_elsewhere):
    """Signed in as an operator while a senior exists -- which is the only way
    to reach this notice rather than the configuration one below it."""
    client, _ = station
    client.post(f"/c/{STEM}/0/defer", follow_redirects=False)

    body = read_in(client, "en").get("/deferred").text

    assert "not answer them" in body


def test_the_page_says_when_nobody_can_answer_them_at_all(station):
    """The state where this queue grows and no error is raised anywhere: every
    operator handed a region back and none of them is senior."""
    client, _ = station
    client.post(f"/c/{STEM}/0/defer", follow_redirects=False)

    body = read_in(client, "en").get("/deferred").text

    assert "Nobody is configured as senior" in body


def test_a_senior_is_not_told_they_cannot_answer(station, senior):
    client, _ = station
    client.post(f"/c/{STEM}/0/defer", follow_redirects=False)

    body = read_in(client, "en").get("/deferred").text

    assert "can answer these" in body
    assert "Nobody is configured as senior" not in body


# ---- the board above the region -----------------------------------------


def test_a_deferral_cannot_release_a_board(queued):
    """The worst-directed defect this feature could have had, and it had it.

    The sequence is ordinary: the model dispositions a region `false_call`, the
    flow escalates it anyway, and a person opens it and says they cannot judge
    it. `dispositions.OPEN_STATUSES` listed only `pending`, so that person's
    answer took the escalation out of the open set while the model's row stayed
    the standing decision -- and the board went from **held to released**.

    An operator refusing to guess is the act that shipped the board. The
    feature built to stop uncertain regions being answered as certain would
    have been the thing certifying them, and no test at the region level could
    see it because nothing at the region level was wrong.
    """
    from aoi_agent.store import dispositions

    with queued() as session:
        candidate = session.query(CandidateRecord).one()
        session.add(
            ReviewDecision(
                candidate_id=candidate.id, verdict="false_call", source="model",
                reviewer=None, reviewer_auth="automated",
                model_digest="sha256:stub", code_version="test",
                decided_at=datetime(2026, 8, 25, 8, 1),
            )
        )
        session.commit()

    assert dispositions.assess(STEM)["disposition"] == "held"

    service.defer_review(REFERENCE, MIKE, note="cannot tell")

    assert dispositions.assess(STEM)["disposition"] == "held", (
        "an operator saying they could not judge a region released the board"
    )


def test_a_deferred_region_still_counts_as_owing_an_answer(queued):
    """The property underneath, stated separately so it survives a rewrite of
    the disposition rule: a region nobody could judge is a region still waiting
    on somebody, not a settled one."""
    from aoi_agent.store import dispositions

    service.defer_review(REFERENCE, MIKE)

    assert dispositions.assess(STEM)["pending_count"] == 1
