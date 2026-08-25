"""What the station says is waiting, against what is actually waiting.

The queue page counted with ``len(escalations.pending())`` until 2026-08-25, and
``pending`` caps at 200. So a queue of 250 rendered **"200 waiting"** with
nothing on screen saying it had been cut -- a number quietly wrong and looking
entirely normal, on the page an operator uses to judge whether they are keeping
up. Every other page carried the same figure in its header badge, so it was one
wrong number in five places.

Nothing failed. No test covered it, because every fixture in this suite holds
one or two regions and the defect only exists above two hundred.

That is what these tests are: the smallest store big enough for the bug to
exist. They are slower than the rest of the file and that is the price of
testing a limit -- a limit cannot be tested below itself.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from aoi_agent.station import app as station_app
from aoi_agent.store import boards, escalations
from aoi_agent.store.models import (
    Board,
    CandidateRecord,
    Escalation,
    ReviewDecision,
    create_all,
    make_session_factory,
)
from conftest import read_in, sign_in

#: One more than ``pending``'s cap. The point is to sit just past the limit:
#: a number far above it would also pass a fix that merely raised the cap, and
#: raising the cap is the fix this test is here to refuse.
OVER_THE_PAGE = 250


@pytest.fixture
def crowded_store(tmp_path, monkeypatch):
    """One board carrying more waiting regions than a page can show."""
    url = f"sqlite:///{tmp_path / 'crowded.db'}"
    create_all(url)
    factory = make_session_factory(url)
    monkeypatch.setattr(boards, "_session_factory", factory)

    raised = datetime(2026, 8, 25, 8, 0)
    with factory() as session:
        board = Board(
            stem="90000001", split="test", lot_id="LOT-9", line_id="L2",
            machine_id="M22", shift="A", inspected_at=raised,
        )
        session.add(board)
        session.flush()
        for index in range(OVER_THE_PAGE):
            candidate = CandidateRecord(
                board_id=board.id, index_on_board=index,
                x1=10, y1=10, x2=40, y2=40, area=900,
                predicted_class="open", confidence=0.55,
                false_call_probability=0.45, ground_truth="open",
            )
            session.add(candidate)
            session.flush()
            session.add(
                Escalation(
                    candidate_id=candidate.id, thread_id=f"thread-{index}",
                    reason="below the escalation threshold", status="pending",
                    # Distinct timestamps, so "oldest first" has something to
                    # order by and the page is a defined 200 rather than an
                    # arbitrary one.
                    raised_at=raised + timedelta(seconds=index),
                    # Every third one lost its explanation, so the unexplained
                    # figure has a value that differs between the page and the
                    # queue -- 83 within the first 200, 84 over all 250.
                    explanation_status="timed_out" if index % 3 == 0 else "ok",
                )
            )
        session.commit()
    return factory


@pytest.fixture
def crowded_client(crowded_store, operators):
    return sign_in(TestClient(station_app.app))


def test_the_store_holds_more_than_one_page(crowded_store):
    """The premise. If `pending` ever stops capping, the tests below would pass
    without exercising anything and this one says so instead."""
    assert escalations.pending_count() == OVER_THE_PAGE
    assert len(escalations.pending()) < OVER_THE_PAGE


def test_the_count_is_the_queue_and_not_the_page(crowded_store):
    """The defect, at the layer where it was introduced."""
    assert escalations.pending_count() == OVER_THE_PAGE


def test_the_queue_page_reports_everyone_who_is_waiting(crowded_client):
    """The defect, where an operator met it: the page said 200 of 250.

    Read in English and asserted against the rendered sentence, not against a
    bare ``"250"`` anywhere in the markup. The first draft of this test did the
    latter and passed under a deliberately broken build, because the page is
    full of numbers -- ``thread-250``, region references, confidences -- and one
    of them will always contain the digits you are looking for. A substring
    search over a whole page is not an assertion about the page.
    """
    body = read_in(crowded_client, "en").get("/").text

    assert f"of {OVER_THE_PAGE} waiting" in body, (
        "the page does not state the real total"
    )


def test_the_queue_page_says_how_much_it_is_not_showing(crowded_client):
    """A truncated list that does not say it is truncated reads as a whole list,
    which is the half of this defect that survives fixing the number."""
    hidden = OVER_THE_PAGE - len(escalations.pending())
    body = read_in(crowded_client, "en").get("/").text

    assert f"{hidden} are not on this page" in body, (
        "the page shows a page of rows and never mentions the rest"
    )


def test_the_header_badge_counts_the_queue_and_not_the_page(crowded_client):
    """It is on every page, so it was the same wrong number in five places."""
    assert crowded_client.get("/queue-count").text.strip() == str(OVER_THE_PAGE)


def test_the_unexplained_figure_covers_the_queue_and_not_the_page(crowded_store):
    """WI-300 requires the absence of a rationale to be *counted*. Counted over
    a screenful it cannot see the shift it exists to make visible."""
    over_page = sum(
        1 for row in escalations.pending()
        if (row["explanation_status"] or "ok") != "ok"
    )

    assert escalations.pending_unexplained_count() > over_page
    assert escalations.pending_unexplained_count() == len(
        [i for i in range(OVER_THE_PAGE) if i % 3 == 0]
    )


# ---- the same defect, one page over ------------------------------------


def test_the_corrections_summary_covers_every_decision_not_the_recent_ones(
    crowded_store,
):
    """The aggregate was worse than the truncated list above it.

    A list that stops is visibly a list. An aggregate that stops still reads as
    "where the model gets corrected" while describing the most recent slice --
    and because the slice is the recent one, a class the operators stopped
    overturning drops out and a reader concludes it was fixed.
    """
    with crowded_store() as session:
        candidates = session.query(CandidateRecord).all()
        for candidate in candidates:
            session.add(
                ReviewDecision(
                    candidate_id=candidate.id, verdict="false_call",
                    source="human", reviewer="mike", reviewer_auth="signed_in",
                    decided_at=datetime(2026, 8, 25, 9, 0),
                )
            )
        session.commit()

    summary = boards.correction_summary()

    assert summary["total"] == OVER_THE_PAGE, (
        "the summary describes only part of the record it claims to describe"
    )
    # Every one of them overruled `open` with `false_call`.
    assert summary["overruled"] == OVER_THE_PAGE


def test_the_queue_shows_when_each_region_started_waiting_labelled_utc(
    crowded_client,
):
    """The ordering rule -- longest wait goes next -- was invisible: the queue
    never showed since when. And an unlabelled clock stored in UTC and read at
    UTC+8 is an eight-hour lie by omission, which the board record and the
    corrections pages already close by saying UTC; this holds the queue to the
    same sentence."""
    body = read_in(crowded_client, "en").get("/").text

    assert "Waiting since (UTC)" in body
    # The fixture raises its escalations at 2026-08-25 08:00 plus seconds; the
    # first page row is the oldest, so its minute-precision stamp is fixed.
    assert "2026-08-25 08:00" in body
