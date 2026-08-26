"""An index of the boards this system settled, and why the station lacked one.

The station's front page is the queue: the regions the agent could *not*
settle. `/board/<stem>` existed and rendered one board's record in full, but
the only route to it was a link on a queued region -- so the 82% the agent
settled had no page, and everything reachable from the front door was a
failure. A reviewer who opened this station read the failures and took them
for the system.

Three properties are held here, and the first is the one worth the file.

1. **The index and the board page never disagree about what stands.** Rows
   accumulate -- a board held on Monday and released on Tuesday is two rows --
   so "this board's disposition" is a rule, not a column. The rule is written
   once, in `_standing_ids`, and this file's job is to fail if a second copy of
   it ever appears and drifts.
2. **The counts are aggregates over the table, not the length of the page.**
   The queue badge counted `len()` of a capped list for a while and was the
   same wrong number in five places at once.
3. **No `ground_truth`, on this page too.** The dict boundary, not a grep of
   the markup.

No Ollama, no GPU, no checkpoint: the model reading is stubbed the way the rest
of the suite stubs it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from conftest import sign_in
from aoi_agent.provenance import DecisionProvenance
from aoi_agent.station import app as station_app
from aoi_agent.store import boards, dispositions
from aoi_agent.store.models import (
    Board,
    CandidateRecord,
    Escalation,
    create_all,
    make_session_factory,
)

DIGEST = "sha256:" + "a" * 16

PROVENANCE = DecisionProvenance(
    model_digest=DIGEST, thresholds={"dismiss": 0.915}, code_version="test"
)

#: Four boards, each with two flagged regions. Two will be dismissed clean and
#: released, two will hold a confirmed defect.
STEMS = ["20085290", "20085291", "20085292", "20085293"]


@pytest.fixture
def store(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'test.db'}"
    create_all(url)
    factory = make_session_factory(url)
    monkeypatch.setattr(boards, "_session_factory", factory)

    with factory() as session:
        for offset, stem in enumerate(STEMS):
            board = Board(
                stem=stem, split="test", lot_id="LOT-2201", line_id="L2",
                machine_id="M22", shift="A",
                inspected_at=datetime(2026, 8, 20, 9, 0) + timedelta(hours=offset),
            )
            session.add(board)
            session.flush()
            for index in range(2):
                session.add(
                    CandidateRecord(
                        board_id=board.id, index_on_board=index,
                        x1=100, y1=120, x2=140, y2=155, area=1400,
                        predicted_class="open", confidence=0.99,
                        false_call_probability=0.01,
                        ground_truth="mousebite",
                    )
                )
        session.commit()
    return factory


def _decide(stem: str, verdicts: tuple[str, str]) -> None:
    for index, verdict in enumerate(verdicts):
        boards.record_decision(
            f"{stem}#{index}", verdict, "agent", provenance=PROVENANCE
        )


def _disposition_all() -> None:
    """Two released, two held -- so no test can pass by returning everything."""
    for stem in STEMS[:2]:
        _decide(stem, ("false_call", "false_call"))
    for stem in STEMS[2:]:
        _decide(stem, ("false_call", "open"))
    for stem in STEMS:
        dispositions.record(stem)


# ---- the rule, written once ---------------------------------------------


def test_the_index_and_the_board_page_agree_on_what_stands(store):
    """The property a second definition of "latest" would break.

    `latest()` reads one board through `history`; `recent()` reads all of them
    through a window function. Two expressions of the same rule is exactly the
    shape that goes wrong quietly -- both return a real row, and they disagree
    only on the boards dispositioned more than once, which are the boards an
    auditor is asking about.
    """
    _disposition_all()
    _decide(STEMS[0], ("open", "open"))
    dispositions.record(STEMS[0])          # now held, and it was released

    by_stem = {row["board_stem"]: row for row in dispositions.recent()}

    assert len(by_stem) == len(STEMS), "one row per board, not one per decision"
    for stem in STEMS:
        assert by_stem[stem] == dispositions.latest(stem)

    assert by_stem[STEMS[0]]["disposition"] == dispositions.HELD


def test_an_earlier_disposition_is_not_what_the_index_shows(store):
    """The same property from the failing side, so the test above cannot pass
    on a store where every board has exactly one row."""
    _decide(STEMS[0], ("false_call", "false_call"))
    dispositions.record(STEMS[0])
    assert dispositions.recent()[0]["disposition"] == dispositions.RELEASED

    _decide(STEMS[0], ("open", "open"))
    dispositions.record(STEMS[0])

    rows = dispositions.recent()
    assert len(rows) == 1
    assert rows[0]["disposition"] == dispositions.HELD


# ---- the counts ----------------------------------------------------------


def test_the_counts_are_taken_over_the_table_and_not_over_the_page(store):
    """`len(recent())` is capped and `board_counts()` must not be.

    A count that agrees with its own page size is worse than no count: the page
    looks internally consistent and is wrong about the line.
    """
    _disposition_all()

    counts = dispositions.board_counts()
    assert counts == {"held": 2, "released": 2, "total": 4, "waiting": 0}
    assert len(dispositions.recent(limit=1)) == 1
    assert dispositions.board_counts()["total"] == 4


def test_the_count_survives_more_boards_than_one_page_holds(store):
    """The mutation the test above does not catch.

    With four boards, `len(recent())` and a `COUNT(*)` agree, so a count wired
    to the listing passes. `recent()` caps at 50 by default, which is the cap
    the wrong version would inherit, so the fixture has to cross it -- this is
    the shape the queue badge actually shipped with.
    """
    extra = 60
    with boards.session_factory()() as session:
        for n in range(extra):
            board = Board(
                stem=f"9900{n:04d}", split="test", lot_id="LOT-2201",
                line_id="L2", machine_id="M22", shift="A",
                inspected_at=datetime(2026, 8, 20, 9, 0) + timedelta(minutes=n),
            )
            session.add(board)
            session.flush()
            session.add(
                CandidateRecord(
                    board_id=board.id, index_on_board=0,
                    x1=100, y1=120, x2=140, y2=155, area=1400,
                    predicted_class="open", confidence=0.99,
                    false_call_probability=0.01, ground_truth="mousebite",
                )
            )
        session.commit()
    for n in range(extra):
        stem = f"9900{n:04d}"
        boards.record_decision(f"{stem}#0", "false_call", "agent",
                               provenance=PROVENANCE)
        dispositions.record(stem)

    total = dispositions.board_counts()["total"]
    assert total == extra
    assert total > len(dispositions.recent()), (
        "the count came from the page, which is capped"
    )


def test_a_board_nobody_has_run_is_in_neither_count(store):
    """Four boards exist and none is dispositioned; none of them is released.

    A board nobody has looked at is not a released board, and the fleet size is
    a fact about `boards`, not about this table.
    """
    assert dispositions.board_counts() == {
        "held": 0, "released": 0, "total": 0, "waiting": 0
    }
    assert dispositions.recent() == []
    assert dispositions.waiting() == []


# ---- filtering and truncation -------------------------------------------


@pytest.mark.parametrize(
    "status,expected", [("held", 2), ("released", 2), (None, 4)]
)
def test_the_filter_returns_only_that_disposition(store, status, expected):
    _disposition_all()
    rows = dispositions.recent(status=status)

    assert len(rows) == expected
    if status is not None:
        assert {row["disposition"] for row in rows} == {status}


def test_the_index_is_ordered_newest_first(store):
    _disposition_all()
    stamps = [row["decided_at"] for row in dispositions.recent()]
    assert stamps == sorted(stamps, reverse=True)


# ---- the page ------------------------------------------------------------


@pytest.fixture
def client(store, monkeypatch, operators):
    monkeypatch.setattr(station_app, "_graph", object())
    return sign_in(TestClient(station_app.app))


def test_the_page_lists_the_boards_and_links_to_each_record(client):
    _disposition_all()
    body = client.get("/boards").text

    for stem in STEMS:
        assert stem in body
        assert f'href="/board/{stem}"' in body


def test_the_page_never_shows_the_answer_key(client):
    """Same boundary as the queue and the board record, for the same reason."""
    _disposition_all()
    body = client.get("/boards").text

    assert "mousebite" not in body
    assert "ground_truth" not in body


def test_the_page_says_how_many_it_is_not_showing(client):
    """The queue's lesson: a page that truncates in silence reports a smaller
    line than the one running."""
    _disposition_all()
    body = client.get("/boards?limit=1").text

    assert "3 are not on this page" in body or "另外 3 片" in body


def test_a_full_page_claims_no_truncation(client):
    _disposition_all()
    body = client.get("/boards").text
    assert "not on this page" not in body


def test_an_unknown_disposition_in_the_url_is_refused(client):
    """Not ignored. A filter that silently matches everything answers a typed
    URL with a plausible page, and the counts above it are then true numbers
    answering a question nobody asked."""
    _disposition_all()
    response = client.get("/boards?status=probably")

    assert response.status_code == 400
    assert "probably" in response.text


def test_the_empty_page_says_what_to_run(client):
    body = client.get("/boards").text
    assert "aoi_agent board" in body


def test_the_index_is_reachable_from_every_page(client):
    """The page existed and was unreachable; that was the whole defect."""
    _disposition_all()
    for path in ("/", "/deferred", "/corrections"):
        assert 'href="/boards"' in client.get(path).text


def test_the_page_needs_a_signed_in_operator(store, monkeypatch, operators):
    monkeypatch.setattr(station_app, "_graph", object())
    response = TestClient(station_app.app).get("/boards", follow_redirects=False)
    assert response.status_code in (302, 303, 307)
    assert "/login" in response.headers["location"]


# ---- the third state: run, and still waiting on a person ----------------


def _queue(store, stem: str, index: int = 0, status: str = "pending") -> None:
    """Put one of a board's regions on the queue, the way a run does."""
    with store() as session:
        candidate = (
            session.query(CandidateRecord)
            .join(Board, CandidateRecord.board_id == Board.id)
            .filter(Board.stem == stem, CandidateRecord.index_on_board == index)
            .one()
        )
        session.add(
            Escalation(
                candidate_id=candidate.id, thread_id=f"{stem}#{index}",
                reason="below the escalation threshold", status=status,
                raised_at=datetime(2026, 8, 26, 13, 21),
            )
        )
        session.commit()


def test_a_board_with_a_region_on_the_queue_is_waiting_not_missing(store):
    """Fifty boards run, twenty-nine on the index. The other twenty-one had a
    region on the queue, so no row, so no line, and the reader's next question
    was where they had gone. They are a third state beside held and released,
    counted next to the total rather than inside it."""
    _queue(store, STEMS[0])

    counts = dispositions.board_counts()
    assert counts["waiting"] == 1
    assert counts["total"] == 0, "waiting is not a disposition and not in the total"
    assert [row["board_stem"] for row in dispositions.waiting()] == [STEMS[0]]
    assert dispositions.recent() == []


def test_a_waiting_board_carries_the_page_s_own_counts_and_no_invented_ones(store):
    _queue(store, STEMS[0])

    row = dispositions.waiting()[0]
    assert row["pending_count"] == dispositions.assess(STEMS[0])["pending_count"]
    assert row["decided_by"] is None and row["decided_at"] is None
    assert row["waiting_since"] == "2026-08-26T13:21:00"
    assert "ground_truth" not in row


def test_a_dispositioned_board_with_a_new_queue_entry_is_counted_once(store):
    """Exclusive buckets. A board with a standing row is held or released
    whatever its queue holds; counting it under waiting as well would make the
    three numbers sum to more boards than were run."""
    _disposition_all()
    _queue(store, STEMS[2])

    counts = dispositions.board_counts()
    assert counts["waiting"] == 0
    assert counts["held"] + counts["released"] == 4


def test_a_deferred_region_keeps_its_board_waiting(store):
    """The deferral path's defect, checked against this reader too: a region
    handed back is still a region waiting on a person."""
    _queue(store, STEMS[1], status="deferred")

    assert dispositions.board_counts()["waiting"] == 1


def test_the_page_lists_the_waiting_boards_under_their_own_filter(client, store):
    _disposition_all()
    _queue(store, STEMS[3])  # a held board with a new entry: not waiting
    with store() as session:
        # a fifth board, run and queued, never dispositioned
        board = Board(
            stem="20085299", split="test", lot_id="LOT-2201", line_id="L2",
            machine_id="M22", shift="A", inspected_at=datetime(2026, 8, 26, 9, 0),
        )
        session.add(board)
        session.flush()
        session.add(CandidateRecord(
            board_id=board.id, index_on_board=0, x1=1, y1=1, x2=9, y2=9, area=64,
            predicted_class="open", confidence=0.5, false_call_probability=0.5,
            ground_truth="open",
        ))
        session.commit()
    _queue(store, "20085299")

    index = client.get("/boards").text
    assert 'href="/boards?status=waiting"' in index

    body = client.get("/boards?status=waiting").text
    assert "20085299" in body
    for stem in STEMS:
        assert f'href="/board/{stem}"' not in body
    assert "ground_truth" not in body
