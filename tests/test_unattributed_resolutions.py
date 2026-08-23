"""Two tables that disagree about what happened to a region.

``escalations`` says five regions were answered. ``review_decisions`` holds no
human verdict for any of them. The station cannot produce that combination --
``resume_review`` writes the decision and then closes the entry, in that order,
so that a crash between the two costs a second look rather than a verdict --
which is what makes these five rows evidence of something that happened outside
the code: five operator labels deleted by hand on 2026-08-22, after five
regions were clicked through by someone without the domain knowledge to judge
them and four of the five turned out wrong.

What is under test is the decision about what to do with them, because every
alternative is worse in a way that is easy to miss:

- deleting the queue rows repeats the mistake that caused this
- re-opening them puts five regions in front of an operator as though the first
  review never happened, days after the boards left the line
- writing a synthetic human decision invents the label that was deleted, and it
  would be indistinguishable from a judgement in the next training round

So they are marked, in a distinct state the station cannot write, with the
original reason kept verbatim underneath. The rules below are the ones a
mistake would be silent in: over-marking teaches a reader to ignore banners,
under-marking leaves a hole in a quality record with nothing beside it.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aoi_agent.store import boards, escalations
from aoi_agent.store.models import (
    Board,
    CandidateRecord,
    Escalation,
    create_all,
    make_session_factory,
)
from mark_unattributed_resolutions import (
    BANNER,
    MARKER,
    is_unattributed,
    mark,
    needs_marking,
)

STEM = "20085293"
REASON = "the re-verification model's confidence of 0.552 is below the 0.915 threshold"


# ---- the rule, on its own ------------------------------------------------


def test_a_closed_entry_with_no_human_decision_is_unattributed():
    assert is_unattributed("resolved", human_decisions=0)


def test_a_closed_entry_backed_by_a_decision_is_the_ordinary_case():
    """Most of the queue's history is this. Marking it would be the mirror
    image of the defect: a correct record flagged as a broken one."""
    assert not is_unattributed("resolved", human_decisions=1)


def test_an_entry_still_waiting_is_not_marked():
    """A pending entry has no decision beneath it *yet*. That is what pending
    means, and it is the state five of these rows would be pushed back into by
    the repair this file exists to refuse."""
    assert not is_unattributed("pending", human_decisions=0)


def test_marking_is_idempotent_so_the_script_can_be_re_run():
    once = mark(REASON)
    assert once.startswith(MARKER)
    assert mark(once) == once
    assert not needs_marking("resolved_unattributed", once, 0)


def test_the_original_reason_survives_verbatim():
    """The record of what the operator was shown is the point of keeping it."""
    assert mark(REASON) == BANNER + REASON
    assert mark(REASON).endswith(REASON)


def test_the_banner_says_what_happened_rather_than_that_something_did():
    """A banner reading "this row is wrong" leaves the reader with a broken
    record and no account of it."""
    assert "no human decision" in BANNER
    assert "deleted by hand" in BANNER


# ---- against a store -----------------------------------------------------


@pytest.fixture
def store(tmp_path, monkeypatch):
    """One board, three regions: one answered properly, one closed with
    nothing behind it, one still waiting."""
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
        for index in range(3):
            session.add(CandidateRecord(
                board_id=board.id, index_on_board=index,
                x1=1, y1=1, x2=2, y2=2, area=1,
                predicted_class="open", confidence=0.55,
                false_call_probability=0.1, ground_truth="open",
            ))
        session.flush()
        for index, status in enumerate(["resolved", "resolved", "pending"]):
            session.add(Escalation(
                candidate_id=index + 1, thread_id=f"{STEM}#{index}",
                reason=REASON, status=status,
            ))
        session.commit()

    # Only the first has an operator's verdict behind it.
    boards.record_decision(f"{STEM}#0", "short", "human", reviewer="mike")
    return factory


def test_only_the_closed_entry_with_nothing_behind_it_is_found(store):
    found = escalations.unattributed_resolutions()
    assert [row["reference"] for row in found] == [f"{STEM}#1"]


def test_marking_moves_it_out_of_resolved_without_moving_it_into_pending(store):
    """The two states the station writes both say something untrue about these
    rows. The third says the true thing."""
    from mark_unattributed_resolutions import main

    main([])

    queued = [row["reference"] for row in escalations.pending()]
    assert queued == [f"{STEM}#2"], "a marked row must not return to the queue"
    assert escalations.counts() == {
        "pending": 1, "resolved": 1, "resolved_unattributed": 1
    }


def test_the_marked_row_still_carries_what_the_operator_was_shown(store):
    from mark_unattributed_resolutions import main

    main([])

    row = escalations.get(f"{STEM}#1")
    assert row["status"] == escalations.RESOLVED_UNATTRIBUTED
    assert row["reason"].startswith(MARKER)
    assert row["reason"].endswith(REASON)


def test_a_dry_run_changes_nothing(store):
    from mark_unattributed_resolutions import main

    main(["--dry-run"])

    assert escalations.get(f"{STEM}#1")["status"] == "resolved"


def test_running_it_twice_marks_nothing_twice(store, capsys):
    from mark_unattributed_resolutions import main

    main([])
    capsys.readouterr()
    main([])

    assert "0 row(s) marked, 1 already marked" in capsys.readouterr().out
    assert escalations.get(f"{STEM}#1")["reason"].count(MARKER) == 1


def test_a_marked_region_is_not_counted_as_still_owed_an_answer(store):
    """A board is held while anyone is still waiting on it. These five were
    reviewed -- badly, and unattributably -- and putting them back in the
    pending count would hold five boards on a queue nobody is working."""
    from aoi_agent.store import dispositions
    from mark_unattributed_resolutions import main

    main([])
    assert dispositions.assess(STEM)["pending_count"] == 2, (
        "the marked region has no decision of its own and counts as pending "
        "for that reason; what must not happen is the marked *queue row* "
        "putting it back in front of a person"
    )
    assert f"{STEM}#1" not in [row["reference"] for row in escalations.pending()]
