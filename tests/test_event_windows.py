"""`query_defect_history` anchored on a machine event, and what it refuses.

The tool answers "did M32 change after its parameter change" as two lookups
the planner composes -- one window before the event, one after -- and never
as a verdict. What this file holds is the shape of those windows, the
interval they carry, and the four ways the question can be malformed:
an anchor without a side, a side without an anchor, an anchor with no machine,
and an anchor on a machine that has no such event -- which includes the
mirror machine the seed deliberately makes *look* changed.

Runs against a small store built here; no dataset, no model.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from aoi_agent.mcp_servers import production
from aoi_agent.store import boards as boards_module
from aoi_agent.store import events
from aoi_agent.store.models import Board, CandidateRecord, create_all, make_session_factory

EVENT_AT = datetime(2026, 8, 5, 12, 0)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Ten boards on M32, five before the event and five after; the ones before
    carry mostly opens, the ones after mostly false calls. M11 has boards and
    no event."""
    url = f"sqlite:///{tmp_path / 'windows.db'}"
    create_all(url)
    factory = make_session_factory(url)
    monkeypatch.setattr(boards_module, "_session_factory", factory)
    with factory() as session:
        for n in range(10):
            when = EVENT_AT + timedelta(hours=(n - 5) * 12)      # -60h .. +48h
            board = Board(stem=f"3000000{n}", split="test", lot_id="LOT-1",
                          line_id="L3", machine_id="M32", shift="A", inspected_at=when)
            session.add(board)
            session.flush()
            classes = ["open", "open", "open", "false_call"] if n < 5 else ["false_call"] * 3 + ["open"]
            for i, klass in enumerate(classes):
                session.add(CandidateRecord(
                    board_id=board.id, index_on_board=i, x1=0, y1=0, x2=10, y2=10, area=100,
                    predicted_class=klass, confidence=0.9,
                    false_call_probability=0.1 if klass != "false_call" else 0.99,
                    ground_truth=klass,
                ))
        other = Board(stem="30000099", split="test", lot_id="LOT-1", line_id="L1",
                      machine_id="M11", shift="A", inspected_at=EVENT_AT)
        session.add(other)
        session.commit()
    events.record("M32", "parameter_change", EVENT_AT)
    return factory


def test_before_and_after_are_two_windows_bounded_by_the_event(store):
    before = production.query_defect_history(machine_id="M32", relative_to="parameter_change", side="before")
    after = production.query_defect_history(machine_id="M32", relative_to="parameter_change", side="after")

    assert "error" not in before and "error" not in after
    assert before["event_at"] == after["event_at"] == EVENT_AT.isoformat()
    assert before["window_end"] == EVENT_AT.isoformat()
    assert after["window_start"] == EVENT_AT.isoformat()
    assert before["boards_inspected"] == 5 and after["boards_inspected"] == 5


def test_the_open_share_moves_the_way_the_boards_do_and_carries_an_interval(store):
    before = production.query_defect_history(machine_id="M32", relative_to="parameter_change", side="before")
    after = production.query_defect_history(machine_id="M32", relative_to="parameter_change", side="after")

    assert before["flagged_regions"] == 20 and after["flagged_regions"] == 20
    assert before["open_share"]["value"] == pytest.approx(15 / 20)
    assert after["open_share"]["value"] == pytest.approx(5 / 20)
    lo_b, hi_b = before["open_share"]["interval_95"]
    lo_a, hi_a = after["open_share"]["interval_95"]
    assert lo_b <= 0.75 <= hi_b and lo_a <= 0.25 <= hi_a
    assert "Wilson" in before["open_share"]["basis"]
    # The tool does not say "improved". It says two intervals.
    assert not any(k in before for k in ("improved", "verdict", "conclusion"))


def test_the_denominator_is_every_flagged_region_not_only_defects(store):
    """False calls are in the denominator on purpose: the planted signal moves
    which boards a machine gets, and a person comparing two windows counts
    everything the AOI flagged."""
    after = production.query_defect_history(machine_id="M32", relative_to="parameter_change", side="after")
    assert after["defects_total"] == 5            # the opens
    assert after["flagged_regions"] == 20         # opens + false calls


@pytest.mark.parametrize("kwargs, needle", [
    ({"machine_id": "M32", "relative_to": "parameter_change"}, "go together"),
    ({"machine_id": "M32", "side": "after"}, "go together"),
    ({"relative_to": "parameter_change", "side": "after"}, "needs machine_id"),
    ({"machine_id": "M32", "relative_to": "parameter_change", "side": "during"}, "unknown side"),
    ({"machine_id": "M32", "relative_to": "lamp_replaced", "side": "after"}, "no 'lamp_replaced' event"),
])
def test_a_malformed_anchor_is_refused_not_approximated(store, kwargs, needle):
    result = production.query_defect_history(**kwargs)
    assert "error" in result and needle in result["error"], result


def test_the_mirror_machine_cannot_be_anchored_on(store):
    """M11 exists and has boards, and in the seeded store it will look as
    though it changed at M32's date. It has no event, so the tool refuses --
    the anchor is a recorded fact, not a pattern in the numbers."""
    result = production.query_defect_history(machine_id="M11", relative_to="parameter_change", side="after")
    assert "error" in result and "M11" in result["error"]


def test_an_unanchored_query_is_unchanged(store):
    """The two new parameters default off, and the old call is the old call."""
    plain = production.query_defect_history(machine_id="M32", days=30)
    assert "error" not in plain
    assert "open_share" not in plain and "event_at" not in plain
    assert plain["filters"]["relative_to"] is None and plain["filters"]["side"] is None


def test_events_lookup_lists_what_was_done_and_points_at_the_comparison(store):
    result = production.query_machine_events(machine_id="M32")
    assert result["count"] == 1
    assert result["events"][0]["kind"] == "parameter_change"
    assert "relative_to" in result["basis"]
    assert production.query_machine_events(machine_id="M11")["count"] == 0


def test_the_planners_domain_for_relative_to_is_the_recorded_kinds(store):
    from aoi_agent.analysis.plan import store_domains

    domains = store_domains()
    assert domains["relative_to"] == {"parameter_change"}
    assert domains["side"] == set(production.SIDES)
