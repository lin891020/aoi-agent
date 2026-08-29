"""A dated window and a top-N cut on the production tools.

The first question the supervisor wrote down for this page was «2026-07-30
缺陷數量前 5 名的機台» -- a date, a ranking, a count. Neither tool could take
it: `days` counted back from the newest inspection and nothing else, and
`query_machine_stats` needed a defect class and returned every machine. The
planner did what the catalogue allowed, which was refuse. This file holds the
two parameters that make the question expressible and the three ways a date
can be malformed.

Runs against a small store built here; no dataset, no model.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aoi_agent.mcp_servers import production
from aoi_agent.store import boards as boards_module
from aoi_agent.store.models import Board, CandidateRecord, create_all, make_session_factory


def _board(session, stem, line, machine, when, classes):
    board = Board(stem=stem, split="test", lot_id="LOT-1", line_id=line,
                  machine_id=machine, shift="A", inspected_at=when)
    session.add(board)
    session.flush()
    for i, klass in enumerate(classes):
        session.add(CandidateRecord(
            board_id=board.id, index_on_board=i, x1=0, y1=0, x2=10, y2=10, area=100,
            predicted_class=klass, confidence=0.9,
            false_call_probability=0.1 if klass != "false_call" else 0.99,
            ground_truth=klass,
        ))


@pytest.fixture
def store(tmp_path, monkeypatch):
    """M22 runs one board a day for nine days, two opens on each; M11 runs one
    board on the 3rd with five opens; M32 one board on the 5th with one short."""
    url = f"sqlite:///{tmp_path / 'dated.db'}"
    create_all(url)
    factory = make_session_factory(url)
    monkeypatch.setattr(boards_module, "_session_factory", factory)
    with factory() as session:
        for day in range(1, 10):
            _board(session, f"2200000{day}", "L2", "M22",
                   datetime(2026, 8, day, 8, 0), ["open", "open", "false_call"])
        _board(session, "11000003", "L1", "M11", datetime(2026, 8, 3, 10, 0), ["open"] * 5)
        _board(session, "32000005", "L3", "M32", datetime(2026, 8, 5, 10, 0), ["short"])
        session.commit()
    return factory


# --- query_defect_history ---------------------------------------------------

def test_a_dated_window_counts_only_the_boards_inside_it(store):
    out = production.query_defect_history(date_from="2026-08-03", date_to="2026-08-04")

    assert out["boards_inspected"] == 3            # M22 on the 3rd and 4th, M11 on the 3rd
    assert out["by_class"] == {"open": 9}
    assert out["window_start"].startswith("2026-08-03T00:00")
    assert out["window_end"].startswith("2026-08-04T23:59")
    assert out["filters"]["date_from"] == "2026-08-03"
    assert out["filters"]["date_to"] == "2026-08-04"


def test_one_date_makes_a_one_day_window(store):
    out = production.query_defect_history(date_from="2026-08-05", date_to="2026-08-05")
    assert out["boards_inspected"] == 2            # M22 and M32 on the 5th
    assert out["by_class"] == {"open": 2, "short": 1}


def test_an_open_ended_date_runs_to_the_edge_of_the_data(store):
    since = production.query_defect_history(date_from="2026-08-09")
    until = production.query_defect_history(date_to="2026-08-01")
    assert since["boards_inspected"] == 1 and since["by_class"] == {"open": 2}
    assert until["boards_inspected"] == 1 and until["by_class"] == {"open": 2}


def test_a_date_and_an_event_anchor_are_two_windows_not_one(store):
    out = production.query_defect_history(
        machine_id="M22", date_from="2026-08-03",
        relative_to="parameter_change", side="before",
    )
    assert "error" in out and "relative_to" in out["error"] and "date" in out["error"]


def test_a_window_that_ends_before_it_starts_is_refused(store):
    out = production.query_defect_history(date_from="2026-08-05", date_to="2026-08-03")
    assert "error" in out


def test_a_date_that_is_not_a_date_is_refused_rather_than_read_as_nothing(store):
    out = production.query_defect_history(date_from="30/07/2026")
    assert "error" in out and "YYYY-MM-DD" in out["error"]


# --- query_machine_stats ----------------------------------------------------

def test_without_a_class_every_machine_is_ranked_by_defects_per_board(store):
    out = production.query_machine_stats()

    assert out["defect_type"] is None
    assert out["ranked_by"] == "per_board"
    assert [m["machine"] for m in out["machines"]] == ["L1-M11", "L2-M22", "L3-M32"]
    assert [m["per_board"] for m in out["machines"]] == [5.0, 2.0, 1.0]
    # A share of "all defects" among all defects is 1 by construction and is
    # not reported as though it were a measurement.
    assert all(m["share_of_defects"] is None for m in out["machines"])


def test_with_a_class_the_ranking_is_still_by_share(store):
    out = production.query_machine_stats(defect_type="open")
    assert out["ranked_by"] == "share_of_defects"
    assert out["machines"][0]["share_of_defects"] == 1.0


def test_top_n_cuts_the_ranking_and_says_how_many_there_were(store):
    out = production.query_machine_stats(top_n=2)
    assert len(out["machines"]) == 2
    assert out["machines_total"] == 3
    assert out["filters"]["top_n"] == 2


def test_a_top_n_below_one_is_refused(store):
    assert "error" in production.query_machine_stats(top_n=0)


def test_machine_stats_take_the_same_dated_window(store):
    out = production.query_machine_stats(date_from="2026-08-03", date_to="2026-08-03")

    assert {m["machine"] for m in out["machines"]} == {"L1-M11", "L2-M22"}
    assert out["window_start"].startswith("2026-08-03T00:00")
    assert out["window_end"].startswith("2026-08-03T23:59")
    assert "days" not in {k for k, v in out["filters"].items() if v is not None} or out["filters"]["days"] is None
