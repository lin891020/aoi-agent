"""The read-only SQL guard: what it refuses, and what the SQL cannot reach.

Every guarantee in `analysis/sql_guard.py`'s docstring has a test here, and
the first one is the one that matters: the answer key is not in the database
the SQL runs against. That is not a filter to bypass; the column is absent.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime

import pytest

from aoi_agent.analysis import sql_guard
from aoi_agent.analysis.sql_guard import EXPOSED, ROW_CAP, RefusedSQL, check, guarded_select
from aoi_agent.store import boards as boards_module
from aoi_agent.store.models import Board, CandidateRecord, create_all, make_session_factory


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Three boards on M22 across two shifts, with a known false-call split."""
    path = tmp_path / "guard.db"
    url = f"sqlite:///{path}"
    create_all(url)
    factory = make_session_factory(url)
    monkeypatch.setattr(boards_module, "_session_factory", factory)
    sql_guard._cache.clear()
    with factory() as session:
        for n, (shift, classes) in enumerate((
            ("A", ["open", "false_call", "false_call"]),
            ("A", ["short", "false_call"]),
            ("C", ["open", "open", "false_call"]),
        )):
            board = Board(stem=f"2200001{n}", split="test", lot_id="LOT-1", line_id="L2",
                          machine_id="M22", shift=shift, inspected_at=datetime(2026, 8, 5, 8 + n))
            session.add(board)
            session.flush()
            for i, klass in enumerate(classes):
                session.add(CandidateRecord(
                    board_id=board.id, index_on_board=i, x1=0, y1=0, x2=10, y2=10, area=100,
                    predicted_class=klass, confidence=0.9,
                    false_call_probability=0.1 if klass != "false_call" else 0.99,
                    ground_truth="TELLTALE-" + klass,
                ))
        session.commit()
    return path


# --- 1. the answer key is absent, not filtered --------------------------------

def test_the_snapshot_holds_no_ground_truth_column_on_any_table(store):
    connection = sql_guard.snapshot()
    for table in EXPOSED:
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        assert "ground_truth" not in columns, table
        assert columns == set(EXPOSED[table]), table


def test_selecting_the_answer_key_fails_at_the_database_not_at_a_filter(store):
    out = guarded_select("SELECT ground_truth FROM candidates")
    assert "error" in out and "no such column" in out["error"]
    out = guarded_select("SELECT * FROM candidates")
    assert "ground_truth" not in out["columns"]
    assert not any("TELLTALE" in str(v) for row in out["rows"] for v in row)


def test_no_table_outside_the_allowlist_exists_on_the_snapshot(store):
    connection = sql_guard.snapshot()
    names = {row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert names == set(EXPOSED)
    assert "analysis_runs" not in names


# --- 2. the connection cannot write ------------------------------------------

def test_the_snapshot_connection_refuses_writes_even_without_the_parser(store):
    connection = sql_guard.snapshot()
    with pytest.raises(sqlite3.OperationalError, match="readonly|query_only"):
        connection.execute("DELETE FROM candidates")
    with pytest.raises(sqlite3.OperationalError):
        connection.execute("CREATE TABLE x (a)")


def test_the_store_file_is_untouched_by_a_query(store):
    before = store.read_bytes()
    guarded_select("SELECT count(*) FROM boards")
    assert store.read_bytes() == before


# --- 3. one statement, and it is a query ---------------------------------------

@pytest.mark.parametrize("sql", [
    "DELETE FROM candidates",
    "UPDATE boards SET shift = 'A'",
    "INSERT INTO machine_events (machine_id, kind) VALUES ('M22', 'x')",
    "DROP TABLE boards",
    "PRAGMA query_only = 0",
    "ATTACH DATABASE '/tmp/x.db' AS other",
    "SELECT 1; DROP TABLE boards",
    "CREATE TABLE t AS SELECT * FROM boards",
    "",
])
def test_anything_but_one_select_is_refused_before_the_connection_sees_it(store, sql):
    with pytest.raises(RefusedSQL):
        check(sql)
    out = guarded_select(sql)
    assert out.get("error", "").startswith("refused")


@pytest.mark.parametrize("sql", [
    "SELECT * FROM analysis_runs",
    "SELECT name FROM sqlite_master",
    "SELECT * FROM src.candidates",
    "SELECT * FROM main.candidates",
])
def test_tables_outside_the_allowlist_and_qualified_names_are_refused(store, sql):
    with pytest.raises(RefusedSQL, match="not exposed|schema-qualified"):
        check(sql)


def test_functions_that_reach_the_filesystem_are_refused_by_name(store):
    with pytest.raises(RefusedSQL, match="load_extension"):
        check("SELECT load_extension('x') FROM boards")
    with pytest.raises(RefusedSQL, match="readfile"):
        check("SELECT readfile('/etc/passwd')")


def test_a_cte_and_a_union_are_still_one_query(store):
    out = guarded_select(
        "WITH m AS (SELECT id FROM boards WHERE machine_id = 'M22') "
        "SELECT count(*) AS n FROM candidates WHERE board_id IN (SELECT id FROM m) "
        "UNION ALL SELECT count(*) FROM boards"
    )
    assert "error" not in out
    assert [row[0] for row in out["rows"]] == [8, 3]


# --- 4. bounded ----------------------------------------------------------------

def test_a_limit_is_imposed_and_a_larger_one_is_cut_to_the_cap(store):
    # One past the cap, so the cut can be reported; see `_capped`.
    assert check("SELECT id FROM candidates").endswith(f"LIMIT {ROW_CAP + 1}")
    assert check("SELECT id FROM candidates LIMIT 5").endswith("LIMIT 5")
    assert check(f"SELECT id FROM candidates LIMIT {ROW_CAP * 10}").endswith(f"LIMIT {ROW_CAP + 1}")


def test_more_rows_than_the_cap_is_said_rather_than_silently_cut(store, monkeypatch):
    monkeypatch.setattr(sql_guard, "ROW_CAP", 3)
    out = guarded_select("SELECT id FROM candidates")
    assert out["row_count"] == 3 and out["truncated"] is True
    out = guarded_select("SELECT id FROM candidates LIMIT 2")
    assert out["row_count"] == 2 and out["truncated"] is False


def test_a_query_that_never_finishes_is_stopped_at_the_time_cap(store, monkeypatch):
    monkeypatch.setattr(sql_guard, "TIME_CAP_S", 0.3)
    started = time.perf_counter()
    out = guarded_select(
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
        "SELECT count(*) FROM c"
    )
    elapsed = time.perf_counter() - started
    assert "error" in out and "stopped after" in out["error"]
    assert elapsed < 2.0
    # The connection is still usable afterwards.
    assert "error" not in guarded_select("SELECT count(*) FROM boards")


# --- 5. auditable, and correct on a real question -------------------------------

def test_the_payload_carries_the_sql_as_written_and_as_run(store):
    sql = "select shift, count(*) as n from boards group by shift"
    out = guarded_select(sql)
    assert out["sql"] == sql
    assert out["sql_run"].upper().startswith("SELECT")
    assert out["sql_run"].endswith(f"LIMIT {ROW_CAP + 1}")
    assert out["basis"]


def test_dismissed_false_calls_per_shift_on_one_machine_the_question_no_typed_tool_takes(store):
    out = guarded_select(
        "SELECT b.shift, COUNT(*) AS flagged, "
        "SUM(c.predicted_class = 'false_call') AS dismissed "
        "FROM candidates c JOIN boards b ON b.id = c.board_id "
        "WHERE b.machine_id = 'M22' GROUP BY b.shift ORDER BY b.shift"
    )
    assert out["columns"] == ["shift", "flagged", "dismissed"]
    assert out["rows"] == [["A", 5, 3], ["C", 3, 1]]


def test_the_snapshot_follows_the_store_when_it_changes(store):
    assert guarded_select("SELECT count(*) FROM boards")["rows"] == [[3]]
    time.sleep(0.01)
    with boards_module.session_factory()() as session:
        session.add(Board(stem="22000099", split="test", lot_id="LOT-2", line_id="L2",
                          machine_id="M21", shift="B", inspected_at=datetime(2026, 8, 6)))
        session.commit()
    assert guarded_select("SELECT count(*) FROM boards")["rows"] == [[4]]


def test_the_tools_docstring_lists_exactly_the_snapshots_tables(store):
    from aoi_agent.mcp_servers.sql_readonly import run_sql

    doc = run_sql.__doc__
    for table, columns in EXPOSED.items():
        assert f"{table}({', '.join(columns)})" in doc
    assert "withheld: ground_truth" in doc
    assert str(ROW_CAP) in doc
