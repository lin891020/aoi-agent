"""One SELECT from the model, against a copy of the store that cannot hurt it.

This module is the exception to the no-SQL rule, and the reason the rule could
be relaxed at all is that nothing here relies on the model being careful. The
guarantees are structural, and each is held by a test in
`tests/test_sql_guard.py`:

1. **The database the SQL runs against does not contain the answer key.**
   `snapshot()` builds an in-memory SQLite database holding only the columns in
   `EXPOSED`, copied out of the store; `ground_truth` is never copied, so no
   query, however phrased, can select it. This is the `result_view` boundary
   again, one level down: not a filter on the way out, an absence on the way in.
2. **The connection cannot write.** `PRAGMA query_only` on the snapshot, and the
   store itself is only ever attached ``mode=ro`` for the copy and detached
   before a query runs. A write that got past the parser would fail at SQLite.
3. **One statement, and it is a query.** `sqlglot` parses the text; anything
   that is not a single SELECT (CTEs and set operations included) is refused
   before the connection sees it. So are tables outside `EXPOSED`, schema-
   qualified names, and the handful of functions that reach the filesystem.
4. **Bounded.** A `LIMIT` of at most `ROW_CAP` is imposed, and a progress
   handler aborts a statement that runs past `TIME_CAP_S` -- a recursive CTE
   with no floor is a denial of service on the machine the station shares with
   the model.
5. **Auditable.** The payload carries the SQL as written and as run. The plan
   stores it, the page shows it, and `synthesis_eval` can read it.

What none of this guards is *meaning*: a query that is valid, read-only,
bounded and wrong returns a plausible number, which is the failure the
typed tools were built to avoid. That is why the planner is told to prefer
them, why the SQL is printed beside its result, and why whether this tool
stays registered is a measurement (`scripts/analysis_eval.py`, with
`AOI_SQL_TOOL=0` for the control) rather than a decision made here.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import Any

import sqlglot
from sqlglot import exp

#: Table -> the columns copied into the snapshot. A column not listed here does
#: not exist on the database the SQL runs against. `boards.split` is dataset
#: bookkeeping and `candidates.ground_truth` is the answer key; neither is
#: production data.
EXPOSED: dict[str, tuple[str, ...]] = {
    "boards": ("id", "stem", "lot_id", "line_id", "machine_id", "shift", "inspected_at"),
    "candidates": ("id", "board_id", "index_on_board", "x1", "y1", "x2", "y2", "area",
                   "predicted_class", "confidence", "false_call_probability"),
    "review_decisions": ("id", "candidate_id", "verdict", "source", "reviewer",
                         "reviewer_auth", "rationale", "measurement",
                         "explanation_status", "decided_at"),
    "escalations": ("id", "candidate_id", "reason", "agent_verdict",
                    "explanation_status", "status", "raised_at", "resolved_at"),
    "board_dispositions": ("id", "board_id", "disposition", "decided_by", "basis",
                           "candidate_count", "confirmed_count", "pending_count",
                           "decided_at"),
    "machine_events": ("id", "machine_id", "kind", "happened_at", "note", "recorded_by"),
}

#: What is withheld, stated so the docstring the planner reads can say so.
WITHHELD: dict[str, tuple[str, ...]] = {
    "candidates": ("ground_truth",),
    "boards": ("split",),
}

ROW_CAP = 200
TIME_CAP_S = 2.0

#: Functions that reach outside the database. `load_extension` is disabled on
#: Python's connections anyway; the others exist only in the SQLite shell's
#: fileio extension. Refused by name regardless, because "not available in
#: this build" is not a property a test can hold.
DENIED_FUNCTIONS = frozenset({
    "load_extension", "readfile", "writefile", "edit", "fsdir", "zipfile",
    "fts3_tokenizer",
})

BASIS = (
    "one SELECT over a read-only copy of the store holding only the columns "
    "listed in the tool's description (no ground truth); at most "
    f"{ROW_CAP} rows; the SQL as run is shown beside the result"
)


class RefusedSQL(ValueError):
    """The text was not a query this guard will run, and this says why."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _table_name(node: exp.Table) -> str:
    return (node.name or "").lower()


def check(sql: str) -> str:
    """The SQL this guard will run for the text given, or `RefusedSQL`.

    Returns the statement with the row cap applied, rendered by the parser, so
    what runs is what was parsed and not what was typed.
    """
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        raise RefusedSQL("no SQL was given")
    if ";" in text:
        raise RefusedSQL("one statement only; a second statement after ';' is refused")

    try:
        statements = sqlglot.parse(text, read="sqlite")
    except sqlglot.errors.ParseError as error:
        raise RefusedSQL(f"could not parse: {error}") from error
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise RefusedSQL("one statement only")
    statement = statements[0]

    if isinstance(statement, exp.Subquery):
        statement = statement.unnest()
    if not isinstance(statement, exp.Query) or isinstance(statement, exp.Command):
        raise RefusedSQL(
            f"only a SELECT may run here; this is {type(statement).__name__.upper()}"
        )

    # Names a CTE introduces are tables this statement may reference.
    defined = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}

    for table in statement.find_all(exp.Table):
        if table.db or table.catalog:
            raise RefusedSQL(f"schema-qualified names are refused: {table.sql()}")
        name = _table_name(table)
        if name in defined:
            continue
        if name not in EXPOSED:
            raise RefusedSQL(
                f"table {name!r} is not exposed here; the tables are "
                f"{', '.join(EXPOSED)}"
            )

    for func in statement.find_all(exp.Func):
        name = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()
        if name in DENIED_FUNCTIONS:
            raise RefusedSQL(f"function {name}() is refused")

    return _capped(statement).sql(dialect="sqlite")


def _capped(statement: exp.Query) -> exp.Query:
    """The statement with a LIMIT of at most `ROW_CAP` + 1.

    One past the cap, so that a result with more rows than are returned can
    be *reported* as cut rather than silently ending at the cap; the extra
    row is dropped and `truncated` set. A caller's own smaller LIMIT stands.
    """
    limit = statement.args.get("limit")
    if limit is not None:
        value = limit.expression
        if isinstance(value, exp.Literal) and value.is_int and int(value.this) <= ROW_CAP:
            return statement
    return statement.limit(ROW_CAP + 1)


# ---------------------------------------------------------------------------
# The snapshot
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_cache: dict[str, tuple[tuple, sqlite3.Connection]] = {}


def store_path() -> str:
    """The SQLite file the board store is on, read off the live engine.

    Off the engine rather than a setting, so a test that points the store at a
    temporary file points this at the same file.
    """
    from aoi_agent.store.boards import session_factory

    engine = session_factory().kw["bind"]
    database = engine.url.database
    if not database or database == ":memory:":
        raise RefusedSQL("the store is not a file, so there is nothing to snapshot")
    return os.path.abspath(database)


def _signature(path: str) -> tuple:
    """What has to be equal for a cached snapshot to still be current."""
    parts = []
    for candidate in (path, path + "-wal"):
        try:
            stat = os.stat(candidate)
            parts.append((stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            parts.append(None)
    return tuple(parts)


def snapshot(path: str | None = None) -> sqlite3.Connection:
    """An in-memory copy of the exposed columns, refreshed when the store moves.

    Built by attaching the store read-only, copying the listed columns table
    by table, and detaching it again -- so by the time a query runs there is
    no route from the connection back to the file, and no table on it that
    was not listed.
    """
    path = path or store_path()
    signature = _signature(path)
    with _lock:
        cached = _cache.get(path)
        if cached and cached[0] == signature:
            return cached[1]

        connection = sqlite3.connect(":memory:", uri=True, check_same_thread=False)
        connection.execute("ATTACH DATABASE ? AS src", (f"file:{path}?mode=ro",))
        try:
            for table, columns in EXPOSED.items():
                # Constants from this module, never text from a caller: the
                # names are the allowlist itself.
                connection.execute(
                    f"CREATE TABLE {table} AS SELECT {', '.join(columns)} FROM src.{table}"
                )
        finally:
            connection.execute("DETACH DATABASE src")
        connection.execute("PRAGMA query_only = 1")
        connection.commit()
        _cache[path] = (signature, connection)
        return connection


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def guarded_select(sql: str) -> dict[str, Any]:
    """Run one SELECT under every guard above; a refusal is a value."""
    try:
        to_run = check(sql)
    except RefusedSQL as refused:
        return {"error": f"refused: {refused}", "sql": sql}

    try:
        connection = snapshot()
    except (RefusedSQL, sqlite3.Error) as error:
        return {"error": f"the store could not be snapshotted: {error}", "sql": sql}

    deadline = time.monotonic() + TIME_CAP_S

    def out_of_time() -> int:
        return 1 if time.monotonic() > deadline else 0

    with _lock:
        connection.set_progress_handler(out_of_time, 1000)
        try:
            cursor = connection.execute(to_run)
            columns = [d[0] for d in cursor.description or ()]
            rows = [list(row) for row in cursor.fetchmany(ROW_CAP + 1)]
        except sqlite3.OperationalError as error:
            message = str(error)
            if "interrupted" in message:
                message = f"stopped after {TIME_CAP_S:.0f}s; narrow the query"
            return {"error": message, "sql": sql, "sql_run": to_run}
        finally:
            connection.set_progress_handler(None, 0)

    truncated = len(rows) > ROW_CAP
    rows = rows[:ROW_CAP]
    return {
        "sql": sql,
        "sql_run": to_run,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
        "basis": BASIS,
    }


def schema_text() -> str:
    """The exposed tables, one line each, for the tool's own docstring."""
    lines = []
    for table, columns in EXPOSED.items():
        withheld = WITHHELD.get(table)
        note = f"  (withheld: {', '.join(withheld)})" if withheld else ""
        lines.append(f"{table}({', '.join(columns)}){note}")
    return "\n".join(lines)
