"""MCP server exposing one read-only SQL tool.

The production server's own docstring says it is deliberately not a
text-to-SQL tool, and that is still true of it. This server is the experiment
beside it: a single SELECT, run through `analysis.sql_guard`, which is the
only path by which model-written SQL reaches anything. Whether the planner
uses it well -- or reaches for it where a typed tool would have stated its
own basis -- is measured by `scripts/analysis_eval.py`, with `AOI_SQL_TOOL=0`
as the control.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from aoi_agent.analysis.sql_guard import ROW_CAP, guarded_select, schema_text

mcp = MCPServer("aoi-sql-readonly")


@mcp.tool()
def run_sql(sql: str) -> dict:
    """Run one read-only SELECT over the production tables listed below.

    Use this only when no other tool expresses the question's dimension -- a
    shift, which machines a lot ran on, how many regions were flagged, what
    a reviewer decided -- and the tables here do. Prefer the typed tools
    whenever one fits: their payloads state their own basis and interval,
    and a SELECT states nothing about itself. Write SQLite syntax, one
    statement, no writes. ``predicted_class = 'false_call'`` is a region this
    system dismissed, not a confirmed false call; there is no ground truth in
    these tables. Results are capped at {row_cap} rows -- aggregate rather
    than list.

    Tables and columns (nothing else exists here):

    {schema}

    Args:
        sql: One SELECT statement in SQLite syntax.
    """
    return guarded_select(sql)


# Derived from the allowlist rather than written beside it: the schema the
# planner is shown is the schema the snapshot holds, by construction.
run_sql.__doc__ = run_sql.__doc__.format(row_cap=ROW_CAP, schema=schema_text())


if __name__ == "__main__":
    mcp.run()
