"""Calling one tool, safely.

A fan-out branch that raises takes the whole superstep with it, and the three
siblings that succeeded are lost along with it. The branches are independent by
construction; letting one abort the others would be a defect, not caution. So a
failure here is a value: `ok=False` with the reason, which the join reports and
the answer names.
"""

from __future__ import annotations

import time
from typing import Any, TypedDict

from aoi_agent.analysis.plan import PLANNABLE_TOOLS, ToolCall


class ToolResult(TypedDict):
    tool: str
    args: dict[str, Any]
    ok: bool
    data: dict | None
    error: str | None
    elapsed_ms: float


def run_call(call: ToolCall) -> ToolResult:
    """Run one planned call and return its outcome, successful or not."""
    name = call.get("tool", "")
    args = call.get("args") or {}
    started = time.perf_counter()

    def finish(ok: bool, data: dict | None, error: str | None) -> ToolResult:
        return {
            "tool": name,
            "args": args,
            "ok": ok,
            "data": data,
            "error": error,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    function = PLANNABLE_TOOLS.get(name)
    if function is None:
        # Validation should have caught this. Failing closed anyway means a
        # gap in validation costs an error message rather than an arbitrary
        # call.
        return finish(False, None, f"{name!r} is not a tool this system exposes")

    try:
        return finish(True, function(**args), None)
    except Exception as error:  # noqa: BLE001 -- the branch must not raise
        return finish(False, None, f"{type(error).__name__}: {error}")
