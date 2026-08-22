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
    #: The plan's justification for *this* call, copied off the call itself.
    #: It travels with the result rather than being paired up by position
    #: afterwards, because the two paths that render a run do not agree on
    #: position: `graph.invoke` applies branch writes in plan order, while
    #: `stream_mode="updates"` hands them back in completion order. Pairing
    #: result *i* with `plan.calls[i].why` therefore printed the fast tool's
    #: row beside the slow tool's reason on the streamed path -- which is the
    #: normal path, and a wrong reason beside a right tool is worse than none.
    why: str
    #: Which call of the plan this was. The plan's own order, so a run reads
    #: the same however it was executed; see `service.in_plan_order`.
    position: int
    ok: bool
    data: dict | None
    error: str | None
    elapsed_ms: float


def run_call(call: ToolCall, position: int = 0) -> ToolResult:
    """Run one planned call and return its outcome, successful or not."""
    name = call.get("tool", "")
    args = call.get("args") or {}
    why = call.get("why") or ""
    started = time.perf_counter()

    def finish(ok: bool, data: dict | None, error: str | None) -> ToolResult:
        return {
            "tool": name,
            "args": args,
            "why": why,
            "position": position,
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
