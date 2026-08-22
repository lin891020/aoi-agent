"""One tool call, and what happens when it goes wrong.

The fan-out's branches are independent, so one of them failing must not take
the others with it. A raised exception would do exactly that, so `run_call`
catches its own failures and returns them as data.
"""

from __future__ import annotations

from aoi_agent.analysis import tools


def call(tool: str, **args) -> dict:
    return {"tool": tool, "args": args, "why": "because"}


def test_a_successful_call_carries_its_data_and_timing(monkeypatch):
    monkeypatch.setitem(
        tools.PLANNABLE_TOOLS, "query_machine_stats", lambda **kw: {"machines": []}
    )
    result = tools.run_call(call("query_machine_stats", defect_type="open", days=7))

    assert result["ok"] is True
    assert result["data"] == {"machines": []}
    assert result["error"] is None
    assert result["elapsed_ms"] >= 0
    assert result["tool"] == "query_machine_stats"
    assert result["args"] == {"defect_type": "open", "days": 7}


def test_a_raising_tool_becomes_a_failed_result_not_an_exception():
    """The whole point. One branch dying must not abort its siblings."""
    def boom(**kw):
        raise RuntimeError("the store is unreachable")

    tools.PLANNABLE_TOOLS["_boom"] = boom
    try:
        result = tools.run_call(call("_boom"))
    finally:
        del tools.PLANNABLE_TOOLS["_boom"]

    assert result["ok"] is False
    assert result["data"] is None
    assert "unreachable" in result["error"]
    assert "RuntimeError" in result["error"]


def test_a_tool_that_is_not_in_the_registry_fails_closed():
    """Validation should have caught this. If it did not, the call must still
    not reach `getattr` on something arbitrary."""
    result = tools.run_call(call("drop_tables"))

    assert result["ok"] is False
    assert "drop_tables" in result["error"]


def test_the_elapsed_time_is_recorded_even_on_failure():
    """A slow failure and a fast one are different operational problems."""
    def boom(**kw):
        raise ValueError("no")

    tools.PLANNABLE_TOOLS["_boom"] = boom
    try:
        result = tools.run_call(call("_boom"))
    finally:
        del tools.PLANNABLE_TOOLS["_boom"]

    assert result["elapsed_ms"] >= 0
