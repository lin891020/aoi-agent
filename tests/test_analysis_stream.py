"""Progress events, and the guarantee that the page works without them."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from aoi_agent.analysis import graph as analysis
from aoi_agent.llm.ollama import ChatResult, Timing
from aoi_agent.station import app as station_app
from aoi_agent.store import analysis as analysis_store
from aoi_agent.store.models import create_all, make_session_factory

PLAN = {
    "interpretation": "M22 against the fleet",
    "assumptions": ["fleet average"],
    "calls": [
        {"tool": "query_machine_stats", "args": {"defect_type": "open", "days": 7},
         "why": "a"},
        {"tool": "search_standards", "args": {"query": "open", "top_k": 2},
         "why": "b"},
    ],
}
DOMAINS = {"line_id": {"L2"}, "machine_id": {"M22"},
           "defect_type": {"open"}, "max_days": 9}


@dataclass
class StubClient:
    plan: dict = field(default_factory=lambda: PLAN)

    def chat(self, messages, **kwargs) -> ChatResult:
        text = json.dumps(self.plan) if kwargs.get("response_format") else "done"
        return ChatResult(text=text, tool_calls=[], thinking="",
                          timing=Timing(1.0, 0.0, 1.0, 1.0, 10, 10))


@pytest.fixture
def client(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'a.db'}"
    create_all(url)
    monkeypatch.setattr("aoi_agent.store.boards._session_factory",
                        make_session_factory(url))
    for name in ("query_machine_stats", "search_standards"):
        monkeypatch.setitem(analysis.PLANNABLE_TOOLS, name, lambda **kw: {"ok": 1})
    monkeypatch.setattr(station_app, "_analysis_graph",
                        analysis.build_analysis_graph(StubClient(), DOMAINS))
    return TestClient(station_app.app)


def events(body: str) -> list[dict]:
    out = []
    for block in body.strip().split("\n\n"):
        name, payload = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
        if name:
            out.append({"event": name, "data": payload})
    return out


def test_the_stream_reports_the_plan_then_each_tool_then_done(client):
    body = client.get("/ask/stream", params={"question": "M22 正常嗎"}).text
    names = [e["event"] for e in events(body)]

    assert names[0] == "plan"
    assert names.count("tool") == 2
    assert names[-1] == "done"


def test_the_plan_event_carries_what_will_run_so_the_page_can_list_it(client):
    body = client.get("/ask/stream", params={"question": "M22 正常嗎"}).text
    first = events(body)[0]["data"]

    assert [c["tool"] for c in first["calls"]] == [
        "query_machine_stats", "search_standards"
    ]
    assert first["interpretation"]


def test_the_done_event_carries_the_run_id_to_navigate_to(client):
    body = client.get("/ask/stream", params={"question": "M22 正常嗎"}).text
    done = events(body)[-1]["data"]

    assert isinstance(done["run_id"], int)
    assert client.get(f"/ask/{done['run_id']}").status_code == 200


def test_the_run_the_stream_persists_is_the_same_one_the_page_renders(client):
    """The stream must not be a second, parallel execution -- that would double
    the cost and could disagree with itself.

    This is also the merge-correctness test: ``stream_mode="updates"`` hands
    back one update per ``Send`` branch, so a naive ``state.update(payload)``
    treats each branch's single-item ``results`` list -- and each node's
    contribution to ``timings_ms`` -- as last-write-wins, keeping only
    whichever update happened to land last. That made the previous version of
    this test order-dependent: it passed only when ``query_machine_stats``
    happened to be the branch folded in last, and failed against
    ``state.update`` otherwise. Asserting both tool names and both a `plan`
    and a `synthesise` timing closes that gap -- any one of the three being
    silently dropped fails it, regardless of arrival order.
    """
    body = client.get("/ask/stream", params={"question": "M22 正常嗎"}).text
    run_id = events(body)[-1]["data"]["run_id"]

    run = analysis_store.get_run(run_id)
    assert sorted(r["tool"] for r in run["results"]) == [
        "query_machine_stats", "search_standards"
    ]
    assert "plan" in run["timings"]
    assert "synthesise" in run["timings"]

    page = client.get(f"/ask/{run_id}").text
    assert "query_machine_stats" in page
    assert "search_standards" in page


def test_a_failing_tool_is_streamed_as_a_failed_tool_event(client, monkeypatch):
    def boom(**kw):
        raise RuntimeError("index unreachable")

    monkeypatch.setitem(analysis.PLANNABLE_TOOLS, "search_standards", boom)
    body = client.get("/ask/stream", params={"question": "M22 正常嗎"}).text
    tools = [e["data"] for e in events(body) if e["event"] == "tool"]

    assert any(t["ok"] is False and "unreachable" in t["error"] for t in tools)


def test_a_rejected_plan_ends_the_stream_with_an_error_event(client, monkeypatch):
    bad = {"interpretation": "i", "assumptions": [],
           "calls": [{"tool": "query_defect_history",
                      "args": {"line_id": "L9"}, "why": "w"}]}
    monkeypatch.setattr(station_app, "_analysis_graph",
                        analysis.build_analysis_graph(StubClient(plan=bad), DOMAINS))
    body = client.get("/ask/stream", params={"question": "L9"}).text
    names = [e["event"] for e in events(body)]

    assert names[-1] == "done", "a rejected plan still produces a viewable run"
    assert any(e["event"] == "error" for e in events(body))


def test_two_calls_to_the_same_tool_get_distinct_rows(client, monkeypatch):
    """Two calls to one tool used to collide on the element id `tool-<name>`
    (keyed by tool name alone), so the second call's `tool` event found
    nothing to update and its row stayed reading "running" forever. Different
    arguments must produce different keys -- one per planned call -- and every
    `tool` event's key must be claimable by exactly one of them, which is the
    actual mechanism that keeps both rows resolving on the page."""
    twice = {
        "interpretation": "two standards lookups",
        "assumptions": [],
        "calls": [
            {"tool": "search_standards", "args": {"query": "open", "top_k": 2},
             "why": "a"},
            {"tool": "search_standards", "args": {"query": "short", "top_k": 2},
             "why": "b"},
        ],
    }
    monkeypatch.setattr(station_app, "_analysis_graph",
                        analysis.build_analysis_graph(StubClient(plan=twice), DOMAINS))
    body = client.get("/ask/stream", params={"question": "two lookups"}).text
    plan_event = next(e for e in events(body) if e["event"] == "plan")
    tool_events = [e["data"] for e in events(body) if e["event"] == "tool"]

    plan_keys = [c["key"] for c in plan_event["data"]["calls"]]
    assert len(set(plan_keys)) == 2, "two different calls must get two different keys"

    tool_keys = sorted(t["key"] for t in tool_events)
    assert tool_keys == sorted(plan_keys), (
        "every tool event must be claimable by exactly one planned call's row"
    )


def test_two_byte_identical_calls_to_the_same_tool_both_resolve(client, monkeypatch):
    """`_tool_base_key` alone (tool + args) is not enough: `validate_plan` has
    no rule against two calls that are byte-identical -- same tool, same
    args -- so content gives them nothing to disambiguate by. This is the
    shape that keying purely on `_tool_base_key` still collides on: both
    `plan` calls carry the exact same base key, and without a further
    tie-break both `tool` events would too, so one page row would never
    resolve. Asserts two distinct keys on the plan side, and that both keys
    get claimed by the two (indistinguishable) tool events -- i.e. both rows
    resolve, not just one of them twice."""
    identical = {
        "interpretation": "same lookup twice",
        "assumptions": [],
        "calls": [
            {"tool": "search_standards", "args": {"query": "open", "top_k": 2},
             "why": "a"},
            {"tool": "search_standards", "args": {"query": "open", "top_k": 2},
             "why": "a"},
        ],
    }
    monkeypatch.setattr(station_app, "_analysis_graph",
                        analysis.build_analysis_graph(StubClient(plan=identical), DOMAINS))
    body = client.get("/ask/stream", params={"question": "same lookup twice"}).text
    plan_event = next(e for e in events(body) if e["event"] == "plan")
    tool_events = [e["data"] for e in events(body) if e["event"] == "tool"]

    plan_keys = sorted(c["key"] for c in plan_event["data"]["calls"])
    assert len(plan_keys) == 2
    assert plan_keys[0] != plan_keys[1], (
        "two byte-identical calls must still get two distinct keys"
    )

    tool_keys = sorted(t["key"] for t in tool_events)
    assert tool_keys == plan_keys, "both rows must resolve, not just one of them twice"


def test_the_form_still_works_with_no_javascript(client):
    """The stream is an enhancement. The station runs on shop-floor browsers."""
    response = client.post("/ask", data={"question": "M22 正常嗎"},
                           follow_redirects=False)
    assert response.status_code == 303
