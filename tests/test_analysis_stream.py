"""Progress events, and the guarantee that the page works without them."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from aoi_agent.analysis import graph as analysis
from aoi_agent.llm.ollama import ChatResult, Timing
from aoi_agent.station import app as station_app
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
    the cost and could disagree with itself."""
    body = client.get("/ask/stream", params={"question": "M22 正常嗎"}).text
    run_id = events(body)[-1]["data"]["run_id"]
    page = client.get(f"/ask/{run_id}").text

    assert "query_machine_stats" in page


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


def test_the_form_still_works_with_no_javascript(client):
    """The stream is an enhancement. The station runs on shop-floor browsers."""
    response = client.post("/ask", data={"question": "M22 正常嗎"},
                           follow_redirects=False)
    assert response.status_code == 303
