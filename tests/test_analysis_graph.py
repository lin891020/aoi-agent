"""The analysis flow, with a stubbed model so it needs neither GPU nor Ollama."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import pytest

from aoi_agent.analysis import graph as analysis
from aoi_agent.llm.ollama import ChatResult, Timing

DOMAINS = {
    "line_id": {"L1", "L2", "L3"},
    "machine_id": {"M11", "M12", "M21", "M22", "M31", "M32"},
    "defect_type": {"open", "short", "mousebite", "spur", "copper", "pin-hole"},
    "max_days": 9,
}

GOOD_PLAN = {
    "interpretation": "M22 against the fleet",
    "assumptions": ["compared with the fleet average"],
    "calls": [
        {"tool": "query_machine_stats", "args": {"defect_type": "open", "days": 7},
         "why": "fleet comparison"},
        {"tool": "search_standards", "args": {"query": "open", "top_k": 2},
         "why": "criteria"},
    ],
}


@dataclass
class StubClient:
    """Returns the plan first and the prose second, recording both prompts."""

    plan: dict = field(default_factory=lambda: GOOD_PLAN)
    answer: str = "M22 runs above the fleet on opens."
    calls: list = field(default_factory=list)

    def chat(self, messages, **kwargs) -> ChatResult:
        self.calls.append(messages)
        text = json.dumps(self.plan) if kwargs.get("response_format") else self.answer
        return ChatResult(text=text, tool_calls=[], thinking="",
                          timing=Timing(1.0, 0.0, 1.0, 1.0, 10, 10))


@pytest.fixture
def stub_tools(monkeypatch):
    """Every tool answers instantly and records that it ran."""
    ran = []

    def make(name, payload):
        def tool(**kwargs):
            ran.append(name)
            time.sleep(0.05)
            return payload
        return tool

    monkeypatch.setitem(
        analysis.PLANNABLE_TOOLS, "query_machine_stats",
        make("query_machine_stats",
             {"defect_type": "open", "fleet_share_of_defects": 0.2,
              "machines": [{"machine": "L2-M22", "share_of_defects": 0.32,
                            "per_board": 2.3}]}),
    )
    monkeypatch.setitem(
        analysis.PLANNABLE_TOOLS, "search_standards",
        make("search_standards", {"passages": [{"document": "WI-201", "text": "x"}]}),
    )
    return ran


def run(client, question="M22 正常嗎", domains=DOMAINS):
    return analysis.build_analysis_graph(client, domains).invoke(
        {"question": question, "results": [], "timings_ms": {}}
    )


def test_a_valid_plan_runs_every_call_and_produces_an_answer(stub_tools):
    state = run(StubClient())

    assert sorted(stub_tools) == ["query_machine_stats", "search_standards"]
    assert len(state["results"]) == 2
    assert state["answer"]
    assert state["plan_errors"] == []


def test_the_results_arrive_through_the_reducer_not_by_overwriting(stub_tools):
    """Two parallel branches both write `results`. Without the reducer this is
    an InvalidUpdateError; with it, both land."""
    state = run(StubClient())
    tools = {r["tool"] for r in state["results"]}

    assert tools == {"query_machine_stats", "search_standards"}


def test_the_branches_run_concurrently(stub_tools):
    """Each stub sleeps 50ms. Sequential would be 100ms plus overhead."""
    started = time.perf_counter()
    run(StubClient())
    elapsed = (time.perf_counter() - started) * 1000

    assert elapsed < 90, f"branches look sequential: {elapsed:.0f}ms"


def test_an_invalid_plan_runs_nothing_and_reports_every_error(stub_tools):
    bad = {
        "interpretation": "i", "assumptions": [],
        "calls": [{"tool": "query_defect_history",
                   "args": {"line_id": "L9", "days": 999}, "why": "w"}],
    }
    state = run(StubClient(plan=bad))

    assert stub_tools == [], "no tool may run when validation fails"
    assert len(state["plan_errors"]) >= 2
    assert state["results"] == []
    assert state["chart_spec"] is None


def test_a_refusal_is_not_an_error(stub_tools):
    """A plan with no calls is the model declining, and it renders as an answer
    rather than as a failure."""
    refusal = {"interpretation": "the store does not cover last year",
               "assumptions": [], "calls": []}
    state = run(StubClient(plan=refusal))

    assert stub_tools == []
    assert state["plan"]["interpretation"]
    assert state["refused"] is True


def test_one_failing_branch_does_not_take_the_others_with_it(monkeypatch, stub_tools):
    def boom(**kwargs):
        raise RuntimeError("index unreachable")

    monkeypatch.setitem(analysis.PLANNABLE_TOOLS, "search_standards", boom)
    state = run(StubClient())

    assert len(state["results"]) == 2
    ok = {r["tool"]: r["ok"] for r in state["results"]}
    assert ok == {"query_machine_stats": True, "search_standards": False}
    assert state["answer"], "a partial answer is still an answer"


def test_the_failure_reaches_the_synthesis_prompt(monkeypatch, stub_tools):
    """The answer can only name what is missing if it is told."""
    def boom(**kwargs):
        raise RuntimeError("index unreachable")

    monkeypatch.setitem(analysis.PLANNABLE_TOOLS, "search_standards", boom)
    client = StubClient()
    run(client)

    synthesis = "\n".join(m["content"] for m in client.calls[-1])
    assert "index unreachable" in synthesis


def test_an_unparseable_plan_is_reported_rather_than_guessed_at(stub_tools):
    class Garbage(StubClient):
        def chat(self, messages, **kwargs):
            self.calls.append(messages)
            return ChatResult(text="I think you want stats?", tool_calls=[],
                              thinking="", timing=Timing(1.0, 0.0, 1.0, 1.0, 10, 10))

    state = run(Garbage())

    assert stub_tools == []
    assert any("parse" in e.lower() for e in state["plan_errors"])


def test_an_unreachable_model_reports_instead_of_crashing(stub_tools):
    import httpx

    class Dead:
        def chat(self, messages, **kwargs):
            raise httpx.ReadTimeout("timed out")

    state = run(Dead())

    assert stub_tools == []
    assert any("ReadTimeout" in e for e in state["plan_errors"])


def test_timings_are_recorded_per_tool_and_for_the_phases(stub_tools):
    state = run(StubClient())

    assert "plan" in state["timings_ms"]
    assert "tools_wall" in state["timings_ms"]
    assert "synthesise" in state["timings_ms"]
    assert state["timings_ms"]["tools_sequential"] >= state["timings_ms"]["tools_wall"]
