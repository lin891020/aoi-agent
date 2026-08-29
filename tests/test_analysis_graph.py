"""The analysis flow, with a stubbed model so it needs neither GPU nor Ollama."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import pytest

from aoi_agent.analysis import graph as analysis
from aoi_agent.analysis import plan as plan_module
from aoi_agent.analysis.plan import capability_summary
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


def test_the_branches_run_concurrently(monkeypatch):
    """Concurrency is asserted as overlapping intervals, not as elapsed time.

    Do not "simplify" this back to a stopwatch. A `< 90ms` bound on two 50ms
    tools measured 60ms once -- about 30ms of headroom on a fanless machine
    that CLAUDE.md says throttles under sustained load. That assertion fails on
    a slow run and passes on a sequential implementation that happens to be
    fast, which is backwards on both counts.

    Overlap is the property itself: two branches whose open intervals intersect
    were in flight at the same time, whatever the machine was doing at the
    time. It is also the only assertion in this file that a `for call in
    plan["calls"]` loop inside one node would fail -- every other test here
    passes against that. This is what makes the Send fan-out load-bearing
    rather than decorative.
    """
    spans: dict[str, tuple[float, float]] = {}

    def make(name):
        def tool(**kwargs):
            started = time.perf_counter()
            time.sleep(0.05)
            spans[name] = (started, time.perf_counter())
            return {}
        return tool

    for name in ("query_machine_stats", "search_standards"):
        monkeypatch.setitem(analysis.PLANNABLE_TOOLS, name, make(name))

    run(StubClient())

    assert len(spans) == 2, f"both branches must have run, got {sorted(spans)}"
    (first_start, first_end), (second_start, second_end) = spans.values()
    overlap_ms = (min(first_end, second_end) - max(first_start, second_start)) * 1000

    assert overlap_ms > 0, (
        f"the branches did not overlap: one finished {-overlap_ms:.0f}ms before "
        f"the other started, which is a sequential run"
    )


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
    """A plan with no calls is the model declining, not failing."""
    refusal = {"interpretation": "the store does not cover last year",
               "assumptions": [], "calls": []}
    state = run(StubClient(plan=refusal))

    assert stub_tools == []
    assert state["plan"]["interpretation"]
    assert state["refused"] is True


def test_a_refusal_does_not_answer_by_repeating_the_question_back(stub_tools):
    """The page shows how the question was read, and then the answer.

    Until 2026-08-23 a refusal put the plan's `interpretation` in both, so
    someone who asked what the system could do was shown their own question
    restated, twice, and was never told either that it could not be answered or
    what could be asked instead. Two headings, one string, no answer.
    """
    reading = "the user is asking what this system can do"
    refusal = {"interpretation": reading, "assumptions": [], "calls": []}
    state = run(StubClient(plan=refusal))

    assert state["answer"] != state["plan"]["interpretation"]
    assert reading not in state["answer"]


def test_a_refusal_says_what_can_be_asked_instead(stub_tools):
    """And says it from the registry, so it cannot go stale."""
    refusal = {"interpretation": "no lookup fits", "assumptions": [], "calls": []}
    state = run(StubClient(plan=refusal))

    for line in capability_summary():
        assert line in state["answer"]


def test_the_capability_summary_comes_from_the_registry_not_from_a_list():
    """A hand-written list of what the system can do rots, and reads as correct
    while it does. This is the same argument that made `model_digest` a hash of
    the checkpoint rather than a string somebody bumps: derived, never declared.
    """
    from aoi_agent.analysis.plan import PLANNABLE_TOOLS

    summary = "\n".join(capability_summary())

    assert len(capability_summary()) == len(PLANNABLE_TOOLS)
    for name in PLANNABLE_TOOLS:
        assert name in summary


def test_a_tool_added_to_the_registry_reaches_the_refusal(monkeypatch, stub_tools):
    """The mutation that proves the line above is load-bearing."""

    def query_solder_paste(line_id: str) -> dict:
        """Read the paste inspection log for one line."""
        return {}

    monkeypatch.setattr(
        plan_module, "REGISTRATIONS",
        (*plan_module.REGISTRATIONS, plan_module.Registration(
            query_solder_paste, identifiers=frozenset({"line_id"}))),
    )
    refusal = {"interpretation": "no lookup fits", "assumptions": [], "calls": []}
    state = run(StubClient(plan=refusal))

    assert "query_solder_paste" in state["answer"]
    assert "Read the paste inspection log for one line." in state["answer"]


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
    assert (
        state["timings_ms"]["tools_sequential"]
        >= state["timings_ms"]["tools_longest_branch"]
    )


def test_the_wall_time_covers_the_scheduling_and_not_only_the_branches(stub_tools):
    """`tools_wall` is the figure the page offers as evidence the fan-out is
    real, and it used to be `max(elapsed_ms)` -- the longest single branch.

    That excludes the superstep's own dispatch, so it reported the tool phase
    as faster than it was: a lower bound printed as a measurement, and an
    understatement is precisely the wrong error for the one number that is
    supposed to be evidence. It is now measured from the plan node returning
    to the join beginning, which cannot be less than the longest branch.
    """
    state = run(StubClient())

    assert (
        state["timings_ms"]["tools_wall"]
        >= state["timings_ms"]["tools_longest_branch"]
    )


def test_a_planner_outage_is_not_reported_as_a_validation_failure(stub_tools):
    """There was no plan to validate, so saying one failed validation sends the
    operator to look for a bad question when the model is simply down."""
    import httpx

    class Dead:
        def chat(self, messages, **kwargs):
            raise httpx.ReadTimeout("timed out")

    state = run(Dead())

    assert "did not validate" not in state["answer"]
    assert "ReadTimeout" in state["answer"]


# --- asking what can be asked -------------------------------------------------

@pytest.mark.parametrize("question", [
    "你能查什麼？", "這個可以問什麼", "有什麼功能", "what can I ask here?",
    "What can you do?", "help",
])
def test_asking_what_can_be_asked_is_answered_without_calling_the_model(stub_tools, question):
    """«你能查什麼» was the first question typed at the station, and it went
    to the planner, which read it as a lookup it could not plan and refused --
    twice restating the question and never once listing what could be asked.
    The answer to this question is the registry, and the registry does not
    need a model to be read."""
    client = StubClient()
    state = run(client, question)

    assert client.calls == [], "no model call answers a question about the registry"
    assert stub_tools == []
    assert state["capability_question"] is True
    for line in capability_summary():
        assert line in state["answer"]


def test_a_capability_answer_does_not_open_by_saying_the_data_is_missing(stub_tools):
    """The ordinary refusal opens with "no lookup answers that". A question
    about what can be asked *is* answered, so it must not."""
    from aoi_agent.i18n import translate

    state = run(StubClient(), "你能查什麼")
    assert translate("analysis.refused.opening") not in state["answer"]
    assert translate("analysis.capabilities.opening") in state["answer"]


def test_an_ordinary_question_still_goes_to_the_planner(stub_tools):
    client = StubClient()
    state = run(client, "M22 的 open 有什麼問題嗎")
    assert len(client.calls) == 2
    assert not state.get("capability_question")


# --- what the model reported, kept beside what the page waited ---------------

def test_the_models_own_inference_time_is_recorded_for_both_calls(stub_tools):
    """`plan` and `synthesise` are wall times. The stub's `Timing` says the
    model spent 1ms evaluating; that figure has to reach the stored run under
    its own key, because on this machine the two differ by the length of
    whatever else is holding the GPU, and the page shows both."""
    state = run(StubClient())
    timings = state["timings_ms"]

    assert timings["plan_eval"] == 1.0
    assert timings["plan_load"] == 0.0
    assert timings["synthesise_eval"] == 1.0
    assert "chart" in timings, "deriving the chart is a stage and is timed"
