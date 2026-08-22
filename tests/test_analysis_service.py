"""Running a question end to end and keeping enough to redraw it."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from aoi_agent.analysis import graph as analysis
from aoi_agent.analysis import service
from aoi_agent.llm.ollama import ChatResult, Timing
from aoi_agent.store import analysis as store_analysis
from aoi_agent.store.boards import session_factory
from aoi_agent.store.models import create_all, make_session_factory

PLAN = {
    "interpretation": "M22 against the fleet",
    "assumptions": ["compared with the fleet average"],
    "calls": [{"tool": "query_machine_stats",
               "args": {"defect_type": "open", "days": 7}, "why": "w"}],
}
DOMAINS = {
    "line_id": {"L1", "L2", "L3"}, "machine_id": {"M22"},
    "defect_type": {"open"}, "max_days": 9,
}


@dataclass
class StubClient:
    plan: dict = field(default_factory=lambda: PLAN)
    answer: str = "M22 sits above the fleet."

    def chat(self, messages, **kwargs) -> ChatResult:
        text = json.dumps(self.plan) if kwargs.get("response_format") else self.answer
        return ChatResult(text=text, tool_calls=[], thinking="",
                          timing=Timing(1.0, 0.0, 1.0, 1.0, 10, 10))


@pytest.fixture
def store(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'a.db'}"
    create_all(url)
    monkeypatch.setattr("aoi_agent.store.boards._session_factory",
                        make_session_factory(url))


@pytest.fixture
def graph(monkeypatch):
    monkeypatch.setitem(
        analysis.PLANNABLE_TOOLS, "query_machine_stats",
        lambda **kw: {"defect_type": "open", "fleet_share_of_defects": 0.2,
                      "machines": [{"machine": "L2-M22", "share_of_defects": 0.32,
                                    "per_board": 2.3}]},
    )
    return analysis.build_analysis_graph(StubClient(), DOMAINS)


def test_a_question_is_answered_and_persisted(store, graph):
    run = service.answer_question(graph, "M22 正常嗎", asked_by="mike")

    assert run["answer"]
    assert run["id"]
    assert run["asked_by"] == "mike"
    assert store_analysis.get_run(run["id"])["question"] == "M22 正常嗎"


def test_the_stored_run_can_redraw_its_chart_without_a_model(store, graph):
    """The point of persisting the spec rather than an image: a run from last
    quarter renders today, and no model is asked to reproduce a plan it would
    not reproduce."""
    run = service.answer_question(graph, "M22 正常嗎")
    stored = store_analysis.get_run(run["id"])

    assert stored["chart"]["kind"] == "bar"
    assert stored["chart"]["series"][0]["points"][0]["x"] == "L2-M22"


def test_the_raw_results_are_kept_beside_the_prose(store, graph):
    """Synthesis can describe correct data incorrectly. A reader can only catch
    that if the data is there to check against."""
    run = service.answer_question(graph, "M22 正常嗎")
    stored = store_analysis.get_run(run["id"])

    assert stored["results"][0]["tool"] == "query_machine_stats"
    assert stored["results"][0]["data"]["machines"]


def test_a_refusal_is_stored_too(store, monkeypatch):
    """What the system declined to answer is as interesting as what it did, and
    the eval script reads refusals from here."""
    refusal = {"interpretation": "the store does not cover last year",
               "assumptions": [], "calls": []}
    graph = analysis.build_analysis_graph(StubClient(plan=refusal), DOMAINS)
    run = service.answer_question(graph, "去年呢")

    assert run["refused"] is True
    assert store_analysis.get_run(run["id"])["refused"] is True


def test_recent_runs_come_back_newest_first(store, graph):
    for question in ("一", "二", "三"):
        service.answer_question(graph, question)

    assert [r["question"] for r in store_analysis.recent_runs(10)][:3] == ["三", "二", "一"]


def test_the_run_records_what_the_fan_out_saved(store, graph):
    run = service.answer_question(graph, "M22 正常嗎")
    stored = store_analysis.get_run(run["id"])

    assert "tools_wall" in stored["timings"]
    assert "tools_sequential" in stored["timings"]
