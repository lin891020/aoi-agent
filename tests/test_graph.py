"""Flow tests with a stubbed model, so they need neither GPU nor Ollama."""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from aoi_agent.graph import flow
from aoi_agent.llm.ollama import ChatResult, Timing


@dataclass
class StubClient:
    """Returns a fixed verdict and records what it was asked."""

    verdict: str = "open"
    confident: bool = True
    rationale: str = "stub"
    raw_text: str | None = None
    prompts: list = None

    def __post_init__(self):
        self.prompts = []

    def chat(self, messages, **kwargs) -> ChatResult:
        self.prompts.append(messages[-1]["content"])
        text = self.raw_text if self.raw_text is not None else json.dumps(
            {
                "verdict": self.verdict,
                "confident": self.confident,
                "rationale": self.rationale,
            }
        )
        return ChatResult(
            text=text,
            tool_calls=[],
            thinking="",
            timing=Timing(1.0, 0.0, 1.0, 1.0, 10, 10),
        )


@pytest.fixture
def stub_tools(monkeypatch):
    """Replace the three MCP tools with fixed answers."""
    state = {"classify": {
        "predicted_class": "open",
        "confidence": 0.55,
        "false_call_probability": 0.10,
        "recommendation": "escalate",
    }}

    def classify_defect(candidate_ref):
        return {"candidate_ref": candidate_ref, **state["classify"]}

    monkeypatch.setattr(flow, "classify_defect", classify_defect)
    monkeypatch.setattr(
        flow, "query_board_context",
        lambda board: {"lot_id": "LOT-1", "line_id": "L2", "machine_id": "M22",
                       "shift": "A", "lot_defects_per_board": 7.2},
    )
    monkeypatch.setattr(
        flow, "query_machine_stats",
        lambda defect, days=30: {
            "defect_type": defect, "fleet_share_of_defects": 0.225,
            "machines": [{"machine": "L2-M22", "share_of_defects": 0.321, "per_board": 2.3}],
        },
    )
    monkeypatch.setattr(
        flow, "search_standards",
        lambda query, top_k=2: {
            "passages": [{"document": "open-circuit", "heading": "Classification",
                          "text": "Any confirmed open is a critical defect."}]
        },
    )
    return state


def run(graph, reference="board#0"):
    config = {"configurable": {"thread_id": reference}}
    return graph.invoke({"candidate_ref": reference}, config=config), config


def test_a_confident_false_call_is_dismissed_without_the_llm(stub_tools):
    stub_tools["classify"] = {
        "predicted_class": "false_call", "confidence": 0.99,
        "false_call_probability": 0.99, "recommendation": "dismiss",
    }
    client = StubClient()
    state, _ = run(flow.build_graph(client, InMemorySaver()))

    assert state["disposition"] == "dismissed"
    assert state["decided_by"] == "model"
    assert state["trace"] == ["classify", "dismiss"]
    assert client.prompts == [], "the LLM must not be consulted on a clear dismissal"


def test_a_confident_defect_is_confirmed_without_the_llm(stub_tools):
    stub_tools["classify"] = {
        "predicted_class": "short", "confidence": 0.99,
        "false_call_probability": 0.001, "recommendation": "review",
    }
    client = StubClient()
    state, _ = run(flow.build_graph(client, InMemorySaver()))

    assert state["verdict"] == "short"
    assert state["trace"] == ["classify", "confirm"]
    assert client.prompts == []


def test_a_confident_open_still_gathers_evidence(stub_tools):
    """`open` never shortcuts, however sure the model is.

    An escaped open ships a dead board, and WI-201 says the class is the
    hardest to tell from a registration artefact.
    """
    stub_tools["classify"] = {
        "predicted_class": "open", "confidence": 0.99,
        "false_call_probability": 0.001, "recommendation": "review",
    }
    state, _ = run(flow.build_graph(StubClient(), InMemorySaver()))

    assert "gather_context" in state["trace"]


def test_an_uncertain_case_gathers_evidence_and_decides(stub_tools):
    state, _ = run(flow.build_graph(StubClient(confident=True), InMemorySaver()))

    assert state["trace"] == ["classify", "gather_context", "reason", "decide"]
    assert state["decided_by"] == "agent"


def test_the_evidence_reaches_the_prompt(stub_tools):
    client = StubClient()
    run(flow.build_graph(client, InMemorySaver()))

    prompt = client.prompts[0]
    assert "M22" in prompt
    assert "critical defect" in prompt
    assert "32.1%" in prompt, "the machine's defect mix should be quoted"


def test_an_unconfident_agent_interrupts_for_a_person(stub_tools):
    graph = flow.build_graph(StubClient(confident=False), InMemorySaver())
    state, config = run(graph)

    assert "__interrupt__" in state
    payload = state["__interrupt__"][0].value
    assert payload["candidate_ref"] == "board#0"
    assert "false_call" in payload["options"]


def test_resuming_records_the_operator_as_the_decider(stub_tools):
    graph = flow.build_graph(StubClient(confident=False), InMemorySaver())
    _, config = run(graph)

    state = graph.invoke(
        Command(resume={"verdict": "short", "reviewer": "mike"}), config=config
    )

    assert state["verdict"] == "short"
    assert state["decided_by"] == "human"
    assert state["human_reviewer"] == "mike"
    assert state["trace"][-2:] == ["escalate", "record_human"]


def test_an_unparseable_verdict_escalates_rather_than_guessing(stub_tools):
    """A response that is not JSON is not a verdict."""
    graph = flow.build_graph(StubClient(raw_text="I think it's fine?"), InMemorySaver())
    state, _ = run(graph)

    assert "__interrupt__" in state


def test_an_unreachable_model_escalates_rather_than_crashing(stub_tools):
    """The GPU being busy must not leave a flagged board undispositioned."""

    class DeadClient:
        def chat(self, messages, **kwargs):
            raise httpx.ReadTimeout("timed out")

    graph = flow.build_graph(DeadClient(), InMemorySaver())
    state, config = run(graph)

    assert "__interrupt__" in state
    assert "did not answer" in state["__interrupt__"][0].value["reason"]

    resumed = graph.invoke(
        Command(resume={"verdict": "open", "reviewer": "mike"}), config=config
    )
    assert resumed["decided_by"] == "human"


def test_timings_separate_the_stages(stub_tools):
    state, _ = run(flow.build_graph(StubClient(), InMemorySaver()))

    assert {"classify", "gather_context", "reason"} <= set(state["timings_ms"])
    assert "reason_eval" in state["timings_ms"], (
        "inference time must be recorded separately from wall time, which "
        "includes queueing behind other jobs on the same GPU"
    )
