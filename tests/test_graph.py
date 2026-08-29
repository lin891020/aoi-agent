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
from conftest import ABOVE_CONFIDENT, IN_THE_EXPLANATION_BAND


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
        self.systems = []

    def chat(self, messages, **kwargs) -> ChatResult:
        self.prompts.append(messages[-1]["content"])
        self.systems.append(messages[0]["content"])
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


#: What the real ``classify_defect`` returns for the weights that read a
#: region. The stub carries it because the stub stands in for that tool, and a
#: fixture that dropped it would make every flow test exercise a path the
#: station cannot take: ``record_decision`` refuses an automated decision that
#: names no model.
STUB_DIGEST = "sha256:0123456789abcdef"


@pytest.fixture
def stub_tools(monkeypatch):
    """Replace the three MCP tools with fixed answers."""
    # 0.55 is below `ESCALATE_BELOW`, so a test that does not set its own
    # confidence is a test of the escalating branch whether it says so or not.
    # Tests whose subject is the branch state the number themselves.
    state = {"classify": {
        "predicted_class": "open",
        "confidence": 0.55,
        "false_call_probability": 0.10,
        "recommendation": "escalate",
        "model_digest": STUB_DIGEST,
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
    def search_standards(query, top_k=2, defect_class=None):
        # Recorded, not ignored. The scope is the whole defence against a
        # pin-hole limit being quoted at an open, and it is passed from here.
        state.setdefault("standards_calls", []).append(
            {"query": query, "top_k": top_k, "defect_class": defect_class}
        )
        return {
            "passages": [{"document": "open-circuit", "heading": "Classification",
                          "text": "Any confirmed open is a critical defect."}]
        }

    monkeypatch.setattr(flow, "search_standards", search_standards)
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
        "predicted_class": "short", "confidence": ABOVE_CONFIDENT,
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


def test_an_uncertain_case_gathers_evidence_and_goes_to_a_person(stub_tools):
    """The stub classifies at 0.55, below `ESCALATE_BELOW`. Evidence is still
    gathered -- the operator reads it -- but nobody settles it automatically."""
    state, _ = run(flow.build_graph(StubClient(confident=True), InMemorySaver()))

    # `escalate` lands in the trace on resume, not here: `interrupt` suspends
    # the node before it records itself.
    assert state["trace"] == ["classify", "gather_context", "reason"]
    assert "__interrupt__" in state


def test_the_agent_branch_cannot_dismiss():
    """The guarantee `ESCALATE_BELOW`'s value buys, held on the routers alone.

    Only two nodes can write `dismissed`. `dismiss_node` is the calibrated
    threshold, swept and budgeted. `decide_node` is the agent branch, and it
    would have to be swept separately -- and after every retrain, silently, or
    it is a second unaudited place where the line's escape budget gets spent.

    Setting `ESCALATE_BELOW` to the dismissal threshold removes the question
    instead of answering it. Reaching `decide` with class `false_call` needs a
    confidence at or above `ESCALATE_BELOW`; for that class the confidence *is*
    `P(false call)`, so such a region was dismissed before the investigation
    began. The band is empty by construction, and this test walks it rather
    than trusting the argument.
    """
    step = 0.005
    values = [round(i * step, 3) for i in range(int(1 / step) + 1)]

    for false_call in values:
        for confidence in values:
            for predicted in ("false_call", "open", "short", "spur"):
                # The classifier's own arithmetic: `confidence` is the top
                # class's probability, so for `false_call` the two are one
                # number, and for any other class it cannot be the smaller.
                if predicted == "false_call" and confidence != false_call:
                    continue
                if predicted != "false_call" and confidence < false_call:
                    continue

                state = {
                    "model_recommendation": (
                        "dismiss"
                        if false_call >= flow.DEFAULT_DISMISS_THRESHOLD
                        else "review"
                    ),
                    "model_confidence": confidence,
                    "model_class": predicted,
                }
                if flow.route_after_classify(state) != "investigate":
                    continue
                if flow.route_after_reason(state) != "decide":
                    continue

                assert flow.decide_node(state)["disposition"] != "dismissed", (
                    f"the agent branch dismissed {predicted} at confidence "
                    f"{confidence}, P(false call) {false_call}"
                )


def test_a_confident_classification_is_decided_after_the_evidence(stub_tools):
    stub_tools["classify"]["confidence"] = IN_THE_EXPLANATION_BAND
    state, _ = run(flow.build_graph(StubClient(confident=True), InMemorySaver()))

    assert state["trace"] == ["classify", "gather_context", "reason", "decide"]
    assert state["decided_by"] == "agent"


def test_the_llm_saying_it_is_confident_does_not_rescue_a_weak_classification(
    stub_tools,
):
    """Measured, the LLM's `confident` flag was worse at selecting who needs a
    person than the classifier's own number. It no longer routes anything."""
    stub_tools["classify"]["confidence"] = 0.61
    sure, unsure = StubClient(confident=True), StubClient(confident=False)

    for client in (sure, unsure):
        state, _ = run(flow.build_graph(client, InMemorySaver()))
        assert "__interrupt__" in state, "confidence decides, not the LLM"


def test_the_classifier_class_stands_when_the_llm_disagrees(stub_tools):
    """It overrode the classifier twelve times in evaluation and was right once."""
    stub_tools["classify"] = {
        "predicted_class": "spur", "confidence": IN_THE_EXPLANATION_BAND,
        "false_call_probability": 0.02, "recommendation": "review",
    }
    state, _ = run(
        flow.build_graph(StubClient(confident=True, verdict="copper"), InMemorySaver())
    )

    assert state["verdict"] == "spur"
    assert state["agent_verdict"] == "copper", "kept for the record, acted on by nobody"


def test_an_unreachable_model_no_longer_forces_an_escalation(stub_tools):
    """It used to, and that was right while the LLM decided. Now the decision
    never depended on it, so an outage costs an explanation, not a verdict --
    and does not put every candidate on the line into the queue."""
    stub_tools["classify"]["confidence"] = IN_THE_EXPLANATION_BAND

    class DeadClient:
        def chat(self, messages, **kwargs):
            raise httpx.ReadTimeout("timed out")

    state, _ = run(flow.build_graph(DeadClient(), InMemorySaver()))

    assert "__interrupt__" not in state
    assert state["decided_by"] == "agent"


def test_the_evidence_reaches_the_prompt(stub_tools):
    client = StubClient()
    run(flow.build_graph(client, InMemorySaver()))

    prompt = client.prompts[0]
    assert "M22" in prompt
    assert "critical defect" in prompt
    assert "32.1%" in prompt, "the machine's defect mix should be quoted"


def test_the_criteria_are_retrieved_scoped_to_the_class_in_hand(stub_tools):
    """The disposition path always has a class, so it never asks the criteria
    the open-ended question. Unscoped, the top passage for `open` was pin-hole's
    "inside a pad: reject", and the model handed that to five operators as the
    rule for an open."""
    run(flow.build_graph(StubClient(), InMemorySaver()))

    calls = stub_tools["standards_calls"]
    assert [call["defect_class"] for call in calls] == ["open"]


def test_a_false_call_asks_the_criteria_as_a_false_call(stub_tools):
    """No work instruction governs a region with no defect, and scoping to
    `false_call` says so: WI-300 and QP-110 answer it, no acceptance limit
    does. Passing the class through unchecked is what keeps that true when the
    classifier's vocabulary grows."""
    stub_tools["classify"] = {
        "predicted_class": "false_call", "confidence": 0.60,
        "false_call_probability": 0.60, "recommendation": "escalate",
    }
    run(flow.build_graph(StubClient(verdict="false_call"), InMemorySaver()))

    assert stub_tools["standards_calls"][0]["defect_class"] == "false_call"


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


def test_an_unparseable_verdict_reaches_a_person_at_a_confidence_of_0_55(stub_tools):
    """A response that is not JSON is not a verdict -- but it is not what sends
    this region to a person either.

    `route_after_reason` reads `model_confidence` and nothing else, so the 0.55
    is the whole reason this interrupts, and it is stated here rather than
    inherited from the fixture. The pair below is the same call above the
    threshold, and it does not escalate."""
    stub_tools["classify"]["confidence"] = 0.55
    graph = flow.build_graph(StubClient(raw_text="I think it's fine?"), InMemorySaver())
    state, _ = run(graph)

    assert "__interrupt__" in state
    assert state["agent_rationale"] == ""
    assert state["explanation_status"] == "unparsed"


def test_an_unparseable_verdict_in_the_explanation_band_is_decided_anyway(stub_tools):
    """The uncomfortable half, and the one the docs used to get wrong.

    "Unparseable verdict -> escalate" was true while the LLM decided. It is not
    true now: the parse failure is recorded in the rationale the operator would
    read, and the region is dispositioned on the classifier's own call, because
    nothing downstream ever consulted the text that failed to parse. The
    failure direction is intact -- what was lost is an explanation, not a
    verdict -- but the sentence describing it was not, so it is written down as
    a test here."""
    stub_tools["classify"]["confidence"] = IN_THE_EXPLANATION_BAND
    graph = flow.build_graph(StubClient(raw_text="I think it's fine?"), InMemorySaver())
    state, _ = run(graph)

    assert "__interrupt__" not in state
    assert state["verdict"] == "open", "the classifier's class, not the unparsed text"
    assert state["decided_by"] == "agent"
    assert state["agent_rationale"] == ""
    assert state["explanation_status"] == "unparsed"


def test_an_unreachable_model_at_0_55_reaches_a_person_rather_than_crashing(stub_tools):
    """The GPU being busy must not leave a flagged board undispositioned.

    Same reading as the two above: the 0.55 escalates, the outage does not.
    Twelve lines up, `test_an_unreachable_model_no_longer_forces_an_escalation`
    runs the identical dead client inside the explanation band and settles. Both
    are green because of where the confidence sits, and now both say so --
    the band's own number moves with the threshold and no longer belongs in a
    sentence."""
    stub_tools["classify"]["confidence"] = 0.55

    class DeadClient:
        def chat(self, messages, **kwargs):
            raise httpx.ReadTimeout("timed out")

    graph = flow.build_graph(DeadClient(), InMemorySaver())
    state, config = run(graph)

    assert "__interrupt__" in state
    payload = state["__interrupt__"][0].value
    assert payload["explanation_status"] == "timed_out"
    # The handover reason is the confidence, and it always was -- the LLM's
    # failure never sent anything to a person. Until 2026-08-23 this field
    # carried `the model did not answer (ReadTimeout)`, which told the operator
    # neither why the region was in front of them nor what to do about it.
    assert "0.550" in payload["reason"]
    assert "escalation threshold" in payload["reason"]
    assert state["agent_rationale"] == ""

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



def test_the_explanation_is_asked_for_in_the_lines_language(stub_tools, monkeypatch):
    """The operator reads the rationale, so it is written in the language the
    line reads -- Traditional Chinese unless `AOI_LINE_LANGUAGE` says
    otherwise. The sentence is the one the analysis prompts use, from the
    same table, so the two paths cannot drift apart on it."""
    from aoi_agent.i18n import LANGUAGE_NOTE, LINE_LANGUAGE_ENV

    monkeypatch.delenv(LINE_LANGUAGE_ENV, raising=False)
    client = StubClient()
    run(flow.build_graph(client, InMemorySaver()))
    assert client.systems[0].endswith(LANGUAGE_NOTE["zh-TW"])
    assert client.systems[0].startswith(flow.SYSTEM_PROMPT), "the measured prompt is unchanged"

    monkeypatch.setenv(LINE_LANGUAGE_ENV, "en")
    client = StubClient()
    run(flow.build_graph(client, InMemorySaver()))
    assert client.systems[0].endswith(LANGUAGE_NOTE["en"])
