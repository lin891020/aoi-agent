"""The re-verification flow.

A graph rather than a loop, because the decision is gated on confidence and one
branch hands control to a person:

    classify
       |
       +-- model is confident it is a false call ------> dismiss
       +-- model is confident it is a defect ----------> confirm
       +-- anything else --> gather context --> reason --+--> decide
                                                         |
                                                         +--> interrupt: ask an operator

The escape budget in QP-110 is met by that last edge, not by the model being
good enough. Cases the model cannot settle are handed over rather than guessed,
so the threshold buys review reduction without buying escapes.

An ordinary while-loop can express this, but the human hand-off is the awkward
part: it has to suspend mid-run, persist everything, and resume days later when
an operator gets to it. That is what LangGraph's checkpointer and ``interrupt``
provide, and it is the reason this is a graph.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from aoi_agent.graph.checkpoint import make_checkpointer
from aoi_agent.graph.state import ReviewState
from aoi_agent.llm.ollama import OllamaClient
from aoi_agent.mcp_servers.classify import classify_defect
from aoi_agent.mcp_servers.production import query_board_context, query_machine_stats
from aoi_agent.mcp_servers.standards import search_standards

DEFAULT_MODEL = "gpt-oss:20b"

#: Above this the model's own class call stands without further evidence.
CONFIDENT = 0.95

#: Within the investigation branch, the classifier's confidence below which the
#: region goes to a person. Derived the same way as
#: ``DEFAULT_DISMISS_THRESHOLD``: the lowest threshold at which this branch adds
#: no escape to the line's budget. See docs/benchmarks.md.
#:
#: This used to be the LLM's own ``confident`` flag. Measured, that flag was
#: worse than the number the classifier had already produced -- it escalated
#: more (61.7% against 48.5%) and kept a less accurate set (91.3% against
#: 94.9%), while its escalated set carried escapes this threshold does not.
#: WI-300 states 0.70 for operator escalation, but 0.70 leaks eight real defects
#: here, and the escape budget in QP-110 outranks it.
ESCALATE_BELOW = 0.90

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["false_call", "open", "short", "mousebite", "spur", "copper", "pin-hole"],
        },
        "confident": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "confident", "rationale"],
}

SYSTEM_PROMPT = """You are a re-verification analyst at a PCB inspection station.

You are given one region an automated optical inspection flagged, a vision
model's reading of it, the production context around the board, and the
acceptance criteria that apply.

Decide what the region is. Set "confident" to false whenever the evidence does
not settle it -- an operator will then look at it themselves. Guessing to avoid
escalating is the one thing you must not do: a dismissed defect leaves the
plant, while an escalation costs someone a few seconds.

Escalate (confident=false) when the acceptance criteria depend on a measurement
you cannot make, when the class is `open` and the evidence is not unambiguous,
or when two classes are similarly likely."""


def _timed(state: ReviewState, name: str, elapsed_ms: float) -> None:
    state.setdefault("timings_ms", {})[name] = round(elapsed_ms, 1)
    state.setdefault("trace", []).append(name)


def classify_node(state: ReviewState) -> dict[str, Any]:
    """Run the vision model. No LLM involved -- this is a measurement."""
    started = time.perf_counter()
    result = classify_defect(state["candidate_ref"])
    if "error" in result:
        raise ValueError(result["error"])

    update = {
        "model_class": result["predicted_class"],
        "model_confidence": result["confidence"],
        "false_call_probability": result["false_call_probability"],
        "model_recommendation": result["recommendation"],
    }
    _timed(state, "classify", (time.perf_counter() - started) * 1000)
    update["timings_ms"] = state["timings_ms"]
    update["trace"] = state["trace"]
    return update


def route_after_classify(state: ReviewState) -> str:
    """Confidence gate.

    Only the unambiguous ends of the distribution skip the investigation. The
    middle is where escapes come from, so it gets the evidence.
    """
    if state["model_recommendation"] == "dismiss":
        return "dismiss"
    if (
        state["model_confidence"] >= CONFIDENT
        and state["model_class"] not in ("open", "false_call")
    ):
        return "confirm"
    return "investigate"


def gather_context_node(state: ReviewState) -> dict[str, Any]:
    """Pull the three kinds of evidence an operator would look at."""
    started = time.perf_counter()
    board = state["candidate_ref"].split("#")[0]
    defect = state["model_class"]

    context = query_board_context(board)
    stats = (
        query_machine_stats(defect, days=30)
        if defect not in ("false_call",)
        else {}
    )
    passages = search_standards(
        f"acceptance criteria and disposition for {defect}", top_k=2
    ).get("passages", [])

    _timed(state, "gather_context", (time.perf_counter() - started) * 1000)
    return {
        "board_context": context,
        "machine_stats": stats,
        "standards": passages,
        "timings_ms": state["timings_ms"],
        "trace": state["trace"],
    }


def make_reason_node(client: OllamaClient):
    def reason_node(state: ReviewState) -> dict[str, Any]:
        """Let the model weigh the evidence and either decide or hand over."""
        started = time.perf_counter()

        machine_note = ""
        stats = state.get("machine_stats") or {}
        if stats.get("machines"):
            this_machine = state["board_context"].get("machine_id")
            row = next(
                (
                    m
                    for m in stats["machines"]
                    if m["machine"].endswith(str(this_machine))
                ),
                None,
            )
            if row:
                machine_note = (
                    f"This board's machine ({row['machine']}) runs "
                    f"{row['share_of_defects']:.1%} of its defects as "
                    f"{stats['defect_type']}, against a fleet average of "
                    f"{stats['fleet_share_of_defects']:.1%}."
                )

        criteria = "\n\n".join(
            f"[{p['document']} / {p['heading']}]\n{p['text']}"
            for p in state.get("standards", [])
        )

        prompt = f"""Region: {state['candidate_ref']}

Vision model reading:
  class      {state['model_class']}
  confidence {state['model_confidence']:.3f}
  P(false call) {state['false_call_probability']:.3f}

Production context:
  lot {state['board_context'].get('lot_id')} on line {state['board_context'].get('line_id')}, machine {state['board_context'].get('machine_id')}, shift {state['board_context'].get('shift')}
  this lot averages {state['board_context'].get('lot_defects_per_board')} defects per board
  {machine_note}

Applicable acceptance criteria:
{criteria}

Give your verdict."""

        try:
            result = client.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                think="low",
                response_format=VERDICT_SCHEMA,
            )
        except (httpx.HTTPError, OSError) as error:
            # The model being unreachable or too slow is an operational
            # problem, not a verdict. Fail towards the human: an escalated
            # candidate costs someone a few seconds, whereas a crashed run
            # leaves a flagged board with no disposition at all.
            _timed(state, "reason", (time.perf_counter() - started) * 1000)
            return {
                "agent_verdict": state["model_class"],
                "agent_confident": False,
                "agent_rationale": f"the model did not answer ({type(error).__name__})",
                "timings_ms": state["timings_ms"],
                "trace": state["trace"],
            }

        try:
            parsed = json.loads(result.text)
        except json.JSONDecodeError:
            # An unparseable verdict is not a verdict. Escalate rather than
            # guess at what the model meant.
            parsed = {
                "verdict": state["model_class"],
                "confident": False,
                "rationale": "the model's response could not be parsed",
            }

        _timed(state, "reason", (time.perf_counter() - started) * 1000)
        state["timings_ms"]["reason_eval"] = round(result.timing.eval_ms, 1)
        return {
            "agent_verdict": parsed["verdict"],
            "agent_confident": bool(parsed["confident"]),
            "agent_rationale": parsed["rationale"],
            "timings_ms": state["timings_ms"],
            "trace": state["trace"],
        }

    return reason_node


def route_after_reason(state: ReviewState) -> str:
    """Route on the classifier's confidence, not on the LLM's opinion of itself.

    The LLM still runs, and what it writes is what the operator reads on the
    review station. It no longer decides who reads it. Measurement put its
    ``confident`` flag behind a plain threshold on the number the classifier had
    already produced, and its re-classifications behind the classifier's own
    call, so both jobs go back to the model that is better at them.

    A consequence worth naming: an LLM that fails no longer forces an
    escalation. It used to, and that was right while the LLM decided -- an
    unanswered call was a decision not made. Now the decision never depended on
    it, so an outage costs the operator an explanation, not a verdict, and does
    not flood the queue with every candidate on the line.
    """
    return (
        "escalate"
        if state["model_confidence"] < ESCALATE_BELOW
        else "decide"
    )


def escalate_node(state: ReviewState) -> dict[str, Any]:
    """Suspend and wait for a person.

    ``interrupt`` stops the run here and checkpoints it. The operator may
    answer in a second or in two days; resuming replays nothing and re-runs no
    tools.
    """
    answer = interrupt(
        {
            "candidate_ref": state["candidate_ref"],
            "reason": state.get("agent_rationale", "model was not confident"),
            "model_class": state["model_class"],
            "model_confidence": state["model_confidence"],
            "agent_verdict": state.get("agent_verdict"),
            "options": ["false_call", "open", "short", "mousebite", "spur", "copper", "pin-hole"],
        }
    )
    state.setdefault("trace", []).append("escalate")
    return {
        "human_verdict": answer.get("verdict"),
        "human_reviewer": answer.get("reviewer", "operator"),
        "escalation_reason": state.get("agent_rationale", ""),
        "trace": state["trace"],
    }


def dismiss_node(state: ReviewState) -> dict[str, Any]:
    state.setdefault("trace", []).append("dismiss")
    return {
        "disposition": "dismissed",
        "verdict": "false_call",
        "decided_by": "model",
        "trace": state["trace"],
    }


def confirm_node(state: ReviewState) -> dict[str, Any]:
    state.setdefault("trace", []).append("confirm")
    return {
        "disposition": "defect_confirmed",
        "verdict": state["model_class"],
        "decided_by": "model",
        "trace": state["trace"],
    }


def decide_node(state: ReviewState) -> dict[str, Any]:
    """Disposition on the classifier's call.

    Not ``agent_verdict``. Over 60 measured candidates the LLM overrode the
    classifier twelve times, was right once, and broke nine the classifier had
    already got right. Its verdict is kept in state for the record and shown to
    the operator as context, but nothing downstream acts on it.
    """
    state.setdefault("trace", []).append("decide")
    verdict = state["model_class"]
    return {
        "disposition": "dismissed" if verdict == "false_call" else "defect_confirmed",
        "verdict": verdict,
        "decided_by": "agent",
        "trace": state["trace"],
    }


def record_human_node(state: ReviewState) -> dict[str, Any]:
    state.setdefault("trace", []).append("record_human")
    verdict = state.get("human_verdict") or state.get("agent_verdict")
    return {
        "disposition": "dismissed" if verdict == "false_call" else "defect_confirmed",
        "verdict": verdict,
        "decided_by": "human",
        "trace": state["trace"],
    }


def build_graph(client: OllamaClient | None = None, checkpointer=None):
    """Compile the flow.

    The checkpointer defaults to the durable one: an escalation that cannot
    outlive the process is not a hand-off, it is a prompt. Tests pass an
    ``InMemorySaver`` when they only need one run's worth of state.
    """
    client = client or OllamaClient(DEFAULT_MODEL)

    graph = StateGraph(ReviewState)
    graph.add_node("classify", classify_node)
    graph.add_node("gather_context", gather_context_node)
    graph.add_node("reason", make_reason_node(client))
    graph.add_node("escalate", escalate_node)
    graph.add_node("dismiss", dismiss_node)
    graph.add_node("confirm", confirm_node)
    graph.add_node("decide", decide_node)
    graph.add_node("record_human", record_human_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {"dismiss": "dismiss", "confirm": "confirm", "investigate": "gather_context"},
    )
    graph.add_edge("gather_context", "reason")
    graph.add_conditional_edges(
        "reason", route_after_reason, {"decide": "decide", "escalate": "escalate"}
    )
    graph.add_edge("escalate", "record_human")
    for terminal in ("dismiss", "confirm", "decide", "record_human"):
        graph.add_edge(terminal, END)

    return graph.compile(checkpointer=checkpointer or make_checkpointer())
