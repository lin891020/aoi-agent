"""The analysis flow.

A second graph, separate from the disposition flow. It has no checkpointer:
nothing suspends, nobody is in the loop, and a question is answered in one
invocation. The disposition flow needs one because of `interrupt`; adopting a
framework feature that has no work to do is what this project spent a day
removing from the other graph.

What the graph is for here is the fan-out. `Send` expands a plan of N calls into
N branches whose count is not known until the plan exists, and an `operator.add`
reducer merges what they return. Without the reducer two branches writing
`results` in one superstep is an `InvalidUpdateError` -- which is the framework
refusing to let a race happen rather than a race happening.

Measured before building: four real tools take 183ms in parallel against 462ms
in sequence, while the two model calls cost around 25 seconds. The fan-out is
the correct structure for independent work and it scales as tools multiply. It
is not a latency optimisation, and nothing here should claim it is.
"""

from __future__ import annotations

import json
import operator
import time
from typing import Annotated, Any, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from aoi_agent.analysis.charts import chart_spec_for
from aoi_agent.analysis.plan import (
    PLAN_SCHEMA,
    PLANNABLE_TOOLS,
    Domains,
    capability_summary,
    store_domains,
    validate_plan,
)
from aoi_agent.analysis.prompts import (
    build_planning_messages,
    build_synthesis_messages,
)
from aoi_agent.analysis.tools import ToolResult, run_call
from aoi_agent.i18n import translate

#: Re-exported so the registry a branch will call is reachable from the flow
#: that fans out over it. Note that nothing here looks a tool up: `run_call`
#: does that, in `tools.py`. Substituting an entry through this name works
#: because all three modules bind the same dict object, not because the lookup
#: happens here -- so rebinding this name rather than mutating the dict would
#: change nothing.
__all__ = ["AnalysisState", "PLANNABLE_TOOLS", "build_analysis_graph"]


class AnalysisState(TypedDict, total=False):
    question: str
    #: The language to write in. Set from the asker's locale and then frozen
    #: onto the run: what the planning call produces is a record of what
    #: happened, and a record does not get rewritten when somebody changes the
    #: switch. Only the synthesised answer is produced again -- see
    #: `analysis/service.py`.
    lang: str
    plan: dict | None
    plan_errors: list[str]
    refused: bool
    #: `time.perf_counter()` at the instant the plan node returned. Not a
    #: duration and not shown anywhere: it is the origin `collect_node`
    #: measures the fan-out's wall time from, and it has to be taken before
    #: the routing that builds the `Send` list rather than inside a branch.
    fan_out_at: float
    results: Annotated[list[ToolResult], operator.add]
    timings_ms: Annotated[dict[str, float], operator.or_]
    chart_spec: dict | None
    answer: str


def make_plan_node(client, domains: Domains):
    def plan_node(state: AnalysisState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            result = client.chat(
                build_planning_messages(state["question"], domains, state.get("lang")),
                think="low",
                response_format=PLAN_SCHEMA,
            )
        except (httpx.HTTPError, OSError) as error:
            return {
                "plan": None,
                "plan_errors": [f"the planner did not answer ({type(error).__name__})"],
                "refused": False,
                "fan_out_at": time.perf_counter(),
                "timings_ms": {"plan": (time.perf_counter() - started) * 1000},
            }

        try:
            plan = json.loads(result.text)
        except json.JSONDecodeError:
            # An unparseable plan is not a plan. Showing the person what came
            # back beats guessing at what was meant.
            return {
                "plan": None,
                "plan_errors": ["the planner's response could not be parsed as a plan"],
                "refused": False,
                "fan_out_at": time.perf_counter(),
                "timings_ms": {"plan": (time.perf_counter() - started) * 1000},
            }

        # A plan with no calls is the model declining, not failing. It carries
        # its reason in `interpretation` and renders as an answer.
        refused = not (plan.get("calls") or [])
        errors = [] if refused else validate_plan(plan, domains)

        return {
            "plan": plan,
            "plan_errors": errors,
            "refused": refused,
            "fan_out_at": time.perf_counter(),
            "timings_ms": {"plan": (time.perf_counter() - started) * 1000},
        }

    return plan_node


def fan_out(state: AnalysisState) -> list[Send] | str:
    """Expand the plan into one branch per call.

    The number of branches comes from the plan, which is why this is `Send` and
    not a fixed set of edges: the graph's shape is not known until the question
    has been read.
    """
    if state.get("plan_errors") or state.get("refused") or not state.get("plan"):
        return "report"
    return [
        Send("run_tool", {"call": call, "position": position})
        for position, call in enumerate(state["plan"]["calls"])
    ]


def run_tool_node(payload: dict) -> dict[str, Any]:
    """Receives one call as its entire state; contributes one result upward."""
    return {"results": [run_call(payload["call"], payload.get("position", 0))]}


def collect_node(state: AnalysisState) -> dict[str, Any]:
    """Join. Builds the chart and records what the fan-out actually cost.

    ``tools_wall`` is measured from the moment the plan node returned to the
    moment this join began -- the whole tool phase, scheduling included. It
    used to be ``max(elapsed_ms)``, the longest single branch, which is a
    *lower* bound: it excludes the superstep's own dispatch, so it reported the
    fan-out as faster than it was. This is the one figure on the page that is
    evidence the fan-out is real, so understating it is exactly the wrong
    error to make; over-inclusive is the safe direction.

    It is not a saving. Four independent tools are parallel because the
    question is, and the two model calls either side of them cost around
    twenty-five seconds regardless. ``tools_sequential`` is beside it so a
    reader can see the ratio for what it is: the shape of the work, at a scale
    the model calls dwarf.
    """
    results = state.get("results") or []
    sequential = sum(r["elapsed_ms"] for r in results)
    started = state.get("fan_out_at")
    if started is None:
        # Only when `collect_node` is called outside the graph -- nothing in
        # the flow reaches here without `plan_node` having stamped it. Falling
        # back to the longest branch keeps a direct call meaningful rather
        # than reporting a wall time of zero.
        wall = max((r["elapsed_ms"] for r in results), default=0.0)
    else:
        wall = (time.perf_counter() - started) * 1000
    return {
        "chart_spec": chart_spec_for(results),
        "timings_ms": {
            "tools_wall": round(wall, 1),
            "tools_longest_branch": round(
                max((r["elapsed_ms"] for r in results), default=0.0), 1
            ),
            "tools_sequential": round(sequential, 1),
        },
    }


def make_synthesise_node(client):
    def synthesise_node(state: AnalysisState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            result = client.chat(
                build_synthesis_messages(
                    state["question"], state.get("plan") or {},
                    state.get("results") or [], state.get("lang"),
                ),
                think="low",
            )
            answer = result.text.strip()
        except (httpx.HTTPError, OSError) as error:
            # The results are already correct and already on screen. Losing the
            # prose costs a reader some effort; losing the results would cost
            # them the answer.
            answer = (
                f"The tools returned their results, but the summary could not be "
                f"written ({type(error).__name__}). The figures below are complete."
            )
        return {
            "answer": answer,
            "timings_ms": {"synthesise": (time.perf_counter() - started) * 1000},
        }

    return synthesise_node


def refusal_answer() -> str:
    """What to show when the planner declined to plan any lookup.

    A dead end is a place someone has to leave, so it names the exit. The list
    is `capability_summary()` rather than a paragraph written here: a
    hand-maintained account of what a system can do goes stale silently, and
    this one would go stale in the place where a reader is already stuck.

    Written in the station's default language. A refusal is not re-derivable
    the way a synthesised answer is -- there are no stored results to write it
    again from -- so, like everything the planning call produced, it keeps the
    language it was made in. See `station/i18n.py`.
    """
    lines = [
        translate("analysis.refused.opening"),
        "",
        translate("analysis.refused.capabilities"),
        "",
        *(f"- {line}" for line in capability_summary()),
    ]
    return "\n".join(lines)


def report_node(state: AnalysisState) -> dict[str, Any]:
    """Terminal for a refusal, a rejected plan, or no plan at all.

    Nothing ran in any of the three, but they are not the same thing and an
    operator acts on them differently. A rejected plan is a question to rephrase
    or a value that does not exist; a planner that never answered is the model
    being down, and telling someone their plan "did not validate" sends them to
    look for a fault in a question that was never read.
    """
    if state.get("refused"):
        # Not the plan's `interpretation`. That string is already on the page
        # under "how it read your question", and putting it here too gave a
        # refusal two headings and one sentence: the reader was shown their own
        # question restated, twice, and told neither that it could not be
        # answered nor what could be asked instead.
        #
        # What replaces it says the question was not answered, and what can
        # be asked -- read off the tool registry, so it cannot go stale.
        return {"answer": refusal_answer(), "chart_spec": None}

    errors = "\n".join(f"- {e}" for e in state.get("plan_errors") or [])
    if state.get("plan") is None:
        # Nothing came back to validate: the planner was unreachable, or what
        # it returned was not a plan.
        return {
            "answer": "Nothing was run because no plan was produced:\n" + errors,
            "chart_spec": None,
        }
    return {
        "answer": "The plan was not run because it did not validate:\n" + errors,
        "chart_spec": None,
    }


def build_analysis_graph(client, domains: Domains | None = None):
    """Compile the flow. No checkpointer: nothing here suspends."""
    domains = domains or store_domains()

    graph = StateGraph(AnalysisState)
    graph.add_node("plan", make_plan_node(client, domains))
    graph.add_node("run_tool", run_tool_node)
    graph.add_node("collect", collect_node)
    graph.add_node("synthesise", make_synthesise_node(client))
    graph.add_node("report", report_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", fan_out, ["run_tool", "report"])
    graph.add_edge("run_tool", "collect")
    graph.add_edge("collect", "synthesise")
    graph.add_edge("synthesise", END)
    graph.add_edge("report", END)

    return graph.compile()
