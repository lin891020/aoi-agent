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

from aoi_agent.i18n import LANGUAGE_NOTE, line_language
from aoi_agent.graph.checkpoint import make_checkpointer
from aoi_agent.graph.rationale_check import unsourced_figures
from aoi_agent.graph.state import ReviewState
from aoi_agent.llm.ollama import EXPLANATION_DEADLINE_S, OllamaClient
from aoi_agent.provenance import DecisionProvenance, code_version
from aoi_agent.mcp_servers.classify import classify_defect
from aoi_agent.mcp_servers.production import query_board_context, query_machine_stats
from aoi_agent.mcp_servers.standards import search_standards
from aoi_agent.vision.inference import DEFAULT_DISMISS_THRESHOLD

DEFAULT_MODEL = "gpt-oss:20b"

#: Seconds within which one escalated region must have a verdict on the record.
#:
#: WI-300 "Response budget", derived from QP-110: analysis that takes longer
#: than an operator would have taken to look has already spent the saving it
#: exists to make. It is a promise about the *verdict*, and the verdict is
#: ``classify_node`` -- 2.5ms per candidate on CPU, measured -- so the budget is
#: met with about three orders of magnitude to spare.
#:
#: It lives here, on the disposition path, and not in ``llm.ollama`` where it
#: sat until 2026-08-23 doubling as the httpx client timeout. That conflation is
#: the defect: a promise to an operator and a resource bound on a client are
#: different things that happened to share a number, and sharing it made a
#: written explanation fail on more than half of the station's calls. The client
#: bound is ``EXPLANATION_DEADLINE_S``, sized from measurement; this number is
#: read from the work instruction and is not to be raised to accommodate a
#: model. What the explanation deadline may not do is push a *verdict* past
#: this: it cannot, because nothing downstream of ``classify_node`` is consulted
#: to produce one.
RESPONSE_BUDGET_S = 10.0

#: Within the investigation branch, the classifier's confidence below which the
#: region goes to a person.
#:
#: Set equal to ``DEFAULT_DISMISS_THRESHOLD``, and that equality is the whole
#: rule: **the agent branch may confirm a defect, it may never dismiss one.**
#: The only way this branch can drop a real defect is ``decide_node`` writing
#: ``dismissed``, which needs ``model_class == "false_call"``; for that class
#: ``model_confidence`` *is* ``false_call_probability``, so the region would
#: have to sit in the band [this threshold, the dismissal threshold). At
#: equality that band is empty, and it is empty by construction rather than by
#: measurement -- it stays empty through a retrain, where a swept number would
#: have to be swept again and silently would not be.
#:
#: This was 0.90 until 2026-08-23, cited as "the lowest threshold adding no
#: escape". ``scripts/threshold_sweep.py`` measured that claim for the first
#: time and it was false twice over: the lowest such threshold on this split is
#: 0.875, and 0.90 was not derived from the sweep at all -- it was a round
#: number that happened to be conservative. Neither number was the one to ship;
#: 0.875 clears the worst observed miss by 0.003, which is the test split read
#: to three decimal places. The change costs 47 more escalations out of 8143
#: candidates, 0.6% of the queue. See docs/benchmarks.md.
#:
#: WI-300 requires escalation below 0.70. That is a floor, not the operative
#: number: used as the operative number it dismisses eight real defects here and
#: puts the line at 0.767% against QP-110's 0.5% budget. ``LOW_CONFIDENCE``
#: carries the floor and this constant must stay at or above it.
ESCALATE_BELOW = DEFAULT_DISMISS_THRESHOLD

#: How far above ``ESCALATE_BELOW`` an explanation is still worth paying for.
#:
#: 0.035 is the width the system shipped with -- 0.95 minus 0.915, back when
#: both ends were literals -- and it is kept because nothing measured argues
#: for another number. What it buys is measured rather than assumed: on the
#: 2026-08-24 split it puts 1,460 of 6,736 automatic dispositions (21.7%) into
#: the queue for a written rationale, about 2.9 a board, and they are the ones
#: nearest the line where the region would instead have gone to a person --
#: which is where an auditor reading the record most wants to know why.
#:
#: The risk this shape carries, named rather than engineered around: the
#: confidences pile up against 1.0, so a future ``ESCALATE_BELOW`` above 0.965
#: makes this band swallow everything and the explanation cost goes to one
#: model call per candidate. ``tests/test_threshold_citations.py`` fails on
#: that, in both directions, against the current predictions.
EXPLAINED_BAND = 0.035

#: Above this the classifier's class stands without the LLM being asked.
#:
#: A cost gate, not a decision gate. ``confirm_node`` and ``decide_node`` write
#: the same verdict -- ``model_class`` -- so anywhere at or above
#: ``ESCALATE_BELOW`` this threshold changes which path a candidate takes and
#: not what happens to it: swept from 0.90 to 0.999 it changes zero
#: dispositions and adds zero escapes. What it buys is the LLM's written
#: rationale on the quality record, at one 20B-model call each.
#:
#: It must not fall below ``ESCALATE_BELOW``. There it stops being free and
#: starts confirming, unreviewed, regions the flow would have handed to a
#: person -- 66 of them at 0.70. That constraint is the citation; the value
#: within it is a dial. ``docs/architecture.md`` cited "WI-300 decision
#: authority" until 2026-08-23, and WI-300 states no such number.
#:
#: **Derived, because declared broke it.** This was the literal 0.95 until
#: 2026-08-24, when a retrain moved ``DEFAULT_DISMISS_THRESHOLD`` to 0.961 and
#: inverted the pair. Thirteen tests went red, which is the loud half; the
#: quiet half is what inversion does, and it is worse than the ordering
#: argument above describes. Above ``CONFIDENT`` the LLM is not asked at all,
#: so with the two crossed the band between them is empty and **every one of
#: the 6,736 automatically dispositioned candidates on this split would have
#: been recorded with no rationale at all** -- the LLM's only remaining job,
#: silently not done, on a flow whose tests all still described it. Writing it
#: as an offset is the same fix ``ESCALATE_BELOW = DEFAULT_DISMISS_THRESHOLD``
#: already is: a retrain cannot invert what it carries along with it.
CONFIDENT = ESCALATE_BELOW + EXPLAINED_BAND

#: ``explanation_status`` when the operator has a written explanation.
EXPLAINED = "ok"

#: ``explanation_status`` when they do not, and what to tell them instead.
#:
#: These are the three ways the only remaining job of the LLM can fail. None of
#: them changes a disposition -- ``decide_node`` reads ``model_class`` and
#: ``route_after_reason`` reads ``model_confidence``, both of which exist before
#: this node is entered -- so each one costs an explanation and nothing else.
#:
#: They are keys, not prose, because WI-300 requires the absence to be *counted*
#: and a station cannot group by a sentence. The sentence beside each is what an
#: operator reads; ``the model did not answer (ReadTimeout)``, which is what the
#: queue used to show, is a stack trace with the stack removed.
NO_EXPLANATION = {
    "timed_out": (
        "No explanation was written: the model did not answer within the "
        f"{EXPLANATION_DEADLINE_S:.0f}-second explanation deadline."
    ),
    "unreachable": (
        "No explanation was written: the model could not be reached."
    ),
    "unparsed": (
        "No explanation was written: the model answered, but not in the format "
        "the station can read."
    ),
}

#: Appended to each of the above. The operator's first question on seeing a
#: missing explanation is whether the disposition beside it is also missing.
DISPOSITION_UNAFFECTED = (
    " The disposition beside it is the re-verification model's and was not "
    "affected -- the explanation is what is lost, not the verdict."
)


def explanation_notice(status: str) -> str:
    """What to show a person where the explanation would have been.

    Not a rationale, and deliberately not stored in the rationale column: WI-300
    forbids filling the gap by any means, so this is rendered from the status
    rather than written into the record as though the model had said it.
    """
    if status == EXPLAINED or status not in NO_EXPLANATION:
        return ""
    return NO_EXPLANATION[status] + DISPOSITION_UNAFFECTED


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

Your job is to explain the evidence to the operator who may have to look at this
region. Write the explanation you would want if the board were in front of you
and you had to decide: what the criteria require, what the production context
suggests, and what would settle the question if it is not settled.

You are not the one who dispositions the region. That is decided on the vision
model's calibrated confidence, which was measured against yours and found
better at it. So do not write that you are flagging the region for review, or
that you are releasing it -- you are doing neither, and an explanation that
announces a disposition you did not make will contradict the record it is
stored against.

Also give your own reading in "verdict" and "confident". They are kept for the
record and compared against the classifier when the layer is re-evaluated;
nothing downstream acts on them. Answer them honestly rather than tactically:
set "confident" false when the evidence genuinely does not settle the class,
not to push the region towards a person."""


def decision_provenance(model_digest: str | None) -> DecisionProvenance | None:
    """What an auditor needs to reproduce a disposition made on this path.

    Assembled here because this is the only place all three parts are in scope
    at once: the checkpoint digest arrives with the reading, the thresholds are
    the constants this module routes on, and the code version is read from the
    tree. Assembling it at write time instead would record whichever checkpoint
    happened to be loaded then, which is not necessarily the one that produced
    the number on the record.

    Returns ``None`` when the reading carries no digest, so the caller fails at
    the point of writing rather than recording a decision attributed to nothing.
    """
    if not model_digest:
        return None
    return DecisionProvenance(
        model_digest=model_digest,
        thresholds={
            "dismiss": DEFAULT_DISMISS_THRESHOLD,
            "escalate_below": ESCALATE_BELOW,
            "confident": CONFIDENT,
        },
        code_version=code_version(),
    )


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
    # Into the state, so it is checkpointed with the run and comes back with
    # it. An operator answering this region in two days' time records their
    # verdict against the model that read it, not against whatever is in
    # ``models/reverifier.pt`` on the day they get to the queue.
    stamp = decision_provenance(result.get("model_digest"))
    if stamp is not None:
        update["provenance"] = stamp.as_dict()
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
    # Scoped to the class in hand. This path always has one -- it is reasoning
    # about a single classified region -- so it never asks the open-ended
    # question, and a limit written for another class cannot reach the prompt
    # as evidence about this one. Unscoped, the top passage for `open` was
    # pin-hole's "inside a pad: reject", and the model duly told five operators
    # to check whether the open was inside a pad. See docs/benchmarks.md.
    passages = search_standards(
        f"acceptance criteria and disposition for {defect}",
        top_k=2,
        defect_class=defect,
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
                    # The operator reads the rationale in the line's language.
                    # Until 2026-08-29 it was written in English under a
                    # Chinese station, and the switch could not move it --
                    # correctly, since it is a record -- so every queue row
                    # read in the wrong language for the person it was for.
                    # The note is the one sentence the analysis prompts use,
                    # from the same table; the rest of the prompt is unchanged
                    # and the deadline it runs under is re-measured
                    # (`scripts/latency_report.py`), because Chinese prose is
                    # more tokens than English for the same content.
                    {"role": "system",
                     "content": f"{SYSTEM_PROMPT}\n\n{LANGUAGE_NOTE[line_language()]}"},
                    {"role": "user", "content": prompt},
                ],
                think="low",
                response_format=VERDICT_SCHEMA,
            )
        except (httpx.HTTPError, OSError) as error:
            # The model being unreachable or too slow is an operational
            # problem, not a verdict, and it is not a rationale either. The
            # rationale stays empty and ``explanation_status`` records what
            # happened, so the station can render a sentence a person can act
            # on and a report can count how often this fires. Writing
            # ``the model did not answer (ReadTimeout)`` into the operator's
            # evidence panel, which is what this did until 2026-08-23, told
            # nobody anything and was counted by nothing.
            _timed(state, "reason", (time.perf_counter() - started) * 1000)
            return {
                "agent_verdict": state["model_class"],
                "agent_confident": False,
                "agent_rationale": "",
                "explanation_status": (
                    "timed_out"
                    if isinstance(error, httpx.TimeoutException)
                    else "unreachable"
                ),
                "timings_ms": state["timings_ms"],
                "trace": state["trace"],
            }

        try:
            parsed = json.loads(result.text)
        except json.JSONDecodeError:
            # The model answered and the answer is unreadable. Same outcome as
            # above: no rationale, a recorded reason, and a disposition that
            # never depended on either.
            _timed(state, "reason", (time.perf_counter() - started) * 1000)
            state["timings_ms"]["reason_eval"] = round(result.timing.eval_ms, 1)
            return {
                "agent_verdict": state["model_class"],
                "agent_confident": False,
                "agent_rationale": "",
                "explanation_status": "unparsed",
                "timings_ms": state["timings_ms"],
                "trace": state["trace"],
            }

        _timed(state, "reason", (time.perf_counter() - started) * 1000)
        state["timings_ms"]["reason_eval"] = round(result.timing.eval_ms, 1)
        return {
            "agent_verdict": parsed["verdict"],
            "agent_confident": bool(parsed["confident"]),
            "agent_rationale": parsed["rationale"],
            # Checked against the prompt the model was actually handed, here,
            # because this is the only place that string exists. Stored on the
            # row beside the rationale so the queue can warn without re-running
            # anything.
            "rationale_flags": unsourced_figures(parsed["rationale"], prompt),
            "explanation_status": EXPLAINED,
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

    The second consequence is the one ``ESCALATE_BELOW``'s value buys: because
    it equals the dismissal threshold, every region this edge sends to
    ``decide`` whose class is ``false_call`` has already been dismissed
    upstream. So this branch cannot dismiss, and cannot add an escape.
    """
    return (
        "escalate"
        if state["model_confidence"] < ESCALATE_BELOW
        else "decide"
    )


def handover_reason(state: ReviewState) -> str:
    """Why this region is in front of a person, stated in its own terms.

    Not the LLM's rationale, which is an explanation of the evidence and is
    sometimes absent. This sentence is always available, because the thing it
    describes -- ``route_after_reason`` reading ``model_confidence`` -- is the
    only thing that puts a region on the queue.
    """
    return (
        f"the re-verification model's confidence of "
        f"{state['model_confidence']:.3f} is below the {ESCALATE_BELOW:.3f} "
        f"escalation threshold"
    )


def escalate_node(state: ReviewState) -> dict[str, Any]:
    """Suspend and wait for a person.

    ``interrupt`` stops the run here and checkpoints it. The operator may
    answer in a second or in two days; resuming replays nothing and re-runs no
    tools.
    """
    reason = state.get("agent_rationale") or handover_reason(state)
    answer = interrupt(
        {
            "candidate_ref": state["candidate_ref"],
            "reason": reason,
            "explanation_status": state.get("explanation_status", EXPLAINED),
            "rationale_flags": list(state.get("rationale_flags") or []),
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
        "escalation_reason": reason,
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
    # The ``dismissed`` arm is unreachable as the routers are set: reaching here
    # with ``false_call`` needs a confidence above ``ESCALATE_BELOW``, which
    # equals the dismissal threshold, and such a region was dismissed before the
    # investigation began. It is kept so this node is correct read on its own --
    # ``test_the_agent_branch_cannot_dismiss`` is what holds the guarantee.
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
