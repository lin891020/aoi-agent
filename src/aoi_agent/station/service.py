"""Starting and resuming one re-verification, independent of who is watching.

The CLI and the web station differ only in how they ask a person for a verdict.
Everything either side of that -- running the flow, noticing it suspended,
queueing it, resuming it, recording the decision -- is the same, and lives here
so the two cannot drift apart.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import Command

from aoi_agent.provenance import DecisionProvenance
from aoi_agent.store import dispositions, escalations
from aoi_agent.store.boards import record_decision

VERDICT_OPTIONS = [
    "false_call",
    "open",
    "short",
    "mousebite",
    "spur",
    "copper",
    "pin-hole",
]


def thread_for(reference: str) -> str:
    """One region, one thread.

    Keying the checkpointer by the region rather than by the run means a
    resumed escalation lands back on the same state the operator was shown,
    whichever process shows it.
    """
    return reference


def _config(reference: str) -> dict:
    return {"configurable": {"thread_id": thread_for(reference)}}


def start_review(graph, reference: str) -> dict[str, Any]:
    """Run one candidate through the flow.

    Returns the flow state. If the run suspended, the state carries
    ``__interrupt__`` and the candidate is now on the queue.

    A candidate already waiting on a person is returned as-is rather than
    re-run: it would cost a second LLM call and could hand back a different
    rationale than the one the operator is currently reading.
    """
    queued = escalations.get(thread_for(reference))
    if queued and queued["status"] == "pending":
        return {"already_pending": queued}

    # ``trace`` and ``timings_ms`` are reset explicitly. The thread may already
    # hold a finished run for this region, and these two channels append rather
    # than replace -- without this a re-review shows the previous run's path
    # concatenated onto its own.
    state = graph.invoke(
        {"candidate_ref": reference, "trace": [], "timings_ms": {}},
        config=_config(reference),
    )

    if "__interrupt__" in state:
        payload = state["__interrupt__"][0].value
        escalations.raise_escalation(
            reference,
            thread_for(reference),
            payload.get("reason", "model was not confident"),
            payload.get("agent_verdict"),
            # Carried onto the queue row so the station can say "no explanation
            # was written, and here is why" instead of showing an error string
            # where a rationale belongs -- and so the absences can be counted,
            # which WI-300's rationale-deadline clause requires and nothing did.
            payload.get("explanation_status"),
        )
    else:
        record_decision(
            reference,
            state["verdict"],
            state["decided_by"],
            rationale=state.get("agent_rationale") or None,
            explanation_status=state.get("explanation_status"),
            # From the run's own state, not looked up now: the digest that read
            # this region is the one that goes on the record beside its number.
            provenance=DecisionProvenance.from_dict(state.get("provenance")),
        )
    return state


def resume_review(graph, reference: str, verdict: str, reviewer: str) -> dict[str, Any]:
    """Hand an operator's verdict back to the suspended run.

    The decision is recorded before the queue entry is closed. If the process
    dies between the two, the candidate stays on the queue and gets looked at
    again -- the opposite order would drop the verdict silently.
    """
    if verdict not in VERDICT_OPTIONS:
        raise ValueError(f"{verdict!r} is not one of {VERDICT_OPTIONS}")

    queued = escalations.get(thread_for(reference))
    state = graph.invoke(
        Command(resume={"verdict": verdict, "reviewer": reviewer}),
        config=_config(reference),
    )
    record_decision(
        reference,
        verdict,
        "human",
        reviewer,
        queued["reason"] if queued else None,
        # The state came back through the checkpointer, so this is the model
        # the operator was actually shown -- which may not be the checkpoint on
        # disk today. A run raised before provenance existed carries none, and
        # the row says ``unavailable`` rather than refusing the answer.
        provenance=DecisionProvenance.from_dict(state.get("provenance")),
    )
    escalations.resolve_escalation(thread_for(reference))
    # Whoever answered the last outstanding region on this board is the
    # authority the board-level record names. That identity is a free-text
    # field until the station has authentication, and the record says so by
    # carrying it verbatim rather than by rounding it up to "a person".
    settle_board(reference, reviewer)
    return state


def settle_board(reference: str, decided_by: str | None = None) -> dict | None:
    """Write the board-level record, once the board has nothing outstanding.

    A line ships boards, and until 2026-08-23 nothing in this store said what
    happened to one. It is written when the board reaches a settled state --
    the end of a board run, or the moment an operator answers its last queued
    region -- and not per region, which would fill the table with rows saying
    the board is still held.

    A board with regions still waiting gets no row from here. Its state is
    "someone is still looking", which the queue already answers, and a
    disposition row per unanswered region would bury the one that matters.
    """
    stem = reference.partition("#")[0]
    assessment = dispositions.assess(stem)
    if assessment is None or assessment["pending_count"]:
        return None
    return dispositions.record(stem, decided_by or "automated")


def flow_state(graph, reference: str) -> dict[str, Any]:
    """What the checkpointer currently holds for a region.

    This is how the station shows an operator the evidence the agent gathered
    without re-running anything: the suspended state is already on disk.
    """
    snapshot = graph.get_state(_config(reference))
    return dict(snapshot.values) if snapshot and snapshot.values else {}
