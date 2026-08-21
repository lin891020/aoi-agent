"""Starting and resuming one re-verification, independent of who is watching.

The CLI and the web station differ only in how they ask a person for a verdict.
Everything either side of that -- running the flow, noticing it suspended,
queueing it, resuming it, recording the decision -- is the same, and lives here
so the two cannot drift apart.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import Command

from aoi_agent.store import escalations
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
        )
    else:
        record_decision(
            reference,
            state["verdict"],
            state["decided_by"],
            rationale=state.get("agent_rationale"),
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
    )
    escalations.resolve_escalation(thread_for(reference))
    return state


def flow_state(graph, reference: str) -> dict[str, Any]:
    """What the checkpointer currently holds for a region.

    This is how the station shows an operator the evidence the agent gathered
    without re-running anything: the suspended state is already on disk.
    """
    snapshot = graph.get_state(_config(reference))
    return dict(snapshot.values) if snapshot and snapshot.values else {}
