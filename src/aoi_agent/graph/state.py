"""State carried through the re-verification flow."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

Disposition = Literal["dismissed", "defect_confirmed", "escalated", "pending"]


class ReviewState(TypedDict, total=False):
    """Everything one candidate accumulates on its way to a verdict."""

    candidate_ref: str

    # Filled by the classify node
    model_class: str
    model_confidence: float
    false_call_probability: float
    model_recommendation: str

    provenance: dict[str, Any]
    """What produced the reading above: checkpoint digest, thresholds, code.

    A plain dict rather than the dataclass, because it is checkpointed and the
    checkpointer serialises what it is given. It is written once, by
    ``classify_node``, and read at the two points a row is committed -- the
    automated write in ``station.service.start_review`` and the operator's own,
    days later, in ``resume_review``. Carrying it in the state is what makes
    the second of those attribute the answer to the model that was actually
    read, rather than to whatever checkpoint is on disk when the operator gets
    to the queue.
    """

    # Filled by the gather_context node
    board_context: dict[str, Any]
    machine_stats: dict[str, Any]
    standards: list[dict[str, Any]]

    # Filled by the reason node
    agent_rationale: str
    agent_verdict: str
    agent_confident: bool

    explanation_status: str
    """``ok``, or why no explanation was written.

    A first-class state rather than an error string in ``agent_rationale``.
    The station's queue held an escalation whose entire evidence panel read
    ``the model did not answer (ReadTimeout)``: it told the operator nothing,
    it read as though it were a rationale, and nothing counted how many others
    there were. ``agent_rationale`` is now empty when no explanation exists --
    WI-300 forbids filling the gap with anything -- and this field carries the
    reason, in a column a report can group by. See ``graph.flow.NO_EXPLANATION``.
    """

    # Filled by escalate / finalize
    escalation_reason: str
    human_verdict: str
    human_reviewer: str

    disposition: Disposition
    verdict: str
    decided_by: Literal["model", "agent", "human"]
    trace: list[str]
    timings_ms: dict[str, float]
