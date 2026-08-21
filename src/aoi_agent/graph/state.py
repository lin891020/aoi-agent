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

    # Filled by the gather_context node
    board_context: dict[str, Any]
    machine_stats: dict[str, Any]
    standards: list[dict[str, Any]]

    # Filled by the reason node
    agent_rationale: str
    agent_verdict: str
    agent_confident: bool

    # Filled by escalate / finalize
    escalation_reason: str
    human_verdict: str
    human_reviewer: str

    disposition: Disposition
    verdict: str
    decided_by: Literal["model", "agent", "human"]
    trace: list[str]
    timings_ms: dict[str, float]
