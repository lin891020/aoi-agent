"""Answering one question, and keeping it.

The CLI and the station both come through here, so the two cannot drift -- the
same arrangement as `station/service.py` for the disposition path.
"""

from __future__ import annotations

from typing import Any

from aoi_agent.store import analysis as store


def answer_question(graph, question: str, asked_by: str | None = "operator") -> dict[str, Any]:
    """Run one question through the analysis graph and persist the result."""
    state = graph.invoke({"question": question, "results": [], "timings_ms": {}})

    run_id = store.save_run(
        question=question,
        plan=state.get("plan"),
        results=state.get("results") or [],
        chart=state.get("chart_spec"),
        answer=state.get("answer", ""),
        timings=state.get("timings_ms") or {},
        refused=bool(state.get("refused")),
        asked_by=asked_by,
    )
    return {**store.get_run(run_id), "plan_errors": state.get("plan_errors") or []}
