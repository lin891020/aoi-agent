"""Answering one question, and keeping it.

Both entrances to the analysis flow come through `persist_run`: `POST /ask`,
which invokes the graph and redirects, and `GET /ask/stream`, which streams the
same run's progress and then saves it. That is not a stylistic preference. The
two paths execute the identical graph but observe it differently -- `invoke`
returns the accumulated state, `stream_mode="updates"` hands back one update per
branch in completion order -- and the stream used to reimplement the eight
argument `save_run` inline. It drifted exactly where a duplicated write always
does: the streamed page paired result *i* with `plan.calls[i].why`, so a fast
tool's row carried a slow tool's justification. One function to persist a run
means the normalising that fixes it cannot be applied to one path and forgotten
on the other.

There is no CLI entry point today. This module is shared between two HTTP
routes, not between a CLI and the station -- do not restate the disposition
path's claim here, since it is not true of this one.
"""

from __future__ import annotations

from typing import Any

from aoi_agent.store import analysis as store


def in_plan_order(results: list[dict]) -> list[dict]:
    """The branches' results, back in the order the plan asked for them.

    A fan-out returns in completion order, and `stream_mode="updates"` reports
    it that way, so three tools at 300/150/10ms accumulate fastest-first. Every
    result carries the `position` its `Send` was given, so the plan's order is
    recoverable rather than guessed at. Stable, and tolerant of a stored run
    written before `position` existed: those all sort as 0 and keep the order
    they were saved in.
    """
    return sorted(results, key=lambda r: r.get("position", 0))


def persist_run(state: dict, question: str, asked_by: str | None) -> int:
    """Write one finished run to the store, and return its id."""
    return store.save_run(
        question=question,
        plan=state.get("plan"),
        results=in_plan_order(state.get("results") or []),
        chart=state.get("chart_spec"),
        answer=state.get("answer", ""),
        timings=state.get("timings_ms") or {},
        refused=bool(state.get("refused")),
        asked_by=asked_by,
    )


def answer_question(graph, question: str, asked_by: str | None = "operator") -> dict[str, Any]:
    """Run one question through the analysis graph and persist the result."""
    state = graph.invoke({"question": question, "results": [], "timings_ms": {}})
    return store.get_run(persist_run(state, question, asked_by))
