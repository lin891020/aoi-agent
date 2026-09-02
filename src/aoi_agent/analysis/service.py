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

from aoi_agent.analysis.graph import synthesise_timed
from aoi_agent.i18n import DEFAULT_LOCALE
from aoi_agent.store import analysis as store


class NothingToWrite(ValueError):
    """The run has no results to write an answer from.

    A refusal's answer is the reason it was refused, and that is not derived
    from any result -- writing it again would be a translation, which is the
    thing the language rule says this path never produces.
    """


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


def persist_run(
    state: dict, question: str, asked_by: str | None, asked_lang: str | None = None
) -> int:
    """Write one finished run to the store, and return its id."""
    return store.save_run(
        question=question,
        plan=state.get("plan"),
        results=in_plan_order(state.get("results") or []),
        chart=state.get("chart_spec"),
        answer=state.get("answer", ""),
        timings=state.get("timings_ms") or {},
        # A question about what can be asked ran no lookup either, but it was
        # answered; the stored flag means "the planner declined", and on the
        # recent list it renders as a chip saying so.
        refused=bool(state.get("refused")) and not state.get("capability_question"),
        asked_by=asked_by,
        # The language the plan was written in, which is what makes the page
        # able to say that section 1 is a record rather than a rendering.
        asked_lang=asked_lang,
    )


def answer_question(
    graph,
    question: str,
    asked_by: str | None = "operator",
    asked_lang: str | None = None,
) -> dict[str, Any]:
    """Run one question through the analysis graph and persist the result."""
    state = graph.invoke({
        "question": question, "results": [], "timings_ms": {},
        "lang": asked_lang or DEFAULT_LOCALE,
    })
    return store.get_run(persist_run(state, question, asked_by, asked_lang))


def answer_again(client, run: dict, lang: str) -> dict[str, Any]:
    """Write a stored run's answer in another language, from the same results.

    The one re-derivable thing on the page. The plan is not re-run and the
    prose is not translated: the stored `results_json` goes down the same
    `synthesise_timed` the first answer came from, with the language changed,
    and what comes back is kept beside the original under its own key. A
    language already held is returned as it is -- `add_answer` only adds, so
    a second press costs nothing and rewrites nothing.

    Until 2026-09-02 nothing called this path. The column, the store function
    and the docs describing it all existed, and the page rendered the original
    answer under whichever heading the switch had set.
    """
    if lang in run["answers"]:
        return run
    if run["refused"] or not run["results"]:
        raise NothingToWrite(f"run {run['id']} has no results to write an answer from")
    text, _timing = synthesise_timed(
        client, run["question"], run["plan"] or {}, run["results"], lang
    )
    return store.add_answer(run["id"], lang, text)
