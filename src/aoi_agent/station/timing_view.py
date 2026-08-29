"""The run's time, stage by stage, for the table under the answer.

The page already printed the per-tool milliseconds and, in one dim line, the
planning and writing times. What a reader watching the progress panel wanted
to know afterwards was different: how long the *model* thought, as against how
long they waited. The two are not the same number on this machine -- a
translation job holding the GPU makes a 13-second inference a 40-second wait --
and a table that showed only the wait would have blamed the model for the
queue.

So every stage has two columns. ``wall`` is what the page waited, measured
here. ``model`` is what Ollama reported as ``eval_duration`` for that call,
which is the figure `measuring-llm-latency` allows to be quoted as the model's
time. Tools and the chart have no model column: nothing infers there. A run
stored before the model figures were recorded shows ``None`` in that column,
and the template renders it as "unrecorded" rather than as zero.
"""

from __future__ import annotations

from typing import TypedDict


class Row(TypedDict):
    key: str
    wall_s: float | None
    model_s: float | None
    #: True when the stage exists on this run but the model column was never
    #: recorded -- as against a stage that has no model column at all.
    model_unrecorded: bool


#: Stage, the `timings_ms` key its wall time is under, and the key its
#: model time is under (None for stages where nothing infers).
STAGES = (
    ("analysis.timing.plan", "plan", "plan_eval"),
    ("analysis.timing.tools", "tools_wall", None),
    ("analysis.timing.chart", "chart", None),
    ("analysis.timing.synthesise", "synthesise", "synthesise_eval"),
)


def _seconds(ms: object) -> float | None:
    if ms is None:
        return None
    try:
        return round(float(ms) / 1000, 2)
    except (TypeError, ValueError):
        return None


def rows(timings: dict | None) -> list[Row]:
    """One row per stage that ran, then the total of the wall column.

    A stage absent from ``timings`` did not run -- a refusal has a plan and
    nothing else -- and gets no row rather than a row of zeros.
    """
    timings = timings or {}
    out: list[Row] = []
    total = 0.0
    for key, wall_key, model_key in STAGES:
        if wall_key not in timings:
            continue
        wall = _seconds(timings.get(wall_key))
        model = _seconds(timings.get(model_key)) if model_key else None
        out.append({
            "key": key,
            "wall_s": wall,
            "model_s": model,
            "model_unrecorded": bool(model_key) and model is None,
        })
        total += wall or 0.0
    if out:
        out.append({"key": "analysis.timing.total", "wall_s": round(total, 2),
                    "model_s": None, "model_unrecorded": False})
    return out
