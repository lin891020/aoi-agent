"""Turning results into a chart specification.

The model does not pick the chart. It picks nothing in this system: measured on
the disposition path, its judgement lost to the classifier's, and there is no
reason to expect it to do better at choosing an axis. The shape of the data
decides.

What comes out is a specification, not an image -- axis labels, series, points.
The page renders it. That is what lets a run stored today be redrawn next
quarter without re-running a plan that would not regenerate identically.
"""

from __future__ import annotations

from typing import Any

from aoi_agent.analysis.tools import ToolResult


def _machine_comparison(data: dict) -> dict | None:
    machines = data.get("machines")
    if not machines:
        return None

    defect = data.get("defect_type", "defects")
    series: list[dict[str, Any]] = [
        {
            "name": f"share of {defect}",
            "points": [
                {"x": m["machine"], "y": round(m["share_of_defects"], 4)}
                for m in machines
            ],
        }
    ]

    fleet = data.get("fleet_share_of_defects")
    if fleet is not None:
        # Carried as a series rather than folded into the bars: without it the
        # reader compares machines against each other and cannot see that every
        # one of them sits above the fleet.
        series.append(
            {
                "name": "fleet average",
                "points": [
                    {"x": m["machine"], "y": round(fleet, 4)} for m in machines
                ],
            }
        )

    return {
        "kind": "bar",
        "title": f"{defect} share by machine",
        "x_label": "machine",
        "y_label": "share of that machine's defects",
        "series": series,
    }


def _defect_breakdown(data: dict) -> dict | None:
    counts = data.get("by_class")
    if not counts:
        return None
    return {
        "kind": "bar",
        "title": "defects by class",
        "x_label": "class",
        "y_label": "count",
        "series": [
            {
                "name": "count",
                "points": [
                    {"x": name, "y": float(value)}
                    for name, value in sorted(counts.items(), key=lambda kv: -kv[1])
                ],
            }
        ],
    }


#: Tried in order. The first result that yields a spec is the chart, because
#: one question gets one chart -- stacking several unrelated ones under a single
#: answer leaves the reader to do the joining.
BUILDERS = {
    "query_machine_stats": _machine_comparison,
    "query_defect_history": _defect_breakdown,
}


def chart_spec_for(results: list[ToolResult]) -> dict | None:
    """The chart for these results, or None when nothing is plottable.

    Failed results are skipped rather than plotted as zero: a missing bar and a
    zero bar read very differently and only one of them is true.
    """
    for result in results:
        if not result.get("ok") or not result.get("data"):
            continue
        builder = BUILDERS.get(result["tool"])
        if builder is None:
            continue
        spec = builder(result["data"])
        if spec is not None:
            return spec
    return None
