"""Turning results into a chart specification.

The model does not pick the chart. It picks nothing in this system: measured on
the disposition path, its judgement lost to the classifier's, and there is no
reason to expect it to do better at choosing an axis. The shape of the data
decides.

What comes out is a specification, not an image -- axis labels, series, points.
The page renders it. That is what lets a run stored today be redrawn next
quarter without re-running a plan that would not regenerate identically.

Two properties of that specification are load-bearing, and both were absent
until 2026-08-23.

**It carries the dimension the plan fanned out over.** A builder is handed every
result from its tool, not one of them. A question that fanned out to three lines
had been answered with a chart of whichever line finished first, titled as
though it were all three -- the largest count in the answer, 302 opens on L2 in
the one class where any confirmed instance is critical, was not on it. The bug
was a `return` inside a loop; what it cost was a reader believing a chart.

**Its labels are keys, not sentences.** The specification is stored and redrawn
long after the run, and a title baked as `defects by class` is an English
sentence sitting in a JSON column: it stays English however the station is set.
Identifiers -- `mousebite`, `L1`, `L2-M22` -- are not text and are carried
through untouched, because they are what the store calls them.
"""

from __future__ import annotations

from typing import Any

from aoi_agent.analysis.tools import ToolResult

#: How a `query_defect_history` result names the entity it was scoped to, most
#: specific first. A plan fanning out over lines and a plan fanning out over
#: machines produce the same shape, and the series must say which it was.
SCOPE_FIELDS = ("machine_id", "line_id", "lot_id")


def _scope_label(filters: dict | None) -> tuple[str, dict]:
    """The label key and arguments for one branch of a fan-out.

    Unscoped is a scope: a single unfiltered lookup is every line together, and
    a series silently named after nothing reads as though it were one line.
    """
    for field in SCOPE_FIELDS:
        value = (filters or {}).get(field)
        if value:
            return f"chart.series.{field}", {field: value}
    return "chart.series.everything", {}


def _defect_breakdown(payloads: list[dict]) -> dict | None:
    """One series per lookup, one bar per class.

    Classes are ordered by their total across every series rather than by any
    one of them, so the ordering does not depend on which branch returned
    first. A class absent from one line is absent from that series rather than
    plotted at zero -- the same rule `chart_spec_for` applies to a failed
    result, one level down. "No spurs on L2" and "L2's spurs were not counted"
    are different statements and a zero bar makes the second one.
    """
    # Two lookups anchored on a machine event are a different question from
    # "which classes": the reader wants one number on each side of a date.
    # The shape of the payload says which question was asked, not the tool.
    if payloads and all(p.get("open_share") and p.get("filters", {}).get("side") for p in payloads):
        return _event_windows(payloads)

    counted = [(p, p.get("by_class") or {}) for p in payloads]
    counted = [(p, counts) for p, counts in counted if counts]
    if not counted:
        return None

    totals: dict[str, float] = {}
    for _, counts in counted:
        for name, value in counts.items():
            totals[name] = totals.get(name, 0) + value
    order = sorted(totals, key=lambda name: -totals[name])

    series: list[dict[str, Any]] = []
    for payload, counts in counted:
        key, args = _scope_label(payload.get("filters"))
        series.append(
            {
                "name_key": key,
                "name_args": args,
                "points": [
                    {"x": name, "y": float(counts[name])}
                    for name in order
                    if name in counts
                ],
            }
        )

    return {
        "kind": "bar",
        "title_key": "chart.title.defects_by_class",
        "x_label_key": "chart.axis.defect_class",
        "y_label_key": "chart.axis.count",
        "series": series,
    }


def _machine_comparison(payloads: list[dict]) -> dict | None:
    """Every machine, one series per lookup, on the axis the tool ranked by.

    The payload says which: ``share_of_defects`` when one class was asked
    about, ``per_board`` when every class was. A chart that read
    ``share_of_defects`` off a ranking by ``per_board`` would plot a column of
    Nones as zero, which is a picture of nothing labelled as a finding.
    """
    usable = [p for p in payloads if p.get("machines")]
    if not usable:
        return None

    def axis(payload: dict) -> str:
        return payload.get("ranked_by") or "share_of_defects"

    series: list[dict[str, Any]] = []
    for payload in usable:
        if axis(payload) == "per_board":
            series.append({
                "name_key": "chart.series.defects_per_board",
                "name_args": {},
                "points": [
                    {"x": m["machine"], "y": round(float(m["per_board"]), 4)}
                    for m in payload["machines"]
                ],
            })
        else:
            series.append({
                "name_key": "chart.series.share_of",
                "name_args": {"defect_type": payload.get("defect_type", "")},
                "points": [
                    {"x": m["machine"], "y": round(m["share_of_defects"], 4)}
                    for m in payload["machines"]
                ],
            })

    first = usable[0]
    per_board = axis(first) == "per_board"
    fleet = first.get("fleet_average_per_board" if per_board else "fleet_share_of_defects")
    if len(usable) == 1 and fleet is not None:
        # Carried as a series rather than folded into the bars: without it the
        # reader compares machines against each other and cannot see that every
        # one of them sits above the fleet.
        #
        # Only when one lookup was made. Several classes already give the
        # reader a baseline -- each other -- and a fleet line per class is four
        # more bars per machine saying what the first four already said.
        series.append(
            {
                "name_key": "chart.series.fleet_average",
                "name_args": {},
                "points": [
                    {"x": m["machine"], "y": round(float(fleet), 4)}
                    for m in first["machines"]
                ],
            }
        )

    return {
        "kind": "bar",
        "title_key": "chart.title.defects_per_board_by_machine" if per_board
                     else "chart.title.share_by_machine",
        "x_label_key": "chart.axis.machine",
        "y_label_key": "chart.axis.defects_per_board" if per_board
                       else "chart.axis.share_of_own_defects",
        "series": series,
    }


#: Tried in order. The first *tool* that yields a spec is the chart, because one
#: question gets one chart -- stacking several unrelated ones under a single
#: answer leaves the reader to do the joining. The branches of one fan-out are
#: not unrelated: they are the same lookup asked about different entities, and
#: they belong on one chart together.
def _false_call_rates(payloads: list[dict]) -> dict | None:
    """Each group's false-call rate, one series per grouping asked about.

    The rate, not the counts: the tool sorts by it, the question that reaches
    this tool is about it, and plotting `flagged` beside `dismissed` would make
    the reader derive the one number the payload already carries. What the bar
    is is the payload's own caveat -- the re-verifier's judgement, not ground
    truth -- and the synthesis prose carries that sentence; the chart only has
    to not claim more, which a rate labelled as dismissal does not.
    """
    usable = [p for p in payloads if p.get("by_group")]
    if not usable:
        return None

    series: list[dict[str, Any]] = [
        {
            "name_key": "chart.series.dismissal_rate_by",
            "name_args": {"group_by": payload.get("filters", {}).get("group_by", "")},
            "points": [
                {"x": row["group"], "y": round(row["false_call_rate"], 4)}
                for row in payload["by_group"]
            ],
        }
        for payload in usable
    ]

    return {
        "kind": "bar",
        "title_key": "chart.title.false_call_rate",
        "x_label_key": "chart.axis.group",
        "y_label_key": "chart.axis.dismissal_rate",
        "series": series,
    }


def _event_windows(payloads: list[dict]) -> dict | None:
    """A machine's open share before and after an event, as two bars.

    One series per machine, one point per side, always in the order before
    then after -- the order the branches returned in carries no meaning and
    a chart that read "after, before" would be read as a reversal. The
    interval is carried on the point as ``y_low``/``y_high`` and the renderer
    may draw it or not; the value is never drawn without it being available,
    because two bars with overlapping intervals are the whole finding and a
    chart of two bare heights would hide it.
    """
    order = {"before": 0, "after": 1}
    by_machine: dict[str, list[dict]] = {}
    for payload in payloads:
        filters = payload.get("filters", {})
        by_machine.setdefault(filters.get("machine_id") or "", []).append(payload)

    series: list[dict[str, Any]] = []
    for machine, group in by_machine.items():
        group = sorted(group, key=lambda p: order.get(p["filters"]["side"], 9))
        points = []
        for payload in group:
            share = payload["open_share"]
            if share.get("value") is None:
                continue
            low, high = share.get("interval_95", (None, None))
            points.append({
                "x": payload["filters"]["side"],
                "y": round(float(share["value"]), 4),
                "y_low": low, "y_high": high,
            })
        if points:
            series.append({
                "name_key": "chart.series.machine_around_event",
                "name_args": {"machine_id": machine,
                              "kind": group[0]["filters"].get("relative_to", "")},
                "points": points,
            })
    if not series:
        return None
    return {
        "kind": "bar",
        "title_key": "chart.title.open_share_around_event",
        "x_label_key": "chart.axis.side_of_event",
        "y_label_key": "chart.axis.open_share",
        "series": series,
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _sql_rows(payloads: list[dict]) -> dict | None:
    """A SELECT's rows as bars, when the result has that shape.

    The shape: a first column of labels and at least one numeric column, in
    at most forty rows -- an aggregate, which is what the tool is for. A
    listing of two hundred candidate rows is not a chart and gets none. One
    series per numeric column (the first three), named by the column, because
    the column name is the only thing the payload says about the number.
    """
    usable = [p for p in payloads if p.get("columns") and p.get("rows")]
    if not usable:
        return None
    payload = usable[0]
    columns, rows = payload["columns"], payload["rows"]
    if len(columns) < 2 or len(rows) > 40:
        return None
    numeric = [
        index for index in range(1, len(columns))
        if all(_is_number(row[index]) or row[index] is None for row in rows)
        and any(_is_number(row[index]) for row in rows)
    ][:3]
    if not numeric:
        return None
    series = [
        {
            "name_key": "chart.series.sql_column",
            "name_args": {"column": columns[index]},
            "points": [
                {"x": str(row[0]), "y": round(float(row[index]), 4)}
                for row in rows if _is_number(row[index])
            ],
        }
        for index in numeric
    ]
    return {
        "kind": "bar",
        "title_key": "chart.title.sql_rows",
        "x_label_key": "chart.axis.sql_column",
        "x_label_args": {"column": columns[0]},
        "y_label_key": "chart.axis.sql_value",
        "series": series,
    }


BUILDERS = {
    "query_machine_stats": _machine_comparison,
    "run_sql": _sql_rows,
    "query_defect_history": _defect_breakdown,
    "query_false_call_rate": _false_call_rates,
}


def chart_spec_for(results: list[ToolResult]) -> dict | None:
    """The chart for these results, or None when nothing is plottable.

    Failed results are skipped rather than plotted as zero: a missing bar and a
    zero bar read very differently and only one of them is true.
    """
    usable = [r for r in results if r.get("ok") and r.get("data")]
    for result in usable:
        builder = BUILDERS.get(result["tool"])
        if builder is None:
            continue
        spec = builder([r["data"] for r in usable if r["tool"] == result["tool"]])
        if spec is not None:
            return spec
    return None
