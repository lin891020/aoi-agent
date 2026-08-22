"""Choosing a chart from the shape of the data, not from the model's opinion.

`chart_spec` is data rather than an image: the page renders it, which is what
lets a stored answer be redrawn months later without re-running a model that
would not produce the same plan twice.
"""

from __future__ import annotations

from aoi_agent.analysis.charts import chart_spec_for


def ok(tool: str, data: dict) -> dict:
    return {"tool": tool, "args": {}, "ok": True, "data": data,
            "error": None, "elapsed_ms": 1.0}


MACHINE_STATS = {
    "defect_type": "open",
    "fleet_share_of_defects": 0.225,
    "machines": [
        {"machine": "L2-M22", "share_of_defects": 0.321, "per_board": 2.3},
        {"machine": "L1-M11", "share_of_defects": 0.190, "per_board": 1.4},
    ],
}


def test_a_comparison_across_entities_becomes_bars():
    spec = chart_spec_for([ok("query_machine_stats", MACHINE_STATS)])

    assert spec["kind"] == "bar"
    assert spec["series"][0]["points"][0]["x"] == "L2-M22"
    assert spec["series"][0]["points"][0]["y"] == 0.321


def test_the_fleet_average_is_carried_as_its_own_series():
    """A bar chart of machine shares with no baseline invites the reader to
    compare machines against each other and miss that all of them are high."""
    spec = chart_spec_for([ok("query_machine_stats", MACHINE_STATS)])
    names = [s["name"] for s in spec["series"]]

    assert any("fleet" in n.lower() for n in names)


def test_a_defect_breakdown_becomes_bars():
    spec = chart_spec_for(
        [ok("query_defect_history", {"counts": {"open": 12, "short": 3}})]
    )

    assert spec["kind"] == "bar"
    assert {p["x"] for p in spec["series"][0]["points"]} == {"open", "short"}


def test_results_with_nothing_plottable_produce_no_chart():
    """Retrieved criteria are prose. Forcing a chart onto them would be
    decoration, and a chart that means nothing is worse than none."""
    spec = chart_spec_for(
        [ok("search_standards", {"passages": [{"document": "WI-201", "text": "..."}]})]
    )

    assert spec is None


def test_failed_results_are_skipped_rather_than_plotted_as_zero():
    """A missing bar and a zero bar read very differently, and only one of them
    is true."""
    failed = {"tool": "query_machine_stats", "args": {}, "ok": False,
              "data": None, "error": "boom", "elapsed_ms": 1.0}
    spec = chart_spec_for([failed])

    assert spec is None


def test_the_first_plottable_result_wins_when_several_are_present():
    """One question, one chart. Stacking several unrelated charts under one
    answer makes the reader do the joining."""
    spec = chart_spec_for(
        [
            ok("search_standards", {"passages": []}),
            ok("query_machine_stats", MACHINE_STATS),
            ok("query_defect_history", {"counts": {"open": 1}}),
        ]
    )

    assert spec["kind"] == "bar"
    assert spec["series"][0]["points"][0]["x"] == "L2-M22"
