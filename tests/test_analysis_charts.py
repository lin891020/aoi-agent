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
    "days": 14,
    "fleet_average_per_board": 1.85,
    "fleet_share_of_defects": 0.225,
    "machines": [
        {"machine": "L2-M22", "boards": 40, "defects": 29, "per_board": 2.3,
         "share_of_defects": 0.321},
        {"machine": "L1-M11", "boards": 35, "defects": 18, "per_board": 1.4,
         "share_of_defects": 0.190},
    ],
}


DEFECT_HISTORY = {
    "filters": {"lot_id": None, "line_id": None, "machine_id": None,
                "defect_type": None, "days": 7},
    "window_end": "2026-08-22T00:00:00",
    "boards_inspected": 120,
    "defects_total": 15,
    "defects_per_board": 0.13,
    "by_class": {"open": 12, "short": 3},
}


#: The counts a real run returned for "比較三條線的缺陷組成" on 2026-08-23, when
#: the page drew one chart titled `defects by class` that was L1 alone and said
#: so nowhere. L2's 302 opens -- the largest figure in the whole answer, and in
#: the one class where any confirmed instance is critical -- was not on it.
BY_LINE = {
    "L1": {"copper": 138, "mousebite": 188, "open": 175,
           "pin-hole": 152, "short": 162, "spur": 151},
    "L2": {"copper": 128, "mousebite": 181, "open": 302,
           "pin-hole": 146, "short": 163, "spur": 129},
    "L3": {"copper": 133, "mousebite": 168, "open": 202,
           "pin-hole": 150, "short": 166, "spur": 158},
}


def history_for(line: str, counts: dict | None = None) -> dict:
    """One `query_defect_history` payload, scoped to a line the way the flow
    scopes it: the filter it was called with travels back in the result."""
    return {
        **DEFECT_HISTORY,
        "filters": {**DEFECT_HISTORY["filters"], "line_id": line},
        "by_class": BY_LINE[line] if counts is None else counts,
    }


def series_named(spec: dict, **name_args) -> dict:
    """The one series whose label arguments match, as ``{class: count}``.

    Matching on the arguments rather than on a rendered string, because the
    rendered string is a translation and this assertion is about the data.
    """
    matching = [
        one for one in spec["series"]
        if all((one.get("name_args") or {}).get(k) == v for k, v in name_args.items())
    ]
    assert len(matching) == 1, (
        f"expected exactly one series for {name_args}, got "
        f"{[one.get('name_args') for one in spec['series']]}"
    )
    return {p["x"]: p["y"] for p in matching[0]["points"]}


def test_a_comparison_across_entities_becomes_bars():
    spec = chart_spec_for([ok("query_machine_stats", MACHINE_STATS)])

    assert spec["kind"] == "bar"
    assert spec["series"][0]["points"][0]["x"] == "L2-M22"
    assert spec["series"][0]["points"][0]["y"] == 0.321


def test_the_fleet_average_is_carried_as_its_own_series():
    """A bar chart of machine shares with no baseline invites the reader to
    compare machines against each other and miss that all of them are high."""
    spec = chart_spec_for([ok("query_machine_stats", MACHINE_STATS)])
    keys = [s["name_key"] for s in spec["series"]]

    assert any("fleet" in key.lower() for key in keys)


def test_several_classes_carry_no_fleet_baseline():
    """The baseline exists because one series has nothing to be read against.
    Several classes are each other's baseline, and a fleet line per class is
    another bar on every machine saying what the others already said."""
    spec = chart_spec_for(
        [
            ok("query_machine_stats", MACHINE_STATS),
            ok("query_machine_stats", {**MACHINE_STATS, "defect_type": "short"}),
        ]
    )
    keys = [s["name_key"] for s in spec["series"]]

    assert len(spec["series"]) == 2
    assert not any("fleet" in key.lower() for key in keys)


def test_a_defect_breakdown_becomes_bars():
    spec = chart_spec_for([ok("query_defect_history", DEFECT_HISTORY)])

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


def test_the_first_plottable_tool_wins_when_several_are_present():
    """One question, one chart. Stacking several unrelated charts under one
    answer makes the reader do the joining.

    "First plottable *tool*", not first result: the branches of one fan-out are
    the same tool asked about different entities, and those belong on one
    chart together. A different tool is a different question.
    """
    spec = chart_spec_for(
        [
            ok("search_standards", {"passages": []}),
            ok("query_machine_stats", MACHINE_STATS),
            ok("query_defect_history", {**DEFECT_HISTORY, "by_class": {"open": 1}}),
        ]
    )

    assert spec["kind"] == "bar"
    assert spec["series"][0]["points"][0]["x"] == "L2-M22"


def test_a_fan_out_across_lines_becomes_one_series_per_line():
    """The chart carries the dimension the plan fanned out over.

    Until 2026-08-23 it did not. `chart_spec_for` returned on the first
    plottable result, so a question that fanned out to three lines was answered
    with a chart of one of them, titled as though it were all three.
    """
    spec = chart_spec_for(
        [ok("query_defect_history", history_for(line)) for line in ("L1", "L2", "L3")]
    )

    assert len(spec["series"]) == 3
    assert series_named(spec, line_id="L1")["mousebite"] == 188
    assert series_named(spec, line_id="L3")["short"] == 166


def test_the_largest_figure_in_the_answer_reaches_the_chart():
    """The bar the old chart dropped, named as the reason this test exists.

    L2's 302 opens is the largest count in the answer and sits in the one class
    where any confirmed instance is critical. A chart built from L1 alone put
    `mousebite` at the top and never showed it.
    """
    spec = chart_spec_for(
        [ok("query_defect_history", history_for(line)) for line in ("L1", "L2", "L3")]
    )

    every_y = [point["y"] for one in spec["series"] for point in one["points"]]

    assert series_named(spec, line_id="L2")["open"] == 302
    assert max(every_y) == 302


def test_a_class_missing_from_one_line_is_absent_rather_than_zero():
    """The rule already applied to failed results, one level down.

    A line with no spurs and a line with an unmeasured spur count read very
    differently, and drawing both as a bar of height zero says the second.
    """
    spec = chart_spec_for(
        [
            ok("query_defect_history", history_for("L1", {"open": 5, "spur": 2})),
            ok("query_defect_history", history_for("L2", {"open": 7})),
        ]
    )

    assert series_named(spec, line_id="L1") == {"open": 5, "spur": 2}
    assert series_named(spec, line_id="L2") == {"open": 7}


def test_labels_are_stored_as_keys_so_a_stored_chart_can_be_redrawn_translated():
    """The specification outlives the language it was drawn in.

    A title baked as `defects by class` is an English sentence in a JSON column:
    the run is redrawn from it next quarter, and it is still English however the
    station is set. Identifiers are not text and stay as they are -- `mousebite`
    and `L1` are what the store calls them.
    """
    spec = chart_spec_for(
        [ok("query_defect_history", history_for(line)) for line in ("L1", "L2")]
    )

    assert spec["title_key"] and "title" not in spec
    assert spec["y_label_key"] and "y_label" not in spec
    assert spec["series"][0]["name_key"] and "name" not in spec["series"][0]
    assert spec["series"][0]["name_args"] == {"line_id": "L1"}
    assert {p["x"] for p in spec["series"][0]["points"]} == set(BY_LINE["L1"])



# ---- two windows around a machine event ---------------------------------


def _window(side, value, low, high, machine="M32"):
    return {
        "filters": {"machine_id": machine, "relative_to": "parameter_change", "side": side},
        "open_share": {"value": value, "interval_95": [low, high]},
        "by_class": {"open": 3},
    }


def test_two_event_windows_become_a_before_after_pair_in_that_order():
    """The branches may return after-then-before; the chart never does."""
    from aoi_agent.analysis.charts import chart_spec_for

    spec = chart_spec_for([
        {"tool": "query_defect_history", "ok": True, "data": _window("after", 0.25, 0.1, 0.45)},
        {"tool": "query_defect_history", "ok": True, "data": _window("before", 0.75, 0.55, 0.9)},
    ])
    assert spec["title_key"] == "chart.title.open_share_around_event"
    (series,) = spec["series"]
    assert [p["x"] for p in series["points"]] == ["before", "after"]
    assert [p["y"] for p in series["points"]] == [0.75, 0.25]
    assert series["points"][0]["y_low"] == 0.55 and series["points"][0]["y_high"] == 0.9
    assert series["name_args"] == {"machine_id": "M32", "kind": "parameter_change"}


def test_an_unanchored_history_lookup_is_still_a_class_breakdown():
    from aoi_agent.analysis.charts import chart_spec_for

    spec = chart_spec_for([{
        "tool": "query_defect_history", "ok": True,
        "data": {"filters": {"machine_id": "M32"}, "by_class": {"open": 3, "short": 1}},
    }])
    assert spec["title_key"] == "chart.title.defects_by_class"


def test_a_window_with_no_flagged_regions_is_left_out_not_drawn_at_zero():
    from aoi_agent.analysis.charts import chart_spec_for

    spec = chart_spec_for([
        {"tool": "query_defect_history", "ok": True, "data": _window("before", 0.5, 0.3, 0.7)},
        {"tool": "query_defect_history", "ok": True, "data": _window("after", None, 0.0, 1.0)},
    ])
    (series,) = spec["series"]
    assert [p["x"] for p in series["points"]] == ["before"]
