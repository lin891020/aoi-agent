"""Where a chart specification becomes pixels.

Two things happen here that nothing upstream can check. Bars are placed by
position, so a chart can be wrong by position without looking wrong; and labels
are resolved out of keys, so a chart stored in one language is redrawn in
whichever language is being read.

The renderer's older properties -- escaping, and returning '' rather than
raising on a specification read back out of a JSON column -- live in
`test_analysis_page.py` beside the route that calls it.
"""

from __future__ import annotations

import re

from aoi_agent.analysis.charts import chart_spec_for
from aoi_agent.station.chart_svg import render_svg

BY_LINE = {
    "L1": {"open": 175, "spur": 151},
    "L2": {"open": 302},
}


def history(line: str) -> dict:
    return {
        "tool": "query_defect_history", "args": {}, "ok": True, "elapsed_ms": 1.0,
        "error": None,
        "data": {
            "filters": {"lot_id": None, "line_id": line, "machine_id": None,
                        "defect_type": None, "days": 7},
            "by_class": BY_LINE[line],
        },
    }


def bar_centres(svg: str) -> list[float]:
    """The x of every plotted bar. Legend swatches are `<rect>` too and are not
    bars -- they carry no `<title>`, which is what this pattern keys on."""
    return [
        float(x)
        for x in re.findall(r'<rect x="([0-9.]+)"[^>]*><title>', svg)
    ]


def slot_labels(svg: str) -> list[str]:
    """The x-axis tick text, in the order it is drawn."""
    return re.findall(r'text-anchor="middle">([^<]+)</text>', svg)


def test_a_series_missing_a_class_leaves_the_slot_empty_rather_than_shifting():
    """The alignment rule, stated as the failure it prevents.

    L2 has no spurs. Placing its bars by their index within its own series --
    which is what the renderer did until 2026-08-23, because every chart until
    then had one series -- would put L2's `open` bar in slot 0 and nothing
    after it, which is right by accident here, and would put a third line's
    second class under the second class of the *first* line. A reader compares
    bar heights under a label. The label has to be the bar's own.
    """
    spec = chart_spec_for([history("L1"), history("L2")])
    svg = render_svg(spec)

    assert slot_labels(svg) == ["open", "spur"]

    # Three bars for two series: L1 has both classes, L2 only `open`.
    centres = sorted(bar_centres(svg))
    assert len(centres) == 3

    # The two `open` bars are neighbours within one slot; `spur` is a slot
    # away. Had L2's bar been placed by its index it would have landed beside
    # L1's `open` either way -- so the claim that separates the two placements
    # is the *gap*: closing it up would make these three evenly spaced.
    within_slot = centres[1] - centres[0]
    across_slots = centres[2] - centres[1]
    assert across_slots > within_slot * 1.5


def test_a_label_key_is_resolved_in_the_locale_being_read():
    spec = chart_spec_for([history("L1"), history("L2")])

    assert "產線 L1" in render_svg(spec, locale="zh-TW")
    assert "Line L1" in render_svg(spec, locale="en")
    assert "各類缺陷數量" in render_svg(spec, locale="zh-TW")
    assert "Defects by class" in render_svg(spec, locale="en")


def test_an_identifier_is_not_translated():
    """`mousebite` and `L1` are what the store calls them, and what the work
    instruction calls them. A chart axis in a third vocabulary is one more
    thing for a reader to reconcile."""
    spec = chart_spec_for([history("L1")])

    for locale in ("zh-TW", "en"):
        svg = render_svg(spec, locale=locale)
        assert "open" in svg and "spur" in svg


def test_a_chart_stored_before_the_keys_existed_is_drawn_as_it_was_stored():
    """Named here rather than left for a reader to find in a redrawn chart.

    A specification written before 2026-08-23 carries the rendered English
    sentence and no key. The language it was drawn in is not recoverable from
    it, so it is shown as it is -- in both locales, unchanged. Inventing a key
    for it would put a Chinese title on a chart whose axis text is English.
    """
    legacy = {
        "kind": "bar",
        "title": "defects by class",
        "y_label": "count",
        "series": [{"name": "count", "points": [{"x": "open", "y": 12.0}]}],
    }

    for locale in ("zh-TW", "en"):
        svg = render_svg(legacy, locale=locale)
        assert "defects by class" in svg
        assert "各類缺陷數量" not in svg


# ---- the ruler, the number, the interval ------------------------------------
#
# Mike's reading of the event-window chart on 2026-08-30: two blue blocks, no
# numbers, no axis -- and a qualifier under the title about overlapping
# intervals that the picture did not draw. Three things a bar chart owes the
# reader, each held here.


def event_window(side: str, value: float, low: float, high: float) -> dict:
    return {
        "tool": "query_defect_history", "args": {}, "ok": True, "elapsed_ms": 1.0,
        "error": None,
        "data": {
            "filters": {"machine_id": "M32", "relative_to": "parameter_change",
                        "side": side},
            "open_share": {"value": value, "interval_95": [low, high]},
            "by_class": {"open": 3},
        },
    }


def test_every_bar_carries_its_value_as_a_label():
    svg = render_svg(chart_spec_for([history("L1")]), locale="en")
    values = re.findall(r'class="value"[^>]*>([^<]+)</text>', svg)
    assert values == ["175", "151"]


def test_the_y_axis_is_a_ruler_with_round_ticks():
    svg = render_svg(chart_spec_for([history("L2")]), locale="en")
    ticks = re.findall(r'class="tick"[^>]*>([^<]+)</text>', svg)
    assert ticks == ["0", "100", "200", "300", "400"], ticks
    assert svg.count('class="grid"') == len(ticks)


def test_an_interval_is_drawn_as_a_whisker_and_the_ruler_reaches_it():
    spec = chart_spec_for([
        event_window("before", 0.2509, 0.21, 0.30),
        event_window("after", 0.1701, 0.13, 0.22),
    ])
    svg = render_svg(spec, locale="en")
    assert svg.count('class="whisker"') == 2
    # The value is on the bar in the form it was stored, not rounded away.
    assert re.findall(r'class="value"[^>]*>([^<]+)</text>', svg) == ["0.2509", "0.1701"]
    # The ruler tops out above the highest interval, not the highest bar.
    ticks = [float(t) for t in re.findall(r'class="tick"[^>]*>([^<]+)</text>', svg)]
    assert ticks[-1] >= 0.30


def test_a_point_without_an_interval_draws_no_whisker():
    svg = render_svg(chart_spec_for([history("L1")]), locale="en")
    assert "whisker" not in svg


def test_a_bar_is_a_mark_not_a_panel():
    """Two bars on a 720 px chart must not be 216 px wide each."""
    spec = chart_spec_for([
        event_window("before", 0.25, 0.2, 0.3), event_window("after", 0.17, 0.1, 0.2),
    ])
    widths = [float(w) for w in re.findall(r'<rect x="[0-9.]+" y="[0-9.]+" width="([0-9.]+)"[^>]*><title>', render_svg(spec))]
    assert widths and max(widths) <= 72
