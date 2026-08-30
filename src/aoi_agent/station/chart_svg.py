"""A bar chart, as inline SVG.

No charting library. The station works with JavaScript off and vendors its one
dependency so it runs on a locked-down shop-floor browser; a chart that needs a
CDN would break both properties for a picture of six bars.

Everything that reaches the markup goes through ``escape``. Series names, point
labels and the title are tool data -- a machine id with an ``&`` in it would
otherwise close nothing and corrupt the document, and the same hole is what
model-authored text would walk through if a title ever came from the model.
Numbers are escaped too rather than trusted to be numbers: the specification is
read back out of a JSON column, and what is in that column next quarter is not
this function's decision to make.
"""

from __future__ import annotations

from html import escape

from aoi_agent.i18n import label_from

PALETTE = ["#60a5fa", "#f59e0b", "#34d399", "#f472b6"]


def _plottable(series: object) -> list[dict]:
    """The series that can actually be drawn, in drawing order.

    A specification is read back out of a JSON column, so "malformed" here is
    not a hypothetical: a series with no points, a point missing its ``y``, a
    ``y`` that is a string. The docstring below promises '' for anything
    unplottable, and a `KeyError` reaching the route would be a 500 on a page
    whose figures are all still correct -- losing the reader the answer to save
    them a picture.
    """
    keep = []
    for one in series if isinstance(series, list) else []:
        if not isinstance(one, dict):
            continue
        points = [
            point
            for point in (one.get("points") or [])
            if isinstance(point, dict)
            and point.get("x") is not None
            and isinstance(point.get("y"), (int, float))
            and not isinstance(point.get("y"), bool)
        ]
        if points:
            # The label fields travel with the series. Rebuilding it as name
            # plus points was right while the name was a finished string; it
            # silently dropped `name_key` and `name_args`, so every series in a
            # translated chart came back blank -- a legend of empty swatches
            # beside bars that were all correct.
            for point in points:
                # The interval travels with the point when the tool gave one;
                # anything else on the point is dropped here, not drawn.
                for side in ("y_low", "y_high"):
                    value = point.get(side)
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        point[side] = None
            keep.append(
                {
                    "name": one.get("name", ""),
                    "name_key": one.get("name_key"),
                    "name_args": one.get("name_args"),
                    # A stored chart from before the field existed still names
                    # the fleet series by its key; that key is what the series
                    # is, so it is drawn as the line it always meant.
                    "role": (
                        "reference"
                        if one.get("role") == "reference"
                        or one.get("name_key") == "chart.series.fleet_average"
                        else None
                    ),
                    "points": points,
                }
            )
    return keep


def _text(value: object) -> str:
    """Anything at all, safe to place in markup or in an attribute."""
    return escape(str(value if value is not None else ""), quote=True)


def _slots(series: list[dict]) -> list:
    """Every x value that appears in any series, in first-seen order.

    Taking these from the first series alone was correct while every chart had
    one, and silently wrong the moment a fan-out produced several: bars were
    placed by their position within their own series, so a class absent from
    one line shifted every later bar of that line one slot left and stood under
    another class's label. Charts are read by position, so a chart that is
    wrong by position is wrong without looking wrong.
    """
    slots: list = []
    for one in series:
        for point in one["points"]:
            if point["x"] not in slots:
                slots.append(point["x"])
    return slots


def _fmt(value: float) -> str:
    """A tick or bar label: an integer as an integer, a fraction with what it needs."""
    if float(value).is_integer():
        return str(int(value))
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _ticks(peak: float, count: int = 4) -> list[float]:
    """Round tick values from 0 to at least ``peak``, ``count`` steps apart.

    Chosen from 1-2-5 multiples of a power of ten, so `0.2509` gets 0.1 steps
    and `302` gets 100 steps -- the ruler a reader would draw by hand.
    """
    if peak <= 0:
        return [0.0]
    import math
    rough = peak / count
    magnitude = 10 ** math.floor(math.log10(rough))
    step = next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= rough)
    ticks = []
    value = 0.0
    while value < peak - 1e-12:
        value = round(value + step, 10)
        ticks.append(value)
    return [0.0] + ticks


def _split_title(title: str) -> tuple[str, str]:
    """The headline, and the parenthetical qualifier it carries, on two lines.

    The chart titles say what the figure is *and* what it is not ("the
    re-verifier's judgement, not ground truth") in one string, because the
    qualifier must not be separable from the number. It can still be drawn on
    its own line: the string is split at the first opening bracket, in either
    script, and the bracket is kept so the qualifier reads as one.
    """
    for bracket in (" (", "（"):
        if bracket in title:
            head, tail = title.split(bracket, 1)
            return head, bracket.strip() + tail
    return title, ""


def render_svg(
    spec: dict, width: int = 720, height: int = 280, locale: str | None = None
) -> str:
    """Render a chart specification. Returns '' for anything unplottable."""
    series = _plottable(spec.get("series"))
    if not series:
        return ""

    # Three header lines above the plot -- title, its qualifier, the y-axis
    # label -- rather than a title that runs off the right edge and a y-label
    # drawn over the first bar, which is what a 90-character English title
    # did to the event-window chart on 2026-08-28.
    pad_left, pad_right, pad_top, pad_bottom = 56, 16, 60, 44
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    slots = _slots(series)
    title = label_from(spec, "title", locale)
    # The ruler has to reach the top of the tallest interval, not just the
    # tallest bar: a whisker drawn off the top of the plot is a whisker the
    # reader cannot compare with the next one.
    peak = max(
        (max(p["y"], p.get("y_high") or p["y"]) for s in series for p in s["points"]),
        default=0.0,
    )
    ticks = _ticks(peak)
    top = ticks[-1] if ticks[-1] > 0 else 1.0
    scale = plot_h / top

    group_w = plot_w / max(1, len(slots))
    # Two bars on a 720 px chart were 216 px wide each and read as blocks of
    # colour rather than as heights; a bar is a mark, not a panel.
    bar_w = min(group_w / (len(series) + 1), 72.0)
    cluster_w = bar_w * len(series)

    headline, qualifier = _split_title(title)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="{_text(title or "chart")}">',
        f'<text x="{pad_left}" y="16" fill="#e7e7ea" font-size="13">'
        f'{_text(headline)}</text>',
        f'<text x="{pad_left}" y="32" fill="#8b8b96" font-size="10">'
        f'{_text(qualifier)}</text>',
        f'<text x="{pad_left}" y="48" fill="#8b8b96" font-size="10">'
        f'{_text(label_from(spec, "y_label", locale))}</text>',
    ]

    # The ruler: a gridline and a tick label per step, drawn before the bars
    # so the bars sit on top of it.
    for tick in ticks:
        y = pad_top + plot_h - tick * scale
        parts.append(
            f'<line class="grid" x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" '
            f'y2="{y:.1f}" stroke="#2e2e35" stroke-dasharray="{"" if tick == 0 else "2,3"}"/>'
            f'<text class="tick" x="{pad_left - 6}" y="{y + 3.5:.1f}" fill="#8b8b96" '
            f'font-size="9" text-anchor="end">{_text(_fmt(tick))}</text>'
        )

    for index, one in enumerate(series):
        colour = PALETTE[index % len(PALETTE)]
        name = label_from(one, "name", locale)
        if one.get("role") == "reference":
            # A baseline: one dashed line at the series' value, labelled at
            # the right edge, and a dashed swatch in the legend. It takes no
            # slot, so the bars keep their width.
            level = one["points"][0]["y"]
            y = pad_top + plot_h - level * scale
            parts.append(
                f'<line class="reference" x1="{pad_left}" y1="{y:.1f}" '
                f'x2="{width - pad_right}" y2="{y:.1f}" stroke="{colour}" '
                f'stroke-width="1.5" stroke-dasharray="6,4"><title>{_text(name)}: '
                f'{_text(_fmt(level))}</title></line>'
                f'<text class="value" text-anchor="end" x="{width - pad_right}" '
                f'y="{y - 4:.1f}" fill="{colour}" font-size="10">{_text(_fmt(level))}</text>'
            )
            parts.append(
                f'<line x1="{pad_left + index * 92}" y1="{height - 9.5}" '
                f'x2="{pad_left + index * 92 + 9}" y2="{height - 9.5}" stroke="{colour}" '
                f'stroke-width="2" stroke-dasharray="3,2"/>'
                f'<text x="{pad_left + index * 92 + 14}" y="{height - 6}" '
                f'fill="#8b8b96" font-size="10">{_text(name)}</text>'
            )
            continue
        for point in one["points"]:
            # By slot, not by enumerate: a series missing a class must leave a
            # gap under that class, not close it up.
            bar_h = point["y"] * scale
            slot_x = pad_left + slots.index(point["x"]) * group_w + group_w / 2
            x = slot_x - cluster_w / 2 + index * bar_w
            y = pad_top + plot_h - bar_h
            low, high = point.get("y_low"), point.get("y_high")
            # The hover text: the value, and the interval when there is one.
            tip = f'{name} {point["x"]}: {_fmt(point["y"])}'
            if low is not None and high is not None:
                tip += f' (95% {_fmt(low)}–{_fmt(high)})'
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{bar_h:.1f}" fill="{colour}" rx="2"><title>'
                f'{_text(tip)}</title></rect>'
            )
            # The number itself, on the bar. A chart without it sends the
            # reader back to the table for the one figure the chart is about.
            label_y = y - 4
            if low is not None and high is not None:
                # The interval as a whisker. Two overlapping whiskers are the
                # finding on the event-window chart; without them the chart
                # showed two heights and the qualifier under the title claimed
                # something the picture did not show.
                cx = x + bar_w / 2
                y_low = pad_top + plot_h - low * scale
                y_high = pad_top + plot_h - high * scale
                parts.append(
                    f'<line class="whisker" x1="{cx:.1f}" y1="{y_low:.1f}" '
                    f'x2="{cx:.1f}" y2="{y_high:.1f}" stroke="#e7e7ea" stroke-width="1.5"/>'
                    f'<line class="whisker-cap" x1="{cx - 5:.1f}" y1="{y_high:.1f}" '
                    f'x2="{cx + 5:.1f}" y2="{y_high:.1f}" stroke="#e7e7ea" stroke-width="1.5"/>'
                    f'<line class="whisker-cap" x1="{cx - 5:.1f}" y1="{y_low:.1f}" '
                    f'x2="{cx + 5:.1f}" y2="{y_low:.1f}" stroke="#e7e7ea" stroke-width="1.5"/>'
                )
                label_y = min(label_y, y_high - 4)
            parts.append(
                f'<text class="value" text-anchor="middle" x="{x + bar_w / 2:.1f}" '
                f'y="{max(label_y, pad_top - 2):.1f}" fill="#e7e7ea" font-size="10">'
                f'{_text(_fmt(point["y"]))}</text>'
            )
        parts.append(
            f'<rect x="{pad_left + index * 92}" y="{height - 14}" width="9" '
            f'height="9" fill="{colour}" rx="2"/>'
            f'<text x="{pad_left + index * 92 + 14}" y="{height - 6}" '
            f'fill="#8b8b96" font-size="10">{_text(name)}</text>'
        )

    for position, label in enumerate(slots):
        x = pad_left + position * group_w + group_w / 2
        parts.append(
            f'<text x="{x:.1f}" y="{pad_top + plot_h + 15}" fill="#8b8b96" '
            f'font-size="10" text-anchor="middle">{_text(label)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)
