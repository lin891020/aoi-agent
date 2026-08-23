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

from aoi_agent.station.i18n import label_from

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
            keep.append(
                {
                    "name": one.get("name", ""),
                    "name_key": one.get("name_key"),
                    "name_args": one.get("name_args"),
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


def render_svg(
    spec: dict, width: int = 720, height: int = 280, locale: str | None = None
) -> str:
    """Render a chart specification. Returns '' for anything unplottable."""
    series = _plottable(spec.get("series"))
    if not series:
        return ""

    pad_left, pad_right, pad_top, pad_bottom = 56, 16, 28, 44
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    slots = _slots(series)
    title = label_from(spec, "title", locale)
    peak = max((p["y"] for s in series for p in s["points"]), default=0.0)
    scale = plot_h / peak if peak > 0 else 0.0  # a flat series must not divide by zero

    group_w = plot_w / max(1, len(slots))
    bar_w = group_w / (len(series) + 1)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="{_text(title or "chart")}">',
        f'<text x="{pad_left}" y="18" fill="#e7e7ea" font-size="13">'
        f'{_text(title)}</text>',
        f'<line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{width - pad_right}" '
        f'y2="{pad_top + plot_h}" stroke="#2e2e35"/>',
    ]

    for index, one in enumerate(series):
        colour = PALETTE[index % len(PALETTE)]
        name = label_from(one, "name", locale)
        for point in one["points"]:
            # By slot, not by enumerate: a series missing a class must leave a
            # gap under that class, not close it up.
            bar_h = point["y"] * scale
            x = pad_left + slots.index(point["x"]) * group_w + index * bar_w + bar_w / 2
            y = pad_top + plot_h - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{bar_h:.1f}" fill="{colour}" rx="2"><title>'
                f'{_text(name)} {_text(point["x"])}: {_text(point["y"])}</title></rect>'
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

    parts.append(
        f'<text x="4" y="{pad_top + 8}" fill="#8b8b96" font-size="10">'
        f'{_text(label_from(spec, "y_label", locale))}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)
