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

PALETTE = ["#60a5fa", "#f59e0b", "#34d399", "#f472b6"]


def _text(value: object) -> str:
    """Anything at all, safe to place in markup or in an attribute."""
    return escape(str(value if value is not None else ""), quote=True)


def render_svg(spec: dict, width: int = 720, height: int = 280) -> str:
    """Render a chart specification. Returns '' for anything unplottable."""
    series = spec.get("series") or []
    if not series or not series[0].get("points"):
        return ""

    pad_left, pad_right, pad_top, pad_bottom = 56, 16, 28, 44
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    labels = [p["x"] for p in series[0]["points"]]
    peak = max((p["y"] for s in series for p in s["points"]), default=0.0)
    scale = plot_h / peak if peak > 0 else 0.0  # a flat series must not divide by zero

    group_w = plot_w / max(1, len(labels))
    bar_w = group_w / (len(series) + 1)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="{_text(spec.get("title", "chart"))}">',
        f'<text x="{pad_left}" y="18" fill="#e7e7ea" font-size="13">'
        f'{_text(spec.get("title", ""))}</text>',
        f'<line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{width - pad_right}" '
        f'y2="{pad_top + plot_h}" stroke="#2e2e35"/>',
    ]

    for index, one in enumerate(series):
        colour = PALETTE[index % len(PALETTE)]
        for position, point in enumerate(one["points"]):
            bar_h = point["y"] * scale
            x = pad_left + position * group_w + index * bar_w + bar_w / 2
            y = pad_top + plot_h - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{bar_h:.1f}" fill="{colour}" rx="2"><title>'
                f'{_text(point["x"])}: {_text(point["y"])}</title></rect>'
            )
        parts.append(
            f'<rect x="{pad_left + index * 92}" y="{height - 14}" width="9" '
            f'height="9" fill="{colour}" rx="2"/>'
            f'<text x="{pad_left + index * 92 + 14}" y="{height - 6}" '
            f'fill="#8b8b96" font-size="10">{_text(one.get("name", ""))}</text>'
        )

    for position, label in enumerate(labels):
        x = pad_left + position * group_w + group_w / 2
        parts.append(
            f'<text x="{x:.1f}" y="{pad_top + plot_h + 15}" fill="#8b8b96" '
            f'font-size="10" text-anchor="middle">{_text(label)}</text>'
        )

    parts.append(
        f'<text x="4" y="{pad_top + 8}" fill="#8b8b96" font-size="10">'
        f'{_text(spec.get("y_label", ""))}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)
