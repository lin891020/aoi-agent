"""Render the two README flow diagrams, light and dark, as standalone SVG.

Two diagrams, one shape each: the disposition graph (`graph/flow.py`) and the
analysis graph (`analysis/graph.py`). They are drawn from the graphs' real node
names and the thresholds the code carries, so a reader comparing the picture
with the source finds the same words. Re-run this after either graph changes;
the SVGs are checked in because GitHub renders them and CI does not run this.

    uv run python scripts/render_diagrams.py     # -> docs/diagrams/*.svg

Design rules follow the diagram-design skill: orthogonal rounded connectors,
masked labels with a visible gap above the stroke, one accent per diagram, a
legend strip at the bottom, every coordinate on a 4 px grid. Fonts are system
stacks rather than web fonts because GitHub serves an SVG inside an ``<img>``,
where nothing external loads.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.graph.flow import CONFIDENT, ESCALATE_BELOW  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "diagrams"

SANS = "'Geist', -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'Geist Mono', 'SF Mono', Menlo, Consolas, monospace"

THEMES = {
    "light": {
        "paper": "#f5f5f5", "ink": "#2d3142", "muted": "#4f5d75", "soft": "#7a8399",
        "rule": "rgba(45,49,66,0.12)", "store": "rgba(45,49,66,0.05)",
        "input": "rgba(79,93,117,0.10)", "accent": "#eb6c36",
        "accent_tint": "rgba(235,108,54,0.08)", "node": "#ffffff",
    },
    "dark": {
        "paper": "#2d3142", "ink": "#f5f5f5", "muted": "#bfc0c0", "soft": "#8e98ac",
        "rule": "rgba(245,245,245,0.12)", "store": "rgba(245,245,245,0.06)",
        "input": "rgba(191,192,192,0.12)", "accent": "#f08a59",
        "accent_tint": "rgba(240,138,89,0.10)", "node": "#393e53",
    },
}


class Canvas:
    def __init__(self, slug: str, width: int, height: int, theme: dict, title: str, desc: str, top: int = 0):
        self.top = top  # first visible y, so a diagram can start below an empty band
        self.t = theme
        self.slug = slug
        self.w, self.h = width, height
        self.arrows: list[str] = []   # drawn first, under the boxes
        self.boxes: list[str] = []
        self.title, self.desc = title, desc

    # ---- primitives -------------------------------------------------------
    def arrow(self, d: str, accent: bool = False, dashed: bool = False) -> None:
        stroke = self.t["accent"] if accent else self.t["muted"]
        marker = "arrow-accent" if accent else "arrow"
        dash = ' stroke-dasharray="4,3"' if dashed else ""
        self.arrows.append(
            f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="1.2"{dash} '
            f'marker-end="url(#{self.slug}-{marker})"/>'
        )

    def label(self, x: int, y: int, text: str, anchor: str = "middle") -> None:
        """A masked arrow label whose mask bottom sits 8 px above ``y``."""
        width = 8 + len(text) * 6.2
        left = x - width / 2 if anchor == "middle" else x
        self.arrows.append(
            f'<rect x="{left:.0f}" y="{y - 24}" width="{width:.0f}" height="14" rx="2" fill="{self.t["paper"]}"/>'
            f'<text x="{x}" y="{y - 13}" fill="{self.t["soft"]}" font-size="8" font-family="{MONO}" '
            f'text-anchor="{anchor}" letter-spacing="0.06em">{text}</text>'
        )

    def side_label(self, x: int, y: int, text: str, left: bool = False) -> None:
        """A label beside a vertical segment, 8 px clear of it on either side."""
        width = 8 + len(text) * 6.2
        if left:
            rect_x, text_x, anchor = x - 8 - width, x - 12, "end"
        else:
            rect_x, text_x, anchor = x + 8, x + 12, "start"
        self.arrows.append(
            f'<rect x="{rect_x:.0f}" y="{y - 7}" width="{width:.0f}" height="14" rx="2" fill="{self.t["paper"]}"/>'
            f'<text x="{text_x}" y="{y + 4}" fill="{self.t["soft"]}" font-size="8" font-family="{MONO}" '
            f'text-anchor="{anchor}" letter-spacing="0.06em">{text}</text>'
        )

    def box(self, x: int, y: int, w: int, h: int, name: str, sub: str = "",
            kind: str = "step", tag: str = "") -> None:
        t = self.t
        fill, stroke, dash = {
            "step": (t["node"], t["ink"], ""),
            "focal": (t["accent_tint"], t["accent"], ""),
            "store": (t["store"], t["muted"], ""),
            "input": (t["input"], t["soft"], ""),
            "terminal": (t["node"], t["ink"], ""),
        }[kind]
        rx = 24 if kind in ("input", "terminal") else 6
        cx, cy = x + w // 2, y + h // 2
        parts = [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{t["paper"]}"/>',
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1"{dash}/>',
        ]
        if tag:
            parts.append(
                f'<rect x="{x + 8}" y="{y + 6}" width="{8 + len(tag) * 5}" height="12" rx="2" fill="transparent" '
                f'stroke="{stroke}" stroke-opacity="0.4" stroke-width="0.8"/>'
                f'<text x="{x + 12 + len(tag) * 2.5:.0f}" y="{y + 15}" fill="{stroke}" fill-opacity="0.8" font-size="7" '
                f'font-family="{MONO}" text-anchor="middle" letter-spacing="0.08em">{tag}</text>'
            )
        name_y = (cy + 2 if not sub else cy - 4) + (4 if tag else 0)
        parts.append(
            f'<text x="{cx}" y="{name_y}" fill="{t["ink"]}" font-size="12" font-weight="600" '
            f'font-family="{SANS}" text-anchor="middle">{name}</text>'
        )
        if sub:
            parts.append(
                f'<text x="{cx}" y="{cy + 12 + (4 if tag else 0)}" fill="{t["muted"]}" font-size="9" '
                f'font-family="{MONO}" text-anchor="middle">{sub}</text>'
            )
        self.boxes.append("".join(parts))

    def diamond(self, cx: int, cy: int, hw: int, hh: int, name: str, sub: str = "",
                focal: bool = False) -> None:
        t = self.t
        stroke = t["accent"] if focal else t["ink"]
        fill = t["accent_tint"] if focal else t["node"]
        pts = f"{cx},{cy - hh} {cx + hw},{cy} {cx},{cy + hh} {cx - hw},{cy}"
        self.boxes.append(
            f'<polygon points="{pts}" fill="{t["paper"]}"/>'
            f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1" stroke-linejoin="round"/>'
            f'<text x="{cx}" y="{cy - 2 if sub else cy + 4}" fill="{t["ink"]}" font-size="12" font-weight="600" '
            f'font-family="{SANS}" text-anchor="middle">{name}</text>'
            + (f'<text x="{cx}" y="{cy + 12}" fill="{t["muted"]}" font-size="9" font-family="{MONO}" '
               f'text-anchor="middle">{sub}</text>' if sub else "")
        )

    def dot(self, cx: int, cy: int) -> None:
        self.boxes.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="{self.t["ink"]}"/>')

    def aside(self, x: int, y: int, text: str, anchor: str = "start") -> None:
        self.boxes.append(
            f'<text x="{x}" y="{y}" fill="{self.t["muted"]}" font-size="12" font-style="italic" '
            f'font-family="Georgia, \'Times New Roman\', serif" text-anchor="{anchor}">{text}</text>'
        )

    def legend(self, y: int, items: list[tuple[str, str]]) -> None:
        t = self.t
        parts = [
            f'<line x1="32" y1="{y - 8}" x2="{self.w - 32}" y2="{y - 8}" stroke="{t["rule"]}" stroke-width="0.8"/>',
            f'<text x="32" y="{y + 8}" fill="{t["muted"]}" font-size="8" font-family="{MONO}" letter-spacing="0.14em">LEGEND</text>',
        ]
        x = 104
        for kind, text in items:
            if kind == "focal":
                parts.append(f'<rect x="{x}" y="{y}" width="16" height="10" rx="2" fill="{t["accent_tint"]}" stroke="{t["accent"]}" stroke-width="1"/>')
            elif kind == "step":
                parts.append(f'<rect x="{x}" y="{y}" width="16" height="10" rx="2" fill="{t["node"]}" stroke="{t["ink"]}" stroke-width="1"/>')
            elif kind == "terminal":
                parts.append(f'<rect x="{x}" y="{y}" width="16" height="10" rx="5" fill="{t["node"]}" stroke="{t["ink"]}" stroke-width="1"/>')
            elif kind == "diamond":
                parts.append(f'<polygon points="{x + 8},{y - 1} {x + 16},{y + 5} {x + 8},{y + 11} {x},{y + 5}" fill="{t["node"]}" stroke="{t["ink"]}" stroke-width="1"/>')
            elif kind == "store":
                parts.append(f'<rect x="{x}" y="{y}" width="16" height="10" rx="2" fill="{t["store"]}" stroke="{t["muted"]}" stroke-width="1"/>')
            elif kind == "dot":
                parts.append(f'<circle cx="{x + 8}" cy="{y + 5}" r="4" fill="{t["ink"]}"/>')
            parts.append(f'<text x="{x + 24}" y="{y + 8}" fill="{t["muted"]}" font-size="8" font-family="{MONO}" letter-spacing="0.06em">{text}</text>')
            x += 32 + int(len(text) * 5.4) + 24
        self.boxes.append("".join(parts))

    # ---- output -----------------------------------------------------------
    def render(self) -> str:
        t = self.t
        s = self.slug
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 {self.top} {self.w} {self.h - self.top}" width="{self.w}" height="{self.h - self.top}" '
            f'role="img" aria-labelledby="{s}-title {s}-desc" font-family="{SANS}">\n'
            f'<title id="{s}-title">{self.title}</title>\n'
            f'<desc id="{s}-desc">{self.desc}</desc>\n'
            f'<defs>'
            f'<marker id="{s}-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">'
            f'<polygon points="0 0, 8 3, 0 6" fill="{t["muted"]}"/></marker>'
            f'<marker id="{s}-arrow-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">'
            f'<polygon points="0 0, 8 3, 0 6" fill="{t["accent"]}"/></marker>'
            f'</defs>\n'
            f'<rect x="0" y="{self.top}" width="{self.w}" height="{self.h - self.top}" fill="{t["paper"]}"/>\n'
            + "\n".join(self.arrows) + "\n" + "\n".join(self.boxes) + "\n</svg>\n"
        )


def disposition(theme_name: str) -> str:
    t = THEMES[theme_name]
    c = Canvas(
        f"disposition-{theme_name}", 960, 776, t,
        "How one flagged region is dispositioned",
        "Flowchart: the re-verifier classifies a flagged region; a confident false call is "
        "dismissed and a confident defect confirmed without a language model; anything else "
        "gathers production context and criteria, the LLM writes a rationale, and the region "
        "is decided on the classifier's confidence or handed to an operator through a durable interrupt.",
    )
    thr = f"{ESCALATE_BELOW:.3f}"
    conf = f"{CONFIDENT:.3f}"

    # arrows first
    c.arrow("M480,72 V108")                                   # start -> classify
    c.arrow("M480,176 V208")                                  # classify -> D1
    c.arrow("M384,248 H144 Q136,248 136,256 V652 Q136,660 144,660 H356")  # dismiss
    c.label(260, 248, f"P(FALSE CALL) ≥ {thr} · DISMISS")
    c.arrow("M576,248 H912 Q920,248 920,256 V652 Q920,660 912,660 H604")  # confirm
    c.label(744, 248, f"DEFECT ≥ {conf}, NOT OPEN · CONFIRM")
    c.arrow("M480,288 V316")                                  # D1 -> gather
    c.side_label(480, 302, "EVERYTHING ELSE")
    c.arrow("M480,380 V404")                                  # gather -> reason
    c.arrow("M480,468 V508")                                  # reason -> D2
    c.arrow("M480,588 V628")                                  # D2 -> verdict
    c.side_label(480, 608, f"CONFIDENCE ≥ {thr} · DECIDE")
    c.arrow("M576,548 H668", accent=True)                     # D2 -> escalate
    c.label(622, 548, "BELOW · ESCALATE")

    # boxes
    c.box(400, 24, 160, 48, "AOI candidate", "template · test · difference", kind="input")
    c.box(352, 112, 256, 64, "ResNet-18 re-verifier", "class + P(false call) · 2.5 ms CPU", tag="CLASSIFY")
    c.diamond(480, 248, 96, 40, "sure enough?")
    c.box(352, 316, 256, 64, "gather context", "3 MCP tools · criteria scoped by class", tag="TOOLS")
    c.box(352, 404, 256, 64, "LLM writes the rationale", "explains, never decides · 60 s limit", tag="LLM")
    c.diamond(480, 548, 96, 40, "sure enough?")
    c.box(360, 632, 240, 56, "classifier's verdict stands", "dismiss · confirm · decide", kind="terminal")
    c.box(672, 516, 224, 64, "an operator answers", "interrupt() · checkpoint · queue", kind="focal", tag="HUMAN")
    c.aside(672, 604, "82.2% of regions never reach the LLM;")
    c.aside(672, 620, "the operator's answer is the next label.")
    c.legend(736, [
        ("terminal", "start / end"), ("step", "node in graph/flow.py"),
        ("diamond", "routes on the classifier"), ("focal", "the one place a person enters"),
    ])
    return c.render()


def analysis(theme_name: str) -> str:
    t = THEMES[theme_name]
    c = Canvas(
        f"analysis-{theme_name}", 992, 568, t,
        "How /ask answers a supervisor's question",
        "Data-flow: a supervisor's question becomes a typed plan of tool calls from one LLM call; "
        "the plan is validated against real tool signatures and the store's value domains and refused "
        "with every error if it fails; valid calls fan out in parallel, results are collected with a "
        "chart derived from their shape, a second LLM call writes prose beside the figures, and the "
        "run is stored so a chart is redrawn from data rather than by re-planning.",
        top=80,
    )
    # arrows
    c.arrow("M184,200 H220")                                  # question -> plan
    c.arrow("M384,200 H404")                                  # plan -> validate
    c.arrow("M480,240 V312")                                  # validate -> refused
    c.side_label(480, 276, "NO · EVERY ERROR SHOWN")
    c.arrow("M552,200 H572")                                  # validate -> dot
    c.arrow("M576,200 H604")                                  # dot -> run tools (middle)
    c.arrow("M576,200 V152 Q576,144 584,144 H604")            # dot -> run tools (top)
    c.arrow("M576,200 V248 Q576,256 584,256 H604")            # dot -> run tools (bottom)
    c.label(576, 132, "SEND ×N · INDEPENDENT FACTS")
    c.arrow("M768,144 H788 Q796,144 796,152 V180 Q796,188 804,188 H808")   # tools -> collect
    c.arrow("M768,200 H808")
    c.arrow("M768,256 H788 Q796,256 796,248 V220 Q796,212 804,212 H808")
    c.arrow("M880,224 V312")                                  # collect -> synthesise
    c.side_label(880, 290, "CHART FROM RESULT SHAPE", left=True)
    c.arrow("M880,364 V428")                                  # synthesise -> page
    c.arrow("M808,460 H740")                                  # page -> store
    c.label(774, 460, "RECORDED")

    # boxes
    c.box(24, 176, 160, 48, "supervisor's question", "zh-TW or en · free text", kind="input")
    c.box(224, 172, 160, 56, "plan", "LLM call #1 · typed plan", tag="LLM")
    c.diamond(480, 200, 72, 40, "valid?", focal=True)
    c.box(400, 312, 160, 52, "refused, with every error", "shown, never retried", kind="terminal")
    c.dot(576, 200)
    c.boxes.append(  # a stack behind the tool node: one node, N branches
        f'<rect x="{616}" y="{124}" width="160" height="136" rx="6" fill="{t["node"]}" stroke="{t["ink"]}" stroke-opacity="0.35" stroke-width="1"/>'
        f'<rect x="{612}" y="{128}" width="160" height="136" rx="6" fill="{t["node"]}" stroke="{t["ink"]}" stroke-opacity="0.55" stroke-width="1"/>'
    )
    c.box(608, 132, 160, 136, "run one tool", "typed args · no SQL", tag="MCP")
    c.boxes.append(
        f'<text x="688" y="234" fill="{t["soft"]}" font-size="8" font-family="{MONO}" text-anchor="middle">a failed branch returns</text>'
        f'<text x="688" y="246" fill="{t["soft"]}" font-size="8" font-family="{MONO}" text-anchor="middle">data, not an exception</text>'
    )
    c.box(808, 176, 144, 48, "collect", "operator.add reducer")
    c.box(808, 312, 144, 52, "synthesise", "LLM call #2 · prose only", tag="LLM")
    c.box(808, 428, 144, 60, "answer page", "prose beside its figures", kind="terminal")
    c.box(580, 428, 160, 60, "analysis_runs", "plan + results, redrawn", kind="store", tag="SQLITE")
    c.aside(24, 468, "The validator checks the tool, the argument names and")
    c.aside(24, 484, "the argument values against what the store holds;")
    c.aside(24, 500, "the LLM chooses lookups and writes the sentence, nothing else.")
    c.legend(528, [
        ("terminal", "start / end"), ("step", "node in analysis/graph.py"),
        ("focal", "the gate that refuses"), ("dot", "Send fan-out"), ("store", "table"),
    ])
    return c.render()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (OUT / f"disposition-flow-{theme}.svg").write_text(disposition(theme))
        (OUT / f"analysis-flow-{theme}.svg").write_text(analysis(theme))
    for path in sorted(OUT.glob("*.svg")):
        print(path.relative_to(OUT.parents[1]), f"{path.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
