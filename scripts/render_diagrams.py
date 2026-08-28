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

SANS = "'Geist', -apple-system, 'PingFang TC', 'Noto Sans TC', 'Microsoft JhengHei', 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'Geist Mono', 'SF Mono', Menlo, Consolas, 'PingFang TC', 'Noto Sans TC', monospace"

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


def _cells(text: str) -> float:
    """Width in latin-character cells: a CJK glyph takes about two."""
    return sum(1.9 if ord(ch) > 0x2E7F else 1.0 for ch in text)


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
        width = 8 + _cells(text) * 6.2
        left = x - width / 2 if anchor == "middle" else x
        self.arrows.append(
            f'<rect x="{left:.0f}" y="{y - 24}" width="{width:.0f}" height="14" rx="2" fill="{self.t["paper"]}"/>'
            f'<text x="{x}" y="{y - 13}" fill="{self.t["soft"]}" font-size="8" font-family="{MONO}" '
            f'text-anchor="{anchor}" letter-spacing="0.06em">{text}</text>'
        )

    def side_label(self, x: int, y: int, text: str, left: bool = False) -> None:
        """A label beside a vertical segment, 8 px clear of it on either side."""
        width = 8 + _cells(text) * 6.2
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
            x += 32 + int(_cells(text) * 5.4) + 24
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


STRINGS = {
    "en": {
        "candidate": ("AOI candidate", "template · test · difference"),
        "classify": ("ResNet-18 re-verifier", "class + P(false call) · 2.5 ms CPU"),
        "sure": "route on confidence",
        "dismiss": "P(FALSE CALL) ≥ {thr} · DISMISS",
        "confirm": "DEFECT ≥ {conf}, NOT OPEN · CONFIRM",
        "else": "EVERYTHING ELSE",
        "gather": ("gather context", "3 MCP tools · criteria scoped to the class"),
        "reason": ("LLM rationale", "explanation only · 60 s deadline"),
        "decide": "CONFIDENCE ≥ {thr} · DECIDE",
        "escalate": "BELOW · ESCALATE",
        "verdict": ("disposition = classifier's class", "dismiss · confirm · decide"),
        "operator": ("operator review", "interrupt() · checkpoint · queue"),
        "aside1": "82.2% of regions are dispositioned without the LLM.",
        "aside2": "The operator's answer is the next training label.",
        "legend1": [("terminal", "start / end"), ("step", "node in graph/flow.py"),
                    ("diamond", "decision on classifier output"), ("focal", "human review")],
        "question": ("supervisor's question", "zh-TW or en · free text"),
        "plan": ("plan", "LLM call #1 · typed plan"),
        "valid": "validation",
        "no": "FAIL · ALL ERRORS SHOWN",
        "send": "SEND ×N · INDEPENDENT FACTS",
        "refused": ("refused", "errors shown · no retry"),
        "tool": ("run tool", "typed arguments · no SQL"),
        "tool_note": ("a failed branch returns data,", "not an exception"),
        "collect": ("collect", "operator.add reducer"),
        "chart": "CHART FROM RESULT SHAPE",
        "synth": ("synthesise", "LLM call #2 · prose"),
        "page": ("answer page", "prose and figures"),
        "store": ("analysis_runs", "plan + results · redraw, not re-run"),
        "recorded": "RECORDED",
        "aside3": ["Validation checks tool name, argument names and",
                   "argument values against the store's domains.",
                   "The LLM selects lookups and writes prose; nothing else."],
        "legend2": [("terminal", "start / end"), ("step", "node in analysis/graph.py"),
                    ("focal", "validation gate"), ("dot", "Send fan-out"), ("store", "table")],
        "title1": "How one flagged region is dispositioned",
        "title2": "How /ask answers a supervisor's question",
    },
    "zh-TW": {
        "candidate": ("AOI 標出的區域", "範本 · 待測板 · 差異圖"),
        "classify": ("ResNet-18 複判模型", "類別 + P(誤判) · CPU 2.5 ms"),
        "sure": "依信心分路",
        "dismiss": "P(誤判) ≥ {thr} · 排除",
        "confirm": "缺陷 ≥ {conf} 且非 open · 確認",
        "else": "其餘",
        "gather": ("取得脈絡", "3 個 MCP 工具 · 標準依類別限定"),
        "reason": ("LLM 產生說明", "僅說明 · 60 秒上限"),
        "decide": "信心 ≥ {thr} · 決定",
        "escalate": "低於 · 交給人",
        "verdict": ("處置 = 分類器判定", "排除 · 確認 · 決定"),
        "operator": ("作業員複判", "interrupt() · checkpoint · 佇列"),
        "aside1": "82.2% 的區域不經 LLM 即完成處置。",
        "aside2": "作業員的判定即下一輪訓練標籤。",
        "legend1": [("terminal", "起點 / 終點"), ("step", "graph/flow.py 的 node"),
                    ("diamond", "依分類器輸出決定"), ("focal", "人工複判")],
        "question": ("主管的問題", "中文或英文 · 自由輸入"),
        "plan": ("規劃", "LLM 呼叫 #1 · 型別化計畫"),
        "valid": "驗證",
        "no": "未通過 · 列出全部錯誤",
        "send": "SEND ×N · 彼此獨立的事實",
        "refused": ("拒絕", "顯示錯誤 · 不重試"),
        "tool": ("執行工具", "型別化參數 · 無 SQL"),
        "tool_note": ("失敗的分支回傳資料，", "而非例外"),
        "collect": ("收集", "operator.add reducer"),
        "chart": "圖從結果的形狀推出",
        "synth": ("合成", "LLM 呼叫 #2 · 文字"),
        "page": ("答案頁", "文字與數字並列"),
        "store": ("analysis_runs", "計畫 + 結果 · 重畫不重跑"),
        "recorded": "寫入",
        "aside3": ["驗證對照工具名稱、參數名稱，",
                   "以及 store 中實際存在的值域。",
                   "LLM 負責選擇查詢與撰寫文字，其餘皆由程式決定。"],
        "legend2": [("terminal", "起點 / 終點"), ("step", "analysis/graph.py 的 node"),
                    ("focal", "驗證關卡"), ("dot", "Send 展開"), ("store", "資料表")],
        "title1": "一個被標出的區域怎麼被處置",
        "title2": "/ask 怎麼回答領班的問題",
    },
}


def disposition(theme_name: str, lang: str = "en") -> str:
    t = THEMES[theme_name]
    L = STRINGS[lang]
    slug = f"disposition-{theme_name}" + ("" if lang == "en" else "-zh")
    c = Canvas(
        slug, 960, 776, t,
        L["title1"],
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
    c.label(260, 248, L["dismiss"].format(thr=thr))
    c.arrow("M576,248 H912 Q920,248 920,256 V652 Q920,660 912,660 H604")  # confirm
    c.label(744, 248, L["confirm"].format(conf=conf))
    c.arrow("M480,288 V316")                                  # D1 -> gather
    c.side_label(480, 302, L["else"])
    c.arrow("M480,380 V404")                                  # gather -> reason
    c.arrow("M480,468 V508")                                  # reason -> D2
    c.arrow("M480,588 V628")                                  # D2 -> verdict
    c.side_label(480, 608, L["decide"].format(thr=thr))
    c.arrow("M576,548 H668", accent=True)                     # D2 -> escalate
    c.label(622, 548, L["escalate"])

    # boxes
    c.box(400, 24, 160, 48, *L["candidate"], kind="input")
    c.box(352, 112, 256, 64, *L["classify"], tag="CLASSIFY")
    c.diamond(480, 248, 96, 40, L["sure"])
    c.box(352, 316, 256, 64, *L["gather"], tag="TOOLS")
    c.box(352, 404, 256, 64, *L["reason"], tag="LLM")
    c.diamond(480, 548, 96, 40, L["sure"])
    c.box(360, 632, 240, 56, *L["verdict"], kind="terminal")
    c.box(672, 516, 224, 64, *L["operator"], kind="focal", tag="HUMAN")
    c.aside(672, 604, L["aside1"])
    c.aside(672, 620, L["aside2"])
    c.legend(736, L["legend1"])
    return c.render()


def analysis(theme_name: str, lang: str = "en") -> str:
    t = THEMES[theme_name]
    L = STRINGS[lang]
    slug = f"analysis-{theme_name}" + ("" if lang == "en" else "-zh")
    c = Canvas(
        slug, 992, 568, t,
        L["title2"],
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
    c.side_label(480, 276, L["no"])
    c.arrow("M552,200 H572")                                  # validate -> dot
    c.arrow("M576,200 H604")                                  # dot -> run tools (middle)
    c.arrow("M576,200 V152 Q576,144 584,144 H604")            # dot -> run tools (top)
    c.arrow("M576,200 V248 Q576,256 584,256 H604")            # dot -> run tools (bottom)
    c.label(576, 132, L["send"])
    c.arrow("M768,144 H788 Q796,144 796,152 V180 Q796,188 804,188 H808")   # tools -> collect
    c.arrow("M768,200 H808")
    c.arrow("M768,256 H788 Q796,256 796,248 V220 Q796,212 804,212 H808")
    c.arrow("M880,224 V312")                                  # collect -> synthesise
    c.side_label(880, 290, L["chart"], left=True)
    c.arrow("M880,364 V428")                                  # synthesise -> page
    c.arrow("M808,460 H740")                                  # page -> store
    c.label(774, 460, L["recorded"])

    # boxes
    c.box(24, 176, 160, 48, *L["question"], kind="input")
    c.box(224, 172, 160, 56, *L["plan"], tag="LLM")
    c.diamond(480, 200, 72, 40, L["valid"], focal=True)
    c.box(400, 312, 160, 52, *L["refused"], kind="terminal")
    c.dot(576, 200)
    c.boxes.append(  # a stack behind the tool node: one node, N branches
        f'<rect x="{616}" y="{124}" width="160" height="136" rx="6" fill="{t["node"]}" stroke="{t["ink"]}" stroke-opacity="0.35" stroke-width="1"/>'
        f'<rect x="{612}" y="{128}" width="160" height="136" rx="6" fill="{t["node"]}" stroke="{t["ink"]}" stroke-opacity="0.55" stroke-width="1"/>'
    )
    c.box(608, 132, 160, 136, *L["tool"], tag="MCP")
    n1, n2 = L["tool_note"]
    c.boxes.append(
        f'<text x="688" y="234" fill="{t["soft"]}" font-size="8" font-family="{MONO}" text-anchor="middle">{n1}</text>'
        f'<text x="688" y="246" fill="{t["soft"]}" font-size="8" font-family="{MONO}" text-anchor="middle">{n2}</text>'
    )
    c.box(808, 176, 144, 48, *L["collect"])
    c.box(808, 312, 144, 52, *L["synth"], tag="LLM")
    c.box(808, 428, 144, 60, *L["page"], kind="terminal")
    c.box(580, 428, 160, 60, *L["store"], kind="store", tag="SQLITE")
    for i, line in enumerate(L["aside3"]):
        c.aside(24, 468 + 16 * i, line)
    c.legend(528, L["legend2"])
    return c.render()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (OUT / f"disposition-flow-{theme}.svg").write_text(disposition(theme))
        (OUT / f"analysis-flow-{theme}.svg").write_text(analysis(theme))
        (OUT / f"disposition-flow-{theme}.zh-TW.svg").write_text(disposition(theme, "zh-TW"))
        (OUT / f"analysis-flow-{theme}.zh-TW.svg").write_text(analysis(theme, "zh-TW"))
    for path in sorted(OUT.glob("*.svg")):
        print(path.relative_to(OUT.parents[1]), f"{path.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
