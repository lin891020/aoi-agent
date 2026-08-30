"""Build the project-journey deck from `scripts/deck_content.py`.

    uv run --with python-pptx --with playwright python scripts/build_deck.py
    uv run --with python-pptx --with playwright python scripts/build_deck.py --embed-video

Three outputs from one source, so they cannot disagree:

* `docs/deck/aoi-agent-journey.zh-TW.pptx` -- 16:9, speaker notes on every
  slide (a plain-language line, the script, and the interviewer's questions).
* `docs/deck/aoi-agent-journey.zh-TW.html` -- the same slides as a page: arrow
  keys to move, `q` for self-test mode (title and questions only), `a` to
  reveal the answers, `n` to fold the notes.
* `docs/deck/study-guide.zh-TW.md` -- the same content as a printable Q&A.

Images are rendered into `docs/deck/img/` on the way: the architecture page
and the two flow diagrams through Playwright, the operating-point curve from
`models/test_predictions.npz` through matplotlib (skipped with a note when
the file is not there), the station screenshots composed from
`docs/screenshots/`.

`--embed-video` writes a second .pptx to `docs/demo/build/` with the demo
recording embedded; that directory is git-ignored, as the recording is.

python-pptx and playwright are not project dependencies -- pass them with
`--with`. Without them the script says so and stops.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from deck_content import CORE_TOTAL, FIVE_LABELS, LAYERS, SLIDES, Slide, by_layer  # noqa: E402

OUT = ROOT / "docs" / "deck"
IMG = OUT / "img"
SCREENSHOTS = ROOT / "docs" / "screenshots"
DIAGRAMS = ROOT / "docs" / "diagrams"
ARCH_PAGE = ROOT / "docs" / "architecture-diagram.html"
PREDICTIONS = ROOT / "models" / "test_predictions.npz"
VIDEO = ROOT / "docs" / "demo" / "aoi-agent-demo-zh.mp4"
VIDEO_LINK = "https://github.com/lin891020/aoi-agent/releases/tag/demo-2026-08-28"
EMBEDDED_OUT = ROOT / "docs" / "demo" / "build" / "aoi-agent-journey.zh-TW.embedded.pptx"

# One palette, both outputs. Dark ground, one accent, a warm mark for mistakes.
INK = "E8E6E1"
DIM = "9AA0A6"
GROUND = "15171A"
PANEL = "1E2125"
ACCENT = "6FC7CF"
WARN = "E2B45F"
FONT = "Noto Sans TC"
MONO = "IBM Plex Mono"


# ----------------------------------------------------------------- images

def render_images(need_playwright: bool = True) -> dict[str, Path]:
    IMG.mkdir(parents=True, exist_ok=True)
    made: dict[str, Path] = {}
    made.update(_render_pages() if need_playwright else {})
    curve = _render_operating_point()
    if curve:
        made["operating_point.png"] = curve
    screens = _compose_screens()
    if screens:
        made["screens.png"] = screens
    for name, fn in (("training_curves.png", _render_training_curves),
                     ("crop_curves.png", _render_crop_curves),
                     ("gate_curves.png", _render_gate_curves)):
        path = fn()
        if path:
            made[name] = path
    return made


def _render_pages() -> dict[str, Path]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed: architecture.png and flows.png not rendered "
              "(pass --with playwright)")
        return {}
    made: dict[str, Path] = {}
    flows_html = IMG / "_flows.html"
    svgs = [
        (DIAGRAMS / "disposition-flow-light.zh-TW.svg").read_text(),
        (DIAGRAMS / "analysis-flow-light.zh-TW.svg").read_text(),
    ]
    flows_html.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<body style='margin:0;background:#f5f5f5;display:grid;grid-template-columns:1fr 1fr;"
        "gap:24px;padding:24px;width:2000px;box-sizing:border-box'>"
        + "".join(f"<div>{svg}</div>" for svg in svgs) + "</body>"
    )
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
        page.goto(ARCH_PAGE.as_uri())
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(800)
        target = IMG / "architecture.png"
        page.screenshot(path=str(target), full_page=True)
        made["architecture.png"] = target
        page = browser.new_page(viewport={"width": 2000, "height": 700}, device_scale_factor=2)
        page.goto(flows_html.as_uri())
        page.wait_for_timeout(800)
        target = IMG / "flows.png"
        page.screenshot(path=str(target), full_page=True)
        made["flows.png"] = target
        browser.close()
    flows_html.unlink(missing_ok=True)
    return made


def _cjk_font(matplotlib) -> None:
    """A face that has the CJK glyphs, whichever of these the machine holds."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans TC", "PingFang TC", "Heiti TC", "Hiragino Sans", "Arial Unicode MS"):
        if name in have:
            matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
            return


def _render_operating_point() -> Path | None:
    if not PREDICTIONS.exists():
        print(f"{PREDICTIONS.relative_to(ROOT)} not found: operating_point.png not rendered "
              "(run scripts/train.py)")
        return None
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    _cjk_font(matplotlib)
    import matplotlib.pyplot as plt
    from aoi_agent.vision.operating_point import best_at_escape_budget, sweep

    data = np.load(PREDICTIONS)
    names = [str(n) for n in data["label_names"]]
    fc = names.index("false_call")
    points = sweep(data["probabilities"][:, fc], data["labels"], fc)
    best = best_at_escape_budget(points, 0.005)
    xs = [p.escape_rate * 100 for p in points]
    ys = [p.review_reduction * 100 for p in points]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160, facecolor="#" + PANEL)
    ax.set_facecolor("#" + PANEL)
    ax.plot(xs, ys, color="#" + ACCENT, linewidth=2)
    ax.axvline(0.5, color="#" + WARN, linestyle="--", linewidth=1.2)
    if best:
        ax.scatter([best.escape_rate * 100], [best.review_reduction * 100],
                   color="#" + WARN, zorder=5, s=50)
        ax.annotate(f"門檻 {best.threshold:.3f}\n{best.review_reduction * 100:.1f}% 省掉 @ "
                    f"{best.escape_rate * 100:.2f}% 漏網",
                    (best.escape_rate * 100, best.review_reduction * 100),
                    xytext=(1.2, best.review_reduction * 100 - 18), color="#" + INK,
                    fontsize=11, arrowprops={"color": "#" + DIM, "arrowstyle": "->"})
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 100)
    ax.set_xlabel("escape rate（漏網率，% of defects）", color="#" + INK)
    ax.set_ylabel("review removed（省掉的人工，% of candidates）", color="#" + INK)
    ax.tick_params(colors="#" + DIM)
    for spine in ax.spines.values():
        spine.set_color("#" + DIM)
    ax.grid(color="#33383D", linestyle=":", linewidth=0.8)
    ax.text(0.55, 3, "QP-110 預算 0.5%", color="#" + WARN, fontsize=10)
    fig.tight_layout()
    target = IMG / "operating_point.png"
    fig.savefig(target, facecolor=fig.get_facecolor())
    plt.close(fig)
    return target


def _dark_axes(ax):
    ax.set_facecolor("#" + PANEL)
    ax.tick_params(colors="#" + DIM)
    for spine in ax.spines.values():
        spine.set_color("#" + DIM)
    ax.grid(color="#33383D", linestyle=":", linewidth=0.8)
    ax.xaxis.label.set_color("#" + INK)
    ax.yaxis.label.set_color("#" + INK)
    ax.title.set_color("#" + INK)


def _history_plot(history_path: Path, target: Path, title: str) -> Path | None:
    if not history_path.exists():
        print(f"{history_path.relative_to(ROOT)} not found: {target.name} not rendered")
        return None
    import json
    import matplotlib
    matplotlib.use("Agg")
    _cjk_font(matplotlib)
    import matplotlib.pyplot as plt
    rows = json.loads(history_path.read_text())
    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=160, facecolor="#" + PANEL)
    for ax, key, label, colour in zip(
        axes, ("train_loss", "val_accuracy", "val_review_reduction"),
        ("train loss", "val accuracy", "val review reduction @ ≤0.5% 漏網"),
        (DIM, DIM, ACCENT),
    ):
        ys = [r[key] for r in rows]
        ax.plot(epochs, ys, marker="o", color="#" + colour, linewidth=2)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("epoch")
        _dark_axes(ax)
        if key != "train_loss":
            ax.set_ylim(0, 1)
        best = max(range(len(ys)), key=lambda i: ys[i]) if key == "val_review_reduction" else None
        if best is not None:
            ax.scatter([epochs[best]], [ys[best]], color="#" + WARN, zorder=5, s=45)
            ax.annotate(f"{ys[best]:.3f} @ epoch {epochs[best]}", (epochs[best], ys[best]),
                        xytext=(0, 10), textcoords="offset points", ha="center",
                        color="#" + WARN, fontsize=9)
    fig.suptitle(title, color="#" + INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(target, facecolor=fig.get_facecolor())
    plt.close(fig)
    return target


def _render_training_curves() -> Path | None:
    return _history_plot(ROOT / "models" / "history.json", IMG / "training_curves.png",
                         "主線複判模型 · ResNet-18 三通道 · 10 epochs")


def _render_crop_curves() -> Path | None:
    return _history_plot(ROOT / "models" / "pcbaoi_reverifier" / "history.json",
                         IMG / "crop_curves.png", "無樣板 crop 複判器 · 同一個 ResNet-18 · 10 epochs")


def _render_gate_curves() -> Path | None:
    import json
    files = {
        "DeepPCB（二值圖，trainval）": ROOT / "eval" / "results" / "gate_check.json",
        "HRIPCB 對齊（照片）": ROOT / "eval" / "results" / "gate_check_hripcb_aligned.json",
        "HRIPCB 對齊＋擾動": ROOT / "eval" / "results" / "gate_check_hripcb_aligned_perturbed.json",
    }
    if not all(f.exists() for f in files.values()):
        print("eval/results/gate_check*.json missing: gate_curves.png not rendered")
        return None
    import matplotlib
    matplotlib.use("Agg")
    _cjk_font(matplotlib)
    import matplotlib.pyplot as plt
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=160, facecolor="#" + PANEL)
    colours = [ACCENT, WARN, "FF8F7C"]
    for (label, path), colour in zip(files.items(), colours):
        runs = json.loads(path.read_text())["runs"]
        thr = [r["threshold"] for r in runs]
        left.plot(thr, [r["recall"] * 100 for r in runs], marker="o", color="#" + colour, label=label)
        right.plot(thr, [r["mean_false_calls"] for r in runs], marker="o", color="#" + colour, label=label)
    left.set_xlabel("灰階門檻"); left.set_ylabel("recall（%）"); left.set_ylim(0, 100)
    right.set_xlabel("灰階門檻"); right.set_ylabel("每張圖的平均誤報數"); right.set_yscale("log")
    for ax in (left, right):
        _dark_axes(ax)
        ax.axvline(60, color="#" + DIM, linestyle="--", linewidth=1)
        ax.legend(facecolor="#" + PANEL, edgecolor="#33383D", labelcolor="#" + INK, fontsize=9)
    left.set_title("門檻 60 是主線的設定", fontsize=11)
    right.set_title("對數尺度", fontsize=11)
    fig.tight_layout()
    target = IMG / "gate_curves.png"
    fig.savefig(target, facecolor=fig.get_facecolor())
    plt.close(fig)
    return target


def _compose_screens() -> Path | None:
    files = [SCREENSHOTS / f"{n}-zh.png" for n in ("queue", "region", "boards", "ask")]
    if not all(f.exists() for f in files):
        print("docs/screenshots/*-zh.png missing: screens.png not rendered")
        return None
    import matplotlib
    matplotlib.use("Agg")
    _cjk_font(matplotlib)
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), dpi=140, facecolor="#" + GROUND)
    for ax, f, label in zip(axes.flat, files, ("待複判", "區域頁", "PCB 處置", "產線查詢")):
        img = mpimg.imread(f)
        h = min(img.shape[0], int(img.shape[1] * 0.62))
        ax.imshow(img[:h])
        ax.set_title(label, color="#" + INK, fontsize=11, loc="left")
        ax.axis("off")
    fig.tight_layout()
    target = IMG / "screens.png"
    fig.savefig(target, facecolor=fig.get_facecolor())
    plt.close(fig)
    return target


# ------------------------------------------------------------------ pptx

def _notes_text(slide: Slide) -> str:
    lines = [f"白話：{slide.plain}", ""]
    lines += [f"講稿 {i}. {s}" for i, s in enumerate(slide.notes, 1)]
    if slide.questions:
        lines += ["", "面試官會問："]
        for q, a in slide.questions:
            lines += [f"Q：{q}", f"A：{a}", ""]
    return "\n".join(lines).rstrip()


def build_pptx(target: Path, images: dict[str, Path], embed_video: bool) -> Path:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Emu, Inches, Pt

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    W, H = prs.slide_width, prs.slide_height
    blank = prs.slide_layouts[6]
    rgb = lambda hexa: RGBColor.from_string(hexa)  # noqa: E731

    def text(slide, left, top, width, height, runs, size=16, color=INK, bold=False,
             align=None, font=FONT, spacing=1.15):
        """`runs`: list of paragraphs; a paragraph is a str or a list of
        (text, {bold, color, size, font}) tuples."""
        box = slide.shapes.add_textbox(left, top, width, height)
        tf = box.text_frame
        tf.word_wrap = True
        first = True
        for para in runs:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.line_spacing = spacing
            if align:
                p.alignment = align
            parts = [(para, {})] if isinstance(para, str) else para
            for chunk, style in parts:
                r = p.add_run()
                r.text = chunk
                r.font.name = style.get("font", font)
                r.font.size = Pt(style.get("size", size))
                r.font.bold = style.get("bold", bold)
                r.font.color.rgb = rgb(style.get("color", color))
        return box

    def ground(slide):
        fill = slide.background.fill
        fill.solid()
        fill.fore_color.rgb = rgb(GROUND)

    def chrome(slide, s: Slide, number: int):
        text(slide, Inches(0.5), Inches(7.0), Inches(6), Inches(0.35),
             [[(s.layer, {"color": DIM, "size": 10, "font": MONO})]])
        tail = f"{number} / {len(SLIDES)}"
        if s.core:
            tail = f"★ 主線 {s.core}/{CORE_TOTAL} · " + tail
        text(slide, Inches(8.8), Inches(7.0), Inches(4.0), Inches(0.35),
             [[(tail, {"color": ACCENT if s.core else DIM, "size": 10, "font": MONO})]],
             align=PP_ALIGN.RIGHT)

    def title_line(slide, s: Slide, size=28):
        text(slide, Inches(0.5), Inches(0.35), Inches(12.3), Inches(1.0),
             [[(s.title, {"bold": True, "size": size})]])

    def bullets(slide, items, left, top, width, height, size=15):
        text(slide, left, top, width, height,
             [[("• ", {"color": ACCENT, "size": size}), (b, {"size": size})] for b in items],
             spacing=1.25)

    def five_box(slide, s: Slide, left, top, width, height):
        rows = list(FIVE_LABELS.items())
        row_h = int(height / len(rows))
        y = top
        for key, label in rows:
            colour = WARN if key == "mistake" else ACCENT
            text(slide, left, y, Inches(1.45), row_h,
                 [[(label, {"bold": True, "color": colour, "size": 12})]])
            body = s.five[key]
            size = 11 if len(body) > 150 else 12.5
            text(slide, left + Inches(1.5), y, width - Inches(1.5), row_h,
                 [[(body, {"size": size})]], spacing=1.1)
            y += row_h

    for number, s in enumerate(SLIDES, 1):
        slide = prs.slides.add_slide(blank)
        ground(slide)
        if s.kind == "title":
            text(slide, Inches(0.8), Inches(2.3), Inches(11.7), Inches(1.4),
                 [[(s.title, {"bold": True, "size": 40})]])
            text(slide, Inches(0.8), Inches(3.9), Inches(11.7), Inches(1.6),
                 [[(b, {"color": DIM, "size": 16})] for b in s.bullets])
        elif s.kind in ("headline", "text", "closing"):
            title_line(slide, s)
            top = Inches(1.5)
            if s.table:
                rows, cols = len(s.table), len(s.table[0])
                row_h = Inches(0.42)
                shape = slide.shapes.add_table(rows, cols, Inches(0.6), top, Inches(12.1), row_h * rows)
                tbl = shape.table
                for r, row in enumerate(s.table):
                    for c, cell_text in enumerate(row):
                        cell = tbl.cell(r, c)
                        cell.text = ""
                        para = cell.text_frame.paragraphs[0]
                        run = para.add_run()
                        run.text = cell_text
                        run.font.name = FONT
                        run.font.size = Pt(10.5 if r else 10)
                        run.font.bold = r == 0
                        run.font.color.rgb = rgb(ACCENT if r == 0 else INK)
                        cell.fill.solid()
                        cell.fill.fore_color.rgb = rgb(PANEL if r % 2 else GROUND)
                        cell.margin_top = cell.margin_bottom = Inches(0.04)
                top = top + row_h * rows + Inches(0.25)
            bullets(slide, s.bullets, Inches(0.6), top, Inches(12.1), H - top - Inches(0.6),
                    size=17 if s.kind == "headline" else (12.5 if s.table else 15))
        elif s.kind in ("image", "video"):
            title_line(slide, s)
            img = images.get(s.image or "")
            if s.kind == "video" and embed_video and VIDEO.exists():
                poster = str(SCREENSHOTS / "region-zh.png")
                slide.shapes.add_movie(str(VIDEO), Inches(0.6), Inches(1.4), Inches(7.4),
                                       Inches(4.2), poster_frame_image=poster,
                                       mime_type="video/mp4")
            elif img:
                pic = slide.shapes.add_picture(str(img), Inches(0.6), Inches(1.4), width=Inches(7.4))
                if pic.height > Inches(4.6):
                    ratio = Inches(4.6) / pic.height
                    pic.height = Inches(4.6)
                    pic.width = Emu(int(pic.width * ratio))
            else:
                text(slide, Inches(0.6), Inches(3.0), Inches(7.4), Inches(1),
                     [[(f"（{s.image} 未產生）", {"color": DIM})]])
            if s.kind == "video" and not embed_video:
                text(slide, Inches(0.6), Inches(5.7), Inches(7.4), Inches(0.5),
                     [[(f"影片：{VIDEO_LINK}", {"color": ACCENT, "size": 11, "font": MONO})]])
            bullets(slide, s.bullets, Inches(8.3), Inches(1.4), Inches(4.6), Inches(4.8), size=12.5)
            text(slide, Inches(0.6), Inches(6.2), Inches(12.2), Inches(0.5),
                 [[(s.caption, {"color": DIM, "size": 10, "font": MONO})]])
        elif s.kind == "five":
            title_line(slide, s, size=24)
            five_box(slide, s, Inches(0.5), Inches(1.35), Inches(12.3), Inches(5.55))
        chrome(slide, s, number)
        slide.notes_slide.notes_text_frame.text = _notes_text(s)

    target.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(target))
    return target


# ------------------------------------------------------------------ html

def build_html(target: Path, images: dict[str, Path]) -> Path:
    def esc(x: object) -> str:
        return html.escape(str(x), quote=True)

    sections = []
    for number, s in enumerate(SLIDES, 1):
        core = (f'<span class="core">★ 主線 {s.core}/{CORE_TOTAL}</span>' if s.core else "")
        body = ""
        if s.kind == "five":
            body = '<dl class="five">' + "".join(
                f'<dt class="{k}">{esc(v)}</dt><dd>{esc(s.five[k])}</dd>'
                for k, v in FIVE_LABELS.items()) + "</dl>"
        if s.table:
            head, *rows = s.table
            body += ('<div class="scroll"><table><thead><tr>' + "".join(f"<th>{esc(h)}</th>" for h in head)
                     + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows)
                     + "</tbody></table></div>")
        if s.bullets:
            body += "<ul>" + "".join(f"<li>{esc(b)}</li>" for b in s.bullets) + "</ul>"
        if s.image and s.image in images:
            body = (f'<figure><img src="img/{esc(s.image)}" alt="{esc(s.caption)}">'
                    f"<figcaption>{esc(s.caption)}</figcaption></figure>") + body
        if s.kind == "video":
            body += (f'<p class="link"><a href="{VIDEO_LINK}">影片（GitHub Release）</a></p>')
        qa = ""
        if s.questions:
            qa = '<div class="qa"><h3>面試官會問</h3>' + "".join(
                f'<div class="q"><p class="question">Q · {esc(q)}</p>'
                f'<p class="answer">{esc(a)}</p></div>' for q, a in s.questions) + "</div>"
        notes = ('<details class="notes"><summary>講稿</summary><p class="plain">'
                 f'{esc(s.plain)}</p><ol>' + "".join(f"<li>{esc(n)}</li>" for n in s.notes)
                 + "</ol></details>")
        sections.append(
            f'<section id="s{number}" data-layer="{esc(s.layer)}">'
            f'<header><span class="layer">{esc(s.layer)}</span>{core}'
            f'<span class="n">{number} / {len(SLIDES)}</span></header>'
            f"<h2>{esc(s.title)}</h2>{body}{qa}{notes}</section>"
        )

    page = f"""<!doctype html>
<html lang="zh-TW">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AOI 複判：一個九天的實驗紀錄</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&family=Noto+Serif+TC:wght@700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{ --ground:#{GROUND}; --panel:#{PANEL}; --ink:#{INK}; --dim:#{DIM}; --accent:#{ACCENT}; --warn:#{WARN}; --rule:#33383D; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink); font-family:"Noto Sans TC",system-ui,sans-serif; line-height:1.6; }}
section {{ min-height:100vh; padding:2.2rem 1.4rem 3rem; max-width:1100px; margin:0 auto; border-bottom:1px solid var(--rule); scroll-margin-top:0; }}
header {{ display:flex; gap:1rem; align-items:baseline; font-family:"IBM Plex Mono",monospace; font-size:.78rem; color:var(--dim); }}
header .core {{ color:var(--accent); }} header .n {{ margin-left:auto; }}
h2 {{ font-family:"Noto Serif TC",serif; font-size:1.6rem; line-height:1.3; margin:.6rem 0 1rem; text-wrap:balance; }}
h3 {{ font-size:.95rem; color:var(--dim); margin:1.4rem 0 .4rem; letter-spacing:.04em; }}
ul {{ padding-left:1.2rem; }} li {{ margin:.35rem 0; max-width:80ch; }}
dl.five {{ display:grid; grid-template-columns:7rem 1fr; gap:.5rem 1rem; margin:0 0 1rem; }}
dl.five dt {{ color:var(--accent); font-weight:700; font-size:.9rem; padding-top:.15rem; }}
dl.five dt.mistake {{ color:var(--warn); }}
dl.five dd {{ margin:0; max-width:85ch; }}
.scroll {{ overflow-x:auto; margin:0 0 1rem; }}
table {{ border-collapse:collapse; width:100%; font-size:.9rem; }}
th {{ text-align:left; color:var(--accent); font-weight:600; font-size:.8rem; padding:.45rem .6rem; border-bottom:1px solid var(--rule); }}
td {{ padding:.45rem .6rem; border-bottom:1px solid var(--rule); vertical-align:top; }}
figure {{ margin:0 0 1rem; background:var(--panel); border:1px solid var(--rule); border-radius:6px; padding:.6rem; }}
figure img {{ max-width:100%; height:auto; display:block; }}
figcaption {{ font-family:"IBM Plex Mono",monospace; font-size:.72rem; color:var(--dim); margin-top:.4rem; }}
.qa {{ background:var(--panel); border:1px solid var(--rule); border-radius:6px; padding:.2rem 1rem .8rem; margin-top:1.2rem; }}
.q {{ margin:.7rem 0; }} .question {{ margin:0; font-weight:500; }} .answer {{ margin:.2rem 0 0; color:var(--ink); border-left:3px solid var(--accent); padding-left:.7rem; max-width:80ch; }}
details.notes {{ margin-top:1rem; color:var(--dim); font-size:.92rem; }} details.notes summary {{ cursor:pointer; }}
.plain {{ color:var(--ink); font-weight:500; }}
.link a {{ color:var(--accent); }}
body.quiz dl.five, body.quiz ul, body.quiz figure, body.quiz details.notes {{ display:none; }}
body.quiz .answer {{ visibility:hidden; }} body.quiz.reveal .answer {{ visibility:visible; }}
#hud {{ position:fixed; right:.8rem; bottom:.8rem; font-family:"IBM Plex Mono",monospace; font-size:.72rem; color:var(--dim); background:var(--panel); border:1px solid var(--rule); border-radius:4px; padding:.3rem .6rem; }}
@media (prefers-reduced-motion: reduce) {{ html {{ scroll-behavior:auto; }} }}
</style>
</head>
<body>
{"".join(sections)}
<div id="hud">← → 翻頁 · q 自測 · a 看答案 · n 講稿</div>
<script>
(function () {{
  const secs = [...document.querySelectorAll('section')];
  let at = 0;
  function go(i) {{ at = Math.max(0, Math.min(secs.length - 1, i)); secs[at].scrollIntoView({{behavior:'smooth'}}); }}
  document.addEventListener('keydown', function (e) {{
    if (e.key === 'ArrowRight' || e.key === ' ') {{ e.preventDefault(); go(at + 1); }}
    else if (e.key === 'ArrowLeft') {{ e.preventDefault(); go(at - 1); }}
    else if (e.key === 'q') {{ document.body.classList.toggle('quiz'); document.body.classList.remove('reveal'); }}
    else if (e.key === 'a') {{ document.body.classList.toggle('reveal'); }}
    else if (e.key === 'n') {{ document.querySelectorAll('details.notes').forEach(d => d.open = !d.open); }}
  }});
  const io = new IntersectionObserver(es => es.forEach(x => {{ if (x.isIntersecting) at = secs.indexOf(x.target); }}), {{threshold: 0.5}});
  secs.forEach(s => io.observe(s));
}})();
</script>
</body>
</html>
"""
    target.write_text(page)
    return target


# ---------------------------------------------------------- study guide

def build_study_guide(target: Path) -> Path:
    lines = ["# AOI 複判專案歷程：問答手冊", "",
             "從 `scripts/deck_content.py` 產生；和投影片、網頁版同一份內容。",
             "每一頁：一句白話 → 為什麼做／怎麼設計／量到什麼／錯在哪／規則 → 面試官會問 → 講稿。", ""]
    core = [s.title for s in sorted((s for s in SLIDES if s.core), key=lambda s: s.core)]
    lines += ["## 十分鐘主線（9 頁）", ""] + [f"{i}. {t}" for i, t in enumerate(core, 1)] + [""]
    for layer, slides in by_layer().items():
        if not slides:
            continue
        lines += [f"## {layer}", ""]
        for s in slides:
            lines += [f"### {s.title}", "", f"**白話：**{s.plain}", ""]
            if s.five:
                for k, label in FIVE_LABELS.items():
                    lines += [f"- **{label}：**{s.five[k]}"]
                lines += [""]
            if s.table:
                head, *rows = s.table
                lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                lines += ["| " + " | ".join(r) + " |" for r in rows] + [""]
            if s.bullets:
                lines += [f"- {b}" for b in s.bullets] + [""]
            if s.questions:
                lines += ["**面試官會問：**", ""]
                for q, a in s.questions:
                    lines += [f"- Q：{q}", f"  - A：{a}"]
                lines += [""]
            lines += ["**講稿：**", ""] + [f"{i}. {n}" for i, n in enumerate(s.notes, 1)] + [""]
    target.write_text("\n".join(lines))
    return target


# ------------------------------------------------------------------ main

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-video", action="store_true",
                        help=f"also write {EMBEDDED_OUT.relative_to(ROOT)} with the demo embedded")
    parser.add_argument("--no-images", action="store_true", help="reuse docs/deck/img as it is")
    args = parser.parse_args()

    try:
        import pptx  # noqa: F401
    except ImportError:
        print("python-pptx is not installed. Run:\n"
              "  uv run --with python-pptx --with playwright python scripts/build_deck.py")
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    if args.no_images:
        images = {p.name: p for p in IMG.glob("*.png")}
    else:
        images = render_images()
    for name, path in images.items():
        print(f"image  {path.relative_to(ROOT)}  {path.stat().st_size // 1024} KB")

    pptx_path = build_pptx(OUT / "aoi-agent-journey.zh-TW.pptx", images, embed_video=False)
    print(f"pptx   {pptx_path.relative_to(ROOT)}  {pptx_path.stat().st_size // 1024} KB")
    html_path = build_html(OUT / "aoi-agent-journey.zh-TW.html", images)
    print(f"html   {html_path.relative_to(ROOT)}  {html_path.stat().st_size // 1024} KB")
    guide = build_study_guide(OUT / "study-guide.zh-TW.md")
    print(f"guide  {guide.relative_to(ROOT)}")
    if args.embed_video:
        if not VIDEO.exists():
            print(f"{VIDEO.relative_to(ROOT)} not found: no embedded deck written")
        else:
            EMBEDDED_OUT.parent.mkdir(parents=True, exist_ok=True)
            build_pptx(EMBEDDED_OUT, images, embed_video=True)
            print(f"pptx   {EMBEDDED_OUT.relative_to(ROOT)}  (video embedded, not committed)")
    summary = {"slides": len(SLIDES), "core": CORE_TOTAL,
               "layers": [(name, len(slides)) for name, slides in by_layer().items()]}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
