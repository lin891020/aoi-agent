"""The demo's shot list: what is on screen, what is framed, and what is said.

One entry per *cue*, not per scene. Until 2026-09-05 the recorder held one
line per scene -- a paragraph of forty Chinese characters burned across the
bottom of the page for fourteen seconds -- and the person watching it could
read the subtitle or the page, not both. A cue is one sentence, wrapped to
at most two lines of the length a subtitle guide allows, shown in a band
*below* the page rather than over it, and it may frame one element on the
page while it is up.

The limits are Netflix's timed-text style guides, which are the only
published numbers for both languages this project renders: Traditional
Chinese 16 characters a line and 9 characters a second; English 42
characters a line and 20 a second. ``tests/test_demo_script.py`` holds every
cue to them, and holds the two languages to the same scenes and the same
number of cues, so a take in either language frames the same things.

The Chinese is written as Chinese, in the register of a line engineer's spoken
briefing, not compressed from the English: Mike read the first draft and
found 判掉, 收不掉, 進預算 -- verbs chosen to fit sixteen characters.

The audience is somebody who has not seen the system. So the first cues say
what an AOI is for and what this thing does about it before any page is
shown, and every page gets a sentence naming what it is before a sentence
about what is on it.

``python scripts/demo_script.py`` writes docs/demo-script.md from this table;
the recorder reads the table directly, so the document a person confirms
and the take the recorder makes cannot disagree.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

#: Netflix Timed Text Style Guide, Chinese (Traditional) and English (USA).
ZH_LINE, EN_LINE = 16, 42
ZH_CPS, EN_CPS = 9.0, 20.0
#: What a person dubbing the silent cut reads at -- slower than the cap, so
#: the hold leaves room to breathe. Chinese in characters, English in words.
ZH_PACE, EN_PACE = 4.5, 2.6
MIN_HOLD = 2.5

ZH_BREAKS = "，。；：、！？"


@dataclass(frozen=True)
class Cue:
    zh: str
    en: str
    #: CSS selector of the one element to frame while the cue is up. ``first:``
    #: takes the first of several; ``sel@@text`` the one whose text contains
    #: ``text`` (``{region}`` is filled with the take's region reference).
    spot: str | None = None
    #: When in its scene the cue is said. ``type``: the recorder types or
    #: clicks while it is up; ``before``: before the scene's model call;
    #: ``wait``: while the model works -- the progress panel is what is on
    #: screen, and whatever wait is left after these is fast-forwarded in the
    #: cut; ``after``: once the result is up.
    phase: str = "after"


PHASES = ("type", "before", "wait", "after")


#: In order. The key is the scene the recorder drives; the cues are said over
#: it, in order, each with its own hold.
SCENES: list[tuple[str, list[Cue]]] = [
    ("intro", [
        Cue("AOI：自動光學檢測，Automated Optical Inspection。", "AOI: Automated Optical Inspection.", "#f-aoi"),
        Cue("AOI 是產線上拍照找瑕疵的機器；設定上寧可誤報，也不漏檢。",
            "It is the camera that checks boards; it over-flags rather than miss a defect.", "#f-aoi"),
        Cue("AOI 標出的區域，六成是誤報，卻每一個都要人工複判。",
            "Six in ten flagged regions are false calls, and every one goes to a person.", "#f-lead"),
        Cue("本系統接在既有的 AOI 之後。", "It sits behind the AOI the line already has.", "#f-system"),
        Cue("視覺模型先判定。", "A vision model decides first.", "#f-model"),
        Cue("沒把握的，由 agent 接手。", "What it is unsure of goes to an agent.", "#f-agent"),
        Cue("兩者都無法判定的區域，才交給作業員。", "Only what neither can settle reaches an operator.", "#f-operator"),
        Cue("作業員的判定會記錄下來，作為下一輪訓練的標註資料。",
            "The operator's verdict is recorded, as the label for the next training round.", "#f-label"),
        Cue("主管也可直接用中文，詢問產線的問題。", "A supervisor can ask the line a question in plain words.", "#f-ask"),
    ]),
    ("cli", [
        Cue("這是工程師看的執行紀錄：一片 PCB，每個區域一行。",
            "The engineer's log of the run: one PCB, one line per flagged region.", "#hdr"),
        Cue("多數區域模型幾毫秒內就判定完成；信心不足的兩個，交給作業員。",
            "Most the model settles in milliseconds; the two it is unsure of go to an operator.", "#queued"),
        Cue("整片 PCB 先扣留，待複判完成再處置。", "The whole board is held until they answer.", "#board"),
    ]),
    ("login", [
        Cue("作業員登入後，每一筆判定都記錄在其名下。",
            "An operator signs in. Every verdict from here on carries this name.", "form.signin-form"),
    ]),
    ("home", [
        Cue("登入後先看到主畫面：這條產線目前的狀態。",
            "The first screen is the line as it stands right now.", ".home h1"),
        Cue("已檢驗的 PCB：已處置、扣留、放行、待判各幾片。",
            "The PCBs that ran: dispositioned, held, released, still waiting.", ".home .stats:first-of-type"),
        Cue("放行為 0 並非異常：示範資料每片都含真實瑕疵。",
            "Released 0 is not a fault: every demo board really carries a defect.", ".home .stats:first-of-type"),
        Cue("以及待人工複判、待資深複判的區域各有幾個。",
            "How many regions wait on a person, and how many went back to a senior.", ".home .stats:last-of-type"),
        Cue("主畫面先呈現整條線的狀態；待複判清單在下一層。",
            "The front page shows the line first; the review queue is one level in.", ".home .doors"),
    ]),
    ("queue", [
        Cue("無法判定的區域都列在這裡；等候最久的排最前。",
            "Every region the agent could not settle is here, longest wait on top.", "table.queue"),
        Cue("每一列一個區域：判定類別、信心值、誤判機率。",
            "Each row: the model's call, how sure it is, how likely a false call.",
            "table.queue tbody tr:first-child"),
        Cue("右側是 agent 的說明；agent 只解釋，不判定。",
            "On the right, the agent's rationale. The agent explains; it never decides.",
            "table.queue tbody tr:first-child td.reason"),
        Cue("實測過：改由 LLM 決定，準確度比分類模型低。",
            "Measured: letting the LLM decide was less accurate than the classifier.",
            "table.queue tbody tr:first-child td.reason"),
    ]),
    ("region", [
        Cue("開啟其中一個區域。", "Open one region.", ".station .head"),
        Cue("最上方：agent 交付人工複判的理由。", "At the top: why the agent handed it over.", ".handover"),
        Cue("golden image、待測 PCB、差異圖。", "Golden image, PCB under test, difference.", "figure.triptych"),
        Cue("左側三張影像，紅框是 AOI 標出的位置。",
            "Three images on the left; the red box is where the AOI flagged.", "figure.triptych"),
        Cue("模型的判讀：類別、信心值，及實際分析的 64 px 小圖。",
            "What the model read: class, confidence, and the 64 px window it actually saw.",
            ".columns .col:nth-child(1)"),
        Cue("這台機台最近的缺陷率（複判後），與其他機台比較。",
            "This machine's recent defect rate after re-verification, against the fleet.",
            ".columns .col:nth-child(2)"),
        Cue("此瑕疵類別的驗收標準，只取該類別的文件。",
            "The acceptance criteria for this class, and only this class.", ".columns .col:nth-child(3)"),
        Cue("作業員以數字鍵輸入判定。", "The operator answers with a number key.", "form.verdict"),
        Cue("資料集附有標準答案，此頁刻意不顯示。",
            "The dataset comes with an answer key; this page deliberately hides it.", "form.verdict p.sub.dim"),
        Cue("若先看到答案，判定就不再是獨立的判斷。",
            "Seen first, the answer would replace the operator's own judgement.", "form.verdict p.sub.dim"),
    ]),
    ("defer", [
        Cue("無法判斷時，按 0。", "Can't tell? Press zero.", "form.defer .defer-button"),
        Cue("這不會記錄為判定；區域改列入待資深複判。",
            "It is not recorded as a verdict. The region moves to the senior list.",
            "table.queue tbody tr@@{region}"),
    ]),
    ("blocked", [
        Cue("改以一般作業員登入，開啟同一個區域。",
            "Sign in as an ordinary operator and open the same region.", ".station .head"),
        Cue("判定按鈕消失，只剩一行說明。", "The buttons are gone; one sentence remains.",
            "section.station > p.sub.warn-line"),
        Cue("退回的區域只有資深人員能判定；這是示範站唯一的權限設定。",
            "Handed-back regions are for seniors; this is the demo's one permission.",
            "section.station > p.sub.warn-line"),
    ]),
    ("boards", [
        Cue("PCB 處置紀錄：已處置、扣留、放行、待判。",
            "PCB dispositions: dispositioned, held, released, waiting.", "section.panel > p:nth-of-type(2)"),
        Cue("上方的統計是整個資料表的合計，不是本頁的列數。",
            "These counts are totals over the whole table, not the rows on this page.",
            "section.panel > p:nth-of-type(2)"),
        Cue("每片 PCB：處置結果、區域數、判定者、時間。",
            "Each PCB: its disposition, how many regions, who decided, and when.",
            "table.queue tbody tr:first-child"),
    ]),
    ("ask", [
        Cue("主管的問題，直接用中文問。", "A supervisor's question, typed in plain language.", "form.ask", "type"),
        Cue("系統先將問題轉成查詢計畫，驗證後才執行。",
            "The question becomes a plan of lookups, validated before anything runs.", "#progress", "wait"),
        Cue("規劃：由模型決定查詢哪些工具、帶哪些參數。",
            "Planning: the model picks which tools to call, and with what.", "#progress", "wait"),
        Cue("驗證：參數不在資料庫的實際值內，即拒絕。",
            "Validation: an argument the store does not hold is refused.", "#progress", "wait"),
        Cue("查詢僅需幾毫秒；其餘等待時間是模型在撰寫回答。",
            "The lookups return in milliseconds; the rest of the wait is the model writing.", "#progress", "wait"),
        Cue("前後兩根長條，信賴區間不重疊：參數變更後 open 佔比下降。",
            "Two bars, before and after, intervals apart: the share of opens fell.", "figure.chart"),
        Cue("圖表依結果的形狀自動產生，不是由模型挑選。",
            "The chart is derived from the shape of the results, not chosen by the model.", "figure.chart"),
        Cue("說明文字緊鄰數據，每個數字都能對回查詢結果。",
            "The prose sits beside the numbers; every figure in it is checked against them.",
            "figure.chart + .answer-block"),
    ]),
    ("control", [
        Cue("再問一個對照組：M31 換燈前後。", "Now a control: the lamp replacement on M31.", "form.ask", "type"),
        Cue("同一套流程，換一台機台、換一個事件。",
            "The same flow, on another machine and another event.", "#progress", "wait"),
        Cue("兩個信賴區間重疊，系統即回答沒有顯著差異。",
            "The intervals overlap, and the system says so.", "figure.chart"),
        Cue("有事件，不代表有影響。", "An event is not an effect.", "figure.chart"),
    ]),
    ("switch", [
        Cue("切換語言。", "Switch the language.", "nav.locale a", "type"),
        Cue("問題與規劃段保留原文，並加上標示。",
            "The question and the plan stay as written, and are labelled.",
            "first:.answer-block h2 .as-asked", "before"),
        Cue("回答不會自動改寫，需按下按鈕才重寫。",
            "The answer does not change by itself; it is written again on request.", "form.rewrite", "before"),
        Cue("重寫走的是同一條實測過的流程，因此需再等模型一次。",
            "The rewrite takes the same measured path, so it waits on the model once.", "form.rewrite", "wait"),
        Cue("以儲存的同一批結果重新撰寫；不是翻譯，原文仍保留。",
            "Written again from the stored results, not translated; the original is kept.",
            "figure.chart + .answer-block"),
    ]),
    ("outro", [
        Cue("人工複判的工作量，減少 55.6%。", "Manual review work down by 55.6%."),
        Cue("漏檢率 0.66%，高於 0.5% 的預算。", "Escape 0.66%, above the 0.5% budget."),
        Cue("漏掉的開路，電性測試可再攔截；其餘在預算內。",
            "The escaped opens are caught by electrical test; the rest is within budget."),
        Cue("README 如實記載：信賴區間、各類別、五組種子。",
            "The README records the interval, each class, and five seeds."),
        Cue("每個門檻都能追溯到產生它的腳本。", "Every threshold traces back to the script that produced it."),
        Cue("每個數字，都在 benchmarks 裡。", "Every figure is in the benchmarks file."),
    ]),
]

CUES: list[tuple[str, int, Cue]] = [(key, i, cue) for key, cues in SCENES for i, cue in enumerate(cues)]


def cue_id(key: str, i: int) -> str:
    return f"{key}.{i}"


def cues_in(key: str, phase: str) -> list[str]:
    """The ids of one scene's cues in one phase, in order."""
    return [cue_id(k, i) for k, i, c in CUES if k == key and c.phase == phase]


def text(cue: Cue, lang: str) -> str:
    return cue.zh if lang == "zh-TW" else cue.en


def hold(cue: Cue, lang: str) -> float:
    """Seconds the silent cut keeps this cue up: long enough to say it at a
    dubbing pace, never shorter than a subtitle's minimum on screen."""
    t = text(cue, lang)
    # A Latin word in a Chinese line is said in about half its letters'
    # worth of characters; a digit is a syllable and counts whole -- the
    # subtitle guide's half-width rule (``width``) is for the band, not the
    # voice, and applied to the hold it left 0.66% with less time than the
    # voice took to say it.
    units = sum(0.5 if ch.isascii() and ch.isalpha() else 1 for ch in t) if lang == "zh-TW" else len(t.split())
    pace = ZH_PACE if lang == "zh-TW" else EN_PACE
    return max(MIN_HOLD, round(units / pace + 0.6, 1))


def width(t: str, lang: str) -> float:
    """Characters as the guide counts them. Chinese counts a full-width
    character as one and a half-width one (Latin, digits, ASCII marks) as a
    half, which is how ``Automated Optical Inspection`` fits beside its
    Chinese name; English counts every character."""
    if lang != "zh-TW":
        return float(len(t))
    return sum(0.5 if ord(ch) < 0x2E80 else 1.0 for ch in t)


def lines(cue: Cue, lang: str) -> list[str]:
    """The cue wrapped for the subtitle band: one line if it fits, else the
    two-line split at a punctuation mark that leaves both halves under the
    limit -- the shortest longer half wins. A cue no split can fit is a cue
    to rewrite, and the test says so."""
    t = text(cue, lang)
    if lang == "zh-TW":
        w = lambda part: width(part, lang)  # noqa: E731 -- the guide's count, used four times below
        if w(t) <= ZH_LINE:
            return [t]
        cuts = [m.end() for m in re.finditer(f"[{ZH_BREAKS}]", t) if 0 < m.end() < len(t)]
        fits = [c for c in cuts if w(t[:c]) <= ZH_LINE and w(t[c:]) <= ZH_LINE]
        cut = min(fits, key=lambda c: max(w(t[:c]), w(t[c:]))) if fits else ZH_LINE
        return [t[:cut], t[cut:]]
    if len(t) <= EN_LINE:
        return [t]
    cuts = [m.start() for m in re.finditer(" ", t)]
    fits = [c for c in cuts if len(t[:c]) <= EN_LINE and len(t[c + 1:]) <= EN_LINE]
    if fits:
        cut = min(fits, key=lambda c: max(len(t[:c]), len(t[c + 1:])))
        return [t[:cut], t[cut + 1:]]
    return [t[i:i + EN_LINE] for i in range(0, len(t), EN_LINE)]


def shot_list() -> str:
    rows = []
    for key, cues in SCENES:
        for i, cue in enumerate(cues):
            when = {"type": "（邊打字）", "before": "", "wait": "（等模型時）", "after": ""}[cue.phase]
            rows.append(f"| {cue_id(key, i)} | {hold(cue, 'zh-TW'):.1f} s | `{cue.spot or ''}`{when} | {cue.zh} | {cue.en} |")
    zh_total = sum(hold(c, "zh-TW") for _, _, c in CUES)
    return (
        "# 示範影片分鏡\n\n"
        "<!-- generated by scripts/demo_script.py; edit the table there, not here -->\n\n"
        f"{len(SCENES)} 幕、{len(CUES)} 句。每一句是一段字幕（中文最多 16 字一行、兩行；英文 42 字元一行），"
        "顯示在頁面**下方的黑帶**裡，不蓋畫面；「框」那一欄是字幕上的時候框起來的元素。"
        f"無聲版每句停留「秒」欄那麼久（中文 {ZH_PACE:g} 字/秒的配音速度），合計約 {zh_total/60:.1f} 分鐘，"
        "還沒算模型回答的等待——等待中先講「（等模型時）」那幾句，剩下的等待在成片裡加速（預設 3 倍，"
        "畫面幾乎不動，右上角標 >> 3x），所以成片比錄的短。\n\n"
        "錄：`uv run --with playwright python scripts/demo_record.py --lang zh-TW --stem 00041208 --index 15 --silent "
        "--base http://127.0.0.1:8111`（先用 `--check` 走一遍每一頁，確認要框的元素都在）。"
        "無聲版帶字幕、沒有聲音；旁邊的 `docs/demo/build/<lang>/script.md` 是同一份台詞加上每句的秒數，配音對著它講。"
        "確認台詞之後 `--narrate-existing` 把旁白（video_transfer 的 Qwen3-TTS／Kokoro）配上去，每句一段、聽回來檢查。\n\n"
        "每次錄影會把示範用的區域按 0 退回；`00041208` 目前待判的是 #15。\n\n"
        "| 句 | 秒 | 框 | 中文 | English |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
        "## 畫面（錄影器做的事）\n\n"
        "| 幕 | 畫面 |\n|---|---|\n"
        "| intro | 片頭卡：標題、一句問題、處置流程圖 |\n"
        "| cli | 終端機：`board 00041208 --queue` 的輸出，只留開頭三個區域、三個進佇列的、最後的 HELD 一行 |\n"
        "| login | `/login` 停 1 秒，打字登入 `mike`（資深） |\n"
        "| home | `/` 主畫面 |\n"
        "| queue | `/queue`，捲一點 |\n"
        "| region | `/c/00041208/<index>`：三聯圖 → 捲到三欄證據 → 捲到判定按鈕 |\n"
        "| defer | 同頁按 `0` → `/deferred` |\n"
        "| blocked | 登出、`watcher`（一般作業員）登入、同一區域、捲到底 |\n"
        "| boards | 再以 `mike` 登入、`/boards` |\n"
        "| ask | `/ask` 打 M32 的問題、等進度面板、等圖 |\n"
        "| control | 再問 M31 換燈、等圖 |\n"
        "| switch | 按另一個語言 → 按「重寫」→ 等新答案 |\n"
        "| outro | 片尾卡：三個數字、repo 網址 |\n"
    )


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "docs" / "demo-script.md"
    out.write_text(shot_list())
    print("wrote", out, f"({len(CUES)} cues)", file=sys.stderr)
