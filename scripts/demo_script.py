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
        Cue("產線的 AOI 為了不漏檢，寧可多報。",
            "A PCB line's AOI would rather over-flag than miss a defect."),
        Cue("標出來的區域，六成是誤報，每一個都要人看。",
            "Six in ten flagged regions are false calls, and every one goes to a person."),
        Cue("這個系統：人前面放一個視覺模型，後面放一個 agent。",
            "This system puts a vision model in front of that person, and an agent behind it."),
        Cue("兩個都收不掉的，才交給人。",
            "Only what neither can settle reaches a person."),
    ]),
    ("cli", [
        Cue("一片 PCB 進來，AOI 標了 30 個區域。",
            "One PCB comes in. The AOI flagged thirty regions.", "#hdr"),
        Cue("一個區域一行，大多數模型幾毫秒就判掉。",
            "One line per region; most of them the model settles in milliseconds.", "#first"),
        Cue("兩個它不敢判的，進佇列，交給人。",
            "The two it was not sure about go to the queue, to a person.", "#queued"),
        Cue("整片 PCB 先扣留，等人判完。",
            "The whole board is held until they answer.", "#board"),
    ]),
    ("login", [
        Cue("作業員登入。之後每個判定都記在這個名字下。",
            "An operator signs in. Every verdict from here on carries this name.", "form.signin-form"),
    ]),
    ("home", [
        Cue("登入後第一眼是主畫面：這條線現在的狀態。",
            "The first screen is the line as it stands right now.", ".home h1"),
        Cue("跑過的 PCB：幾片已處置、扣留、放行、待判。",
            "The PCBs that ran: dispositioned, held, released, still waiting.", ".home .stats:first-of-type"),
        Cue("幾個區域還在等人看，幾個被退回給資深。",
            "How many regions wait on a person, and how many went back to a senior.", ".home .stats:last-of-type"),
        Cue("失敗清單不在第一眼，要點進去才是。",
            "The list of failures is one click in, not the front door.", ".home .doors"),
    ]),
    ("queue", [
        Cue("agent 收不掉的區域都在這裡，等最久的排最前。",
            "Every region the agent could not settle is here, longest wait on top.", "table.queue"),
        Cue("每一列：模型的判定、信心、誤判機率。",
            "Each row: the model's class, its confidence, the false-call probability.",
            "table.queue tbody tr:first-child"),
        Cue("右邊是 agent 寫的說明。agent 只解釋，不決定。",
            "On the right, the agent's rationale. The agent explains; it never decides.",
            "table.queue tbody tr:first-child td.reason"),
    ]),
    ("region", [
        Cue("點進一個區域。", "Open one region.", ".station .head"),
        Cue("最上面：agent 為什麼把它交給人。", "At the top: why the agent handed it over.", ".handover"),
        Cue("golden image、待測 PCB、差異圖。", "Golden image, PCB under test, difference.", "figure.triptych"),
        Cue("左邊三張圖，紅框是 AOI 標的位置。",
            "Three images on the left; the red box is where the AOI flagged.", "figure.triptych"),
        Cue("模型讀到什麼：類別、信心，和它實際看的 64 px 小圖。",
            "What the model read: class, confidence, and the 64 px window it actually saw.",
            ".columns .col:nth-child(1)"),
        Cue("這台機器最近的缺陷率，對比全廠。",
            "This machine's recent defect rate, against the fleet.", ".columns .col:nth-child(2)"),
        Cue("這一類的驗收標準，只給這一類的。",
            "The acceptance criteria for this class, and only this class.", ".columns .col:nth-child(3)"),
        Cue("作業員按數字鍵判定。", "The operator answers with a number key.", "form.verdict"),
        Cue("這一頁沒有答案：按下去的，就是下一輪訓練的標籤。",
            "No answer key on this page: what gets pressed is the next training label.",
            "form.verdict p.sub.dim"),
    ]),
    ("defer", [
        Cue("真的看不出來？按 0。", "Can't tell? Press zero.", "form.defer .defer-button"),
        Cue("它不會被記成判定。區域換到待資深複判。",
            "It is not recorded as a verdict. The region moves to the senior list.",
            "table.queue tbody tr@@{region}"),
    ]),
    ("blocked", [
        Cue("換一般作業員登入，開同一個區域。",
            "Sign in as an ordinary operator and open the same region.", ".station .head"),
        Cue("按鈕不見了，只剩一句話。", "The buttons are gone; one sentence remains.",
            "section.station > p.sub.warn-line"),
        Cue("退回的區域只有資深能答，這是整個站唯一的權限。",
            "Handed-back regions are for seniors only. That is the station's one permission.",
            "section.station > p.sub.warn-line"),
    ]),
    ("boards", [
        Cue("PCB 處置紀錄：已處置、扣留、放行、待判。",
            "PCB dispositions: dispositioned, held, released, waiting.", "section.panel > p:nth-of-type(2)"),
        Cue("數字是對整張表數的，不是對這一頁。",
            "The counts are over the whole table, not over this page.", "section.panel > p:nth-of-type(2)"),
        Cue("每一片：怎麼處置、幾個區域、誰判的、什麼時候。",
            "Each PCB: its disposition, how many regions, who decided, and when.",
            "table.queue tbody tr:first-child"),
    ]),
    ("ask", [
        Cue("主管的問題，直接用中文問。", "A supervisor's question, typed in plain language.", "form.ask", "type"),
        Cue("系統先把問題變成一份查詢計畫，驗證過才跑。",
            "The question becomes a plan of lookups, validated before anything runs.", "#progress", "wait"),
        Cue("規劃：模型挑要查哪些工具、帶什麼參數。",
            "Planning: the model picks which tools to call, and with what.", "#progress", "wait"),
        Cue("驗證：參數對不上資料庫真有的值，就拒絕。",
            "Validation: an argument the store does not hold is refused.", "#progress", "wait"),
        Cue("查詢幾毫秒就回來；剩下的等待，是模型在寫答案。",
            "The lookups return in milliseconds; the rest of the wait is the model writing.", "#progress", "wait"),
        Cue("前後兩根柱，區間沒重疊：參數變更後 open 比例掉了。",
            "Two bars, before and after, intervals apart: the share of opens fell.", "figure.chart"),
        Cue("圖是從結果的形狀畫的，不是模型挑的。",
            "The chart is derived from the shape of the results, not chosen by the model.", "figure.chart"),
        Cue("文字寫在數字旁邊，每個數字都對得到結果。",
            "The prose sits beside the numbers; every figure in it is checked against them.",
            "figure.chart + .answer-block"),
    ]),
    ("control", [
        Cue("再問一個對照組：M31 換燈前後。", "Now a control: the lamp replacement on M31.", "form.ask", "type"),
        Cue("同一份流程，換一台機器、換一個事件。",
            "The same flow, on another machine and another event.", "#progress", "wait"),
        Cue("兩根區間重疊，系統就直接說沒差。", "The intervals overlap, and the system says so.", "figure.chart"),
        Cue("有事件，不代表有影響。", "An event is not an effect.", "figure.chart"),
    ]),
    ("switch", [
        Cue("切換語言。", "Switch the language.", "nav.locale a", "type"),
        Cue("問題和規劃段保留原文，標示出來。",
            "The question and the plan stay as written, and are labelled.",
            "first:.answer-block h2 .as-asked", "before"),
        Cue("答案不會自己變，要按一下才重寫。",
            "The answer does not change by itself; it is written again on request.", "form.rewrite", "before"),
        Cue("重寫走的是同一條量過的路，所以要等模型一次。",
            "The rewrite takes the same measured path, so it waits on the model once.", "form.rewrite", "wait"),
        Cue("用儲存的同一批結果再寫一次：不是翻譯，原文也留著。",
            "Written again from the stored results, not translated; the original is kept.",
            "figure.chart + .answer-block"),
    ]),
    ("outro", [
        Cue("省掉 55.6% 的人工複判。", "55.6% of manual review removed."),
        Cue("漏檢 0.66%，超過千分之五的預算。", "0.66% escape, over the 0.5% budget."),
        Cue("README 就這樣寫：區間、分類別、五個種子、失效條件。",
            "The README says so: the interval, per class, five seeds, where it stops working."),
        Cue("每個門檻都引得到一支腳本。", "Every threshold cites a script."),
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
    units = len(t) if lang == "zh-TW" else len(t.split())
    pace = ZH_PACE if lang == "zh-TW" else EN_PACE
    return max(MIN_HOLD, round(units / pace + 0.6, 1))


def lines(cue: Cue, lang: str) -> list[str]:
    """The cue wrapped for the subtitle band: one line if it fits, else the
    two-line split at a punctuation mark that leaves both halves under the
    limit -- the shortest longer half wins. A cue no split can fit is a cue
    to rewrite, and the test says so."""
    t = text(cue, lang)
    if lang == "zh-TW":
        if len(t) <= ZH_LINE:
            return [t]
        cuts = [m.end() for m in re.finditer(f"[{ZH_BREAKS}]", t) if 0 < m.end() < len(t)]
        fits = [c for c in cuts if len(t[:c]) <= ZH_LINE and len(t[c:]) <= ZH_LINE]
        cut = min(fits, key=lambda c: max(len(t[:c]), len(t[c:]))) if fits else ZH_LINE
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
