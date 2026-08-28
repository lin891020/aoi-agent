"""Record the demo video: Playwright drives the station, `say` narrates, ffmpeg muxes.

    uv run --with playwright python scripts/demo_record.py --lang zh-TW --stem 00041208
    uv run --with playwright python scripts/demo_record.py --lang en    --stem 00041208

Needs the station on :8110, Ollama with gpt-oss:20b, operators `mike` (senior,
passphrase in AOI_DEMO_SENIOR_SECRET) and `watcher` (operator, in
AOI_DEMO_OPERATOR_SECRET), and a board whose region ``<stem>#8`` is on the queue.
The board is run through the CLI first so the terminal scene shows real output.
Everything lands under docs/demo/ (gitignored): the intermediate webm, narration
and subtitles under build/<lang>/, and aoi-agent-demo-<lang>.mp4 beside them.

macOS only: narration comes from `say` (Meijia for zh-TW, Samantha for en) and
the mux uses the Homebrew ffmpeg-full build for its subtitle filter.
"""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

_args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_args.add_argument("--lang", default="zh-TW", choices=("zh-TW", "en"))
_args.add_argument("--stem", required=True, help="a board whose region #8 is on the queue")
_args.add_argument("--base", default="http://127.0.0.1:8110")
ARGS = _args.parse_args()

LANG = ARGS.lang
BASE = ARGS.base
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo" / "build" / LANG
NARR = OUT / "narration"
STEM = ARGS.stem
REGION = f"{STEM}#8"
SENIOR = ("mike", os.environ.get("AOI_DEMO_SENIOR_SECRET", ""))
OPERATOR = ("watcher", os.environ.get("AOI_DEMO_OPERATOR_SECRET", ""))
FFMPEG = "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"
VOICE = {"zh-TW": "Meijia", "en": "Samantha"}[LANG]
TAG = "zh" if LANG == "zh-TW" else "en"

Q_M32 = {"zh-TW": "M32 參數變更前後，open 的比例有沒有變？",
         "en": "Did the parameter change on M32 move its share of opens?"}[LANG]
Q_M31 = {"zh-TW": "M31 換燈前後，open 的比例有沒有變？",
         "en": "Did the lamp replacement on M31 move its share of opens?"}[LANG]

SCENES = {
 "zh-TW": [
  ("cli",      f"一片板進來。AOI 標了三十個區域；視覺模型在幾毫秒內排除了二十八個，剩下兩個它不敢判，交給人。"),
  ("queue",    "這是等人看的清單。每一列有模型的判定、信心、誤判機率，還有 agent 寫的說明。它只解釋，不決定。等最久的排最前面。"),
  ("region",   "打開一個區域：範本、待測板、差異圖並排；右邊是這台機器的缺陷率，和這一類的驗收標準。這一頁不顯示答案，因為作業員的答案就是下一輪的訓練標籤。"),
  ("defer",    "看不出來就按零。它不會變成一個判定；區域換到另一個隊伍，等資深的人。"),
  ("blocked",  "換一般作業員登入，打開同一個區域：按鈕不見了。退回的區域只有資深能答，這是這個站唯一的權限。"),
  ("boards",   "板的索引：已定案、扣住、放行、等待中。分母在這裡，不在佇列。"),
  ("ask",      "主管的問題：M32 參數變更前後，open 的比例有沒有變。系統把它變成一份型別化的查詢計畫，先驗證，再平行執行。"),
  ("ask_done", "前後兩根柱，區間不重疊；圖是從結果的形狀畫出來的，文字寫在數字旁邊。"),
  ("control",  "對照組：M31 換燈前後。"),
  ("control_done", "兩根區間重疊。工具會說沒差，不是有事件就有影響。"),
  ("switch",   "切換語言：問題和規劃段保留原文並標示；答案從同一批結果重寫，不是翻譯。每個門檻都引得到一支腳本，每個數字都在 benchmarks 裡。"),
 ],
 "en": [
  ("cli",      "One board comes in. The AOI flagged thirty regions; the vision model dismissed twenty-eight in milliseconds and handed two it could not settle to a person."),
  ("queue",    "This is the queue: each row carries the model's class, its confidence, the false-call probability, and the agent's rationale. The agent explains; it does not decide. Longest wait first."),
  ("region",   "One region: template, board under test and difference side by side; the machine's defect rate and the acceptance criteria for this class on the right. The page never shows the answer key, because the operator's answer is the next training label."),
  ("defer",    "Press zero when you cannot tell. It is not recorded as a verdict; the region moves to a second list for a senior reviewer."),
  ("blocked",  "Signed in as an ordinary operator, the same region has no buttons. Handed-back regions are answered by a senior only — the station's one permission."),
  ("boards",   "The board index: dispositioned, held, released and waiting. The denominator lives here, not on the queue."),
  ("ask",      "A supervisor's question: did the parameter change on M32 move its share of opens. It becomes a typed plan of lookups, validated before anything runs, then executed in parallel."),
  ("ask_done", "Two bars, before and after, with intervals that do not overlap. The chart is derived from the result shape; the prose sits beside the figures."),
  ("control",  "A control: the lamp replacement on M31."),
  ("control_done", "Overlapping intervals — no difference shown. The tool says so rather than reading an event as an effect."),
  ("switch",   "Switching language: the question and the plan stay as written and are labelled; the answer is written again from the same results, not translated. Every threshold cites a script and every figure is in the benchmarks file."),
 ],
}[LANG]


def tts() -> dict[str, float]:
    NARR.mkdir(parents=True, exist_ok=True)
    durations = {}
    for key, text in SCENES:
        path = NARR / f"{key}.aiff"
        subprocess.run(["say", "-v", VOICE, "-r", "175" if LANG == "en" else "190", "-o", str(path), text], check=True)
        info = subprocess.run(["afinfo", str(path)], capture_output=True, text=True).stdout
        secs = float([l for l in info.splitlines() if "estimated duration" in l][0].split(":")[1].split("sec")[0])
        durations[key] = secs
    return durations


def cli_transcript() -> list[str]:
    """Run the board through the flow and keep what the CLI printed, minus the
    HTTP client's log lines, for the terminal scene."""
    out = subprocess.run(["uv", "run", "python", "-m", "aoi_agent", "board", STEM, "--queue"],
                         capture_output=True, text=True, cwd=ROOT).stdout
    out = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", out)
    return [l for l in out.splitlines() if l.strip() and "INFO" not in l and "http" not in l and "HTTP/1.1" not in l]


def terminal_page() -> Path:
    lines = cli_transcript()
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;background:#0f1115;color:#d7dae0;font:15px/1.5 "SF Mono",Menlo,monospace;padding:28px 36px}}
.prompt{{color:#7ee787}} .dim{{color:#8b949e}} .q{{color:#f0b429}} .d{{color:#8b949e}}
</style></head><body><div id="t"><span class="prompt">$</span> uv run python -m aoi_agent board {STEM} --queue</div>
<script>
const lines = {json.dumps(lines)};
const t = document.getElementById('t'); let i = 0;
function tick(){{ if (i >= lines.length) return; const l = lines[i++]; const d = document.createElement('div');
  d.textContent = l; if (l.includes('QUEUED') || l.includes('escalated')) d.className='q'; else if (l.trim().startsWith('path') || l.trim().startsWith('classify')) d.className='d';
  t.appendChild(d); window.scrollTo(0, document.body.scrollHeight);
  setTimeout(tick, l.includes('reason:') ? 900 : 55); }}
setTimeout(tick, 900);
</script></body></html>"""
    path = OUT / "terminal.html"
    path.write_text(html)
    return path


def main() -> None:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    durations = tts()
    term = terminal_page()
    timeline: list[dict] = []

    def login(pg, name, secret):
        pg.context.clear_cookies()
        pg.goto(f"{BASE}/login"); pg.fill("input[name=name]", name); pg.fill("input[name=secret]", secret)
        pg.click("button[type=submit], input[type=submit]"); pg.wait_for_load_state("networkidle")
        pg.goto(f"{BASE}/locale/{LANG}?next=/"); pg.wait_for_load_state("networkidle")

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        ctx = browser.new_context(viewport={"width": 1280, "height": 800}, color_scheme="dark",
                                  record_video_dir=str(OUT), record_video_size={"width": 1280, "height": 800})
        pg = ctx.new_page()
        t0 = time.monotonic()

        def scene(key, action):
            start = time.monotonic() - t0
            action()
            elapsed = time.monotonic() - t0 - start
            hold = max(0.0, durations[key] + 1.0 - elapsed)
            pg.wait_for_timeout(int(hold * 1000))
            timeline.append({"key": key, "start": round(start, 2), "end": round(time.monotonic() - t0, 2)})

        def slow_scroll(px, steps=8):
            for _ in range(steps):
                pg.mouse.wheel(0, px // steps); pg.wait_for_timeout(180)

        # 1 terminal
        scene("cli", lambda: (pg.goto(term.as_uri()), pg.wait_for_timeout(7000)))
        # 2 queue
        scene("queue", lambda: (login(pg, *SENIOR), pg.goto(f"{BASE}/"), pg.wait_for_load_state("networkidle"), pg.wait_for_timeout(1500), slow_scroll(500)))
        # 3 region
        scene("region", lambda: (pg.goto(f"{BASE}/c/{STEM}/8"), pg.wait_for_load_state("networkidle"), pg.wait_for_timeout(2500), slow_scroll(700, 10), pg.wait_for_timeout(800)))
        # 4 defer with 0
        def defer():
            pg.goto(f"{BASE}/c/{STEM}/8"); pg.wait_for_load_state("networkidle")
            pg.mouse.wheel(0, 1600); pg.wait_for_timeout(1200)
            pg.keyboard.press("0"); pg.wait_for_load_state("networkidle"); pg.wait_for_timeout(1000)
            pg.goto(f"{BASE}/deferred"); pg.wait_for_load_state("networkidle")
        scene("defer", defer)
        # 5 blocked as watcher
        scene("blocked", lambda: (login(pg, *OPERATOR), pg.goto(f"{BASE}/c/{STEM}/8"), pg.wait_for_load_state("networkidle"), pg.wait_for_timeout(1500), pg.mouse.wheel(0, 1600), pg.wait_for_timeout(1500)))
        # 6 boards
        scene("boards", lambda: (login(pg, *SENIOR), pg.goto(f"{BASE}/boards"), pg.wait_for_load_state("networkidle"), pg.wait_for_timeout(1500), slow_scroll(300, 4)))
        # 7 ask
        def ask(question):
            pg.goto(f"{BASE}/ask"); pg.wait_for_load_state("networkidle")
            pg.click("input[name=question]"); pg.type("input[name=question]", question, delay=45)
            pg.wait_for_timeout(600); pg.press("input[name=question]", "Enter")
            pg.wait_for_selector("figure.chart", timeout=180000)
        scene("ask", lambda: ask(Q_M32))
        scene("ask_done", lambda: (pg.wait_for_timeout(500), pg.locator("figure.chart").scroll_into_view_if_needed(), pg.wait_for_timeout(1500)))
        scene("control", lambda: ask(Q_M31))
        scene("control_done", lambda: (pg.wait_for_timeout(500), pg.locator("figure.chart").scroll_into_view_if_needed(), pg.wait_for_timeout(1500)))
        # 9 language switch on the answer page
        other = "en" if LANG == "zh-TW" else "zh-TW"
        def switch():
            pg.mouse.wheel(0, -4000); pg.wait_for_timeout(600)
            pg.goto(f"{BASE}/locale/{other}?next={pg.url.replace(BASE, '')}"); pg.wait_for_load_state("networkidle")
            try:
                pg.wait_for_selector("figure.chart", timeout=120000)
            except Exception:
                pass
            pg.wait_for_timeout(1500); slow_scroll(900, 8)
        scene("switch", switch)

        pg.wait_for_timeout(1500)
        video_path = pg.video.path()
        ctx.close(); browser.close()

    (OUT / "timeline.json").write_text(json.dumps(timeline, indent=1))
    # subtitles
    def ts(s): h = int(s // 3600); m = int(s % 3600 // 60); sec = s % 60; return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")
    text = dict(SCENES)
    srt = []
    for i, t in enumerate(timeline, 1):
        end = min(t["end"], t["start"] + durations[t["key"]] + 1.5)
        srt.append(f"{i}\n{ts(t['start'])} --> {ts(end)}\n{text[t['key']]}\n")
    (OUT / "subs.srt").write_text("\n".join(srt))
    # audio: each narration delayed to its scene start, mixed
    inputs, delays = [], []
    for i, t in enumerate(timeline):
        inputs += ["-i", str(NARR / f"{t['key']}.aiff")]
        delays.append(f"[{i+1}:a]adelay={int(t['start']*1000)}|{int(t['start']*1000)}[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(timeline))) + f"amix=inputs={len(timeline)}:normalize=0[narr]"
    font = "PingFang TC" if LANG == "zh-TW" else "Helvetica Neue"
    subs = str(OUT / "subs.srt").replace(":", "\\:")
    final = ROOT / "docs" / "demo" / f"aoi-agent-demo-{TAG}.mp4"
    cmd = [FFMPEG, "-y", "-i", str(video_path), *inputs,
           "-filter_complex", ";".join(delays) + ";" + mix + f";[0:v]subtitles='{subs}':force_style='FontName={font},FontSize=15,PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BorderStyle=4,BackColour=&H90000000,MarginV=28'[v]",
           "-map", "[v]", "-map", "[narr]", "-c:v", "libx264", "-crf", "22", "-preset", "medium", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-shortest", str(final)]
    subprocess.run(cmd, check=True, capture_output=True)
    print("wrote", final, f"{final.stat().st_size/1e6:.1f} MB; scenes:", len(timeline), "; length", timeline[-1]["end"], "s")


if __name__ == "__main__":
    main()
