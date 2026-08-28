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
_args.add_argument("--tts", default="kokoro", choices=("kokoro", "say"),
                   help="kokoro: Kokoro-82M through ~/Projects/video_transfer's backend (neural, both languages); say: macOS")
_args.add_argument("--video-transfer", default=str(Path.home() / "Projects" / "video_transfer"))
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
VOICE = {"zh-TW": "Meijia", "en": "Samantha"}[LANG]            # macOS say
KOKORO_VOICE = {"zh-TW": "zf_xiaoxiao", "en": "af_heart"}[LANG]  # Kokoro: first letter is the language
TAG = "zh" if LANG == "zh-TW" else "en"

Q_M32 = {"zh-TW": "M32 參數變更前後，open 的比例有沒有變？",
         "en": "Did the parameter change on M32 move its share of opens?"}[LANG]
Q_M31 = {"zh-TW": "M31 換燈前後，open 的比例有沒有變？",
         "en": "Did the lamp replacement on M31 move its share of opens?"}[LANG]

SCENES = {
 "zh-TW": [
  ("cli",      "好，一片 PCB 剛進來。AOI 標了三十個區域，視覺模型幾毫秒就排掉二十八個；剩下兩個它不敢判，就交給人。"),
  ("queue",    "這一頁就是等人看的清單。每一列都有模型的判定、信心、誤判機率，還有 agent 寫的一段說明。注意，agent 只負責解釋，不做決定。誰等最久，誰排前面。"),
  ("region",   "點進一個區域。左邊是黃金樣板、待測 PCB 和差異圖並排；右邊是這台機器的缺陷率，還有這一類的驗收標準。這一頁故意不顯示答案，因為作業員按下去的答案，就是下一輪訓練的標籤。"),
  ("defer",    "真的看不出來？按零。它不會被記成判定，區域會換到另一個隊伍，交給資深的人。"),
  ("blocked",  "換一般作業員登入，打開同一個區域，按鈕不見了。退回的區域只有資深能答，這是整個站唯一的權限。"),
  ("boards",   "這是 PCB 處置紀錄：已定案、扣住、放行、等待中。分母在這裡，不在待複判清單。"),
  ("ask",      "主管問：M32 參數變更前後，open 的比例有沒有變？系統先把問題變成一份查詢計畫，驗證過才跑，幾個查詢是平行的。"),
  ("ask_done", "前後兩根柱，區間沒有重疊。圖是從結果的形狀畫出來的，文字就寫在數字旁邊。"),
  ("control",  "再問一個對照組：M31 換燈前後。"),
  ("control_done", "這次兩根區間重疊，系統就直接說沒差。有事件，不代表有影響。"),
  ("switch",   "最後切換語言。問題和規劃段保留原文、標示出來；答案是用同一批結果重寫的，不是翻譯。每個門檻都引得到腳本，每個數字都在 benchmarks 裡。"),
 ],
 "en": [
  ("cli",      "A board just came in. The AOI flagged thirty regions; the vision model cleared twenty-eight of them in milliseconds, and the two it wasn't sure about go to a person."),
  ("queue",    "This is the review queue. Every row has the model's class, its confidence, the false-call probability, and a short rationale from the agent. The agent explains — it never decides. Whoever has waited longest is on top."),
  ("region",   "Open one region. Template, PCB under test and difference side by side; on the right, this machine's defect rate and the acceptance criteria for the class. The answer key is deliberately not on this page, because whatever the operator presses becomes the next training label."),
  ("defer",    "Can't tell? Press zero. It isn't recorded as a verdict; the region moves to a second list for a senior reviewer."),
  ("blocked",  "Sign in as an ordinary operator, open the same region, and the buttons are gone. Handed-back regions are for seniors only — that's the station's one permission."),
  ("boards",   "PCB dispositions: settled, held, released, waiting. The denominator lives here, not on the queue."),
  ("ask",      "A supervisor asks: did the parameter change on M32 move its share of opens? The question becomes a plan of lookups, validated before anything runs, then executed in parallel."),
  ("ask_done", "Two bars, before and after, and the intervals don't overlap. The chart comes from the shape of the results; the prose sits right beside the numbers."),
  ("control",  "Now a control: the lamp replacement on M31."),
  ("control_done", "This time the intervals overlap, and the system says so. An event is not an effect."),
  ("switch",   "Finally, switch the language. The question and the plan stay as written and are labelled; the answer is written again from the same results, not translated. Every threshold cites a script, and every figure is in the benchmarks file."),
 ],
}[LANG]


def _duration(path: Path) -> float:
    info = subprocess.run(["afinfo", str(path)], capture_output=True, text=True).stdout
    return float([l for l in info.splitlines() if "estimated duration" in l][0].split(":")[1].split("sec")[0])


def tts() -> dict[str, float]:
    """One narration file per scene, and its length. Kokoro-82M through
    ~/Projects/video_transfer's TTS backend by default -- the same neural voice
    that project dubs with -- with macOS `say` as the fallback."""
    NARR.mkdir(parents=True, exist_ok=True)
    if ARGS.tts == "say":
        for key, text in SCENES:
            subprocess.run(["say", "-v", VOICE, "-r", "175" if LANG == "en" else "190",
                            "-o", str(NARR / f"{key}.aiff"), text], check=True)
        return {key: _duration(NARR / f"{key}.aiff") for key, _ in SCENES}
    spec = NARR / "lines.json"
    spec.write_text(json.dumps([{"key": k, "text": t} for k, t in SCENES], ensure_ascii=False))
    runner = (
        "import json, sys\nfrom pathlib import Path\nfrom video_pipeline.tts import KokoroBackend\n"
        "spec, out, voice = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]\n"
        "b = KokoroBackend()\n"
        "for line in json.loads(spec.read_text()):\n"
        "    b.synthesize(line['text'], out / (line['key'] + '.wav'), speaker=voice)\n"
    )
    subprocess.run(["uv", "run", "--project", ARGS.video_transfer, "python", "-c", runner,
                    str(spec), str(NARR), KOKORO_VOICE],
                   check=True, cwd=ARGS.video_transfer, env={**os.environ, "VT_TTS_BACKEND": "kokoro"},
                   capture_output=True)
    return {key: _duration(NARR / f"{key}.wav") for key, _ in SCENES}


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
        ext = "aiff" if ARGS.tts == "say" else "wav"
        inputs += ["-i", str(NARR / f"{t['key']}.{ext}")]
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
