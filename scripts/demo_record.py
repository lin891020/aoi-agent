"""Record the demo video: Playwright drives the station, `say` narrates, ffmpeg muxes.

    uv run --with playwright python scripts/demo_record.py --lang zh-TW --stem 00041208
    uv run --with playwright python scripts/demo_record.py --lang en    --stem 00041208
    uv run --with playwright python scripts/demo_record.py --lang zh-TW --stem 00041208 --index 15 --silent
                                        # video only, holds from HOLD_S, script.md for dubbing by hand

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
_args.add_argument("--stem", required=True, help="a board with a region on the queue")
_args.add_argument("--index", type=int, default=8, help="which of that board's regions is on the queue")
_args.add_argument("--silent", action="store_true",
                   help="no narration and no subtitles: the video alone, each scene held for HOLD_S "
                        "seconds, plus script.md (scene, start, end, the line to say) for dubbing over it")
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
INDEX = ARGS.index
REGION = f"{STEM}#{INDEX}"
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
  ("home",     "登入後第一眼是這條線的分母：跑過的 PCB 幾片定案、幾片扣住、幾片放行、幾片還在等人，還有幾個區域等人看。失敗清單不在第一眼，要點進去才是。"),
  ("queue",    "這一頁就是等人看的清單。每一列都有模型的判定、信心、誤判機率，還有 agent 寫的一段說明。注意，agent 只負責解釋，不做決定。誰等最久，誰排前面。"),
  ("region",   "點進一個區域。左邊是黃金樣板、待測 PCB 和差異圖並排；右邊是這台機器的缺陷率，還有這一類的驗收標準。這一頁故意不顯示答案，因為作業員按下去的答案，就是下一輪訓練的標籤。"),
  ("defer",    "真的看不出來？按零。它不會被記成判定，區域會換到另一個隊伍，交給資深的人。"),
  ("blocked",  "換一般作業員登入，打開同一個區域，按鈕不見了。退回的區域只有資深能答，這是整個站唯一的權限。"),
  ("boards",   "這是 PCB 處置紀錄：已定案、扣住、放行、等待中。分母在這裡，不在待複判清單。"),
  ("ask",      "主管問：M32 參數變更前後，open 的比例有沒有變？系統先把問題變成一份查詢計畫，驗證過才跑，幾個查詢是平行的。"),
  ("ask_done", "前後兩根柱，區間沒有重疊。圖是從結果的形狀畫出來的，文字就寫在數字旁邊。"),
  ("control",  "再問一個對照組：M31 換燈前後。"),
  ("control_done", "這次兩根區間重疊，系統就直接說沒差。有事件，不代表有影響。"),
  ("switch",   "最後切換語言。問題和規劃段保留原文、標示出來；答案要按一下才重寫——用儲存的同一批結果再寫一次，不是翻譯，原文也留著。每個門檻都引得到腳本，每個數字都在 benchmarks 裡。"),
 ],
 "en": [
  ("cli",      "A board just came in. The AOI flagged thirty regions; the vision model cleared twenty-eight of them in milliseconds, and the two it wasn't sure about go to a person."),
  ("home",     "Sign in and the first screen is the line's denominator: how many boards settled, held, released, still waiting, and how many regions wait on a person. The list of failures is one click in, not the front door."),
  ("queue",    "This is the review queue. Every row has the model's class, its confidence, the false-call probability, and a short rationale from the agent. The agent explains — it never decides. Whoever has waited longest is on top."),
  ("region",   "Open one region. Template, PCB under test and difference side by side; on the right, this machine's defect rate and the acceptance criteria for the class. The answer key is deliberately not on this page, because whatever the operator presses becomes the next training label."),
  ("defer",    "Can't tell? Press zero. It isn't recorded as a verdict; the region moves to a second list for a senior reviewer."),
  ("blocked",  "Sign in as an ordinary operator, open the same region, and the buttons are gone. Handed-back regions are for seniors only — that's the station's one permission."),
  ("boards",   "PCB dispositions: settled, held, released, waiting. The denominator lives here, not on the queue."),
  ("ask",      "A supervisor asks: did the parameter change on M32 move its share of opens? The question becomes a plan of lookups, validated before anything runs, then executed in parallel."),
  ("ask_done", "Two bars, before and after, and the intervals don't overlap. The chart comes from the shape of the results; the prose sits right beside the numbers."),
  ("control",  "Now a control: the lamp replacement on M31."),
  ("control_done", "This time the intervals overlap, and the system says so. An event is not an effect."),
  ("switch",   "Finally, switch the language. The question and the plan stay as written and are labelled; the answer is written again only when asked, from the same stored results, not translated, and the original is kept. Every threshold cites a script, and every figure is in the benchmarks file."),
 ],
}[LANG]


#: Seconds a scene stays on screen in the silent cut, once its action is done --
#: room to say the line in SCENES at a speaking pace, not a synthetic one.
HOLD_S = {"cli": 14.0, "home": 12.0, "queue": 14.0, "region": 40.0, "defer": 12.0, "blocked": 12.0,
          "boards": 12.0, "ask": 8.0, "ask_done": 14.0, "control": 6.0, "control_done": 12.0, "switch": 18.0}


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
    durations = dict(HOLD_S) if ARGS.silent else tts()
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
        # 1b the front door: the denominator, then the queue one click in
        scene("home", lambda: (login(pg, *SENIOR), pg.goto(f"{BASE}/"), pg.wait_for_load_state("networkidle"), pg.wait_for_timeout(1500)))
        # 2 queue
        scene("queue", lambda: (pg.goto(f"{BASE}/queue"), pg.wait_for_load_state("networkidle"), pg.wait_for_timeout(1500), slow_scroll(500)))
        # 3 region
        def region():
            # The scene the narration spends longest on, paced in three holds:
            # the rationale and the triptych, then the model's reading beside
            # the context and the criteria, then the verdict form and the note
            # that the answer key is not on this page.
            pg.goto(f"{BASE}/c/{STEM}/{INDEX}"); pg.wait_for_load_state("networkidle"); pg.wait_for_timeout(11000)
            slow_scroll(700, 10); pg.wait_for_timeout(9000)
            slow_scroll(900, 10); pg.wait_for_timeout(1000)
        scene("region", region)
        # 4 defer with 0
        def defer():
            pg.goto(f"{BASE}/c/{STEM}/{INDEX}"); pg.wait_for_load_state("networkidle")
            pg.mouse.wheel(0, 1600); pg.wait_for_timeout(1200)
            pg.keyboard.press("0"); pg.wait_for_load_state("networkidle"); pg.wait_for_timeout(1000)
            pg.goto(f"{BASE}/deferred"); pg.wait_for_load_state("networkidle")
        scene("defer", defer)
        # 5 blocked as watcher
        scene("blocked", lambda: (login(pg, *OPERATOR), pg.goto(f"{BASE}/c/{STEM}/{INDEX}"), pg.wait_for_load_state("networkidle"), pg.wait_for_timeout(1500), pg.mouse.wheel(0, 1600), pg.wait_for_timeout(1500)))
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
            pg.wait_for_timeout(1500); slow_scroll(900, 8)
            # The switch renders chrome; the answer is written again only when
            # asked, because a GET on a stored run must not cost a model call.
            # Press the button, wait out the one synthesis call, and read the
            # answer that comes back under its badge.
            button = pg.locator("form.rewrite button")
            if button.count():
                button.scroll_into_view_if_needed(); pg.wait_for_timeout(800)
                button.click()
                pg.wait_for_load_state("networkidle", timeout=180000)
                pg.locator("div.prose").scroll_into_view_if_needed()
                # The answer written again is the scene's point, and the call
                # that produces it takes most of the scene: hold on it for
                # its own time rather than the seconds left over.
                pg.wait_for_timeout(9000)
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
    if ARGS.silent:
        # The video alone: no narration track, no burned subtitles. What goes
        # beside it is the script -- one row per scene with the seconds the
        # scene is on screen and the line to say over it -- so a person can
        # dub it without operating the station by hand.
        final = ROOT / "docs" / "demo" / f"aoi-agent-demo-{TAG}-silent.mp4"
        subprocess.run([FFMPEG, "-y", "-i", str(video_path), "-an", "-c:v", "libx264", "-crf", "22",
                        "-preset", "medium", "-pix_fmt", "yuv420p", str(final)], check=True, capture_output=True)
        rows = "\n".join(f"| {i} | {t['key']} | {t['start']:.0f}–{t['end']:.0f} s | {text[t['key']]} |"
                          for i, t in enumerate(timeline, 1))
        (OUT / "script.md").write_text(
            f"# {final.name} — 配音腳本 / dubbing script\n\n每一幕畫面停留的秒數，和要在那段時間裡講的話。"
            f"影片沒有聲音、沒有字幕；`subs.srt` 是同一份台詞的字幕檔，可以疊上去對時間。\n\n"
            f"| # | 幕 | 秒 | 台詞 |\n|---|---|---|---|\n{rows}\n")
        print("wrote", final, f"{final.stat().st_size/1e6:.1f} MB; scenes:", len(timeline),
              "; length", timeline[-1]["end"], "s; script:", OUT / "script.md")
        return
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
