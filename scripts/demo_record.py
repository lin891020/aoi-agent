"""Record the demo video: Playwright drives the station cue by cue, ffmpeg puts
the subtitles in a band under the page, and the narration -- when there is
one -- comes through ~/Projects/video_transfer's own TTS path.

    uv run --with playwright python scripts/demo_record.py --lang zh-TW --stem 00041208 --index 15 --check
                                        # walk the non-mutating pages and confirm every framed element exists
    uv run --with playwright python scripts/demo_record.py --lang zh-TW --stem 00041208 --index 15 --silent
                                        # the take: video + subtitles, no audio; script.md beside it
    uv run --with playwright python scripts/demo_record.py --lang zh-TW --stem 00041208 --index 15 --narrate-existing
                                        # dub that take: one wav per cue, listened back, mixed in

The shot list is scripts/demo_script.py (docs/demo-script.md is generated
from it): thirteen scenes, fifty-odd *cues*, each one sentence with its own
hold and, usually, one element on the page it frames. Until 2026-09-05 the
recorder held one paragraph per scene and burned it across the page; Mike
watched the English take and said three things: the subtitle covered what it
was describing, the login flashed past, and the script assumed the viewer
knew what an AOI queue was. So: subtitles of at most two short lines in a
100 px band *below* the 1280x800 page, a spotlight (dimmed page, one framed
element -- the driver.js pattern, injected rather than imported) while a cue
is up, a one-second pause on the sign-in form, and a title card before any
page that says what the system is for.

Needs a station (`--base`; record against a second instance, never the one
somebody is using), Ollama with gpt-oss:20b, operators `mike` (senior,
passphrase in AOI_DEMO_SENIOR_SECRET) and `watcher` (operator, in
AOI_DEMO_OPERATOR_SECRET), and a board whose region `<stem>#<index>` is on the
queue. The board is run through the CLI first so the terminal scene shows
real output. Everything lands under docs/demo/ (gitignored): the raw webm,
cards, narration and subtitles under build/<lang>/, and the mp4 beside them.
A take hands the demo region back (the defer scene presses 0) and asks two
questions on /ask; nothing else in the store changes.

macOS only: the mux uses the Homebrew ffmpeg-full build for its subtitle
filter, and `--tts say` uses the system voices.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_script import CUES, SCENES, cue_id, hold, lines, text

_args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_args.add_argument("--lang", default="zh-TW", choices=("zh-TW", "en"))
_args.add_argument("--stem", required=True, help="a board with a region on the queue")
_args.add_argument("--index", type=int, default=15, help="which of that board's regions is on the queue")
_args.add_argument("--silent", action="store_true",
                   help="no narration: the video with its subtitles, each cue held for its reading time, "
                        "plus script.md (cue, seconds, the line) for dubbing over it")
_args.add_argument("--narrate-existing", action="store_true",
                   help="do not drive the station: take build/<lang>/take.webm and timeline.json from the last "
                        "take, synthesise one narration per cue, and mux")
_args.add_argument("--check", action="store_true",
                   help="drive the pages a take does not change (cards, terminal, login, home, queue, region, "
                        "boards, the /ask form) and report every framed element that is missing; no video")
_args.add_argument("--base", default="http://127.0.0.1:8110")
_args.add_argument("--tts", default="auto", choices=("auto", "qwen3tts", "kokoro", "say"),
                   help="auto: video_transfer's own split -- Qwen3-TTS for Chinese (the one voice measured to read "
                        "mixed Chinese/English), Kokoro for English; or name a backend; say: macOS, no verification")
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
REPO = "github.com/lin891020/aoi-agent"

#: The page, and the band under it where the subtitles go.
W, H, BAND = 1280, 800, 100
#: Scenes a take changes something in -- `--check` does not run these.
MUTATING = {"defer", "blocked", "ask", "control", "switch"}

CUE = {cue_id(k, i): c for k, i, c in CUES}
SCENE_OF = {cue_id(k, i): k for k, i, _ in CUES}

Q_M32 = {"zh-TW": "M32 參數變更前後，open 的比例有沒有變？",
         "en": "Did the parameter change on M32 move its share of opens?"}[LANG]
Q_M31 = {"zh-TW": "M31 換燈前後，open 的比例有沒有變？",
         "en": "Did the lamp replacement on M31 move its share of opens?"}[LANG]


def _duration(path: Path) -> float:
    info = subprocess.run(["afinfo", str(path)], capture_output=True, text=True, check=False).stdout
    return float([x for x in info.splitlines() if "estimated duration" in x][0].split(":")[1].split("sec")[0])


def tts() -> dict[str, float]:
    """One narration file per cue, and its length.

    Through ~/Projects/video_transfer's own path -- `source env.sh local`,
    `get_tts_backend(name, language=...)`, `dubscript.to_dub_text()` -- and not
    around it. Until 2026-09-05 this constructed `KokoroBackend()` directly and
    skipped both: the Chinese take read "好一片剛進來，照標了三十個區域" for
    "一片 PCB 剛進來，AOI 標了三十個區域" -- PCB gone, AOI turned into 照, M32
    into 32 -- and every sentence stayed grammatical, so nobody heard it go.
    Kokoro is the voice video_transfer measured at 1/96 English words surviving
    a mixed sentence; Qwen3-TTS is the one it keeps for Chinese (93/96), and
    its backend listens every piece back through Whisper as it goes.

    Each finished cue is listened back once more here and its identifiers
    (every Latin token of a Chinese cue; PCB, AOI, M32 and their kind in an
    English one) are checked against what was heard, so a word that silently
    vanished is printed rather than shipped.
    """
    NARR.mkdir(parents=True, exist_ok=True)
    if ARGS.tts == "say":
        for cid, c in CUE.items():
            subprocess.run(["say", "-v", VOICE, "-r", "175" if LANG == "en" else "190",
                            "-o", str(NARR / f"{cid}.aiff"), text(c, LANG)], check=True)
        return {cid: _duration(NARR / f"{cid}.aiff") for cid in CUE}
    lang = "zh" if LANG == "zh-TW" else "en"
    name = {"zh": "qwen3tts", "en": "kokoro"}[lang] if ARGS.tts == "auto" else ARGS.tts
    spec = NARR / "lines.json"
    spec.write_text(json.dumps([{"key": cid, "text": text(c, LANG)} for cid, c in CUE.items()], ensure_ascii=False))
    runner = (
        "import json, sys\nfrom pathlib import Path\n"
        "from video_pipeline import dubscript\n"
        "from video_pipeline.tts import get_tts_backend\n"
        "from video_pipeline.transcribe import transcribe\n"
        "spec, out, voice, name, lang = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]\n"
        "import re, shutil\n"
        # What must be heard back: in a Chinese line every Latin token is an
        # identifier; in an English line only the identifiers are (PCB, AOI,
        # M32) -- Whisper writes thirty as 30 and labelled as labeled, and
        # a check over every word would spend the retries on spelling.
        "TOKEN = r'[A-Za-z][A-Za-z0-9]+' if lang == 'zh' else r'\\b[A-Z][A-Z0-9]+'\n"
        "def missing_of(sent, heard):\n"
        "    flat = re.sub(r'[\\s\\-_]', '', heard).lower()\n"
        "    return [t for t in re.findall(TOKEN, sent) if t.lower() not in flat]\n"
        "b = get_tts_backend(name, language=lang)\n"
        "prior = json.loads((out / 'heard.json').read_text()) if (out / 'heard.json').exists() else {}\n"
        "heard = {}\n"
        "for line in json.loads(spec.read_text()):\n"
        "    key, text = line['key'], dubscript.to_dub_text(line['text'])\n"
        "    wav, old = out / (key + '.wav'), prior.get(key)\n"
        "    if wav.exists() and old and old.get('sent') == text and old.get('backend') == b.name and not missing_of(text, old['heard']):\n"
        "        heard[key] = old; continue\n"
        "    best = None\n"
        "    for attempt in range(1, 4):\n"
        "        take = out / (key + '.take.wav')\n"
        "        b.synthesize(text, take, speaker=voice)\n"
        "        h = ' '.join(s.text for s in transcribe(take, language=lang).segments)\n"
        "        entry = {'backend': b.name, 'sent': text, 'heard': h, 'missing': missing_of(text, h), 'attempts': attempt}\n"
        "        if best is None or len(entry['missing']) < len(best['missing']):\n"
        "            best = entry; shutil.copy(take, wav)\n"
        "        if not entry['missing']: break\n"
        "    take.unlink(missing_ok=True); heard[key] = best\n"
        "(out / 'heard.json').write_text(json.dumps(heard, ensure_ascii=False, indent=1))\n"
    )
    cmd = (f"source env.sh local >/dev/null && uv run --project . python -c {shlex.quote(runner)} "
           f"{shlex.quote(str(spec))} {shlex.quote(str(NARR))} {KOKORO_VOICE} {name} {lang}")
    done = subprocess.run(["bash", "-c", cmd], cwd=ARGS.video_transfer, capture_output=True, text=True, check=False)
    if done.returncode:
        sys.exit(f"video_transfer's TTS failed ({name}):\n" + done.stderr[-3000:])
    heard = json.loads((NARR / "heard.json").read_text())
    for key, entry in heard.items():
        if entry["missing"]:
            print(f"warning: {key}: not heard back: {', '.join(entry['missing'])}\n  sent:  {entry['sent']}\n"
                  f"  heard: {entry['heard']}", file=sys.stderr)
    print(f"narration: {name}; {len(heard)} cues; listen-back in {NARR / 'heard.json'}")
    return {cid: _duration(NARR / f"{cid}.wav") for cid in CUE}


# --- the pages the recorder makes itself ------------------------------------

CARD_CSS = """
body{margin:0;background:#0b0d12;color:#e6e8ee;font:18px/1.5 -apple-system,"PingFang TC","Helvetica Neue",sans-serif}
.card{box-sizing:border-box;width:1280px;height:800px;padding:40px 80px 32px;display:flex;flex-direction:column;gap:12px}
h1{font-size:38px;font-weight:600;margin:0;letter-spacing:-.01em}
p.lead{font-size:21px;color:#aeb4c2;margin:0;max-width:1000px}
.diagram{flex:1;display:flex;align-items:center;justify-content:center;min-height:0}
.diagram svg{width:100%;height:auto;max-height:100%}
.figures{display:flex;gap:56px;margin-top:22px}
.figures div{display:flex;flex-direction:column;gap:6px}
.figures b{font-size:64px;font-weight:600;line-height:1;font-variant-numeric:tabular-nums}
.figures span{font-size:20px;color:#aeb4c2;max-width:320px}
.foot{margin-top:auto;color:#8b93a5;font:20px "SF Mono",Menlo,monospace}
"""


def card_page(name: str, title: str, lead: str, svg: Path | None = None,
              figures: list[tuple[str, str]] | None = None, foot: str = "") -> Path:
    body = f"<h1>{title}</h1><p class='lead'>{lead}</p>"
    if svg is not None:
        body += f"<div class='diagram'>{svg.read_text()}</div>"
    if figures:
        body += "<div class='figures'>" + "".join(f"<div><b>{n}</b><span>{label}</span></div>" for n, label in figures) + "</div>"
    if foot:
        body += f"<p class='foot'>{foot}</p>"
    path = OUT / f"{name}.html"
    path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CARD_CSS}</style></head>"
                    f"<body><div class='card'>{body}</div></body></html>")
    return path


def intro_page() -> Path:
    title = {"zh-TW": "AOI 複判站", "en": "AOI re-verification station"}[LANG]
    lead = {"zh-TW": "AOI 標出來的區域，六成是誤報，每一個都要人看。一個視覺模型、一個 agent，收不掉的才交給人。",
            "en": "Six in ten regions an AOI flags are false calls, and every one goes to a person. "
                  "A vision model, then an agent; only what neither settles reaches a person."}[LANG]
    svg = ROOT / "docs" / "diagrams" / ("disposition-flow-dark.zh-TW.svg" if LANG == "zh-TW" else "disposition-flow-dark.svg")
    return card_page("intro", title, lead, svg=svg, foot=REPO)


def outro_page() -> Path:
    zh = LANG == "zh-TW"
    return card_page(
        "outro",
        "數字都在 README" if zh else "The numbers are in the README",
        "DeepPCB 測試集，門檻不在報成績的那份資料上挑。" if zh else "DeepPCB test split; the threshold was chosen off the split it is reported against.",
        figures=[("55.6%", "人工複判省掉" if zh else "of manual review removed"),
                 ("0.66%", "漏檢（預算 0.5%）" if zh else "escape (budget 0.5%)"),
                 ("85.9%", "的區域不經 LLM" if zh else "of regions never reach an LLM")],
        foot=REPO,
    )


def cli_transcript() -> list[str]:
    """Run the board through the flow and keep what the CLI printed, minus the
    HTTP client's log lines, for the terminal scene."""
    out = subprocess.run(["uv", "run", "python", "-m", "aoi_agent", "board", STEM, "--queue"],
                         capture_output=True, text=True, cwd=ROOT, check=False).stdout
    out = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", out)
    return [x for x in out.splitlines() if x.strip() and "INFO" not in x and "http" not in x and "HTTP/1.1" not in x]


def terminal_page() -> Path:
    """The CLI's output, folded to what the cues point at: the header, the
    first three regions the model settled, one line standing for the rest,
    the regions that went to the queue, and the board's own line."""
    raw = cli_transcript()
    (OUT / "cli.txt").write_text("\n".join(raw) + "\n")  # what the fold below was made from
    head = [x for x in raw if "AOI candidates" in x][:1]
    foot = [x for x in raw if x.strip().startswith(("board ", "under model")) and "candidates" not in x]
    # A block is one region: its verdict line and the indented lines under it.
    # An escalated region prints `escalated: <ref>`, its reason and two notes
    # *before* its QUEUED line, so that block stays open until the line with
    # the reference arrives.
    blocks: list[list[str]] = []
    open_escalation = False
    for row in raw:
        s = row.strip()
        if row in head or row in foot:
            continue
        if s.startswith("escalated:"):
            blocks.append([row])
            open_escalation = True
        elif re.match(r"\d{8}#\d+\s", s):
            if open_escalation:
                blocks[-1].append(row)
                open_escalation = False
            else:
                blocks.append([row])
        elif blocks:
            blocks[-1].append(row)
    queued = [b for b in blocks if any("QUEUED" in x for x in b)]
    settled = [b for b in blocks if b not in queued]
    first, rest = settled[:3], settled[3:]
    fold = f"      … {len(rest)} more regions settled by the model or the agent, a few ms each …"

    def div(id_, rows):
        return {"id": id_, "lines": [x for b in rows for x in b]}
    parts = [div("hdr", [head]), div("first", first), div("fold", [[fold]]), div("queued", queued), div("board", [foot])]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;background:#0f1115;color:#d7dae0;font:16px/1.55 "SF Mono",Menlo,monospace;padding:28px 36px}}
.prompt{{color:#7ee787}} .q{{color:#f0b429}} .d{{color:#8b949e}} #fold{{color:#8b949e;font-style:italic}}
div div{{white-space:pre}} .hidden{{display:none}}
</style></head><body><div><span class="prompt">$</span> uv run python -m aoi_agent board {STEM} --queue</div>
{"".join(f'<div id="{p["id"]}"></div>' for p in parts)}
<script>
const parts = {json.dumps(parts)};
const todo = []; for (const p of parts) for (const l of p.lines) todo.push([p.id, l]);
let i = 0;
function tick(){{ if (i >= todo.length) return; const [id, l] = todo[i++]; const d = document.createElement('div');
  d.textContent = l; if (l.includes('QUEUED') || l.includes('escalated')) d.className='q';
  else if (l.trim().startsWith('path') || l.trim().startsWith('classify')) d.className='d';
  document.getElementById(id).appendChild(d); setTimeout(tick, 60); }}
setTimeout(tick, 700);
</script></body></html>"""
    path = OUT / "terminal.html"
    path.write_text(html)
    return path


# --- the spotlight -----------------------------------------------------------

SPOT_JS = """(el) => {
  document.querySelectorAll('.demo-spot').forEach(e => e.remove());
  if (!el) return;
  const vh = window.innerHeight, pad = 8;
  let r = el.getBoundingClientRect();
  // Bring the element's top into view (under the sticky nav) when it is
  // taller than the screen or not wholly on it: scrollIntoView on a tall
  // column landed on its empty bottom half.
  if (r.height > vh - 130 || r.top < 70 || r.bottom > vh) {
    window.scrollTo({top: window.scrollY + r.top - 84, behavior: 'instant'});
    r = el.getBoundingClientRect();
  }
  const top = Math.max(r.top - pad, 66), bottom = Math.min(r.bottom + pad, vh - 6);
  const d = document.createElement('div'); d.className = 'demo-spot';
  Object.assign(d.style, {position: 'fixed', left: (r.left - pad) + 'px', top: top + 'px',
    width: (r.width + 2 * pad) + 'px', height: (bottom - top) + 'px', border: '2px solid #f0b429',
    borderRadius: '6px', boxShadow: '0 0 0 9999px rgba(0,0,0,0.5)', pointerEvents: 'none', zIndex: 2147483647});
  document.body.appendChild(d);
}"""


def main() -> None:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    durations = {cid: hold(c, LANG) for cid, c in CUE.items()} if (ARGS.silent or ARGS.check) else tts()
    intro, outro = intro_page(), outro_page()
    term = terminal_page()
    timeline: list[dict] = []
    missing: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        video = {} if ARGS.check else {"record_video_dir": str(OUT), "record_video_size": {"width": W, "height": H}}
        ctx = browser.new_context(viewport={"width": W, "height": H}, color_scheme="dark", **video)
        pg = ctx.new_page()
        t0 = time.monotonic()
        now = lambda: time.monotonic() - t0  # noqa: E731

        def target(spec: str):
            first = spec.startswith("first:")
            sel, _, txt = spec.removeprefix("first:").partition("@@")
            loc = pg.locator(sel)
            if txt:
                loc = loc.filter(has_text=txt.replace("{region}", REGION))
            n = loc.count()
            if n == 0 or (n > 1 and not first):
                return None, n
            return loc.first, n

        def cue(cid: str, during=None) -> None:
            c = CUE[cid]
            start = now()
            if c.spot:
                loc, n = target(c.spot)
                if loc is None:
                    msg = f"{cid}: {c.spot!r} matches {n} elements on {pg.url}"
                    if ARGS.check:
                        missing.append(msg)
                    else:
                        sys.exit(f"cannot frame {msg}")
                else:
                    loc.scroll_into_view_if_needed()
                    pg.evaluate(SPOT_JS, loc.element_handle())
                    pg.wait_for_timeout(250)
            if during is not None:
                during()
            remaining = 0.15 if ARGS.check else durations[cid] - (now() - start)
            if remaining > 0:
                pg.wait_for_timeout(int(remaining * 1000))
            pg.evaluate(SPOT_JS, None)
            timeline.append({"id": cid, "key": SCENE_OF[cid], "start": round(start, 2), "end": round(now(), 2)})

        def login(name: str, secret: str, cid: str | None = None) -> None:
            # A second on the form before anything is typed: the viewer has to
            # see that there is a sign-in at all.
            pg.context.clear_cookies()
            pg.goto(f"{BASE}/login")
            pg.wait_for_load_state("networkidle")
            pg.goto(f"{BASE}/locale/{LANG}?next=/login")
            pg.wait_for_load_state("networkidle")
            pg.wait_for_timeout(1000)

            def type_in():
                pg.type("input[name=name]", name, delay=70)
                pg.wait_for_timeout(300)
                pg.type("input[name=secret]", secret, delay=70)
                pg.wait_for_timeout(500)
            if cid:
                cue(cid, during=type_in)
            else:
                type_in()
            pg.click("button[type=submit], input[type=submit]")
            pg.wait_for_load_state("networkidle")

        def ask(question: str, cid: str) -> None:
            pg.goto(f"{BASE}/ask")
            pg.wait_for_load_state("networkidle")
            pg.wait_for_timeout(600)

            def type_in():
                pg.click("input[name=question]")
                pg.type("input[name=question]", question, delay=45)
            cue(cid, during=type_in)
            pg.press("input[name=question]", "Enter")

        def card(page: Path, ids: list[str]) -> None:
            pg.goto(page.as_uri())
            pg.wait_for_timeout(800)
            for cid in ids:
                cue(cid)

        def scene_intro():
            card(intro, ["intro.0", "intro.1", "intro.2", "intro.3"])

        def scene_cli():
            pg.goto(term.as_uri())
            pg.wait_for_timeout(1400)
            for cid in ("cli.0", "cli.1", "cli.2", "cli.3"):
                cue(cid)

        def scene_login():
            login(*SENIOR, cid="login.0")

        def scene_home():
            pg.goto(f"{BASE}/")
            pg.wait_for_load_state("networkidle")
            pg.wait_for_timeout(800)
            for cid in ("home.0", "home.1", "home.2", "home.3"):
                cue(cid)

        def scene_queue():
            pg.goto(f"{BASE}/queue")
            pg.wait_for_load_state("networkidle")
            pg.wait_for_timeout(800)
            for cid in ("queue.0", "queue.1", "queue.2"):
                cue(cid)

        def scene_region():
            pg.goto(f"{BASE}/c/{STEM}/{INDEX}")
            pg.wait_for_load_state("networkidle")
            pg.wait_for_timeout(800)
            for i in range(9):
                cue(f"region.{i}")

        def scene_defer():
            cue("defer.0")
            # The key handler submits the defer form; wait for that navigation
            # itself rather than issuing another one under it (net::ERR_ABORTED).
            with pg.expect_navigation(timeout=15000):
                pg.keyboard.press("0")
            pg.wait_for_load_state("networkidle")
            if "/deferred" not in pg.url:
                pg.goto(f"{BASE}/deferred")
                pg.wait_for_load_state("networkidle")
            pg.wait_for_timeout(600)
            cue("defer.1")

        def scene_blocked():
            login(*OPERATOR)
            pg.goto(f"{BASE}/c/{STEM}/{INDEX}")
            pg.wait_for_load_state("networkidle")
            pg.wait_for_timeout(800)
            for cid in ("blocked.0", "blocked.1", "blocked.2"):
                cue(cid)

        def scene_boards():
            login(*SENIOR)
            pg.goto(f"{BASE}/boards")
            pg.wait_for_load_state("networkidle")
            pg.wait_for_timeout(800)
            for cid in ("boards.0", "boards.1", "boards.2"):
                cue(cid)

        def scene_ask():
            if ARGS.check:
                pg.goto(f"{BASE}/ask")
                pg.wait_for_load_state("networkidle")
                cue("ask.0")
                return
            ask(Q_M32, "ask.0")
            pg.wait_for_selector("#progress", state="visible", timeout=30000)
            cue("ask.1")
            cue("ask.2")
            pg.wait_for_selector("figure.chart", timeout=180000)
            pg.wait_for_timeout(600)
            for cid in ("ask.3", "ask.4", "ask.5"):
                cue(cid)

        def scene_control():
            ask(Q_M31, "control.0")
            pg.wait_for_selector("figure.chart", timeout=180000)
            pg.wait_for_timeout(600)
            cue("control.1")
            cue("control.2")

        def scene_switch():
            cue("switch.0")
            pg.click("nav.locale a")
            pg.wait_for_load_state("networkidle")
            pg.wait_for_timeout(800)
            cue("switch.1")
            cue("switch.2")
            # The switch renders chrome; the answer is written again only when
            # asked, because a GET on a stored run must not cost a model call.
            pg.click("form.rewrite button")
            pg.wait_for_load_state("networkidle", timeout=180000)
            pg.wait_for_timeout(600)
            cue("switch.3")

        def scene_outro():
            card(outro, [f"outro.{i}" for i in range(5)])

        flow = [("intro", scene_intro), ("cli", scene_cli), ("login", scene_login), ("home", scene_home),
                ("queue", scene_queue), ("region", scene_region), ("defer", scene_defer), ("blocked", scene_blocked),
                ("boards", scene_boards), ("ask", scene_ask), ("control", scene_control), ("switch", scene_switch),
                ("outro", scene_outro)]
        assert [k for k, _ in flow] == [k for k, _ in SCENES], "the recorder and the shot list disagree on the scenes"
        for key, run in flow:
            if ARGS.check and key in MUTATING and key != "ask":
                continue
            run()
        pg.wait_for_timeout(1500)
        if ARGS.check:
            ctx.close()
            browser.close()
            checked = {t["id"] for t in timeline}
            skipped = [cid for cid, c in CUE.items() if c.spot and cid not in checked]
            print(f"checked {len(checked)} cues; not checked (a take changes these pages): {', '.join(skipped)}")
            if missing:
                sys.exit("missing:\n  " + "\n  ".join(missing))
            print("every framed element found")
            return
        video_path = pg.video.path()
        ctx.close()
        browser.close()

    take = OUT / "take.webm"
    shutil.move(video_path, take)
    (OUT / "timeline.json").write_text(json.dumps(timeline, indent=1))
    finish(take, timeline, durations)


def finish(video_path: Path, timeline: list[dict], durations: dict[str, float]) -> None:
    """Subtitles from the timeline into the band, then the silent cut or the narrated one."""
    def ts(s):
        h, m, sec = int(s // 3600), int(s % 3600 // 60), s % 60
        return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")
    # Where each cue's narration starts: its cue's start, unless the previous
    # one is still talking -- then 0.3 s after it ends. The silent cut keeps
    # its cue starts exactly, since its holds *are* the timeline.
    starts, prev_end = [], -10.0
    for t in timeline:
        start = t["start"] if ARGS.silent else max(t["start"], prev_end + 0.3)
        starts.append(start)
        prev_end = start + durations[t["id"]]
        if not ARGS.silent and prev_end > t["end"] + 1.0:
            print(f"warning: {t['id']} narration runs {prev_end - t['end']:.1f}s past its cue", file=sys.stderr)
    srt = []
    for i, (t, start) in enumerate(zip(timeline, starts), 1):
        end = t["end"] if ARGS.silent else max(start + durations[t["id"]], min(t["end"], start + durations[t["id"]] + 1.0))
        srt.append(f"{i}\n{ts(start)} --> {ts(end)}\n" + "\n".join(lines(CUE[t["id"]], LANG)) + "\n")
    (OUT / "subs.srt").write_text("\n".join(srt))
    font = "PingFang TC" if LANG == "zh-TW" else "Helvetica Neue"
    subs = str(OUT / "subs.srt").replace(":", "\\:")
    # libass sizes against a 288-line script by default, so FontSize=13 is
    # about 40 px on a 900 px frame: two lines and a margin in the 100 px band.
    band = (f"[0:v]pad={W}:{H + BAND}:0:0:color=0x0b0d12,subtitles='{subs}':force_style="
            f"'FontName={font},FontSize=13,Alignment=2,MarginV=12,BorderStyle=1,Outline=1,Shadow=0,"
            f"PrimaryColour=&H00F2F2F2,OutlineColour=&H00000000'[v]")
    encode = ["-c:v", "libx264", "-crf", "22", "-preset", "medium", "-pix_fmt", "yuv420p"]
    if ARGS.silent:
        final = ROOT / "docs" / "demo" / f"aoi-agent-demo-{TAG}-silent.mp4"
        subprocess.run([FFMPEG, "-y", "-i", str(video_path), "-filter_complex", band, "-map", "[v]", "-an",
                        *encode, str(final)], check=True, capture_output=True)
        rows = "\n".join(f"| {t['id']} | {t['start']:.1f}–{t['end']:.1f} | {text(CUE[t['id']], LANG)} |" for t in timeline)
        (OUT / "script.md").write_text(
            f"# {final.name} — 配音腳本 / dubbing script\n\n每一句字幕在畫面上的秒數，和要在那段時間裡講的話。"
            f"影片有字幕、沒有聲音；`subs.srt` 是同一份字幕檔。\n\n| 句 | 秒 | 台詞 |\n|---|---|---|\n{rows}\n")
        print("wrote", final, f"{final.stat().st_size/1e6:.1f} MB; cues:", len(timeline),
              "; length", timeline[-1]["end"], "s; script:", OUT / "script.md")
        return
    ext = "aiff" if ARGS.tts == "say" else "wav"
    inputs, delays = [], []
    for i, (t, start) in enumerate(zip(timeline, starts)):
        inputs += ["-i", str(NARR / f"{t['id']}.{ext}")]
        delays.append(f"[{i+1}:a]adelay={int(start*1000)}|{int(start*1000)}[a{i}]")
    # `apad` after the mix: `-shortest` otherwise ends the file where the last
    # cue's audio ends, and the last card holds longer than its line.
    mix = "".join(f"[a{i}]" for i in range(len(timeline))) + f"amix=inputs={len(timeline)}:normalize=0[mix];[mix]apad[narr]"
    final = ROOT / "docs" / "demo" / f"aoi-agent-demo-{TAG}.mp4"
    cmd = [FFMPEG, "-y", "-i", str(video_path), *inputs,
           "-filter_complex", ";".join(delays) + ";" + mix + ";" + band,
           "-map", "[v]", "-map", "[narr]", *encode, "-c:a", "aac", "-b:a", "128k", "-shortest", str(final)]
    subprocess.run(cmd, check=True, capture_output=True)
    print("wrote", final, f"{final.stat().st_size/1e6:.1f} MB; cues:", len(timeline), "; length", timeline[-1]["end"], "s")


def narrate_existing() -> None:
    """Narrate the take already recorded, on its own timeline: one wav per
    cue at the cue's start, the same subtitles in the band. Nothing is driven
    and nothing in the store changes."""
    take = OUT / "take.webm"
    timeline = json.loads((OUT / "timeline.json").read_text()) if (OUT / "timeline.json").exists() else []
    if not take.exists() or [t["id"] for t in timeline] != list(CUE):
        sys.exit(f"no take with a matching timeline under {OUT}; record one with --silent first")
    finish(take, timeline, tts())


if __name__ == "__main__":
    narrate_existing() if ARGS.narrate_existing else main()
