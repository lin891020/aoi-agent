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
real output. Everything lands under docs/demo/ (gitignored): the master
take, cards, narration and subtitles under build/<lang>/, and the cut beside them.
A take hands the demo region back (the defer scene presses 0) and asks two
questions on /ask; nothing else in the store changes.

macOS only: the mux uses the Homebrew ffmpeg-full build for its subtitle
filter, and `--tts say` uses the system voices.
"""
from __future__ import annotations

import argparse
import base64
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
from demo_script import CUES, SCENES, cue_id, cues_in, hold, lines, text

_args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_args.add_argument("--lang", default="zh-TW", choices=("zh-TW", "en"))
_args.add_argument("--stem", required=True, help="a board with a region on the queue")
_args.add_argument("--index", type=int, default=15, help="which of that board's regions is on the queue")
_args.add_argument("--silent", action="store_true",
                   help="no narration: the video with its subtitles, each cue held for its reading time, "
                        "plus script.md (cue, seconds, the line) for dubbing over it")
_args.add_argument("--narrate-existing", action="store_true",
                   help="do not drive the station: take build/<lang>/take.mp4 and timeline.json from the last "
                        "take, synthesise one narration per cue, and mux")
_args.add_argument("--check", action="store_true",
                   help="drive the pages a take does not change (cards, terminal, login, home, queue, region, "
                        "boards, the /ask form) and report every framed element that is missing; no video")
_args.add_argument("--base", default="http://127.0.0.1:8110")
_args.add_argument("--tts", default="auto", choices=("auto", "qwen3tts", "kokoro", "say"),
                   help="auto: video_transfer's own split -- Qwen3-TTS for Chinese (the one voice measured to read "
                        "mixed Chinese/English), Kokoro for English; or name a backend; say: macOS, no verification")
_args.add_argument("--video-transfer", default=str(Path.home() / "Projects" / "video_transfer"))
_args.add_argument("--fast-forward", type=float, default=3.0,
                   help="speed of the cut through a wait no cue covers (the model working, the page barely "
                        "moving); a badge in the corner says so")
_args.add_argument("--fast-max", type=float, default=10.0,
                   help="seconds a fast-forwarded wait may still take on screen; the speed rises past "
                        "--fast-forward to keep under it")
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

#: The page in CSS pixels, captured at SCALE device pixels per CSS pixel and
#: encoded at OUT_W wide; the band under it is where the subtitles go. The
#: capture is Chrome's own screencast, frame by frame, not Playwright's video
#: recorder: that one is VP8 at a fixed ~1 Mbps, and a page of 13 px text
#: came out of it at 460 kbps -- legible, and mush. Mike asked whether the
#: blur was his player. It was not.
W, H, SCALE = 1280, 800, 2
OUT_W = 1920
OUT_H = H * OUT_W // W
BAND = 100 * OUT_W // W
FPS_CAP = 15.0
#: Scenes a take changes something in -- `--check` does not run these.
MUTATING = {"defer", "blocked", "ask", "control", "switch"}

CUE = {cue_id(k, i): c for k, i, c in CUES}
SCENE_OF = {cue_id(k, i): k for k, i, _ in CUES}


def ids(key: str) -> list[str]:
    return [cue_id(k, i) for k, i, _ in CUES if k == key]

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
p.lead{font-size:21px;color:#aeb4c2;margin:0;max-width:1120px;text-wrap:balance}
.body{flex:1;display:flex;flex-direction:column;justify-content:center;padding-bottom:40px}
.flow{display:flex;align-items:stretch}
.flow .box{box-sizing:border-box;width:176px;padding:16px 14px;border:2px solid #2f3748;border-radius:10px;background:#141926;display:flex;flex-direction:column;gap:7px}
.flow .box b{font-size:27px;font-weight:600;line-height:1.15}
.flow .box i{font-style:normal;font-size:17px;color:#d0d4dd;line-height:1.35}
.flow .box small{font-size:14px;color:#8b93a5;line-height:1.3}
.flow .arrow{align-self:center;width:40px;text-align:center;font-size:32px;color:#6f778a;flex:none}
.flow .system{display:flex;align-items:stretch;padding:36px 12px 12px;border:2px dashed #4a5470;border-radius:14px;position:relative}
.flow .system>span{position:absolute;top:9px;left:16px;font-size:15px;color:#f0b429;letter-spacing:.04em}
.ask{display:flex;align-items:center;gap:18px;margin:22px 0 0 auto;padding:14px 22px;border:2px solid #2f3748;border-radius:10px;background:#141926;width:fit-content}
.ask b{font-size:22px;font-weight:600}
.ask i{font-style:normal;font-size:17px;color:#d0d4dd}
.figures{display:flex;gap:56px;margin-top:22px}
.figures div{display:flex;flex-direction:column;gap:6px}
.figures b{font-size:64px;font-weight:600;line-height:1;font-variant-numeric:tabular-nums}
.figures span{font-size:20px;color:#aeb4c2;max-width:320px}
.data{margin:0 0 6px;color:#aeb4c2;font-size:19px;width:fit-content;padding:4px 8px 4px 0}
.foot{margin-top:auto;color:#8b93a5;font:20px "SF Mono",Menlo,monospace}
"""


def card_page(name: str, title: str, lead: str, html: str = "",
              figures: list[tuple[str, str]] | None = None, foot: str = "") -> Path:
    body = f"<h1>{title}</h1><p class='lead' id='f-lead'>{lead}</p>{html}"
    if figures:
        body += "<div class='figures'>" + "".join(f"<div><b>{n}</b><span>{label}</span></div>" for n, label in figures) + "</div>"
    if foot:
        body += f"<p class='foot'>{foot}</p>"
    path = OUT / f"{name}.html"
    path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CARD_CSS}</style></head>"
                    f"<body><div class='card'>{body}</div></body></html>")
    return path


def intro_page() -> Path:
    """The flow as five boxes a viewer can read from across a room, each with
    an id the intro's cues frame in turn. It replaced the README's flow
    diagram scaled into the card, which forty seconds of narration pointed
    at nothing on and which the screencast could not keep sharp."""
    zh = LANG == "zh-TW"
    title = "AOI 複判站" if zh else "AOI re-verification station"
    lead = ("AOI 標出的區域，六成是誤報，卻每一個都要人工複判。本系統在 DeepPCB 上減少 55.6% 的複判工作。" if zh else
            "Six in ten regions an AOI flags are false calls, and every one goes to a person. "
            "On DeepPCB this system removes 55.6% of that review.")
    # Said out loud once (intro.9) and written where the cue can frame it:
    # the dataset is public, the AOI is a simulator, the line records are
    # seeded. The page footnotes say so; a viewer who finds it there first
    # reads the intro's "behind the AOI the line already has" as a claim.
    data = ("示範資料：DeepPCB 公開資料集；AOI 與產線紀錄為模擬。" if zh else
            "Demo data: the public DeepPCB set; the AOI and the line records are simulated.")

    def box(id_: str, name: str, what: str, note: str) -> str:
        return f"<div class='box' id='{id_}'><b>{name}</b><i>{what}</i><small>{note}</small></div>"
    arrow = "<div class='arrow'>&rarr;</div>"
    if zh:
        boxes = [box("f-aoi", "AOI", "自動光學檢測", "Automated Optical Inspection<br>拍照找瑕疵；寧可誤報，不漏檢"),
                 box("f-model", "視覺模型", "先判定", "多數區域幾毫秒內處置"),
                 box("f-agent", "agent", "沒把握的，由它接手", "解釋給人看"),
                 box("f-operator", "作業員", "兩者都無法判定的，才交給人", "在複判站作答"),
                 box("f-label", "訓練標註", "作業員的判定記錄下來", "下一輪訓練的資料")]
        system, ask = "本系統 · 接在既有的 AOI 之後", "<b>主管</b><i>直接用中文問產線的問題 &rarr; 產線查詢</i>"
    else:
        boxes = [box("f-aoi", "AOI", "Automated Optical Inspection", "the camera that checks boards;<br>over-flags rather than miss"),
                 box("f-model", "vision model", "decides first", "most regions in milliseconds"),
                 box("f-agent", "agent", "takes what it is unsure of", "explains, for a person"),
                 box("f-operator", "operator", "only what neither can settle", "answers at the station"),
                 box("f-label", "training label", "the verdict is recorded", "for the next round")]
        system, ask = "this system · behind the AOI the line already has", "<b>supervisor</b><i>asks the line a question in plain words &rarr; line analytics</i>"
    flow = (f"<div class='body'><div class='flow'>{boxes[0]}{arrow}<div class='system' id='f-system'><span>{system}</span>"
            f"{boxes[1]}{arrow}{boxes[2]}{arrow}{boxes[3]}</div>{arrow}{boxes[4]}</div>"
            f"<div class='ask' id='f-ask'>{ask}</div></div><p class='data' id='f-data'>{data}</p>")
    return card_page("intro", title, lead, html=flow, foot=REPO)


def outro_page() -> Path:
    zh = LANG == "zh-TW"
    return card_page(
        "outro",
        "數字都在 README" if zh else "The numbers are in the README",
        "DeepPCB 測試集；判定門檻不在報告成績的那份資料上選定。" if zh else "DeepPCB test split; the threshold was chosen off the split it is reported against.",
        figures=[("55.6%", "人工複判工作量減少" if zh else "of manual review removed"),
                 ("0.66%", "漏檢率（預算 0.5%）" if zh else "escape (budget 0.5%)"),
                 ("85.9%", "的區域不經過 LLM" if zh else "of regions never reach an LLM")],
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
#hdr,#first,#fold,#queued,#board{{width:fit-content;max-width:1150px;padding:6px 14px;margin:6px 0 6px -14px}}
div div{{white-space:pre-wrap;overflow-wrap:anywhere}} .hidden{{display:none}}
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
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    durations = {cid: hold(c, LANG) for cid, c in CUE.items()} if (ARGS.silent or ARGS.check) else tts()
    intro, outro = intro_page(), outro_page()
    term = terminal_page()
    timeline: list[dict] = []
    missing: list[str] = []

    frames_dir = OUT / "frames"
    frames: list[tuple[float, Path]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        ctx = browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=SCALE, color_scheme="dark")
        pg = ctx.new_page()
        t0 = time.monotonic()
        now = lambda: time.monotonic() - t0  # noqa: E731
        if not ARGS.check:
            shutil.rmtree(frames_dir, ignore_errors=True)
            frames_dir.mkdir(parents=True)
            cdp = ctx.new_cdp_session(pg)

            def on_frame(params):
                # Chrome sends a frame on every repaint and nothing while the
                # page is still; each one is acked, and one in every 1/FPS_CAP
                # seconds is kept, stamped on arrival in the take's own clock.
                # Not with the frame's own `metadata.timestamp`: that is the
                # renderer's clock, and a take that crosses from file:// cards
                # to the station's pages crosses renderer processes -- the
                # first take mastered that way ran 29 s long and drifted
                # scene by scene.
                cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})
                ts = now()
                data = base64.b64decode(params["data"])
                if frames and ts - frames[-1][0] < 1.0 / FPS_CAP:
                    # Within a tick of the last kept frame: this one *replaces*
                    # it rather than being dropped. A static page arrives as
                    # two frames a few ms apart -- the blank commit and the
                    # paint -- and dropping the second left the intro card
                    # black for twenty seconds and the outro card never shown.
                    frames[-1][1].write_bytes(data)
                    return
                path = frames_dir / f"{len(frames):06d}.jpg"
                path.write_bytes(data)
                frames.append((ts, path))
            cdp.on("Page.screencastFrame", on_frame)
            cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": W * SCALE,
                                              "maxHeight": H * SCALE, "everyNthFrame": 1})

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

        def cue(cid: str, during=None, optional: bool = False) -> None:
            c = CUE[cid]
            start = now()
            if c.spot:
                try:
                    loc, n = target(c.spot)
                    if loc is None:
                        msg = f"{cid}: {c.spot!r} matches {n} elements on {pg.url}"
                        if ARGS.check:
                            missing.append(msg)
                        elif optional:
                            print(f"note: {msg}; said without a frame", file=sys.stderr)
                        else:
                            sys.exit(f"cannot frame {msg}")
                    else:
                        # A wait cue is said over a page that may be mid-navigation;
                        # two seconds is the most framing it may cost before the
                        # cue goes up unframed.
                        patience = 2000 if optional else 30000
                        loc.scroll_into_view_if_needed(timeout=patience)
                        pg.evaluate(SPOT_JS, loc.element_handle(timeout=patience))
                        pg.wait_for_timeout(250)
                except PlaywrightError as exc:  # the page moved under a wait cue: say it unframed
                    if not optional:
                        raise
                    print(f"note: {cid}: {exc.__class__.__name__} while framing; said without a frame", file=sys.stderr)
            if during is not None:
                during()
            remaining = 0.15 if ARGS.check else durations[cid] - (now() - start)
            if remaining > 0:
                pg.wait_for_timeout(int(remaining * 1000))
            try:
                pg.evaluate(SPOT_JS, None)
            except PlaywrightError:
                pass  # the page navigated away with the frame on it
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
            card(intro, ids("intro"))

        def scene_cli():
            pg.goto(term.as_uri())
            pg.wait_for_timeout(1400)
            for cid in ids("cli"):
                cue(cid)

        def scene_login():
            login(*SENIOR, cid="login.0")

        def scene_home():
            pg.goto(f"{BASE}/")
            pg.wait_for_load_state("networkidle")
            pg.wait_for_timeout(800)
            for cid in ids("home"):
                cue(cid)

        def scene_queue():
            pg.goto(f"{BASE}/queue")
            pg.wait_for_load_state("networkidle")
            pg.wait_for_timeout(800)
            for cid in ids("queue"):
                cue(cid)

        def scene_region():
            pg.goto(f"{BASE}/c/{STEM}/{INDEX}")
            pg.wait_for_load_state("networkidle")
            pg.wait_for_timeout(800)
            for cid in ids("region"):
                cue(cid)

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
            for cid in ids("blocked"):
                cue(cid)

        def scene_boards():
            login(*SENIOR)
            pg.goto(f"{BASE}/boards")
            pg.wait_for_load_state("networkidle")
            pg.wait_for_timeout(800)
            for cid in ids("boards"):
                cue(cid)

        def say_while_waiting(key: str, done) -> None:
            # The cues for the wait, until the result is up. Whatever wait is
            # left after the last one is fast-forwarded in the cut; a cue the
            # result overtakes is simply not said.
            for cid in cues_in(key, "wait"):
                if done():
                    break
                cue(cid, optional=True)

        def chart_up() -> bool:
            return pg.locator("figure.chart").count() > 0

        def scene_ask():
            if ARGS.check:
                pg.goto(f"{BASE}/ask")
                pg.wait_for_load_state("networkidle")
                cue("ask.0")
                return
            ask(Q_M32, "ask.0")
            pg.wait_for_selector("#progress", state="visible", timeout=30000)
            say_while_waiting("ask", chart_up)
            pg.wait_for_selector("figure.chart", timeout=180000)
            pg.wait_for_timeout(600)
            for cid in cues_in("ask", "after"):
                cue(cid)

        def scene_control():
            ask(Q_M31, "control.0")
            pg.wait_for_selector("#progress", state="visible", timeout=30000)
            say_while_waiting("control", chart_up)
            pg.wait_for_selector("figure.chart", timeout=180000)
            pg.wait_for_timeout(600)
            for cid in cues_in("control", "after"):
                cue(cid)

        def scene_switch():
            cue("switch.0")
            pg.click("nav.locale a")
            pg.wait_for_load_state("networkidle")
            pg.wait_for_timeout(800)
            for cid in cues_in("switch", "before"):
                cue(cid)
            # The switch renders chrome; the answer is written again only when
            # asked, because a GET on a stored run must not cost a model call.
            # The POST holds until the model is done, so the click must not
            # wait on it: the old page stays up meanwhile, and the wait cues
            # are said over it.
            pg.click("form.rewrite button", no_wait_after=True)
            say_while_waiting("switch", lambda: pg.locator("form.rewrite").count() == 0)
            pg.wait_for_load_state("networkidle", timeout=180000)
            pg.wait_for_selector("form.rewrite", state="detached", timeout=180000)
            pg.wait_for_timeout(600)
            for cid in cues_in("switch", "after"):
                cue(cid)

        def scene_outro():
            card(outro, ids("outro"))

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
        end = now() + 0.5
        cdp.send("Page.stopScreencast")
        ctx.close()
        browser.close()

    (OUT / "timeline.json").write_text(json.dumps(timeline, indent=1))
    take = master(frames, end)
    finish(take, timeline, durations)


def master(frames: list[tuple[float, Path]], end: float) -> Path:
    """The take as one constant-rate H.264 file at OUT_W wide: for every
    fortieth of a second up to `end`, the latest screencast frame on screen
    at that instant, as a hard link, so the master's clock is the take's by
    construction and no demuxer's idea of a duration comes into it. The cut
    is made from this and `--narrate-existing` reads it again; the frames go
    once the length has been checked."""
    if not frames:
        sys.exit("no frames arrived from the screencast")
    seq = frames[0][1].parent / "seq"
    shutil.rmtree(seq, ignore_errors=True)
    seq.mkdir()
    ticks = int(round(max(end, frames[-1][0] + 0.5) * 25))
    i = 0
    for k in range(ticks):
        while i + 1 < len(frames) and frames[i + 1][0] <= k / 25:
            i += 1
        os.link(frames[i][1], seq / f"{k:06d}.jpg")
    take = OUT / "take.mp4"
    subprocess.run([FFMPEG, "-y", "-framerate", "25", "-i", str(seq / "%06d.jpg"),
                    "-vf", f"scale={OUT_W}:{OUT_H}:flags=lanczos", "-c:v", "libx264", "-crf", "16",
                    "-preset", "medium", "-pix_fmt", "yuv420p", str(take)], check=True, capture_output=True)
    probe = subprocess.run([str(Path(FFMPEG).with_name("ffprobe")), "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(take)], capture_output=True, text=True, check=False).stdout.strip()
    if abs(float(probe) - ticks / 25) > 0.5:
        sys.exit(f"master is {probe} s for {ticks / 25:.1f} s of take; frames kept under {frames[0][1].parent}")
    shutil.rmtree(frames[0][1].parent, ignore_errors=True)
    print(f"master: {take} from {len(frames)} frames over {ticks / 25:.1f} s, {take.stat().st_size/1e6:.1f} MB")
    return take


def cuts(timeline: list[dict]) -> list[tuple[float, float | None, float]]:
    """The take in segments with a speed each: real time while a cue is up and
    for a second either side of it; the wait between two cues, when it is
    longer than six seconds, at `--fast-forward` or faster. The page barely
    moves during those waits -- a spinner and a phase label -- so what the
    viewer needs is the start and the end, not the middle."""
    segs: list[tuple[float, float | None, float]] = []
    cursor = 0.0
    for a, b in zip(timeline, timeline[1:]):
        s, e = a["end"] + 1.0, b["start"] - 1.0
        if e - s >= 6.0:
            segs.append((cursor, s, 1.0))
            segs.append((s, e, max(ARGS.fast_forward, (e - s) / ARGS.fast_max)))
            cursor = e
    segs.append((cursor, None, 1.0))
    return segs


def remap(segs):
    """Take time -> cut time, for the subtitles, the script and the narration."""
    marks, out = [], 0.0
    for s, e, f in segs:
        marks.append((s, e, f, out))
        if e is not None:
            out += (e - s) / f

    def to_out(t: float) -> float:
        for s, e, f, o in marks:
            if e is None or t < e:
                return o + (t - s) / f
        return out
    return to_out


def finish(video_path: Path, timeline: list[dict], durations: dict[str, float]) -> None:
    """The cut: waits fast-forwarded, subtitles in the band, then silent or narrated."""
    def ts(s):
        h, m, sec = int(s // 3600), int(s % 3600 // 60), s % 60
        return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")
    segs = cuts(timeline)
    to_out = remap(segs)
    (OUT / "cuts.json").write_text(json.dumps(
        [{"from": s, "to": e, "speed": round(f, 2), "at": round(to_out(s), 2)} for s, e, f in segs], indent=1))
    # Where each cue's narration starts, in cut time: its cue's start, unless
    # the previous one is still talking -- then 0.3 s after it ends. The
    # silent cut keeps its cue starts exactly, since its holds *are* the
    # timeline.
    starts, prev_end = [], -10.0
    for c in timeline:
        at = to_out(c["start"])
        start = at if ARGS.silent else max(at, prev_end + 0.3)
        starts.append(start)
        prev_end = start + durations[c["id"]]
        if not ARGS.silent and prev_end > to_out(c["end"]) + 1.0:
            print(f"warning: {c['id']} narration runs {prev_end - to_out(c['end']):.1f}s past its cue", file=sys.stderr)
    srt = []
    for i, (c, start) in enumerate(zip(timeline, starts), 1):
        until = to_out(c["end"])
        end = until if ARGS.silent else max(start + durations[c["id"]], min(until, start + durations[c["id"]] + 1.0))
        srt.append(f"{i}\n{ts(start)} --> {ts(end)}\n" + "\n".join(lines(CUE[c["id"]], LANG)) + "\n")
    (OUT / "subs.srt").write_text("\n".join(srt))

    # The video: every segment trimmed and re-timed, joined, then the band.
    n = len(segs)
    graph = [f"[0:v]split={n}" + "".join(f"[i{k}]" for k in range(n))]
    for k, (s, e, f) in enumerate(segs):
        trim = f"trim=start={s:.3f}" + (f":end={e:.3f}" if e is not None else "")
        graph.append(f"[i{k}]{trim},setpts=(PTS-STARTPTS)/{f:.4f}[c{k}]")
    graph.append("".join(f"[c{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=0,fps=25[vc]")
    font = "PingFang TC" if LANG == "zh-TW" else "Helvetica Neue"
    subs = str(OUT / "subs.srt").replace(":", "\\:")
    # libass sizes against a 288-line script by default, so FontSize=13 is
    # about 40 px on a 900 px frame: two lines and a margin in the 100 px band.
    band = (f"[vc]pad={OUT_W}:{OUT_H + BAND}:0:0:color=0x0b0d12,subtitles='{subs}':force_style="
            f"'FontName={font},FontSize=13,Alignment=2,MarginV=12,BorderStyle=1,Outline=1,Shadow=0,"
            f"PrimaryColour=&H00F2F2F2,OutlineColour=&H00000000'")
    for s, e, f in segs:
        if f > 1.0:
            band += (f",drawtext=text='>> {f:g}x':fontfile=/System/Library/Fonts/Helvetica.ttc:"
                     f"fontsize={26 * OUT_W // W}:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8:"
                     f"x=w-tw-{20 * OUT_W // W}:y={64 * OUT_W // W}:"
                     f"enable='between(t,{to_out(s):.2f},{to_out(e):.2f})'")
    graph.append(band + "[v]")
    encode = ["-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    length = to_out(timeline[-1]["end"])
    if ARGS.silent:
        final = ROOT / "docs" / "demo" / f"aoi-agent-demo-{TAG}-silent.mp4"
        subprocess.run([FFMPEG, "-y", "-i", str(video_path), "-filter_complex", ";".join(graph), "-map", "[v]", "-an",
                        *encode, str(final)], check=True, capture_output=True)
        rows = "\n".join(f"| {c['id']} | {to_out(c['start']):.1f}–{to_out(c['end']):.1f} | {text(CUE[c['id']], LANG)} |"
                          for c in timeline)
        fast = ", ".join(f"{to_out(s):.0f}–{to_out(e):.0f} s ({f:g}x)" for s, e, f in segs if f > 1.0) or "none"
        (OUT / "script.md").write_text(
            f"# {final.name} — 配音腳本 / dubbing script\n\n每一句字幕在成片裡的秒數，和要在那段時間裡講的話。"
            f"影片有字幕、沒有聲音；`subs.srt` 是同一份字幕檔。加速的段落（右上角有標）：{fast}。\n\n"
            f"| 句 | 秒 | 台詞 |\n|---|---|---|\n{rows}\n")
        print("wrote", final, f"{final.stat().st_size/1e6:.1f} MB; cues:", len(timeline),
              f"; length {length:.1f} s (take {timeline[-1]['end']:.1f} s); script:", OUT / "script.md")
        return
    ext = "aiff" if ARGS.tts == "say" else "wav"
    inputs, delays = [], []
    for i, (c, start) in enumerate(zip(timeline, starts)):
        inputs += ["-i", str(NARR / f"{c['id']}.{ext}")]
        delays.append(f"[{i+1}:a]adelay={int(start*1000)}|{int(start*1000)}[a{i}]")
    # The last lines of a voice slower than the holds run past the picture;
    # the picture's last frame is held until they are done, plus a breath.
    spoken_end = max(start + durations[c["id"]] for c, start in zip(timeline, starts))
    master_len = float(subprocess.run([str(Path(FFMPEG).with_name("ffprobe")), "-v", "error", "-show_entries",
                                       "format=duration", "-of", "csv=p=0", str(video_path)],
                                      capture_output=True, text=True, check=False).stdout.strip() or 0)
    extra = spoken_end + 1.0 - to_out(master_len)
    if extra > 0:
        graph[-2] = graph[-2].replace(",fps=25[vc]", f",fps=25,tpad=stop_mode=clone:stop_duration={extra:.2f}[vc]")
        print(f"tail held {extra:.1f}s for the last lines", file=sys.stderr)
    # `apad` after the mix: `-shortest` otherwise ends the file where the last
    # cue's audio ends, and the last card holds longer than its line.
    mix = "".join(f"[a{i}]" for i in range(len(timeline))) + f"amix=inputs={len(timeline)}:normalize=0[mix];[mix]apad[narr]"
    final = ROOT / "docs" / "demo" / f"aoi-agent-demo-{TAG}.mp4"
    cmd = [FFMPEG, "-y", "-i", str(video_path), *inputs,
           "-filter_complex", ";".join(delays) + ";" + mix + ";" + ";".join(graph),
           "-map", "[v]", "-map", "[narr]", *encode, "-c:a", "aac", "-b:a", "128k", "-shortest", str(final)]
    subprocess.run(cmd, check=True, capture_output=True)
    print("wrote", final, f"{final.stat().st_size/1e6:.1f} MB; cues:", len(timeline), f"; length {length:.1f} s")


def narrate_existing() -> None:
    """Narrate the take already recorded, on its own timeline: one wav per
    cue at the cue's start, the same subtitles in the band. Nothing is driven
    and nothing in the store changes."""
    take = OUT / "take.mp4"
    timeline = json.loads((OUT / "timeline.json").read_text()) if (OUT / "timeline.json").exists() else []
    if not take.exists() or not timeline or any(c["id"] not in CUE for c in timeline):
        sys.exit(f"no take with a matching timeline under {OUT}; record one with --silent first")
    finish(take, timeline, tts())


if __name__ == "__main__":
    narrate_existing() if ARGS.narrate_existing else main()
