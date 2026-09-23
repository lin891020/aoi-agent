"""Why does one unchanged model answer differently on another day?

Three readings of the planner on unchanged code spanned 23-28 of 42 answered
(2026-08-30, 09-17, 09-19) and five back-to-back readings on 2026-09-22 were
identical -- the same score and the same 28 misses, five times. So the variance
is between sessions and not between runs, and the 09-23 entry drew the rule
that two arms have to be measured inside one session to be comparable.

What it could not say is why, and two candidates were already dead: the server
process has not restarted since 2026-08-27 (`/opt/homebrew/var/log/ollama.log`,
version 0.32.13 throughout) and every load behind those four readings carries
the same parameters -- 25/25 layers on GPU, `n_ctx` 32768, batch 512/512, flash
attention on, KV cache `q8_0`, a 12036.67 MiB GPU buffer. A five-minute probe
then found that the same question answered after different preceding requests
produces *different prose* and identical tool calls, each condition exactly
reproducible.

This asks the question the eval actually scores: **can the planned calls be
moved, inside one session, by changing what ran before them?**

Four conditions, in this order, one measurement each:

* **C1** baseline -- the model unloaded first, so the slot starts empty;
* **C2** straight after C1, nothing in between;
* **C3** after a fixed interposed workload (the same ten long questions, twenty
  times, hard-coded here so the interposition cannot itself drift);
* **C4** after `ollama stop` and a pause, so the slot is rebuilt.

The measure is **plan identity against C1**: of the seventy questions, how many
planned exactly the same tools with exactly the same arguments. The eval's own
score is reported beside it and is the coarser reading -- two different plans
can score the same.

**The verdict, written before the run.**

**C2 is the noise floor, not a prediction.** The rule first written here read
C3 against C1 directly and expected C2 to match C1 outright; a six-question
smoke run moved one question in C2, with nothing at all in between. A repeat is
therefore not free, which the eval's own determinism figure had already said
(59 of 70 plan the same tools across a run's three repeats). So each condition
is read against C2's move count rather than against zero, and the rule was
changed before any measured run, which is the only time it may be.

* **the sequence is the cause** -- C3 moves at least 3 questions more than C2;
* **the reload is the cause** -- C4 does, and C3 does not;
* **neither** -- neither C3 nor C4 moves more than 2 beyond C2.

Three is the increment because the three separate-day readings differ from each
other by 2 to 5 questions: a smaller move would not be the size of the thing
being explained. `kv_cache_type()` only records what the server was started
with; changing it is the scheduling script's job, not this one's.

**The steps mode, added the same day the conditions ran.** The four conditions
answered "neither" on a batch-512 load and, by accident, something else on a
batch-1024 one: the second arm was meant to change the KV cache type and did
not (`launchctl kickstart` does not re-read a plist's environment), but the
server had 18.6 GiB free when it loaded and chose `n_batch` 1024, and 22 of 70
plans went with it. So the variable is forced now rather than found: each step
unloads the model, sets `num_batch` and `use_mmap` on every request, measures
the seventy, and then reads the load the server actually made out of its own
log -- a step whose requested batch is not the batch in the log refuses to be
counted. Checkpoint churn (`created` / `erased` / `restored`) is read per step
from the same log, because the restores stopped on the 1024 load and that is
the mechanism candidate.

    uv run python scripts/planner_session_probe.py --steps 512 1024 512 1024 mmap0 512
    uv run python scripts/planner_session_probe.py --conditions C1 C2 --questions 6 --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aoi_agent.analysis.plan import PLAN_SCHEMA, store_domains  # noqa: E402
from aoi_agent.analysis.prompts import build_planning_messages  # noqa: E402
from aoi_agent.graph.flow import DEFAULT_MODEL  # noqa: E402
from aoi_agent.llm.ollama import OllamaClient  # noqa: E402

QUESTIONS = ROOT / "tests" / "fixtures" / "analysis_questions_independent.json"
OUT_JSON = ROOT / "models" / "planner_session_probe.json"
CONDITIONS = ("C1", "C2", "C3", "C4")

#: How far past the noise floor a condition has to move to count. The three
#: separate-day readings differ from one another by 2 to 5 questions, so 2 or
#: fewer is inside what a day of nothing produces. Read as an increment over
#: C2 -- a straight repeat with nothing in between -- because a repeat moves
#: questions on its own: one of six in the smoke run, and the eval's own
#: determinism figure is 59 of 70 across a run's three repeats.
MOVED = 3

#: The interposition, fixed so that it cannot itself drift between arms: ten
#: questions of the shape the station answers, asked in this order, this many
#: times. They are not the fixture's -- interposing the measurement's own
#: questions would warm exactly the prefixes being measured.
INTERPOSED = [
    "把 L1 這個禮拜每一台機台的缺陷組成整理給我，順便說明各類的允收規定。",
    "M22 最近的 open 比例跟其他機台比起來如何？請說明你怎麼算的。",
    "這個月每一條線的複判量各是多少？哪一條最花人力？",
    "LOT-2608003 這個批號從投入到現在的缺陷紀錄，請完整列出來。",
    "夜班跟白班的缺陷組成有沒有明顯不同？請分機台說明。",
    "short 這一類的允收標準是什麼？工單裡怎麼寫的？",
    "最近一次換參數之後，M32 的缺陷率有沒有改變？",
    "把三條線的 false call 比例排序，並說明資料涵蓋哪幾天。",
    "哪一台機台的 mousebite 最多？請一併說明它的班別分布。",
    "pin-hole 跟 spurious copper 的判定差在哪裡？請引用工單。",
]
INTERPOSED_ROUNDS = 20


def kv_cache_type() -> str:
    """What the running server was started with, read from its own log.

    The plist is what the *next* start will use; the log line is what this
    process actually has. A probe that reported the plist would label an arm by
    an intention rather than by the server serving it.
    """
    try:
        log = Path("/opt/homebrew/var/log/ollama.log").read_text(errors="replace")
    except OSError:
        return "unknown"
    for line in reversed(log.splitlines()):
        if "OLLAMA_KV_CACHE_TYPE:" in line:
            return line.split("OLLAMA_KV_CACHE_TYPE:")[1].split()[0]
    return "unknown"


def load_parameters() -> dict:
    """The current load's parameters, from the same log, for the entry to carry."""
    wanted = {"offloaded": "load_tensors: offloaded ", "n_ctx": "llama_context: n_ctx  ",
              "n_batch": "llama_context: n_batch  ", "flash_attn": "llama_context: flash_attn  "}
    found: dict[str, str] = {}
    try:
        lines = Path("/opt/homebrew/var/log/ollama.log").read_text(errors="replace").splitlines()
    except OSError:
        return found
    for line in reversed(lines):
        for key, marker in wanted.items():
            if key not in found and marker in line:
                found[key] = line.split("=")[-1].strip() if "=" in line else line.strip()
        if len(found) == len(wanted):
            break
    return found


def unload(model: str, pause: float = 5.0) -> None:
    subprocess.run(["ollama", "stop", model], capture_output=True, check=False)
    time.sleep(pause)


def plan_calls(client: OllamaClient, question: str, domains) -> str:
    """The planned tools and arguments, as one comparable string.

    Only the calls. The first version of this probe compared the whole
    response, which includes `interpretation`, and reported two conditions as
    planning differently when all they did was word the same plan differently.
    What the eval scores -- and what a wrong plan costs -- is the calls.
    """
    # Composed outside the try on purpose: a prompt this probe failed to build
    # is a defect in the probe, and reporting it in the table as "the model did
    # not answer" would put it where nobody looks for it.
    messages = build_planning_messages(question, domains, "zh-TW")
    try:
        result = client.chat(messages, think="low", response_format=PLAN_SCHEMA)
    # swallow: a call that fails is one of the outcomes being compared -- it has
    # to land in the table as "(no answer: ...)" beside the plans, and a probe
    # that aborted on the first timeout would lose the four conditions measured
    # before it. The exception type is kept in the cell.
    except Exception as error:      # noqa: BLE001
        return f"(no answer: {type(error).__name__})"
    text = getattr(result, "text", None) or getattr(result, "content", "") or str(result)
    try:
        calls = json.loads(text).get("calls", [])
    except (json.JSONDecodeError, AttributeError):
        return "(unparsed)"
    return " + ".join(
        f"{call.get('tool')}({json.dumps(call.get('args', {}), sort_keys=True, ensure_ascii=False)})"
        for call in calls) or "(refused)"


def interpose(client: OllamaClient, domains, rounds: int = INTERPOSED_ROUNDS, log=print) -> int:
    """The fixed workload between C2 and C3. Its answers are thrown away."""
    asked = 0
    for round_index in range(rounds):
        for question in INTERPOSED:
            plan_calls(client, question, domains)
            asked += 1
        log(f"  interposed round {round_index + 1}/{rounds}")
    return asked


def measure(client: OllamaClient, questions: list[dict], domains, log=print) -> dict[str, str]:
    plans = {}
    began = time.perf_counter()
    for index, question in enumerate(questions, 1):
        plans[question["id"]] = plan_calls(client, question["question"], domains)
        if index % 10 == 0:
            log(f"    {index}/{len(questions)} ({(time.perf_counter() - began) / 60:.1f} min)")
    return plans


def differences(baseline: dict[str, str], other: dict[str, str]) -> list[str]:
    """Question ids whose planned calls are not identical."""
    return sorted(qid for qid in baseline if baseline[qid] != other.get(qid))


def verdict(moves: dict[str, int], moved: int = MOVED, total: int = 70) -> tuple[str, str]:
    """The pre-registered reading: each condition against C2's noise floor."""
    if any(condition not in moves for condition in ("C2", "C3", "C4")):
        return "incomplete", ("Incomplete: the verdict reads C3 and C4 against C2's noise floor "
                              "and one of the three was not measured.")
    floor = moves["C2"]
    over = {c: moves[c] - floor for c in ("C3", "C4")}
    if over["C3"] >= moved:
        return "the_sequence", (
            f"The sequence is the cause: a straight repeat moves {floor} of {total} questions "
            f"and the same measurement after a fixed interposed workload moves {moves['C3']}, "
            f"{over['C3']} more -- inside one session, on one loaded model, with nothing in the "
            "repository changed.")
    if over["C4"] >= moved:
        return "the_reload", (
            f"The reload is the cause: the interposed workload moves {moves['C3']} against the "
            f"repeat's {floor}, inside the noise, while unloading and reloading the model moves "
            f"{moves['C4']} -- {over['C4']} past the floor.")
    return "neither", (
        f"Neither: a straight repeat moves {floor} of {total} questions, the interposed workload "
        f"{moves['C3']} and a reload {moves['C4']}, so neither is more than {moved - 1} past the "
        "floor and the spread across days has a cause this run does not contain.")


def publishable(conditions: list[str], questions: int) -> bool:
    """Only the full four-condition run over the whole fixture may append."""
    return tuple(conditions) == CONDITIONS and questions == 70


# ---- the steps mode: the runner as a controlled variable ----------------------

LOG_PATH = Path("/opt/homebrew/var/log/ollama.log")
#: The server log is stamped in local time, so the windows are read in it.
LOCAL = datetime.now(UTC).astimezone().tzinfo
GIN_CHAT = re.compile(r'\[GIN\] (\d{4}/\d{2}/\d{2}) - (\d{2}:\d{2}:\d{2}) \|.*"/api/chat"')
STEP_MOVED = 3


def parse_step(spec: str) -> dict:
    """`512`, `1024`, `mmap0` (512 without mmap) or `mmap1`."""
    if spec in ("512", "1024"):
        return {"label": f"batch {spec}", "num_batch": int(spec), "use_mmap": True}
    if spec in ("mmap0", "mmap1"):
        return {"label": f"batch 512, mmap {'on' if spec == 'mmap1' else 'off'}",
                "num_batch": 512, "use_mmap": spec == "mmap1"}
    raise SystemExit(f"unknown step {spec!r}; one of 512, 1024, mmap0, mmap1")


def window_facts(lines: list[str], day: str, start: str, end: str) -> dict:
    """What the server log says happened between two clock times on one day.

    The load block (n_batch, load_mode) is the last one that began inside the
    window; checkpoint lines are attributed to the next `/api/chat` request
    line after them, which is how the runner interleaves them.
    """
    facts = {"n_batch": None, "load_mode": None, "created": 0, "erased": 0, "restored": 0,
             "requests": 0}
    pending = {"created": 0, "erased": 0, "restored": 0}
    stamp = re.compile(r"time=(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})")
    inside_load = False
    for line in lines:
        m = stamp.search(line)
        if m and m.group(1) == day.replace("/", "-"):
            inside_load = start <= m.group(2) <= end and "system memory" in line and "free=" in line \
                or (inside_load and "loaded runners" not in line)
            if inside_load and "system memory" in line:
                facts["n_batch"] = facts["load_mode"] = None
        if inside_load:
            b = re.search(r"llama_context: n_batch *= (\d+)", line)
            if b:
                facts["n_batch"] = int(b.group(1))
            lm = re.search(r"load_mode = (\w+)", line)
            if lm:
                facts["load_mode"] = lm.group(1)
        if "checkpoint" in line:
            for key in pending:
                if key in line:
                    pending[key] += 1
        g = GIN_CHAT.search(line)
        if g:
            if g.group(1) == day and start <= g.group(2) <= end:
                facts["requests"] += 1
                for key in pending:
                    facts[key] += pending[key]
            pending = {"created": 0, "erased": 0, "restored": 0}
    return facts


def step_verdict(steps: list[dict], moved: int = STEP_MOVED, total: int = 70) -> tuple[str, str]:
    """The pre-registered reading of a `512 1024 512 1024 mmap0 512` sequence.

    Every step is read against the first baseline. Baselines must agree with
    it within the noise (2 or fewer), both forced-1024 steps must move at
    least `moved` questions and agree with *each other* within the noise, and
    the mmap step is read on its own. Anything else is reported as it came.
    """
    by = {}
    for step in steps:
        by.setdefault(step["spec"], []).append(step)
    bases = by.get("512", [])
    b1024 = by.get("1024", [])
    mm = by.get("mmap0", [])
    if len(bases) < 2 or len(b1024) < 2:
        return "incomplete", "Incomplete: the rule needs two baselines and two forced-1024 steps."
    if any(b["moved_vs_baseline"] > moved - 1 for b in bases[1:]):
        return "baseline_unstable", (
            f"Baseline unstable: a repeated batch-512 load moved "
            f"{max(b['moved_vs_baseline'] for b in bases[1:])} of {total} against the first, past "
            "the noise, so nothing else in this sequence can be read.")
    pair = b1024[1].get("moved_vs_same_spec")
    if all(b["moved_vs_baseline"] >= moved for b in b1024) and pair is not None and pair <= moved - 1:
        label, text = "batch_is_a_cause", (
            f"Batch size is a cause: two independent loads forced to 1024 move "
            f"{b1024[0]['moved_vs_baseline']} and {b1024[1]['moved_vs_baseline']} of {total} plans "
            f"against the 512 baseline and differ from each other on {pair}, while repeated 512 "
            f"loads differ from the baseline on at most "
            f"{max(b['moved_vs_baseline'] for b in bases[1:])}.")
    elif all(b["moved_vs_baseline"] >= moved for b in b1024):
        label, text = "batch_moves_but_not_reproducibly", (
            f"Batch 1024 moves {b1024[0]['moved_vs_baseline']} and {b1024[1]['moved_vs_baseline']} "
            f"plans but the two 1024 loads differ from each other on {pair}, so the load carries "
            "something besides its batch size.")
    else:
        label, text = "batch_is_not_a_cause", (
            f"Batch size is not a cause here: forced-1024 loads move "
            f"{[b['moved_vs_baseline'] for b in b1024]} of {total}, inside the noise.")
    if mm:
        n = mm[0]["moved_vs_baseline"]
        text += (f" Loading without mmap moves {n}: "
                 + ("a second cause." if n >= moved else "not a cause on its own."))
    return label, text


def run_steps(specs: list[str], questions: list[dict], domains, model: str, pause: float,
              log=print) -> list[dict]:
    """Each step: unload, force the runner's options, measure, verify against the log."""
    day = datetime.now(LOCAL).strftime("%Y/%m/%d")
    steps: list[dict] = []
    for index, spec in enumerate(specs):
        step = {"index": index, "spec": spec, **parse_step(spec)}
        unload(model, pause=pause)
        client = OllamaClient(model, load_options={"num_batch": step["num_batch"],
                                                   "use_mmap": step["use_mmap"]})
        log(f"step {index} ({step['label']}): measuring")
        started = datetime.now(LOCAL).strftime("%H:%M:%S")
        step["plans"] = measure(client, questions, domains, log=log)
        ended = datetime.now(LOCAL).strftime("%H:%M:%S")
        time.sleep(3)   # the runner writes the last request's log lines after answering
        facts = window_facts(LOG_PATH.read_text(errors="replace").splitlines(), day, started, ended) \
            if LOG_PATH.exists() else {}
        step.update({"started": started, "ended": ended, "log": facts})
        if facts.get("n_batch") != step["num_batch"]:
            raise SystemExit(f"step {index}: requested num_batch {step['num_batch']} but the "
                             f"server loaded with {facts.get('n_batch')}; refusing to count it")
        expected_mode = "mmap" if step["use_mmap"] else "none"
        if facts.get("load_mode") not in (expected_mode, None):
            raise SystemExit(f"step {index}: requested use_mmap={step['use_mmap']} but the "
                             f"server loaded with load_mode={facts.get('load_mode')}")
        baseline = steps[0]["plans"] if steps else step["plans"]
        step["moved_vs_baseline_ids"] = differences(baseline, step["plans"])
        step["moved_vs_baseline"] = len(step["moved_vs_baseline_ids"])
        same = [x for x in steps if x["spec"] == spec]
        step["moved_vs_same_spec"] = len(differences(same[-1]["plans"], step["plans"])) if same else None
        log(f"  moved vs baseline {step['moved_vs_baseline']}, log n_batch {facts.get('n_batch')} "
            f"load_mode {facts.get('load_mode')} restored {facts.get('restored')}")
        steps.append(step)
    return steps


def render_steps(record: dict) -> str:
    lines = [
        f"## {record['date']} · commit {record['commit']}",
        "",
        "### The runner as a controlled variable: batch size and mmap, forced, one at a time",
        "",
        f"The four-condition probe answered *neither* on a batch-512 load and, by accident, moved "
        f"22 of 70 on a batch-1024 one; the server chooses the batch by free memory (18.6 GiB on "
        f"this machine, 136 loads, no exception). Here the runner is forced through the request "
        f"options and read back from the server's log after every step; a step whose requested "
        f"batch is not the loaded one is refused. Same {record['questions']} questions, one repeat, "
        f"plans compared call by call against the first baseline. KV cache `{record['kv_cache']}`, "
        f"{record['wall_minutes']:.0f} min. `scripts/planner_session_probe.py --steps "
        f"{' '.join(record['specs'])}`.",
        "",
        f"**Verdict, by the rule written before the run: {record['verdict_sentence']}**",
        "",
        "| step | runner | moved vs first baseline | vs previous same runner | n_batch in log "
        "| load_mode | checkpoints restored / created |",
        "|---|---|---|---|---|---|---|",
    ]
    for st in record["steps"]:
        f = st["log"]
        same = "—" if st["moved_vs_same_spec"] is None else str(st["moved_vs_same_spec"])
        lines.append(f"| {st['index']} | {st['label']} | **{st['moved_vs_baseline']}** | {same} | "
                     f"{f.get('n_batch')} | {f.get('load_mode')} | {f.get('restored')} / {f.get('created')} |")
    lines.append("")
    for st in record["steps"]:
        if st["moved_vs_baseline_ids"]:
            lines.append(f"Step {st['index']} ({st['label']}) moved: {', '.join(st['moved_vs_baseline_ids'])}.")
    lines += [
        "",
        "**What this does not establish.** One model and one machine; two loads per forced setting, "
        "which separates a setting from a load but not from the day. The checkpoint counts are "
        "read beside the plans, not manipulated -- the server exposes no switch for them -- so "
        "they are consistent with the mechanism and do not prove it. And the historical readings "
        "this began from were all batch-512 loads, so whatever moved them is not the batch; the "
        "mmap step is the one remaining logged difference among them.",
        "",
    ]
    return "\n".join(lines)


def render(record: dict) -> str:
    moves = record["moves"]
    lines = [
        f"## {record['date']} · commit {record['commit']}",
        "",
        "### What moves the planner between sessions: the sequence, or the load?",
        "",
        f"Five back-to-back readings agreed question for question on 2026-09-22 and three "
        f"separate-day readings spanned 23-28 of 42, with the server process unrestarted since "
        f"2026-08-27 and every load carrying the same parameters. This asks whether the planned "
        f"*calls* can be moved inside one session. Four conditions in order, "
        f"{record['questions']} questions each, one repeat, plans compared call by call against "
        f"C1; KV cache `{record['kv_cache']}`, load {record['load']}. "
        f"{record['wall_minutes']:.0f} min. `scripts/planner_session_probe.py`.",
        "",
        f"**Verdict, by the rule written before the run: {record['verdict_sentence']}**",
        "",
        "| condition | what ran before it | questions planning differently from C1 |",
        "|---|---|---|",
        "| C1 | the model unloaded, slot empty | — (baseline) |",
        f"| C2 | C1, nothing else -- the noise floor | **{moves.get('C2', '—')}** |",
        f"| C3 | {record['interposed']} interposed questions | **{moves.get('C3', '—')}** |",
        f"| C4 | `ollama stop`, then a pause | **{moves.get('C4', '—')}** |",
        "",
    ]
    if record.get("moved_ids"):
        for condition, ids in record["moved_ids"].items():
            if ids:
                lines.append(f"Moved in {condition}: {', '.join(ids)}.")
        lines.append("")
    lines += [
        "**What this does not establish.** One model, one machine, one night, one repeat per "
        "question -- a question that plans two ways on its own would show here as a move, and "
        "the 2026-09-22 series is the reason to expect it would not. The interposed workload is "
        "fixed but arbitrary: it says that *some* preceding work moves the planner, never which "
        "property of it. And this measures the planned calls, which is what the eval scores; the "
        "prose moved under every condition already and nothing here counts that.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=int, default=70)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--steps", nargs="+", default=None,
                        help="the steps mode: a sequence such as 512 1024 512 1024 mmap0 512")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rounds", type=int, default=INTERPOSED_ROUNDS)
    parser.add_argument("--pause", type=float, default=30.0,
                        help="seconds between unloading the model and asking again (C4)")
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "docs" / "benchmarks.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.steps and not args.dry_run and not (args.questions == 70 and len(args.steps) >= 4):
        print("the steps mode appends only a full-fixture sequence of at least four steps; "
              "pass --dry-run", file=sys.stderr)
        return 2
    if not args.steps and not args.dry_run and not publishable(args.conditions, args.questions):
        print("only the full run (C1 C2 C3 C4 over all 70 questions) may append; pass --dry-run",
              file=sys.stderr)
        return 2

    questions = json.loads(QUESTIONS.read_text())[:args.questions]
    domains = store_domains()
    client = OllamaClient(args.model)
    started = time.perf_counter()

    if args.steps:
        steps = run_steps(args.steps, questions, domains, args.model, args.pause)
        label, sentence = step_verdict(steps, total=len(questions))
        record = {
            "date": datetime.now(UTC).date().isoformat(),
            "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                     text=True, check=False).stdout.strip() or "unknown",
            "model": args.model, "questions": len(questions), "specs": args.steps,
            "kv_cache": kv_cache_type(), "verdict": label, "verdict_sentence": sentence,
            "steps": steps, "wall_minutes": (time.perf_counter() - started) / 60,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        body = render_steps(record)
        print("\n" + body)
        if args.dry_run:
            print("(dry run -- nothing appended)")
            return 0
        args.benchmarks.write_text(args.benchmarks.read_text().rstrip() + "\n\n" + body.rstrip() + "\n")
        print(f"appended to {args.benchmarks.relative_to(ROOT)}")
        return 0

    plans: dict[str, dict[str, str]] = {}

    for condition in args.conditions:
        if condition == "C1":
            unload(args.model)
        if condition == "C3":
            print("interposing the fixed workload")
            interpose(client, domains, args.rounds)
        if condition == "C4":
            unload(args.model, pause=args.pause)
        print(f"{condition}: measuring")
        plans[condition] = measure(client, questions, domains)

    baseline = plans.get("C1", {})
    moved_ids = {c: differences(baseline, plans[c]) for c in plans if c != "C1"}
    moves = {c: len(ids) for c, ids in moved_ids.items()}
    label, sentence = verdict(moves, total=len(questions))
    record = {
        "date": datetime.now(UTC).date().isoformat(),
        "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                 text=True, check=False).stdout.strip() or "unknown",
        "model": args.model, "questions": len(questions), "conditions": args.conditions,
        "kv_cache": kv_cache_type(), "load": load_parameters(),
        "interposed": len(INTERPOSED) * args.rounds,
        "moves": moves, "moved_ids": moved_ids,
        "verdict": label, "verdict_sentence": sentence, "plans": plans,
        "wall_minutes": (time.perf_counter() - started) / 60,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    body = render(record)
    print("\n" + body)
    if args.dry_run:
        print("(dry run -- nothing appended)")
        return 0
    args.benchmarks.write_text(args.benchmarks.read_text().rstrip() + "\n\n" + body.rstrip() + "\n")
    print(f"appended to {args.benchmarks.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
