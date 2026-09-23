"""Was the planner's 28 -> 23 a change we made, or drift? Two sentences, one load.

Four readings of the independent seventy sit on two planner prompts and the
entry that compared them did not know it. `docs/benchmarks.md` for 2026-09-19
says the planner prompt was last changed on 2026-08-30; `bbe1df8` on
2026-09-02 changed `i18n.LANGUAGE_NOTE`, which `build_planning_messages`
appends to every planning call. Rendered against today's store, the prompt at
`68c7e23` (the 28/42 reading) hashes differently from the one at `6c09a60`,
`9eba15f` and HEAD (23, 26, 25 x 5), and those three are identical to the
byte. So 28 was read under one sentence and everything since under another.

This script does not run the planner. `scripts/analysis_eval.py --plan-only
--num-batch 512 --raw` runs it four times on one load, alternating
`AOI_LANGUAGE_NOTE=2026-08-30` and `current` (ABAB), and this script reads the
four files, the server log for the window, and returns a verdict written
before the run:

* **baseline_unstable** -- the two runs of one arm differ by more than one
  answered question, or plan more than two questions differently. Nothing
  below is read.
* **the_sentence** -- the old sentence answers at least 3 more of the 42 than
  the current one, averaged over its two runs. Three is the size of the move
  that was called drift.
* **not_the_sentence** -- the arms differ by at most 1.
* **unresolved** -- a difference of 2, which neither rule owns.

And a manipulation check that overrides all of them: the window must hold
exactly one model load and it must be `n_batch = 512`, or the run is
`invalid` -- the two settings plan 22 of 70 differently, and a reload
between arms would put the difference on the runner rather than the sentence.

Written before the run: the old sentence should read about 28 and the current
one about 25 on today's load, and A13 and A35 -- the two class-naming
questions the 09-17 run lost -- should come back under the old sentence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from planner_session_probe import LOG_PATH, window_facts

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("2026-08-30", "current")
ANSWERED_FLOOR = 1          # same-arm runs may differ by this many answered
MOVED_FLOOR = 2             # ... and plan this many questions differently
MOVED = 3                   # the old sentence must answer this many more
REQUIRED_BATCH = 512
PROMPT_HASH_2026_08_30_COMMIT = "68c7e23"


def summarise(rows: list[dict]) -> dict:
    """What one eval run says, in the eval's own terms."""
    answerable = [r for r in rows if not r.get("expect_refusal")]
    refusable = [r for r in rows if r.get("expect_refusal")]
    return {
        "answered": sum(1 for r in answerable if r["ok"]),
        "answerable": len(answerable),
        "refused": sum(1 for r in refusable if r["ok"]),
        "refusable": len(refusable),
        "no_plan": sum(1 for r in rows if r.get("no_plan")),
        "no_plan_ids": sorted(r["id"] for r in rows if r.get("no_plan")),
        "planned": {r["id"]: r["planned"] for r in rows},
        "ok": {r["id"]: bool(r["ok"]) for r in rows},
    }


def moved(a: dict, b: dict) -> list[str]:
    """Questions the two runs planned differently, by the scored calls."""
    return sorted(q for q in a["planned"] if a["planned"][q] != b["planned"].get(q))


def prompt_hash(arm: str) -> str:
    """The planning prompt under one arm, hashed, with today's store."""
    from aoi_agent.analysis.plan import store_domains
    from aoi_agent.analysis.prompts import build_planning_messages
    saved = os.environ.get("AOI_LANGUAGE_NOTE")
    os.environ["AOI_LANGUAGE_NOTE"] = arm
    try:
        msgs = build_planning_messages("最近 short 有比較多嗎?", store_domains())
    finally:
        if saved is None:
            os.environ.pop("AOI_LANGUAGE_NOTE", None)
        else:
            os.environ["AOI_LANGUAGE_NOTE"] = saved
    text = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in msgs)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def loads_in(lines: list[str], day: str, start: str, end: str) -> list[int | None]:
    """Every model load that began inside the window, by its n_batch."""
    stamp = re.compile(r"time=(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})")
    loads: list[int | None] = []
    open_load = False
    for line in lines:
        m = stamp.search(line)
        if m and "system memory" in line and "free=" in line:
            open_load = m.group(1) == day and start <= m.group(2) <= end
            if open_load:
                loads.append(None)
            continue
        if open_load:
            b = re.search(r"llama_context: n_batch *= (\d+)", line)
            if b:
                loads[-1] = int(b.group(1))
            if "loaded runners" in line:
                open_load = False
    return loads


def verdict(runs: list[dict], loads: list[int | None]) -> tuple[str, str]:
    """The pre-registered ladder, manipulation check first."""
    if loads != [REQUIRED_BATCH]:
        return ("invalid",
                f"The window holds {len(loads)} model load(s) at n_batch {loads}; the "
                f"comparison needs exactly one, at {REQUIRED_BATCH}. Not read.")
    by_arm = {arm: [r for r in runs if r["arm"] == arm] for arm in ARMS}
    if any(len(v) != 2 for v in by_arm.values()):
        return ("incomplete", "Each arm needs two runs.")
    for arm, pair in by_arm.items():
        gap = abs(pair[0]["answered"] - pair[1]["answered"])
        drift = len(moved(pair[0], pair[1]))
        if gap > ANSWERED_FLOOR or drift > MOVED_FLOOR:
            return ("baseline_unstable",
                    f"The two `{arm}` runs differ by {gap} answered and plan {drift} "
                    f"questions differently, past the {ANSWERED_FLOOR}/{MOVED_FLOOR} floor "
                    "one load was measured to hold. Nothing below is read.")
    old = sum(r["answered"] for r in by_arm["2026-08-30"]) / 2
    new = sum(r["answered"] for r in by_arm["current"]) / 2
    diff = old - new
    if diff >= MOVED:
        return ("the_sentence",
                f"The 2026-08-30 sentence answers {diff:+.1f} of 42 against the current one "
                f"on the same load ({old:.1f} against {new:.1f}), at least the {MOVED} the "
                "09-19 entry called drift. The 28 -> 23 move was the prompt change of "
                "2026-09-02, not the runner.")
    if diff <= 1:
        return ("not_the_sentence",
                f"The arms differ by {diff:+.1f} of 42 ({old:.1f} against {new:.1f}); the "
                "class-list sentence does not move the planner, and the 28 -> 23 spread "
                "keeps the cause this run does not contain.")
    return ("unresolved",
            f"The arms differ by {diff:+.1f} of 42 ({old:.1f} against {new:.1f}), between "
            "the two rules. Not called either way.")


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def render(record: dict) -> str:
    runs = record["runs"]
    lines = [
        f"## {record['date']} · commit {record['commit']}",
        "",
        "### Correction, and the two sentences read on one load — was 28 -> 23 the prompt?",
        "",
        "**Correction to the 2026-09-19 entry.** It says the planner prompt was last "
        "changed on 2026-08-30. `bbe1df8` (2026-09-02) changed `i18n.LANGUAGE_NOTE` to "
        "list the six classes, and `build_planning_messages` appends that sentence to "
        "every planning call. Rendered against today's store, the planning prompt at "
        f"`{PROMPT_HASH_2026_08_30_COMMIT}` (the 28/42 reading) hashes "
        f"`{record['prompt_hash']['2026-08-30']}` and the prompt at `6c09a60`, `9eba15f` "
        f"and HEAD hashes `{record['prompt_hash']['current']}` -- the three later readings "
        "(23, 26, 25 x 5) were made under one prompt and the first under another. "
        "The sentence \"nothing in this repository had moved between the two\" was "
        "false, and the ±5 spread it priced is two prompts and one unknown, not one "
        "unknown.",
        "",
        f"**The run.** `scripts/analysis_eval.py --plan-only --repeats 1 --num-batch "
        f"{REQUIRED_BATCH}` four times on one load, `AOI_LANGUAGE_NOTE` alternating "
        "`2026-08-30`, `current`, `2026-08-30`, `current`; the server log for the window "
        f"{record['window']['day']} {record['window']['start']}-{record['window']['end']} "
        f"holds {len(record['loads'])} load(s) at n_batch {record['loads']}. "
        "`scripts/language_note_arms.py` reads the four `--raw` files; rule and "
        "prediction in its docstring, written first.",
        "",
        "| run | sentence | answered / 42 | refused / 28 | no plan | planned differently from run 1 |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(runs):
        diff = len(moved(runs[0], r)) if i else "—"
        lines.append(f"| {i + 1} | `{r['arm']}` | {r['answered']} | {r['refused']} | "
                     f"{r['no_plan']} | {diff} |")
    lines += ["", f"**Verdict: {record['verdict']}.** {record['verdict_sentence']}", ""]
    if record["verdict"] not in ("invalid", "incomplete"):
        old, new = runs[0], runs[1]
        gained = sorted(q for q in old["ok"] if old["ok"][q] and not new["ok"].get(q))
        lost = sorted(q for q in old["ok"] if not old["ok"][q] and new["ok"].get(q))
        lines += [
            f"Questions right under the 2026-08-30 sentence and wrong under the current "
            f"one (run 1 against run 2): {', '.join(gained) or 'none'}. The reverse: "
            f"{', '.join(lost) or 'none'}. Planned differently between the two sentences: "
            f"{len(moved(old, new))} of 70 ({', '.join(moved(old, new)) or 'none'}); between "
            f"the two runs of the old sentence {len(moved(runs[0], runs[2]))}, of the "
            f"current {len(moved(runs[1], runs[3]))}.",
            "",
        ]
    timed_out = {r["arm"]: sorted({q for run in runs if run["arm"] == r["arm"]
                                    for q in run["no_plan_ids"]}) for r in runs}
    current = [r["answered"] for r in runs if r["arm"] == "current"]
    lines += [
        "**Read beside the verdict.** A question with no plan is a miss, and the "
        f"questions that produced none were: `2026-08-30` "
        f"{', '.join(timed_out['2026-08-30']) or 'none'}; `current` "
        f"{', '.join(timed_out['current']) or 'none'} -- a deadline, not a plan, so the "
        "old sentence's count is a floor by that many. The sentence does move plans "
        "without moving the score: the questions planned differently between the arms "
        "are listed above, and the same load's own repeat floor is beside them. And the "
        "current sentence on this load reads "
        f"{' and '.join(str(n) for n in current)} of 42 where the five same-day readings "
        "of 2026-09-22 read 25 on the same sentence and the same batch: one more "
        "between-session reading, on a prompt and a runner setting both now held fixed.",
        "",
        "**What this does not establish.** One load, one day, `--repeats 1` where the "
        "historical readings used 3 -- read as comparable because five same-load "
        "readings on 2026-09-22 agreed to the question and a straight repeat moved 0-1 "
        "of 70 on 2026-09-23. The 23/26/25 spread among the three readings on the "
        "current sentence is not addressed by this run: all three were 512 loads on one "
        "prompt, and whatever separates them is still not in any measurement here.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=4, required=True, metavar="ARM:FILE",
                        help="four `arm:path` pairs in the order they ran")
    parser.add_argument("--day", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "docs" / "benchmarks.md")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "language_note_arms.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runs = []
    for spec in args.runs:
        arm, _, path = spec.partition(":")
        if arm not in ARMS:
            print(f"unknown arm {arm!r}", file=sys.stderr)
            return 2
        runs.append({"arm": arm, "file": path, **summarise(json.loads(Path(path).read_text()))})
    if [r["arm"] for r in runs] != ["2026-08-30", "current", "2026-08-30", "current"]:
        print("the runs must be ABAB: 2026-08-30, current, 2026-08-30, current", file=sys.stderr)
        return 2

    lines = LOG_PATH.read_text(errors="replace").splitlines()
    loads = loads_in(lines, args.day, args.start, args.end)
    facts = window_facts(lines, args.day.replace("-", "/"), args.start, args.end)
    label, sentence = verdict(runs, loads)
    record = {
        "date": args.day, "commit": git_head(),
        "window": {"day": args.day, "start": args.start, "end": args.end},
        "loads": loads, "log_facts": facts,
        "prompt_hash": {arm: prompt_hash(arm) for arm in ARMS},
        "runs": [{k: v for k, v in r.items() if k != "planned"} | {"planned": r["planned"]}
                 for r in runs],
        "verdict": label, "verdict_sentence": sentence,
        "written": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    body = render(record)
    print(body)
    args.out.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"\n-> {args.out}")
    if args.dry_run:
        print("(dry run: nothing appended)")
        return 0
    args.benchmarks.write_text(args.benchmarks.read_text().rstrip() + "\n\n" + body.rstrip() + "\n")
    print(f"appended to {args.benchmarks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
