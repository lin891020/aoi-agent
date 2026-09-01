"""Is the operator's rationale true of the evidence it was written from?

Two answers in this project's own question bank say the same thing: the LLM's
written explanation has never been evaluated, and it is the next thing to
measure. `scripts/agent_eval.py` scored the *verdict* and took it off the
decision path; what the model still writes is the only sentence on a queue row
an operator reads, and nothing has ever asked whether it is true.

The question is deliberately the analysis path's question, one path over.
**Not** "is the rationale good", which needs a person who knows the line. "Is
the rationale true of what the model was shown" -- and because the reason node
composes one prompt from the classifier's reading, the production context and
the retrieved criteria, most of that is set comparison and arithmetic.

**No model judges another model here.** A model judge is the same class of
instrument as the thing under test, graded by nobody, and its agreement with
the writer is indistinguishable from a shared blind spot. That is
`synthesis_eval.py`'s rule and it holds here for the same reason.

Three kinds are checked -- exactly, reproducibly, from what the run stored:

- `unsourced_figure` -- a number in the rationale that renders from no number
  in the prompt. `graph/rationale_check.py` already computes this per run and
  shows it to the operator; nothing has ever aggregated it over a population.
- `foreign_document` -- the rationale cites a `WI-xxx` that retrieval never
  returned. This is the 2026-08-23 incident's shape: eight stored explanations
  quoting a rule from another class's document. Retrieval is scoped now, so a
  non-zero count here would mean the model is naming documents from memory.
- `no_explanation` -- the run produced no rationale at all
  (`explanation_status` is `timed_out`, `unreachable` or `unparsed`).

Three are pattern matches that raise a candidate for a person, published with
the sentence attached so the judgement is auditable rather than asserted:

- `class_not_named` -- the classifier's class never appears. `i18n.LANGUAGE_NOTE`
  instructs both languages to spell identifiers as the data spells them, so
  this is also a check of whether that instruction is obeyed.
- `other_class_named` -- a different defect class appears. Often legitimate
  ("not a short"), which is why it is a flag.
- `limit_for_a_zero_tolerance_class` -- an acceptance-limit-shaped phrase with
  a figure, on an `open` or `short`, whose work instructions admit none.

Both languages, always. A rationale is a record written once in the line's
language, so the two runs are two populations rather than one translated -- but
they share the candidates, the classifier's readings and the retrieved
criteria, which is what makes the comparison a cross-check. Publishing a
single-language report is refused, for `synthesis_eval.py`'s reason: a figure
written from one surface reads as a claim about the system and is a claim about
half of it.

    uv run python scripts/rationale_eval.py                     # ~35 min
    uv run python scripts/rationale_eval.py --candidates 6 --dry-run

Ground truth is never read. This scores the prose against the prompt, not
against the board.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from aoi_agent.graph.flow import DEFAULT_MODEL, build_graph  # noqa: E402
from aoi_agent.i18n import LINE_LANGUAGE_ENV  # noqa: E402
from aoi_agent.llm.ollama import EXPLANATION_DEADLINE_S, OllamaClient  # noqa: E402
from agent_eval import investigated  # noqa: E402

#: Other work on this machine cannot change what a rationale says, but it can
#: turn a call into a timeout -- and `no_explanation` is one of the counted
#: kinds. `ollama ps` alone is not the check (see CLAUDE.md): a torch job
#: saturates the same silicon while that comes back clean.
BUSY = ("python", "torch", "ffmpeg", "train.py")


def contention() -> dict:
    """What else was on this machine, recorded rather than assumed."""
    def command(argv: list[str]) -> str:
        try:
            return subprocess.check_output(argv, text=True, timeout=20).strip()
        except Exception as error:  # noqa: BLE001 - a missing tool is not a failure
            return f"<{type(error).__name__}>"

    resident = command(["ollama", "ps"])
    table = command(["ps", "-Ao", "pid,ppid,%cpu,comm"])
    # This process is the busiest python on the machine while it runs, and the
    # first draft duly reported itself. `reverifier_latency.py` learned the same
    # lesson from `pgrep -f` matching the shell that was waiting on it.
    mine = {os.getpid(), os.getppid()}
    heavy = []
    for line in table.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, cpu, comm = parts
        if int(pid) in mine or int(ppid) in mine:
            continue
        if any(name in comm.lower() for name in BUSY) and float(cpu) > 20.0:
            heavy.append(f"{cpu} {comm}")
    return {"ollama_ps": resident, "busy_processes": heavy[:8]}

LANGUAGES = ("zh-TW", "en")

#: The six defect classes the store labels with. `false_call` is not one: a
#: rationale saying a region is a false call is not naming a defect class.
DEFECT_CLASSES = ("open", "short", "mousebite", "spur", "copper", "pin-hole")

DOCUMENT = re.compile(r"\bWI-\d{3}\b")

#: An acceptance limit is a figure attached to a permission. Both languages,
#: because the rationale is written in the line's and the check must not be
#: blinder in one of them.
LIMIT_PHRASE = re.compile(
    r"(acceptable|allowable|permitted|up to|within|tolerance|limit of"
    r"|可接受|允許|容許|上限|以內|不超過|限值)",
    re.IGNORECASE,
)
FIGURE = re.compile(r"\d")

#: Classes whose work instructions admit no acceptable size at all.
ZERO_TOLERANCE = ("open", "short")

CHECKED = ("unsourced_figure", "foreign_document", "no_explanation")
FLAGGED = ("class_not_named", "other_class_named", "limit_for_a_zero_tolerance_class")


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


#: The model writes `pin\u2011hole` with a non-breaking hyphen often enough that
#: matching the class name on the ASCII spelling alone reported a class as
#: unnamed when it was named and typeset. Every dash-like codepoint folds to
#: `-` before the class names are looked for.
DASHES = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uff0d"})


def normalise_dashes(text: str) -> str:
    return text.translate(DASHES)


def sourced_documents(state: dict) -> set[str]:
    """The document numbers the retrieved passages actually put in front of the
    model -- read out of the passage text, because that is what the model was
    shown. A document's slug (`reverification-procedure`) and the number its
    body cites (`WI-201`) are two different names and the store carries only
    the first."""
    text = " ".join(p.get("text", "") for p in (state.get("standards") or []))
    return set(DOCUMENT.findall(text))


def sentences(text: str) -> list[str]:
    """Split on sentence ends, and not inside a number. `21.1%` is one figure;
    splitting it produced the fragment `1%\uff0c\u4f46\u4ecd\u5c6c\u65bc\u53ef\u63a5\u53d7\u7bc4\u570d` and with it ten
    `limit_for_a_zero_tolerance_class` flags that were all a machine's defect
    *rate* being described as normal."""
    return re.split(r"(?<!\d)\.(?!\d)|[\u3002\uff0e\n]", text)


def findings(state: dict, model_class: str) -> dict[str, list[str]]:
    """Every kind this run raises, with what raised it."""
    rationale = (state.get("agent_rationale") or "").strip()
    status = state.get("explanation_status", "unknown")
    found: dict[str, list[str]] = {}

    if not rationale:
        return {"no_explanation": [status]}

    unsourced = list(state.get("rationale_flags") or [])
    if unsourced:
        found["unsourced_figure"] = unsourced

    # Against the numbers the retrieved *text* names, not the slugs the store
    # files it under. The first draft compared `WI-\d{3}` citations against
    # {"open-circuit", "reverification-procedure"} -- two vocabularies that
    # never intersect -- so every citation was foreign by construction and the
    # 2026-09-01 run's six were all legitimate. A check that cannot return zero
    # is not a check.
    sourced = sourced_documents(state)
    cited = set(DOCUMENT.findall(rationale))
    foreign = sorted(cited - sourced)
    if foreign:
        found["foreign_document"] = foreign

    lowered = normalise_dashes(rationale).lower()
    if model_class in DEFECT_CLASSES and model_class not in lowered:
        found["class_not_named"] = [model_class]
    others = [c for c in DEFECT_CLASSES if c != model_class and c in lowered]
    if others:
        found["other_class_named"] = others

    if model_class in ZERO_TOLERANCE:
        for sentence in sentences(rationale):
            if LIMIT_PHRASE.search(sentence) and FIGURE.search(sentence):
                found.setdefault("limit_for_a_zero_tolerance_class", []).append(sentence.strip())
    return found


def run_language(sample: list[dict], language: str, args, log) -> list[dict]:
    """One pass over the candidates with the line speaking `language`."""
    previous = os.environ.get(LINE_LANGUAGE_ENV)
    os.environ[LINE_LANGUAGE_ENV] = language
    try:
        graph = build_graph(OllamaClient(args.model, timeout=EXPLANATION_DEADLINE_S),
                            InMemorySaver())
        rows = []
        for position, case in enumerate(sample, 1):
            state = graph.invoke(
                {"candidate_ref": case["reference"], "trace": [], "timings_ms": {}},
                config={"configurable": {"thread_id": f"rationale-{language}-{case['reference']}"}},
            )
            found = findings(state, case["model_class"])
            rows.append({
                "language": language,
                "reference": case["reference"],
                "model_class": case["model_class"],
                "explanation_status": state.get("explanation_status", "unknown"),
                "rationale": (state.get("agent_rationale") or "").strip(),
                "retrieved_documents": sorted({p["document"] for p in (state.get("standards") or [])}),
                # The slug is what the store files a document under; the number
                # is what the model may cite. Both are stored so that a reading
                # taken later can re-derive `foreign_document` without the run.
                "sourced_documents": sorted(sourced_documents(state)),
                "findings": found,
            })
            marks = ",".join(sorted(found)) or "clean"
            log(f"  [{position:>3}/{len(sample)}] {language} {case['reference']:<16} "
                f"{case['model_class']:<10} {marks}", flush=True)
        return rows
    finally:
        if previous is None:
            os.environ.pop(LINE_LANGUAGE_ENV, None)
        else:
            os.environ[LINE_LANGUAGE_ENV] = previous


def tally(rows: list[dict]) -> dict:
    counts = Counter()
    for row in rows:
        for kind in row["findings"]:
            counts[kind] += 1
    clean = sum(1 for row in rows if not row["findings"])
    return {"rows": len(rows), "clean": clean,
            "counts": {kind: counts.get(kind, 0) for kind in CHECKED + FLAGGED}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--candidates", type=int, default=60)
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "rationale_eval.json")
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "docs" / "benchmarks.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if len(args.languages) < 2 and not args.dry_run:
        print("publishing a single-language rationale report is refused: the "
              "rationale is written in the line's language and a count over one "
              "of them reads as a claim about both. Pass --dry-run to look at "
              "one anyway.", file=sys.stderr)
        return 2

    sample = investigated(args.candidates)
    if not sample:
        print("no investigated candidates in the store", file=sys.stderr)
        return 1
    print(f"{len(sample)} candidates x {len(args.languages)} languages")

    before = contention()
    started = time.perf_counter()
    rows: list[dict] = []
    for language in args.languages:
        rows += run_language(sample, language, args, print)
    wall = time.perf_counter() - started

    per_language = {language: tally([r for r in rows if r["language"] == language])
                    for language in args.languages}
    overall = tally(rows)
    record = {"commit": commit(), "model": args.model, "candidates": len(sample),
              "languages": args.languages, "wall_seconds": round(wall, 1),
              "checked": list(CHECKED), "flagged": list(FLAGGED),
              "per_language": per_language, "overall": overall,
              "contention": {"before": before, "after": contention()},
              "cases": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    def row_for(kind: str) -> str:
        cells = " | ".join(str(per_language[l]["counts"][kind]) for l in args.languages)
        return f"| `{kind}` | {cells} | {overall['counts'][kind]} |"

    seen = sorted({p for phase in record["contention"].values() for p in phase["busy_processes"]})
    busy = ", ".join(f"`{p}`" for p in seen[:4])
    lines = [
        f"## {date.today().isoformat()} · commit {commit()}",
        "",
        "### Is the operator's rationale true of what the model was shown?",
        "",
        f"The verdict came off the LLM on 2026-08-23 and what it still writes -- "
        f"the one sentence on a queue row an operator reads -- had never been "
        f"scored. Two answers in this project's own question bank said so. "
        f"{len(sample)} candidates the router sends to investigation, run in "
        f"{' and '.join(args.languages)}, {wall / 60:.0f} min on `{args.model}`. "
        f"Ground truth is not read: this scores the prose against the prompt, "
        f"not against the board. `scripts/rationale_eval.py`.",
        "",
        "**No model judges another model.** Three kinds are checked exactly from "
        "what the run stored; three are pattern matches published with the "
        "sentence attached, for a person to resolve.",
        "",
        "| kind | " + " | ".join(args.languages) + " | both |",
        "|---|" + "---|" * (len(args.languages) + 1),
    ] + [row_for(k) for k in CHECKED] \
      + ["|" + " |" * (len(args.languages) + 2)] \
      + [row_for(k) for k in FLAGGED] + [
        "",
        "Clean by language: " + ", ".join(
            f"{l} {per_language[l]['clean']}/{per_language[l]['rows']}"
            for l in args.languages) + ".",
        "",
        (f"**The machine was not quiet**: {busy} was running, so a "
         f"`no_explanation` here may be a timeout rather than the model. "
         f"Recorded rather than hidden; the other five kinds read a rationale "
         f"that exists and are unaffected.\n"
         if busy else
         "The machine was checked for other work before and after the run -- "
         "`ollama ps` and the process table, because the first alone comes back "
         "clean while a torch job saturates the same silicon -- and was quiet "
         "both times.\n"),
        "",
        "**What this does not establish.** It asks whether the prose is true of "
        "the prompt, never whether the prompt was the right thing to show or "
        "whether the verdict beneath it was right -- `agent_eval.py` is the "
        "second and nothing is the first. A grounded figure used in a wrong "
        "comparison passes, which is `rationale_check.py`'s stated boundary and "
        "is inherited here. And the flags are patterns: their counts are a floor "
        "on what a pattern can raise, not a rate.",
        "",
    ]
    body = "\n".join(lines)
    if args.dry_run:
        print("\n" + body)
    else:
        with args.benchmarks.open("a") as handle:
            handle.write("\n" + body)
        print(f"\nappended to {args.benchmarks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
