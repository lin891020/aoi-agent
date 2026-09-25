"""Can Jev tell whether a question is answerable with the tools we have?

The planner's hardest call is not which arguments to pass; it is whether to
plan at all. On the independent seventy, 28 questions should be refused --
the data is not held, no tool returns the quantity, no entity is named, or
the request is an action -- and gpt-oss refuses 17 of them while also
refusing three it should have answered. This asks a model that does nothing
but answer typed questions whether it can make that call better.

Jev (TypeSafe AI) is given one *state* -- the question, the tools as one line
each, and a few facts about the data -- and one `noul` question: can this be
answered by the listed tools? It returns P(yes). For the questions that should
be answered, one more `noul` per tool asks whether the answer needs that tool.

Three arms, each question three times (to see whether a resend moves):

* **full** -- question + tools + facts;
* **tools_only** -- question + tools, no facts (does "the store holds nine
  days" or "no tool returns a reviewer" have to be *told*?);
* **blind** -- the question replaced by a placeholder, tools and facts kept.
  A model that scores the same here is answering the base rate, not the
  question.

Two gold standards, both reported: the fixture as written (28 refuse / 42
answer, by authors who never saw the prompt) and the fixture after the
2026-08-29 adjudication, which found five refusals a route through `run_sql`
and `query_false_call_rate` (S02, S06, S11, S12, A04) -- Jev is shown both
tools, so the second is its fair reading and the first is the historical one.

The comparator is gpt-oss on the same seventy: refused = a plan with no
calls, read from the stored batch-512 runs (the prompt has not changed since
2026-09-02) plus one run made today, and reported as a range because the
planner moves between sessions.

Verdict, written before the run, on the adjudicated gold, threshold 0.5,
full arm, F1 on the refuse class:

* **not_reading** -- full F1 minus blind F1 is under 0.10: it is guessing the
  base rate, whatever the other numbers say.
* **unstable** -- more than 10 questions whose P(answerable) has a standard
  deviation over 0.05 across the three resends.
* **useful_as_gate** -- Jev F1 is at least 0.05 above gpt-oss's best run and
  at least 0.15 above blind.
* **worse** -- Jev F1 is 0.05 or more below gpt-oss's worst run.
* **no_better** -- anything else.

Predicted first: gpt-oss around F1 0.77 on the adjudicated gold (it refuses
judgement questions like S18/S24/S28), Jev around 0.80, blind around 0.55;
tool-set hit around 0.7 against gpt-oss's 0.64, with the two aggregate tools
(`query_defect_history` / `query_machine_stats`) the ones it will not tell
apart. What this does not do: gate anything. The result is a measurement, not
a stage in front of `sql_guard` or the planner, and the docstring of the
client says why a cloud model needs its version stored per reading.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from aoi_agent.analysis.plan import capability_summary, store_domains
from aoi_agent.llm.jev import JevClient, noul

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "tests" / "fixtures" / "analysis_questions_independent.json"
ARMS = ("full", "tools_only", "blind")
THRESHOLD = 0.5
REPEATS = 3
BOOTSTRAPS = 1000
UNSTABLE_SD = 0.05
UNSTABLE_QUESTIONS = 10

#: The five refusals the 2026-08-29 adjudication found a route for once
#: `run_sql` and `query_false_call_rate` existed. Jev is shown both tools.
ADJUDICATED_GAINS = frozenset({"S02", "S06", "S11", "S12", "A04"})

#: The question, held constant. Changing a word here is a new experiment: Jev's
#: probability is a property of the wording (docs.typesafe.ai, "Noul").
BLIND_QUESTION = "（一個關於這條產線的問題；原文已移除）"
ANSWERABLE = noul(
    "Can this question be answered by calling one or more of the listed tools, "
    "with arguments the question itself supplies?",
    {"true": "Every figure or record the question asks for is returned by a listed "
             "tool, filtered only by things the question names, inside the data held.",
     "false": "The question asks for a quantity no tool returns, for a period outside "
              "the data held, names no board, machine, line, lot or event to filter on, "
              "asks for something to be changed rather than looked up, or is too vague "
              "to turn into arguments."})


def needs_tool(tool: str) -> dict:
    return noul(f"Does answering this question need a call to `{tool}`?",
                {"true": "A figure or record the question asks for comes from this tool.",
                 "false": "This tool's output is not needed to answer, even if it is related."})


#: What the tools do not hold, stated because a refusal often rests on it and
#: no docstring says it. Hand-written, and the `tools_only` arm measures what
#: it is worth.
ABSENCES = (
    "No tool returns who reviewed a board, or whether a reviewer was right.",
    "False-call counts and rates come only from query_false_call_rate; the other "
    "aggregates exclude false calls.",
    "Every tool reads. Nothing changes a board, a queue entry, a record or a schema, "
    "and nothing exports or sends anything.",
    "No tool forecasts, and no tool holds yield, scrap, customer returns or "
    "confirmed escapes.",
)


def tool_lines() -> list[str]:
    """Each tool as one line: its name and the first line of its docstring."""
    return capability_summary()


def facts(domains) -> str:
    span = domains.get("date_span") or ("?", "?")
    return "\n".join([
        f"The store holds {domains['max_days']} days of inspections, {span[0]} to {span[1]}.",
        f"Lines: {', '.join(sorted(domains['line_id']))}. "
        f"Machines: {', '.join(sorted(domains['machine_id']))}.",
        f"Defect classes: {', '.join(sorted(domains['defect_type']))}.",
        f"Recorded machine event kinds: {', '.join(sorted(domains['relative_to']))}.",
        *ABSENCES,
    ])


def build_state(question: str, arm: str, tools: list[str], fact_text: str) -> str:
    """The state Jev is shown. Never the planner's system prompt: only the
    question, the tools and the facts, so what it reads is what it is scored on."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; arms are {ARMS}")
    shown = BLIND_QUESTION if arm == "blind" else question
    parts = [f"Question from a line supervisor:\n{shown}",
             "Tools available (name — what it returns):\n" + "\n".join(f"- {t}" for t in tools)]
    if arm != "tools_only":
        parts.append("Facts about the data:\n" + fact_text)
    return "\n\n".join(parts)


def questions_for(tool_names: list[str]) -> dict[str, dict]:
    return {"answerable": ANSWERABLE, **{f"needs__{t}": needs_tool(t) for t in tool_names}}


# ---- gold ----------------------------------------------------------------------


def gold(rows: list[dict], scheme: str) -> dict[str, bool]:
    """id -> should refuse. `adjudicated` moves the five gains to the answer side."""
    if scheme not in ("fixture", "adjudicated"):
        raise ValueError(scheme)
    out = {}
    for r in rows:
        refuse = bool(r.get("expect_refusal"))
        if scheme == "adjudicated" and r["id"] in ADJUDICATED_GAINS:
            refuse = False
        out[r["id"]] = refuse
    return out


def expected_tool_sets(row: dict) -> list[set[str]]:
    """Every accepted tool set for an answerable question, as `analysis_eval` reads it."""
    sets = [set(row.get("expect_tools") or [])]
    for alternative in row.get("expect_any_of") or []:
        sets.append(set(alternative.get("expect_tools") or []))
    return [s for s in sets if s] or [set()]


# ---- metrics --------------------------------------------------------------------


def refusal_metrics(pred_refuse: dict[str, bool], truth: dict[str, bool],
                    ids: list[str] | None = None) -> dict[str, float]:
    ids = list(ids if ids is not None else truth)
    tp = sum(1 for q in ids if pred_refuse[q] and truth[q])
    fp = sum(1 for q in ids if pred_refuse[q] and not truth[q])
    fn = sum(1 for q in ids if not pred_refuse[q] and truth[q])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def ece(probs: dict[str, float], truth_yes: dict[str, bool], bins: int = 10,
        ids: list[str] | None = None) -> float:
    """Expected calibration error of P(yes) against whether yes was right."""
    ids = list(ids if ids is not None else probs)
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [q for q in ids if lo <= probs[q] < hi or (b == bins - 1 and probs[q] == 1.0)]
        if not members:
            continue
        confidence = sum(probs[q] for q in members) / len(members)
        accuracy = sum(1 for q in members if truth_yes[q]) / len(members)
        total += len(members) / len(ids) * abs(confidence - accuracy)
    return total


def noise_floor_ece(probs: dict[str, float], n_boot: int, seed: int = 0) -> float:
    """What a perfectly calibrated model with these probabilities scores at this n.

    Outcomes are drawn from the probabilities themselves, so any ECE left is
    sampling noise; a measured ECE is read against this, not against zero.
    """
    import random
    rng = random.Random(seed)
    ids = list(probs)
    floors = []
    for _ in range(n_boot):
        simulated = {q: rng.random() < probs[q] for q in ids}
        floors.append(ece(probs, simulated))
    return statistics.fmean(floors)


def bootstrap_ci(metric, ids: list[str], n_boot: int, seed: int = 0) -> tuple[float, float]:
    import random
    rng = random.Random(seed)
    values = sorted(metric(rng.choices(ids, k=len(ids))) for _ in range(n_boot))
    return values[int(0.025 * n_boot)], values[min(int(0.975 * n_boot), n_boot - 1)]


def toolset_metrics(predicted: dict[str, set[str]], rows: list[dict],
                    answerable_ids: list[str]) -> dict[str, float]:
    """hit = some accepted tool set is inside the predicted set; precision over predicted."""
    by_id = {r["id"]: r for r in rows}
    hits, precisions = 0, []
    for q in answerable_ids:
        accepted = expected_tool_sets(by_id[q])
        pred = predicted.get(q, set())
        if any(s <= pred for s in accepted):
            hits += 1
        best = max((len(s & pred) / len(pred)) if pred else 0.0 for s in accepted)
        precisions.append(best)
    n = len(answerable_ids) or 1
    return {"hit": hits / n, "precision": statistics.fmean(precisions) if precisions else 0.0,
            "hits": hits, "n": len(answerable_ids)}


def resend_sd(readings: dict[str, list[float]]) -> dict[str, float]:
    return {q: (statistics.pstdev(v) if len(v) > 1 else 0.0) for q, v in readings.items()}


# ---- gpt-oss, read from stored runs --------------------------------------------

PLANNED_TOOLS = re.compile(r"([a-z_]+)\(")


def gptoss_from_raw(path: Path) -> dict[str, dict]:
    rows = json.loads(Path(path).read_text())
    return {r["id"]: {"refused": r["planned"] == "(refused)",
                      "tools": set(PLANNED_TOOLS.findall(r["planned"]))} for r in rows}


def gptoss_from_log(path: Path, ids: list[str]) -> dict[str, dict]:
    """The eval's stdout, for runs that kept no `--raw` (the 2026-09-22 five)."""
    lines = Path(path).read_text(errors="replace").splitlines()
    out: dict[str, dict] = {}
    for i, line in enumerate(lines):
        m = re.match(r"\s*\[\s*(\d+)/(\d+)\]\s+(ok|MISS)", line)
        if not m:
            continue
        qid = ids[int(m.group(1)) - 1]
        for j in range(i + 1, min(i + 4, len(lines))):
            if "planned " in lines[j]:
                planned = lines[j].split("planned ", 1)[1].strip()
                out[qid] = {"refused": planned == "(refused)",
                            "tools": set(PLANNED_TOOLS.findall(planned))}
                break
    return out


# ---- the run ---------------------------------------------------------------------


def run(client: JevClient, rows: list[dict], arms: tuple[str, ...], repeats: int,
        tools: list[str], fact_text: str, log=print, existing: list[dict] | None = None) -> list[dict]:
    """Every (arm, repeat, question) once; records already present are kept."""
    records = list(existing or [])
    done = {(r["arm"], r["repeat"], r["id"]) for r in records}
    names = [t.split(" — ")[0] for t in tools]
    questions = questions_for(names)
    total = len(arms) * repeats * len(rows)
    for arm in arms:
        for repeat in range(repeats):
            for row in rows:
                key = (arm, repeat, row["id"])
                if key in done:
                    continue
                state = build_state(row["question"], arm, tools, fact_text)
                answer = client.ask(state, questions)
                records.append({
                    "arm": arm, "repeat": repeat, "id": row["id"],
                    "p_answerable": answer.noul("answerable"),
                    "p_tools": {t: answer.noul(f"needs__{t}") for t in names},
                    "model": answer.model, "latency_s": round(answer.latency_s, 3),
                    "input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens,
                    "asked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                })
                log(f"  [{len(records):>3}/{total}] {arm:<10} r{repeat} {row['id']} "
                    f"P(answerable)={answer.noul('answerable'):.2f} {answer.latency_s * 1000:.0f} ms")
    return records


def readings(records: list[dict], arm: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in sorted(records, key=lambda r: r["repeat"]):
        if r["arm"] == arm:
            out.setdefault(r["id"], []).append(r["p_answerable"])
    return out


def mean_probs(records: list[dict], arm: str) -> dict[str, float]:
    return {q: statistics.fmean(v) for q, v in readings(records, arm).items()}


def predicted_tools(records: list[dict], arm: str, threshold: float) -> dict[str, set[str]]:
    sums: dict[str, dict[str, list[float]]] = {}
    for r in records:
        if r["arm"] != arm:
            continue
        for t, p in r["p_tools"].items():
            sums.setdefault(r["id"], {}).setdefault(t, []).append(p)
    return {q: {t for t, v in tools.items() if statistics.fmean(v) >= threshold}
            for q, tools in sums.items()}


def analyse(records: list[dict], rows: list[dict], gptoss: dict[str, dict[str, dict]],
            threshold: float = THRESHOLD, n_boot: int = BOOTSTRAPS) -> dict:
    ids = [r["id"] for r in rows]
    result: dict = {"threshold": threshold, "arms": {}, "gptoss": {}, "gold": {}}
    for scheme in ("fixture", "adjudicated"):
        truth = gold(rows, scheme)
        result["gold"][scheme] = {"refuse": sum(truth.values()), "answer": len(truth) - sum(truth.values())}
        for arm in ARMS:
            probs = mean_probs(records, arm)
            if not probs:
                continue
            pred = {q: probs[q] < threshold for q in ids if q in probs}
            present = [q for q in ids if q in probs]
            metrics = refusal_metrics(pred, truth, present)
            truth_yes = {q: not truth[q] for q in present}
            entry = {**metrics,
                     "f1_ci": bootstrap_ci(lambda s: refusal_metrics(pred, truth, s)["f1"], present, n_boot),
                     "ece": ece(probs, truth_yes, ids=present),
                     "ece_floor": noise_floor_ece({q: probs[q] for q in present}, min(n_boot, 200))}
            if scheme == "adjudicated":
                answerable = [q for q in present if not truth[q]]
                entry["toolset"] = toolset_metrics(predicted_tools(records, arm, threshold), rows, answerable)
            result["arms"].setdefault(arm, {})[scheme] = entry
        for name, run_ in gptoss.items():
            pred = {q: run_[q]["refused"] for q in ids if q in run_}
            present = [q for q in ids if q in run_]
            entry = refusal_metrics(pred, truth, present)
            if scheme == "adjudicated":
                answerable = [q for q in present if not truth[q]]
                entry["toolset"] = toolset_metrics({q: run_[q]["tools"] for q in present}, rows, answerable)
            result["gptoss"].setdefault(name, {})[scheme] = entry
    sd = resend_sd(readings(records, "full"))
    result["resend"] = {"unstable_questions": sorted(q for q, s in sd.items() if s > UNSTABLE_SD),
                        "max_sd": max(sd.values(), default=0.0)}
    result["models"] = sorted({r["model"] for r in records})
    lat = [r["latency_s"] for r in records]
    result["latency_s"] = {"p50": statistics.median(lat) if lat else None,
                           "p90": sorted(lat)[int(0.9 * len(lat))] if lat else None}
    result["tokens"] = {"input": sum(r["input_tokens"] for r in records),
                        "output": sum(r["output_tokens"] for r in records)}
    return result


def verdict(analysis: dict) -> tuple[str, str]:
    full = analysis["arms"].get("full", {}).get("adjudicated")
    blind = analysis["arms"].get("blind", {}).get("adjudicated")
    gpt = [v["adjudicated"]["f1"] for v in analysis["gptoss"].values() if "adjudicated" in v]
    if not (full and blind and gpt):
        return "incomplete", "The full and blind arms and at least one gpt-oss run are needed."
    unstable = analysis["resend"]["unstable_questions"]
    if len(unstable) > UNSTABLE_QUESTIONS:
        return ("unstable", f"{len(unstable)} questions move more than {UNSTABLE_SD} in P(answerable) "
                            "across resends; nothing below is read until that is understood.")
    gap = full["f1"] - blind["f1"]
    if gap < 0.10:
        return ("not_reading", f"Full-state F1 {full['f1']:.2f} is within {gap:.2f} of the blind arm's "
                               f"{blind['f1']:.2f}: it is answering the base rate, not the question.")
    if full["f1"] >= max(gpt) + 0.05 and gap >= 0.15:
        return ("useful_as_gate", f"Jev F1 {full['f1']:.2f} against gpt-oss {min(gpt):.2f}-{max(gpt):.2f} "
                                  f"and blind {blind['f1']:.2f}: it reads the question and beats the planner "
                                  "at deciding whether to plan.")
    if full["f1"] < min(gpt) - 0.05:
        return ("worse", f"Jev F1 {full['f1']:.2f} is below gpt-oss's worst run {min(gpt):.2f} by more than 0.05.")
    return ("no_better", f"Jev F1 {full['f1']:.2f} sits with gpt-oss {min(gpt):.2f}-{max(gpt):.2f}; "
                         "a second model that agrees with the first buys nothing at the gate.")


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def render(record: dict) -> str:
    a = record["analysis"]
    lines = [f"## {record['date']} · commit {record['commit']}", "",
             "### Can Jev tell whether a question is answerable with the tools we have?", "",
             f"`scripts/jev_refusal_eval.py`, Jev through the {record['door']} "
             f"(model string{'s' if len(a['models']) != 1 else ''} reported: "
             f"{', '.join(a['models']) or 'none'}), the independent seventy, "
             f"{record['repeats']} resends per question per arm, threshold {a['threshold']} on "
             "P(answerable), refuse = below it. The verdict rule and the prediction are in the "
             "script's docstring, written before the run. gpt-oss rows are the planner's plans "
             "on the same seventy, refused = no call, batch-512 loads only.", ""]
    for scheme in ("fixture", "adjudicated"):
        g = a["gold"][scheme]
        lines += [f"**{scheme} gold** ({g['refuse']} refuse / {g['answer']} answer)", "",
                  "| arm | precision | recall | F1 (95% CI) | ECE (noise floor) | tool-set hit |",
                  "|---|---|---|---|---|---|"]
        for arm in ARMS:
            e = a["arms"].get(arm, {}).get(scheme)
            if not e:
                continue
            ts = e.get("toolset")
            lines.append(f"| Jev {arm} | {e['precision']:.2f} | {e['recall']:.2f} | {e['f1']:.2f} "
                         f"({e['f1_ci'][0]:.2f}–{e['f1_ci'][1]:.2f}) | {e['ece']:.3f} ({e['ece_floor']:.3f}) | "
                         f"{(f'{ts['hit']:.2f} ({ts['hits']}/{ts['n']})' if ts else '—')} |")
        for name, runs in a["gptoss"].items():
            e = runs.get(scheme)
            if not e:
                continue
            ts = e.get("toolset")
            lines.append(f"| gpt-oss {name} | {e['precision']:.2f} | {e['recall']:.2f} | {e['f1']:.2f} | — | "
                         f"{(f'{ts['hit']:.2f} ({ts['hits']}/{ts['n']})' if ts else '—')} |")
        lines.append("")
    r = a["resend"]
    lines += [f"**Verdict: {record['verdict']}.** {record['verdict_sentence']}", "",
              f"Resends: {len(r['unstable_questions'])} questions with sd > {UNSTABLE_SD} on the full arm "
              f"(max sd {r['max_sd']:.3f}){(': ' + ', '.join(r['unstable_questions'])) if r['unstable_questions'] else ''}. "
              f"Latency p50 {a['latency_s']['p50']:.2f} s, p90 {a['latency_s']['p90']:.2f} s; "
              f"{a['tokens']['input']:,} input tokens over {record['requests']} requests.", ""]
    if record.get("zh_en"):
        z = record["zh_en"]
        lines += [f"Chinese against English wording, {len(z)} questions: "
                  f"{sum(1 for v in z.values() if abs(v['zh'] - v['en']) > 0.2)} differ by more than 0.20 "
                  f"in P(answerable) ({', '.join(f'{q} {v['zh']:.2f}/{v['en']:.2f}' for q, v in z.items() if abs(v['zh'] - v['en']) > 0.2) or 'none'}). "
                  "Not in the verdict; a reading of whether the model reads Chinese the way it reads English.", ""]
    lines += ["**What this does not establish.** Seventy questions from one store, one day, one cloud "
              "model version; the tool-set question is asked per tool as a yes/no and cannot express "
              "arguments, so it is a recall on tool names against the fixture's accepted sets and not a "
              "plan. Nothing here gates anything: Jev is not in front of the planner or `sql_guard`, "
              "and the adversarial-text caveat in the client's docstring is why. The `tools_only` arm "
              "says what the hand-written facts are worth; it does not say the facts are complete.", "",
              "<details><summary>Per-question, full arm, adjudicated gold</summary>", "",
              "| id | gold | P(answerable) | Jev | gpt-oss (runs) | tools ≥ threshold |", "|---|---|---|---|---|---|"]
    truth = record["per_question"]
    for q in truth:
        p = truth[q]
        lines.append(f"| {q} | {'refuse' if p['refuse'] else 'answer'} | {p['p']:.2f} | "
                     f"{'refuse' if p['p'] < a['threshold'] else 'answer'} | {p['gptoss']} | {', '.join(sorted(p['tools'])) or '—'} |")
    lines += ["", "</details>"]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--limit", type=int, default=None, help="first N questions, for a smoke run")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--gptoss-raw", nargs="*", type=Path, default=[],
                        help="analysis_eval --raw files (batch-512 loads only)")
    parser.add_argument("--gptoss-log", nargs="*", type=Path, default=[],
                        help="analysis_eval stdout logs, for runs that kept no --raw")
    parser.add_argument("--zh-en", type=Path, default=None,
                        help="JSON {id: english_wording}; each is asked once on the full arm")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="a previous --out to continue")
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "docs" / "benchmarks.md")
    parser.add_argument("--print-state", action="store_true", help="show the three states for the first question and exit")
    parser.add_argument("--dry-run", action="store_true", help="run and analyse, append nothing")
    args = parser.parse_args()

    rows = json.loads(args.questions.read_text())
    if args.limit:
        rows = rows[:args.limit]
    ids = [r["id"] for r in rows]
    tools = tool_lines()
    fact_text = facts(store_domains())

    if args.print_state:
        for arm in ARMS:
            print(f"===== {arm}\n{build_state(rows[0]['question'], arm, tools, fact_text)}\n")
        print("===== questions\n" + json.dumps(questions_for([t.split(' — ')[0] for t in tools]), indent=1, ensure_ascii=False))
        return 0

    client = JevClient.from_env()
    existing = json.loads(args.resume.read_text())["records"] if args.resume else []
    records = run(client, rows, tuple(args.arms), args.repeats, tools, fact_text, existing=existing)

    zh_en = {}
    if args.zh_en:
        names = [t.split(" — ")[0] for t in tools]
        by_id = {r["id"]: r for r in rows}
        for q, english in json.loads(args.zh_en.read_text()).items():
            if q not in by_id:
                continue
            en = client.ask(build_state(english, "full", tools, fact_text), questions_for(names)).noul("answerable")
            zh = statistics.fmean(readings(records, "full").get(q, [float("nan")]))
            zh_en[q] = {"zh": zh, "en": en}

    gptoss = {}
    for p in args.gptoss_raw:
        gptoss[p.stem.replace("language_note_", "")] = gptoss_from_raw(p)
    all_ids = [r["id"] for r in json.loads(args.questions.read_text())]
    for p in args.gptoss_log:
        gptoss[p.stem.replace("planner_drift_", "")] = gptoss_from_log(p, all_ids)

    analysis = analyse(records, rows, gptoss, args.threshold)
    label, sentence = verdict(analysis)
    truth = gold(rows, "adjudicated")
    probs = mean_probs(records, "full")
    ptools = predicted_tools(records, "full", args.threshold)
    per_question = {q: {"refuse": truth[q], "p": probs.get(q, float("nan")),
                        "gptoss": "/".join("R" if run_.get(q, {}).get("refused") else "A" for run_ in gptoss.values()) or "—",
                        "tools": ptools.get(q, set())} for q in ids if q in probs}
    record = {"date": datetime.now().astimezone().date().isoformat(), "commit": git_head(),
              "door": client.door, "repeats": args.repeats, "requests": len(records),
              "analysis": analysis, "verdict": label, "verdict_sentence": sentence,
              "zh_en": zh_en, "per_question": per_question, "records": records}
    out = args.out or ROOT / "models" / f"jev_refusal_{record['date']}.json"
    out.write_text(json.dumps(record, indent=1, ensure_ascii=False, default=sorted))
    body = render(record)
    print("\n" + body)
    print(f"\n-> {out}")
    if args.dry_run or args.limit:
        print("(not appended)" + (": --limit runs never publish" if args.limit else ""))
        return 0
    args.benchmarks.write_text(args.benchmarks.read_text().rstrip() + "\n\n" + body.rstrip() + "\n")
    print(f"appended to {args.benchmarks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
