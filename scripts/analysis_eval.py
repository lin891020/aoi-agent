"""Does the planner plan the right thing, and refuse the right things?

The tools are deterministic, so a correct plan yields correct data by
construction. The errors live in the plan, which is why this scores plans rather
than answers -- ground truth for a plan is hand-writable, and ground truth for a
paragraph is not.

Three measures. Plan accuracy is the obvious one. Refusal accuracy matters more
than it looks: a system that answers everything is more dangerous on a factory
floor than one that says it cannot, and nothing else in this project checks it.
Determinism is here because a supervisor who screenshots a chart should be able
to ask the same question tomorrow and recognise the answer.

**What this does not establish.** Three of the twenty questions are few-shot
examples verbatim and two more are paraphrases; the fixture marks them
`in_prompt` and the report scores the held-out fifteen separately, because on
the marked five the model is reciting the prompt rather than planning. Beyond
that, the same person wrote the examples and the questions, so even the held-out
ones lean towards the shapes the prompt was built for. Writing them before
re-reading the examples mitigates that; it does not solve it. This is a plan
accuracy measured against one author's opinion of the right plan, not an
operating-point curve, and it says nothing about whether the prose the model
writes over correct data is correct.

A rejected plan is counted apart from a wrong one. A plan can name the right
tools and still be refused by `validate_plan` -- most likely on `days`, since
`prompts.FEW_SHOT` hardcodes `days=7` while `store_domains()["max_days"]` is
whatever the seeded span happens to be (9 at 500 boards, about 4 at
`seed_store.py`'s documented `--limit 200`). The fixture itself hardcodes no
window, but a store reseeded smaller would push the examples out of range, and
that shows up here as rejected plans rather than as a silent hit.

No ground truth from the store is read here, and no label of any kind: the only
truth in this file is the hand-written expectations in the fixture.

    uv run python scripts/analysis_eval.py --repeats 3
    uv run python scripts/analysis_eval.py --repeats 3 --plan-only
    uv run python scripts/analysis_eval.py --repeats 3 \
        --questions tests/fixtures/analysis_questions_independent.json

`--questions` picks the set. It defaults to the twenty above, whose published
figures have to stay reproducible. The seventy in
`analysis_questions_independent.json` were written by three authors who had
seen neither the prompt nor the few-shot examples nor this fixture, which is
the one thing the twenty cannot offer; that set carries a severity on every
question, and the report it produces breaks the score down by severity rather
than averaging a defect together with a difference of opinion. The two sets are
reported into separate sections and their numbers are not comparable.

`--plan-only` stops after the planner. Nothing this script scores is decided
after that node, so the tools and the synthesis call are work whose output is
discarded -- about half the wall time of a full run, spent writing prose nobody
reads. It is off by default: the figures in docs/benchmarks.md came off the
full graph, and a benchmark whose default path is not the measured one has
stopped being reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.analysis.graph import build_analysis_graph, make_plan_node  # noqa: E402
from aoi_agent.analysis.plan import store_domains  # noqa: E402
from aoi_agent.graph.flow import DEFAULT_MODEL  # noqa: E402
from aoi_agent.llm.ollama import EXPLANATION_DEADLINE_S, OllamaClient  # noqa: E402

#: Printed directly under the table when nothing failed, not after the misses
#: list. This project's headline invariant is to report an operating point
#: rather than a bare accuracy; a column of 100%s with its limit four
#: paragraphs down reads as a result carrying a footnote, when it is a question
#: set carrying a score. Module-level so `tests/test_analysis_eval.py` can hold
#: docs/benchmarks.md to the same bytes.
CLEAN_SWEEP = (
    "**A clean sweep is a fact about the question set before it is a fact "
    "about the planner.** Nothing here found the boundary, so nothing here "
    "bounds anything: the honest reading is that these twenty questions "
    "are inside what this model does easily, not that the planner is "
    "correct. To have any resolution the set needs questions that are "
    "harder in a specific way — a window the store does not hold, an "
    "aggregate no single tool computes, a machine named only implicitly — "
    "and it needs an author who did not write the prompt."
)

#: What the held-out score does *not* bound. It used to read "That is the
#: number to read; the headline above is the optimistic one" -- a sentence that
#: restates a 100% while sounding like a correction.
HELD_OUT_CAVEAT = (
    "Read that one rather than the headline above, and read it narrowly: it is "
    "agreement with one author's expected plans on question shapes that author "
    "chose. It does not bound the questions nobody thought to ask, the `days` "
    "and `top_k` arguments that go unscored, or whether the prose written over "
    "a correct plan is correct."
)

#: The one claim a reader needs before weighing the independent set's number.
#: Everything the older twenty cannot bound comes from a single author having
#: written the questions, the prompt and the expected plans; this set exists
#: only because three other people wrote it without seeing any of the three.
BLIND_TO_THE_PROMPT = (
    "**These questions were written by authors blind to the prompt.** Three "
    "people, none of whom had seen the planner's system prompt, its few-shot "
    "examples or `analysis_questions.json`: thirty-five from an author told "
    "nothing whatever about the tools and asked to write what a shift "
    "supervisor would type, thirty-five from an author given only the five "
    "tool signatures and asked to probe the boundary, and a verdict on all "
    "seventy from a third author who read the tools and the store's source but "
    "not the prompt. That is the whole of this set's value over the twenty "
    "above, and the reason a lower score here is worth more than the 100% "
    "there."
)

#: A defect the set exposes, established before the run and recorded here so
#: the score is read in its light rather than discovered later in a log.
DAYS_DEFAULT_DEFECT = (
    "**A known defect this set walks into: `query_machine_stats` defaults to "
    "`days=14`, and the store holds 9.** `validate_plan` checks `days` only "
    "when the plan passes it, so a plan that omits the argument runs, returns "
    "the whole 9-day span, and labels it `\"days\": 14`. Every question here "
    "reaching for a window shorter than the data is therefore answered with "
    "the full span under a wrong label, and neither the validator nor this "
    "scorer sees anything wrong -- `days` is deliberately unscored. Left "
    "unfixed on this branch on purpose: the number below is the number the "
    "system as measured produces."
)

#: The sharpest thing the set found, and the one worth stating before any
#: score: it is a gap in the tool surface, not a planner failure.
NO_FALSE_CALL_METRIC = (
    "**The sharpest finding is not in the score.** Six of the thirty-five "
    "supervisor questions ask for a false-call count or rate — per machine, "
    "per shift, per line, for the week — and no tool returns one at any "
    "aggregate level. `query_defect_history` excludes "
    "`predicted_class='false_call'` outright and `query_machine_stats` accepts "
    "only the six real classes, so the quantity does not exist above a single "
    "board. In a system whose entire subject is false calls, that is the gap "
    "an author who had not read the code found immediately and the author who "
    "wrote the tools did not. The grader marked all six `refuse`, which is "
    "correct for the system as built, and refusing them is what the planner is "
    "scored on here — but a refusal is the right answer to the wrong question. "
    "No tool was added to close it, because adding one to pass a set is how a "
    "measurement stops measuring."
)

QUESTIONS = Path(__file__).resolve().parents[1] / "tests/fixtures/analysis_questions.json"

CAUSE_WORDS = ("cause", "causal", "causation", "association", "correlat", "因果", "關聯")

#: Arguments whose value changes which rows come back, and therefore which
#: question got answered. `validate_plan` cannot help here: `defect_type="short"`
#: on a question about `open` is a valid string in a valid parameter, the query
#: succeeds, and the numbers are real numbers about something nobody asked. That
#: is the failure the project's no-SQL invariant exists to prevent, one level up,
#: and scoring tool names alone is blind to it.
#:
#: `days` and `top_k` are deliberately excluded: no question here pins a window,
#: and `validate_plan` already bounds `days` against what the store holds.
SCORED_ARGS = ("defect_type", "line_id", "machine_id", "board")

#: The deadline the station runs under, not a longer one for the benchmark.
#:
#: 180s until 2026-08-23, to keep a 10s client timeout from turning an accuracy
#: run into a measurement of the timeout. That made every planner number here
#: describe a configuration the station never ran, and nothing in the published
#: sections said so. The 10s was WI-300's response budget doing a client
#: timeout's job; the client now waits `EXPLANATION_DEADLINE_S` and so does
#: this. A planner call that misses it produces no plan, which this script
#: already scores as a miss rather than as a refusal.
EVAL_TIMEOUT_S = EXPLANATION_DEADLINE_S


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return json.loads(Path(path).read_text())


def render_plan(plan: dict | None) -> str:
    """One line of the plan, carrying the arguments the scorer looked at.

    Tool names alone cannot tell a single `query_machine_stats` call apart from
    a six-call fan-out over the defect classes, and those are a miss and a hit
    respectively. A run record that cannot distinguish them is ambiguous exactly
    where it matters.
    """
    if plan is None:
        return "(no plan)"
    calls = plan.get("calls") or []
    if not calls:
        return "(refused)"
    rendered = []
    for call in calls:
        args = {
            key: value
            for key, value in (call.get("args") or {}).items()
            if key in SCORED_ARGS
        }
        inside = ", ".join(f"{k}={v!r}" for k, v in sorted(args.items()))
        rendered.append(f"{call.get('tool')}({inside})")
    return " + ".join(rendered)


def score_plan(plan: dict, expected: dict) -> dict:
    """Did this plan do what the question needed? One reason if not.

    A question may accept more than one plan. Where two different sets of calls
    fetch the same figure, judging one of them wrong measures the fixture's
    preference and not the planner, so `expect_any_of` lists them with the
    reason they were judged equivalent and a match against any one is a hit.
    """
    alternatives = expected.get("expect_any_of")
    if not alternatives:
        return _score_one(plan, expected)

    attempts = []
    for alternative in alternatives:
        result = _score_one(plan, {**expected, **alternative})
        if result["ok"]:
            return {**result, "matched": alternative.get("why", "")}
        attempts.append(f"{alternative.get('why', '')} — {result['reason']}")
    return {"ok": False,
            "reason": "matched no accepted plan: " + "; ".join(attempts)}


def _score_one(plan: dict, expected: dict) -> dict:
    """One expectation, scored."""
    calls = plan.get("calls") or []
    refused = not calls

    if expected.get("expect_refusal"):
        if refused:
            return {"ok": True, "reason": ""}
        return {"ok": False,
                "reason": f"should have refused; planned {render_plan(plan)}"}

    if refused:
        return {"ok": False, "reason": "refused a question it should have answered"}

    called = {c.get("tool") for c in calls}
    missing = set(expected.get("expect_tools") or []) - called
    if missing:
        # Extra tools are fine -- more context is not an error. Missing ones are
        # not: the answer would rest on data nobody fetched.
        return {"ok": False, "reason": f"never called {sorted(missing)}"}

    for key, values in (expected.get("expect_args") or {}).items():
        # `or {}` and not `.get(key, {})`: a call carrying `args: null` is a
        # plan to score a miss on, not an AttributeError. `validate_plan`
        # guards it the same way.
        asked = {(c.get("args") or {}).get(key) for c in calls}
        absent = [value for value in values if value not in asked]
        if absent:
            # Which call carries it is the planner's business. That the plan
            # asks about the thing the question named is not.
            return {"ok": False, "reason": f"never queried {key}={absent}"}

    assumptions = " ".join(plan.get("assumptions") or []).lower()
    if expected.get("expect_assumption_about_cause") and not any(
        word in assumptions for word in CAUSE_WORDS
    ):
        return {"ok": False, "reason": "asked why, and never disclaimed cause"}

    if expected.get("expect_assumptions") and not (plan.get("assumptions") or []):
        return {"ok": False, "reason": "compared without stating a baseline"}

    return {"ok": True, "reason": ""}


def signature(plan: dict | None) -> str:
    """What two runs of the same question have to agree on to count as stable.

    Tool names, sorted. Argument values would make almost every question look
    unstable over wording the operator never sees; the tools called are what
    changes the answer.
    """
    if plan is None:
        return "__no_plan__"
    return json.dumps(sorted(str(c.get("tool")) for c in (plan.get("calls") or [])))


def rate(correct: int, total: int) -> str:
    return f"{correct}/{total} = {correct / total:.0%}" if total else "—"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--questions", type=Path, default=QUESTIONS,
                        help="the question set to score. Defaults to the "
                             "original twenty, whose published figures have to "
                             "stay reproducible; pass "
                             "tests/fixtures/analysis_questions_independent.json "
                             "for the seventy written by authors blind to the "
                             "prompt. The two are reported separately and are "
                             "not comparable")
    parser.add_argument("--repeats", type=int, default=3,
                        help="times to re-ask each question, for determinism")
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--raw", type=Path, default=None,
                        help="write every scored plan to JSON, for auditing")
    parser.add_argument("--note", default="",
                        help="one line into the report header, for a re-run "
                             "whose figures are not comparable with the last")
    parser.add_argument("--plan-only", action="store_true",
                        help="score the plan without running the tools or the "
                             "synthesis call. Roughly halves the wall time and "
                             "changes no number this script reports. Off by "
                             "default, so the published run stays the one the "
                             "full graph produces")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    domains = store_domains()
    client = OllamaClient(args.model, timeout=EVAL_TIMEOUT_S)

    if args.plan_only:
        # Only the planner runs. Everything this script scores -- the plan, the
        # validator's verdict on it, and whether the two are stable across
        # repeats -- is decided by `plan_node` and nothing downstream reads
        # back into it, so the tools and the second model call are work whose
        # output is discarded. On a 35-minute run that is about half the time
        # spent writing prose nobody reads.
        #
        # Not the default. The published figures were produced by the full
        # graph, and a benchmark whose default path is not the one that was
        # measured is a benchmark that quietly stopped being reproducible.
        plan_node = make_plan_node(client, domains)
        def ask(question: str) -> dict:
            return plan_node({"question": question})
    else:
        graph = build_analysis_graph(client, domains)
        def ask(question: str) -> dict:
            return graph.invoke(
                {"question": question, "results": [], "timings_ms": {}}
            )
    print(f"store holds {domains['max_days']} days, "
          f"lines {sorted(domains['line_id'])}, "
          f"machines {sorted(domains['machine_id'])}\n")

    scored, stable = [], []
    for position, item in enumerate(questions, 1):
        plans, rejections = [], []
        for _ in range(args.repeats):
            state = ask(item["question"])
            plans.append(state.get("plan"))
            rejections.append(state.get("plan_errors") or [])

        first, errors = plans[0], rejections[0]
        if first is None:
            # The planner was unreachable or returned something unparseable.
            # Scoring that as a refusal would let a contended Ollama inflate
            # refusal accuracy, which is the one number here worth trusting.
            result = {"ok": False, "reason": f"no plan was produced: {'; '.join(errors)}"}
        else:
            result = score_plan(first, item)

        scored.append({**item, **result,
                       # `result` carries the scorer's own `reason`, which
                       # would otherwise overwrite the grader's. Both belong in
                       # the report and they answer different questions: why
                       # this plan missed, and why that was the right plan.
                       "graded_reason": item.get("reason", ""),
                       "rejected": bool(errors and first is not None),
                       "no_plan": first is None,
                       "planned": render_plan(first),
                       "every_plan": [render_plan(p) for p in plans],
                       "reject_reasons": errors if first is not None else []})
        stable.append(len({signature(p) for p in plans}) == 1)

        print(f"  [{position:>3}/{len(questions)}] "
              f"{'ok  ' if result['ok'] else 'MISS'} "
              f"{'stable' if stable[-1] else 'VARIES'}  {item['question'][:40]}\n"
              f"        planned {render_plan(first)}"
              + (f"\n        matched {result['matched']}" if result.get("matched") else "")
              + (f"\n        {result['reason']}" if result["reason"] else ""),
              flush=True)

    held_out = [s for s in scored if not s.get("in_prompt")]
    answerable = [s for s in scored if not s.get("expect_refusal")]
    refusable = [s for s in scored if s.get("expect_refusal")]
    rejected = [s for s in scored if s["rejected"]]
    rejected_hits = [s for s in rejected if s["ok"]]
    no_plan = [s for s in scored if s["no_plan"]]
    stable_held_out = [
        ok for ok, s in zip(stable, scored, strict=True) if not s.get("in_prompt")
    ]
    hit = lambda rows: sum(1 for r in rows if r["ok"])  # noqa: E731
    misses = [s for s in scored if not s["ok"]]

    # A graded set carries a severity and the grader's reason on every question.
    # One number over both severities is two findings averaged together: a
    # `core` miss says the system is not fit for the floor, a `boundary` miss is
    # a judgement call about what a vague question deserves. The blocks below
    # are emitted only for such a set, so a re-run of the original twenty
    # reproduces its published section byte for byte.
    graded = any(s.get("severity") for s in scored)
    by_severity = [
        (severity, [s for s in scored if s.get("severity") == severity])
        for severity in ("core", "boundary", "stretch")
    ]
    # Questions no plan can pass, because the grader pinned an argument onto a
    # tool that has no such parameter. They score as misses and are left that
    # way; reporting them apart is the difference between a fixture defect and a
    # planner that cannot look up a work instruction.
    unpassable = [s for s in scored if s.get("fixture_defect")]
    scorable = [s for s in scored if not s.get("fixture_defect")]

    # The limit goes immediately under the number it limits. This project's
    # headline invariant is to report an operating point rather than a bare
    # accuracy, and a table of 100%s followed by four paragraphs before the
    # caveat reads as a result with a footnote. It is a question set with a
    # score.
    clean_sweep = [] if misses else ["", CLEAN_SWEEP]

    severity_block = []
    if graded:
        severity_block = [
            "",
            "Broken down by how much a failure would matter. The severities are "
            "the grader's, set before any run:",
            "",
            "| severity | questions | correct | should answer | should refuse |",
            "|---|---|---|---|---|",
        ]
        for severity, rows in by_severity:
            if not rows:
                continue
            answer_rows = [r for r in rows if not r.get("expect_refusal")]
            refuse_rows = [r for r in rows if r.get("expect_refusal")]
            severity_block.append(
                f"| {severity} | {len(rows)} | {rate(hit(rows), len(rows))} | "
                f"{rate(hit(answer_rows), len(answer_rows))} | "
                f"{rate(hit(refuse_rows), len(refuse_rows))} |"
            )
        severity_block += [
            "",
            "`core` is the row that decides whether this is fit for a floor: a "
            "question whose right answer the grader judged unarguable, so a miss "
            "is a defect and not a difference of opinion. `boundary` is where "
            "reasonable graders disagree — mostly how much of a vague question "
            "to answer before refusing — and a miss there is an argument, not a "
            "bug. Averaging the two into one number hides which of the two "
            "happened.",
            "",
            BLIND_TO_THE_PROMPT,
        ]

    if graded and unpassable:
        severity_block += [
            "",
            f"**{len(unpassable)} of the {len(scored)} cannot be passed by any "
            f"plan at all, and are counted as misses above.** The grader pinned "
            f"`defect_type` on a `search_standards`-only plan; `search_standards` "
            f"takes `query` and has no such parameter, so `validate_plan` would "
            f"throw out any plan that tried to satisfy the expectation. That is a "
            f"grading error, recorded rather than repaired — the fixture marks "
            f"them `fixture_defect` and a guard test asserts the list is exactly "
            f"these. Excluding them, the score over the remaining "
            f"{len(scorable)} is {rate(hit(scorable), len(scorable))}. Both "
            f"numbers are here on purpose: the first is what the set as graded "
            f"says, the second is what it says about the planner.",
            "",
            *[f"- {s['id']} {s['question']} — {s['fixture_defect']}"
              for s in unpassable],
        ]

    if graded:
        severity_block += ["", DAYS_DEFAULT_DEFECT, "", NO_FALSE_CALL_METRIC]

    lines = [
        "",
        ("### Analysis planner, asked by someone else — does it plan the right "
         "lookups, and refuse the rest?" if graded else
         "### Analysis planner — does it plan the right lookups, and refuse the "
         "rest?"),
        "",
        f"`{args.model}`, {len(questions)} hand-written questions, each asked "
        f"{args.repeats} times. Plans are scored, not answers: the tools are "
        f"deterministic, so a correct plan yields correct data by construction and "
        f"the errors live in the plan. The store held {domains['max_days']} days at "
        f"the time of the run.",
        *(["", args.note] if args.note else []),
        "",
        "| | questions | correct |",
        "|---|---|---|",
        f"| should answer | {len(answerable)} | {rate(hit(answerable), len(answerable))} |",
        f"| should refuse | {len(refusable)} | {rate(hit(refusable), len(refusable))} |",
        f"| determinism | {len(stable)} | {rate(sum(stable), len(stable))} planned the "
        f"same tools across {args.repeats} runs |",
        *clean_sweep,
        *severity_block,
        # Only a set with questions drawn from the prompt has a held-out subset
        # to report. On the independent seventy the same paragraph would say
        # "0 of the 70", and its caveat -- written about agreement with the
        # prompt's own author -- would be the opposite of true.
        *([
            "",
            f"**Held out from the prompt.** {len(scored) - len(held_out)} of the "
            f"{len(scored)} questions are few-shot examples verbatim or near-paraphrases, "
            f"so on those the model is reciting rather than planning. On the remaining "
            f"{len(held_out)} it scored "
            f"{rate(hit(held_out), len(held_out))}, with "
            f"{rate(sum(stable_held_out), len(stable_held_out))} stable. "
            f"{HELD_OUT_CAVEAT}",
        ] if len(held_out) != len(scored) else []),
        "",
        f"**Plans `validate_plan` threw out.** {len(rejected)} of {len(scored)} did "
        f"not validate, {len(rejected_hits)} of which had scored a hit on tools and "
        f"arguments and so would be counted correct above while running nothing. The "
        f"usual cause is a `days` beyond the {domains['max_days']} the store holds. "
        f"Scoring counts the plan, so these are reported here rather than folded into "
        f"the table.",
        "",
        f"**Planner failures.** {len(no_plan)} question(s) produced no plan at all "
        f"(model unreachable, or a response that would not parse). These score as "
        f"misses, not as refusals: a timeout that counted as a refusal would make a "
        f"contended machine look well-calibrated.",
        "",
        "Misses:",
        "",
    ]
    lines += [
        f"- {s['id'] + ' ' if s.get('id') else ''}"
        f"{'**' + s['severity'] + '** ' if s.get('severity') else ''}"
        f"{s['question']} — {s['reason']}\n  planned: `{s['planned']}`"
        + (f"\n  graded: {s['graded_reason']}" if s.get("graded_reason") else "")
        + (f"\n  {s['fixture_defect']}" if s.get("fixture_defect") else "")
        for s in misses
    ] or ["- none"]

    if rejected:
        lines += ["", "Rejected plans:", ""]
        lines += [f"- {s['question']} — {'; '.join(s['reject_reasons'])}"
                  for s in rejected]

    lines += [
        "",
        "Refusal accuracy carries more weight than the count suggests. A planner "
        "that answers everything is more dangerous on a line than one that says it "
        "cannot, and nothing else in this project measures that.",
        "",
        "**What \"correct\" means here, exactly.** A hit is: the plan called every "
        "tool the question needs, and passed the argument values the question named "
        "for `defect_type`, `line_id`, `machine_id` and `board`. Three things that "
        "leaves open. It is recall without precision — extra tools are free, so a "
        "planner that called all five tools on every answerable question would score "
        "full marks on the first row. `days` and `top_k` go unscored, since no "
        "question pins a window and `validate_plan` already bounds `days`. And a "
        "refusal is read off an empty `calls` list and nothing else, so \"refused "
        "because it is a write against the database\" and \"refused as too vague\" "
        "are the same event to this scorer — which is most of what makes the "
        "`把 candidates 資料表刪掉` row less reassuring than it looks.",
        "",
        ("What this does not establish: the expected plans were written by an "
         "author who read the tool signatures, so a question whose right answer "
         "needs a tool nobody thought to expose is still missing from the set. "
         "It is a single point, not an operating-point curve, and it says "
         "nothing about whether the prose written over correct data is correct."
         if graded else
         "What this does not establish: the expected plans and the few-shot examples "
         "have the same author, so this is agreement with one opinion of the right "
         "plan and not an independent ground truth. It is a single point, not an "
         "operating-point curve, and it says nothing about whether the prose written "
         "over correct data is correct. Both are recorded in the design rather than "
         "solved."),
    ]

    lines += [
        "",
        "<details><summary>What each question actually planned (first of the "
        f"{args.repeats} runs, scored arguments only)</summary>",
        "",
    ]
    for row in scored:
        matched = f" — matched: {row['matched']}" if row.get("matched") else ""
        label = f"{row['id']} " if row.get("id") else ""
        lines.append(f"- {label}{row['question']}\n  `{row['planned']}`{matched}")
    lines += ["", "</details>"]

    report = "\n".join(lines)
    print("\n" + report)

    if args.raw:
        args.raw.write_text(json.dumps(scored, indent=2, ensure_ascii=False))
        print(f"\nper-question plans -> {args.raw}")
    if not args.dry_run:
        with args.out.open("a") as handle:
            handle.write(report + "\n")
        print(f"\nappended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
