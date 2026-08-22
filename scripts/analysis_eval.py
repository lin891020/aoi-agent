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
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.analysis.graph import build_analysis_graph  # noqa: E402
from aoi_agent.analysis.plan import store_domains  # noqa: E402
from aoi_agent.graph.flow import DEFAULT_MODEL  # noqa: E402
from aoi_agent.llm.ollama import OllamaClient  # noqa: E402

QUESTIONS = Path(__file__).resolve().parents[1] / "tests/fixtures/analysis_questions.json"

CAUSE_WORDS = ("cause", "causal", "causation", "association", "correlat", "因果", "關聯")

#: Accuracy, not latency. A 10s client timeout would measure the timeout.
EVAL_TIMEOUT_S = 180.0


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return json.loads(Path(path).read_text())


def score_plan(plan: dict, expected: dict) -> dict:
    """Did this plan do what the question needed? One reason if not."""
    calls = plan.get("calls") or []
    refused = not calls

    if expected.get("expect_refusal"):
        if refused:
            return {"ok": True, "reason": ""}
        return {"ok": False,
                "reason": f"should have refused; planned {[c['tool'] for c in calls]}"}

    if refused:
        return {"ok": False, "reason": "refused a question it should have answered"}

    called = {c["tool"] for c in calls}
    missing = set(expected.get("expect_tools") or []) - called
    if missing:
        # Extra tools are fine -- more context is not an error. Missing ones are
        # not: the answer would rest on data nobody fetched.
        return {"ok": False, "reason": f"never called {sorted(missing)}"}

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
    return json.dumps(sorted(c["tool"] for c in (plan.get("calls") or [])))


def rate(correct: int, total: int) -> str:
    return f"{correct}/{total} = {correct / total:.0%}" if total else "—"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--repeats", type=int, default=3,
                        help="times to re-ask each question, for determinism")
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    questions = load_questions()
    domains = store_domains()
    graph = build_analysis_graph(
        OllamaClient(args.model, timeout=EVAL_TIMEOUT_S), domains
    )
    print(f"store holds {domains['max_days']} days, "
          f"lines {sorted(domains['line_id'])}, "
          f"machines {sorted(domains['machine_id'])}\n")

    scored, stable = [], []
    for position, item in enumerate(questions, 1):
        plans, rejections = [], []
        for _ in range(args.repeats):
            state = graph.invoke(
                {"question": item["question"], "results": [], "timings_ms": {}}
            )
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

        scored.append({**item, **result, "rejected": bool(errors and first is not None),
                       "no_plan": first is None,
                       "reject_reasons": errors if first is not None else []})
        stable.append(len({signature(p) for p in plans}) == 1)

        print(f"  [{position:>3}/{len(questions)}] "
              f"{'ok  ' if result['ok'] else 'MISS'} "
              f"{'stable' if stable[-1] else 'VARIES'}  {item['question'][:40]}  "
              f"{result['reason']}", flush=True)

    held_out = [s for s in scored if not s.get("in_prompt")]
    answerable = [s for s in scored if not s.get("expect_refusal")]
    refusable = [s for s in scored if s.get("expect_refusal")]
    rejected = [s for s in scored if s["rejected"]]
    no_plan = [s for s in scored if s["no_plan"]]
    stable_held_out = [
        ok for ok, s in zip(stable, scored, strict=True) if not s.get("in_prompt")
    ]
    hit = lambda rows: sum(1 for r in rows if r["ok"])  # noqa: E731

    lines = [
        "",
        "### Analysis planner — does it plan the right lookups, and refuse the rest?",
        "",
        f"`{args.model}`, {len(questions)} hand-written questions, each asked "
        f"{args.repeats} times. Plans are scored, not answers: the tools are "
        f"deterministic, so a correct plan yields correct data by construction and "
        f"the errors live in the plan. The store held {domains['max_days']} days at "
        f"the time of the run.",
        "",
        "| | questions | correct |",
        "|---|---|---|",
        f"| should answer | {len(answerable)} | {rate(hit(answerable), len(answerable))} |",
        f"| should refuse | {len(refusable)} | {rate(hit(refusable), len(refusable))} |",
        f"| determinism | {len(stable)} | {rate(sum(stable), len(stable))} planned the "
        f"same tools across {args.repeats} runs |",
        "",
        f"**Held out from the prompt.** Five of the twenty questions are few-shot "
        f"examples verbatim or near-paraphrases, so on those the model is reciting "
        f"rather than planning. On the remaining {len(held_out)} it scored "
        f"{rate(hit(held_out), len(held_out))}, with "
        f"{rate(sum(stable_held_out), len(stable_held_out))} stable. That is the "
        f"number to read; the headline above is the optimistic one.",
        "",
        f"**Plans that scored a hit and still could not run.** {len(rejected)} named "
        f"the right tools and were rejected by `validate_plan` — most often a `days` "
        f"beyond the {domains['max_days']} the store holds. Scoring counts the plan, "
        f"so these are reported here rather than folded into the table.",
        "",
        f"**Planner failures.** {len(no_plan)} question(s) produced no plan at all "
        f"(model unreachable, or a response that would not parse). These score as "
        f"misses, not as refusals: a timeout that counted as a refusal would make a "
        f"contended machine look well-calibrated.",
        "",
        "Misses:",
        "",
    ]
    misses = [s for s in scored if not s["ok"]]
    lines += [f"- {s['question']} — {s['reason']}" for s in misses] or ["- none"]

    if not misses:
        # A benchmark nothing fails has not been passed, it has been outgrown.
        # Saying so here rather than in a commit message keeps the next reader
        # from quoting a clean sweep as a bound.
        lines += [
            "",
            "**A clean sweep is a fact about the question set before it is a fact "
            "about the planner.** Nothing here found the boundary, so nothing here "
            "bounds anything: the honest reading is that these twenty questions "
            "are inside what this model does easily, not that the planner is "
            "correct. To have any resolution the set needs questions that are "
            "harder in a specific way — a window the store does not hold, an "
            "aggregate no single tool computes, a machine named only implicitly — "
            "and it needs an author who did not write the prompt.",
        ]

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
        "What this does not establish: the expected plans and the few-shot examples "
        "have the same author, so this is agreement with one opinion of the right "
        "plan and not an independent ground truth. It is a single point, not an "
        "operating-point curve, and it says nothing about whether the prose written "
        "over correct data is correct. Both are recorded in the design rather than "
        "solved.",
    ]

    report = "\n".join(lines)
    print("\n" + report)
    if not args.dry_run:
        with args.out.open("a") as handle:
            handle.write(report + "\n")
        print(f"\nappended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
