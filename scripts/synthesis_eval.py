"""Is the prose true of the results it was written from?

`scripts/analysis_eval.py` scores the plan and stops there, on the reasoning
that the tools are deterministic so a correct plan yields correct data. That is
true and it leaves the last link unmeasured: after the tools return, a second
model call writes the sentences a supervisor reads, and nothing established
whether those sentences are true of the payload they were written from.

Not hypothetical. On 2026-08-23 the criteria retrieval was returning another
class's rules and the synthesis layer repeated them faithfully -- eight stored
explanations cited a rule no document contains. The retrieval is fixed. What
the incident exposed is that a wrong sentence over right data, and a right
sentence over wrong data, both leave this system silently.

The question is deliberately narrow. **Not** "is the answer true", which needs
a person who knows the line. "Is the prose true of the results" -- and because
the results are deterministic and stored beside the prose, most of that is
arithmetic. A figure quoted in a sentence either renders from a value in the
payload or it does not. The entity it is attached to either holds it or does
not. `aoi_agent.analysis.claims` is where that comparison lives; this script
runs the graph, applies it, and reports the five kinds apart rather than as one
accuracy.

Two of the five are checked -- no judgement anywhere in the verdict, and a
reader can reproduce each one from the stored payload. Three are pattern
matches that raise a candidate for a person: a cause word, a trend word, a rule
asserted about a class nothing retrieved. Those are published as flags with the
sentence attached, every one of them, so the judgement is auditable rather than
asserted. **No model judges another model here.** A model judge would be the
same class of instrument as the thing under test, graded by nobody, and its
agreement would be indistinguishable from a shared blind spot.

    uv run python scripts/synthesis_eval.py
    uv run python scripts/synthesis_eval.py --limit 8 --dry-run

The question set is the seventy in `analysis_questions_independent.json`,
reused rather than rewritten because its authorship is its value: three people
who had seen neither the planner's prompt nor its examples. Only the runs that
reach the synthesis node have prose over results to score -- a refusal, a
rejected plan and a planner failure all terminate in `report_node`, which
writes a fixed string.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.analysis.claims import (  # noqa: E402
    CHECKED_KINDS,
    KINDS,
    Grounding,
    check,
    gaps,
    numbers_in,
    perturbations,
    sentences,
)
from aoi_agent.analysis.graph import build_analysis_graph  # noqa: E402
from aoi_agent.analysis.plan import store_domains  # noqa: E402
from aoi_agent.graph.flow import DEFAULT_MODEL  # noqa: E402
from aoi_agent.llm.ollama import EXPLANATION_DEADLINE_S, OllamaClient  # noqa: E402
from reverifier_latency import (  # noqa: E402
    competing_processes,
    ollama_ps,
    process_table,
    resident_models,
)

QUESTIONS = (
    Path(__file__).resolve().parents[1]
    / "tests/fixtures/analysis_questions_independent.json"
)
CONTROL = (
    Path(__file__).resolve().parents[1]
    / "tests/fixtures/synthesis_wrong_summary.json"
)

#: The deadline the station runs at, not a longer one for the benchmark. Same
#: reasoning as `analysis_eval.EVAL_TIMEOUT_S`: a run measured at a timeout
#: nothing ships describes a configuration nobody has.
EVAL_TIMEOUT_S = EXPLANATION_DEADLINE_S

#: One line per kind, printed under the count. A taxonomy whose entries are
#: named but not defined is a taxonomy a reader has to take on trust.
WHAT_EACH_KIND_IS = {
    "fabricated_figure":
        "a number in the prose that renders from no value in the payload. The "
        "reader cannot catch this: it looks like the figures beside it.",
    "misattributed_figure":
        "a number that is in the payload, attached to an entity that does not "
        "hold it — the right rate against the wrong machine, line or class. "
        "Worse than a fabrication in one way: every figure audits clean.",
    "unsupported_claim":
        "a cause, or a movement over time. No plannable tool returns a time "
        "series, so a trend claim is unsupported by the shape of the tool "
        "surface and not merely by this run's numbers.",
    "unhedged_gap":
        "a tool failed, errored or returned nothing, and the prose reads as "
        "complete.",
    "misquoted_criterion":
        "a rule asserted about a class no retrieved passage governs or "
        "mentions, or attributed to a work instruction never retrieved. This "
        "is the 2026-08-23 incident, made checkable.",
}

#: What the checked half cannot be argued with, and what the judged half is.
THE_SPLIT = (
    "**Two of the five kinds carry no judgement at all.** A figure is compared "
    "against the stored payload and the comparison is reproducible from the "
    "raw file this run wrote; nothing about it is an opinion. The other three "
    "are pattern matches — a cause word, a trend word, a class name beside a "
    "normative word — and each one is a candidate a person settles. Every "
    "candidate is printed below with its sentence, so the judgement is "
    "auditable rather than asserted."
)

#: The limit that matters most and is easiest to leave out.
NOT_INDEPENDENT = (
    "**The checker and the system have the same author.** That is the sharpest "
    "limit on this number and no run removes it: the same person chose what "
    "counts as a fabrication and wrote the layer being scored, so a failure "
    "mode neither of them thought about is invisible to both. Two things "
    "mitigate it and neither closes it. The questions are the independent "
    "seventy, written by three authors who had seen none of this. And the "
    "checker is required to fail a summary corrupted on purpose — "
    "`tests/fixtures/synthesis_wrong_summary.json` carries one instance of "
    "every kind and `tests/test_synthesis_claims.py` fails if any kind stops "
    "firing, or if a faithful summary over the same results starts firing. A "
    "taxonomy nothing can fail reads exactly like a clean result, and that "
    "control is what tells the two apart."
)

#: What the first pass actually measured, which was the checker. Stated in the
#: report because a clean result whose instrument was corrected on the way to
#: it is a different claim from a clean result taken first time.
WHAT_ADJUDICATION_FOUND = (
    "**The first pass of this checker raised 43 findings, and 41 of them were "
    "the checker's fault.** Adjudicating them against the payloads — which is "
    "what a judged flag is for — turned up five defects, every one of which "
    "made the instrument shout rather than made the model wrong. `M12` was "
    "read as the number 12, so a sentence that merely named six machines "
    "produced six swap findings; twenty-two of the thirty-seven attribution "
    "findings were nothing but that. A Chinese answer never split into "
    "sentences, because `。` is not followed by a space, so a figure was "
    "attributed to whichever entity was named last anywhere in the paragraph. "
    "`19 copper, 22 mousebite` was read backwards, shifting a whole list by "
    "one. A fleet average quoted beside a machine name was called that "
    "machine's. And a share the model divided out of two stored figures was "
    "called a fabrication. Each correction is a commit with a test either "
    "side of it — the shape it now passes, and the swap in that same shape it "
    "must still catch — because five changes that each make a checker quieter "
    "are exactly how a checker goes blind. The numbers below come from a "
    "fresh run scored by the corrected checker."
)

#: What the measure is *of*, stated before the number so it cannot be read
#: wider than it is.
FIDELITY_NOT_TRUTH = (
    "**This measures fidelity to the results, not truth about the line.** A "
    "tool that returns a 9-day window labelled `\"days\": 14` — which "
    "`query_machine_stats` does, and docs/benchmarks.md already records — is "
    "repeated faithfully by the model and passes every check here. Correctly: "
    "that defect belongs to the tool, and a prose checker that flagged it "
    "would be scoring the wrong layer. The claim this section supports is "
    "that a supervisor reading the sentence is reading the payload, not that "
    "the payload is right."
)


def contention(model: str) -> list[str]:
    """Everything sharing this machine that is not the model under test."""
    others = [
        name for name in resident_models(ollama_ps())
        if not name.startswith(model.split(":")[0])
    ]
    return [f"resident model: {name}" for name in others] + competing_processes(
        process_table(), os.getpid()
    )


def rate(part: int, whole: int) -> str:
    return f"{part}/{whole} = {part / whole:.0%}" if whole else "—"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N questions, for a smoke run")
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument("--raw", type=Path, default=None,
                        help="write every answer and finding to JSON, so a "
                             "reader can re-adjudicate the judged kinds")
    parser.add_argument("--dry-run", action="store_true",
                        help="print, do not append -- and do not refuse on a "
                             "contended machine, since nothing is published")
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text())
    if args.limit:
        questions = questions[: args.limit]

    ps_before = ollama_ps()
    busy_before = contention(args.model)
    print(f"ollama ps before:\n{ps_before}\n")
    print("busy processes before:\n  " + ("\n  ".join(busy_before) or "(none)") + "\n")
    if busy_before and not args.dry_run:
        print(
            "something else is holding this machine; a contended run is to be "
            "discarded, not published. Re-run when it is quiet, or use --dry-run.",
            file=sys.stderr,
        )
        return 1

    domains = store_domains()
    client = OllamaClient(args.model, timeout=EVAL_TIMEOUT_S)
    client.warm_up()
    graph = build_analysis_graph(client, domains)

    print(f"store holds {domains['max_days']} days, "
          f"lines {sorted(domains['line_id'])}, "
          f"machines {sorted(domains['machine_id'])}\n")

    started = time.perf_counter()
    scored, unscored = [], []
    for position, item in enumerate(questions, 1):
        state = graph.invoke(
            {"question": item["question"], "results": [], "timings_ms": {}}
        )
        results = state.get("results") or []
        answer = state.get("answer", "")
        if not results:
            # `report_node`'s fixed string: a refusal, a rejected plan or a
            # planner failure. There is no payload for the prose to be true or
            # false of, so scoring it would be scoring the planner again.
            unscored.append({**item, "answer": answer,
                             "why": "refused" if state.get("refused")
                             else "no plan ran"})
            print(f"  [{position:>3}/{len(questions)}] --   {item['question'][:44]}")
            continue

        findings, waved, derived = check(answer, state.get("plan"), results)
        grounding = Grounding(results)
        accepted, tried, accepted_dp, tried_dp = perturbations(answer, grounding)
        figures = [
            value for sentence in sentences(answer)
            for value, *_ in numbers_in(sentence)
        ]
        scored.append({
            "id": item.get("id"),
            "question": item["question"],
            "severity": item.get("severity"),
            "plan": state.get("plan"),
            "results": results,
            "answer": answer,
            "findings": [f.__dict__ for f in findings],
            "restated_from_plan": waved,
            "derived": derived,
            "figures": len(figures),
            "sentences": len(sentences(answer)),
            "gaps": gaps(results),
            "perturbation": [accepted, tried, accepted_dp, tried_dp],
        })
        counts = Counter(f.kind for f in findings)
        print(f"  [{position:>3}/{len(questions)}] "
              f"{'ok  ' if not findings else 'FLAG'} "
              f"{len(figures):>3} figures  {item['question'][:38]}"
              + (f"\n        {dict(counts)}" if counts else ""), flush=True)

    minutes = (time.perf_counter() - started) / 60
    ps_after = ollama_ps()
    busy_after = contention(args.model)
    print(f"\nollama ps after:\n{ps_after}\n")
    print("busy processes after:\n  " + ("\n  ".join(busy_after) or "(none)"))
    if busy_after and not args.dry_run:
        print("the machine was contended by the end of the run; discarding.",
              file=sys.stderr)
        return 1

    report = render(scored, unscored, args, minutes, ps_before, ps_after,
                    busy_before, busy_after)
    print("\n" + report)

    if args.raw:
        args.raw.write_text(json.dumps(scored, indent=2, ensure_ascii=False))
        print(f"\nevery answer and finding -> {args.raw}")
    if not args.dry_run:
        with args.out.open("a") as handle:
            handle.write(report + "\n")
        print(f"\nappended to {args.out}")
    return 0


def render(scored, unscored, args, minutes, ps_before, ps_after,
           busy_before, busy_after) -> str:
    every = [f for row in scored for f in row["findings"]]
    counts = Counter(f["kind"] for f in every)
    checked = sum(counts[kind] for kind in CHECKED_KINDS)
    judged = len(every) - checked
    clean = [row for row in scored if not row["findings"]]
    clean_checked = [
        row for row in scored
        if not [f for f in row["findings"] if f["kind"] in CHECKED_KINDS]
    ]
    figures = sum(row["figures"] for row in scored)
    restated = sum(row["restated_from_plan"] for row in scored)
    derived = sum(len(row["derived"]) for row in scored)
    accepted = sum(row["perturbation"][0] for row in scored)
    tried = sum(row["perturbation"][1] for row in scored)
    accepted_dp = sum(row["perturbation"][2] for row in scored)
    tried_dp = sum(row["perturbation"][3] for row in scored)
    control = json.loads(CONTROL.read_text())

    lines = [
        "",
        "### The prose over the results — is the sentence true of the payload?",
        "",
        f"`{args.model}`, the {len(scored) + len(unscored)} independent questions "
        f"run through the whole analysis graph, of which **{len(scored)} reached "
        f"the synthesis node** and have prose written over a payload to score. "
        f"The other {len(unscored)} terminated in `report_node` — a refusal, a "
        f"plan `validate_plan` threw out, or no plan at all — which writes a "
        f"fixed string over no results. Ran in {minutes:.0f} min. One pass per "
        f"question, so this is a single point and not a distribution.",
        "",
        FIDELITY_NOT_TRUTH,
        "",
        WHAT_ADJUDICATION_FOUND,
        "",
        f"**{rate(len(clean), len(scored))} of the scored answers carry no "
        f"finding of any kind**, and {rate(len(clean_checked), len(scored))} "
        f"carry no *checked* finding — nothing fabricated and nothing "
        f"misattributed across {figures} figures quoted in "
        f"{sum(row['sentences'] for row in scored)} sentences.",
        "",
        "| kind | findings | answers affected | decided by |",
        "|---|---|---|---|",
    ]
    for kind in KINDS:
        affected = sum(
            1 for row in scored if any(f["kind"] == kind for f in row["findings"])
        )
        how = "comparison" if kind in CHECKED_KINDS else "a person, on a flag"
        lines.append(f"| `{kind}` | {counts[kind]} | {affected} | {how} |")
    lines += ["", *[f"- `{kind}` — {text}" for kind, text in WHAT_EACH_KIND_IS.items()]]

    lines += [
        "",
        THE_SPLIT,
        "",
        f"**How much of this was checkable rather than judged.** Of {figures} "
        f"figures quoted across the scored answers, every one was compared "
        f"against the payload — that is 100% of the figure claims, and figures "
        f"are what a supervisor acts on. Of the {len(every)} findings, "
        f"{checked} came from a comparison and {judged} from a flag somebody "
        f"had to settle. What is *not* covered at all: a sentence that quotes "
        f"every figure correctly and characterises them wrongly — \"M22 is the "
        f"worst machine\" over a payload where it is second — is outside every "
        f"kind here, because the claim is a reading of the numbers rather than "
        f"a number. That is the boundary, and it is where a person still has "
        f"to look.",
        "",
        f"**Two latitudes the checker grants, counted rather than assumed.** "
        f"{restated} figure(s) were accepted as restated from the plan and "
        f"{derived} as a ratio of two figures the payload holds — \"18.1% of "
        f"L1's defects\" over a payload storing 175 and 966. The division is "
        f"checked, not assumed: a value is excused only when a pair producing "
        f"it exists. A derived figure is exempt from the entity check as well, "
        f"and that is a real gap rather than a convenience — a quotient "
        f"carries no entity, so a rate computed for the wrong machine is "
        f"outside what this can see.",
        "",
        f"**{restated} figure(s) were waved through as restated from the plan.** "
        f"`SYNTHESIS_PROMPT` orders the plan's assumptions repeated in the "
        f"prose, so a window the planner chose comes back in the answer having "
        f"never been in a payload. Counting those as fabrications would score "
        f"the synthesis node for the planner's work, which is "
        f"`scripts/analysis_eval.py`'s job. They are counted here instead of "
        f"being silent.",
        "",
        f"**How lenient the figure check is, measured rather than argued.** "
        f"Every figure that grounded was re-asked at 1.3x and 0.7x. "
        f"{rate(accepted, tried)} of those perturbed figures still grounded — "
        f"but that number is carried by small integers, since the payloads are "
        f"full of box coordinates and a coordinate moved 30% lands on another "
        f"coordinate. Restricted to figures written with a decimal point, "
        f"which is where a rate or a share lives, "
        f"{rate(accepted_dp, tried_dp)} survived perturbation. A rate that is "
        f"wrong by 30% does not pass this checker; a small count sometimes "
        f"does.",
        "",
        NOT_INDEPENDENT,
        "",
        "```",
        "ollama ps before the run",
        ps_before or "(empty)",
        "",
        "busy processes before the run",
        "\n".join(busy_before) or "(none)",
        "",
        "ollama ps after the run",
        ps_after or "(empty)",
        "",
        "busy processes after the run",
        "\n".join(busy_after) or "(none)",
        "```",
        "",
        f"**The control.** {' '.join(control['why'].split())}",
        "",
        *[f"- `{kind}` — {text}" for kind, text in control["expected_kinds"].items()],
        "",
        "Every finding, so the judged ones can be re-adjudicated:",
        "",
    ]

    if not every:
        lines.append("- none")
    for row in scored:
        for finding in row["findings"]:
            head = f"- {row['id']} `{finding['kind']}` — {finding['claim']}"
            lines.append(
                head
                + f"\n  results say: {finding['evidence']}"
                + (f"\n  sentence: {finding['sentence']}" if finding["sentence"] else "")
            )

    lines += [
        "",
        "What this does not establish. It is one pass over "
        f"{len(scored)} answers, not a distribution, and the model is sampled "
        "rather than deterministic — a second run would produce different "
        "sentences over the identical payloads. The three judged kinds are "
        "pattern matches with a person behind them, so their counts are a "
        "floor on what a pattern can raise and not a rate. Nothing here scores "
        "whether the answer was *useful*, whether it answered the question "
        "asked, or whether the plan fetched the right data — that last is "
        "`scripts/analysis_eval.py`, which scored 55/70 on the same set. And "
        "the checker is not independent of the system; the control fixture is "
        "what stands in for independence, and it is a weaker thing.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
