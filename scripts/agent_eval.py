"""Does the agent layer earn its place?

The headline numbers in docs/benchmarks.md are the vision model's. Nothing has
ever measured the layer above it, so the design's central bet is unevidenced:
that an LLM weighing production context and acceptance criteria settles cases
the classifier could not, and knows when it cannot settle them either.

Two questions, and the second matters more.

**Does it beat the classifier it is second-guessing?** The comparison is on the
17.8% the router sends to investigation, not on the whole queue. On that subset
the vision model is right 62.9% of the time, and that is the number to beat.
Beating it by rewriting confident classifications is worth little; the point is
the ones the classifier got wrong.

**Is the escalation calibrated?** The flow's failure story rests on
``confident=false`` landing on the genuinely hard cases. If accuracy on what it
escalated matches accuracy on what it kept, the flag carries no signal and the
escalations are noise -- the layer would be spending operator time at random.
A calibrated layer escalates the cases it would have got wrong.

Escapes are counted separately because the errors are not symmetric: calling a
real defect ``false_call`` ships a bad board, and no amount of accuracy
elsewhere pays for it.

    uv run python scripts/agent_eval.py --candidates 60

Ground truth is read here and nowhere else. The agent never sees it, the store's
dict helpers never expose it, and the review station never renders it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from sqlalchemy import select  # noqa: E402

from aoi_agent.graph.flow import CONFIDENT, DEFAULT_MODEL, build_graph  # noqa: E402
from aoi_agent.llm.ollama import OllamaClient  # noqa: E402
from aoi_agent.store.boards import session_factory  # noqa: E402
from aoi_agent.store.models import Board, CandidateRecord  # noqa: E402
from aoi_agent.vision.inference import DEFAULT_DISMISS_THRESHOLD  # noqa: E402

#: Accuracy, not latency. WI-300's 10s budget would time out most calls and turn
#: the run into a measurement of the timeout. Latency lives in latency_report.py.
EVAL_TIMEOUT_S = 180.0

#: Candidates whose box only fragments a real defect. Training holds them out
#: rather than labelling them spurious, and scoring has to do the same: neither
#: `false_call` nor the defect's class is the right answer for them.
HELD_OUT = "fragment"


def investigated(limit: int) -> list[dict]:
    """A spread of candidates the router sends to the LLM.

    Sampled by stride across the store's order rather than taking the first N,
    which would draw them all from the same few boards and the same lot.
    """
    with session_factory()() as session:
        rows = session.execute(
            select(
                Board.stem,
                CandidateRecord.index_on_board,
                CandidateRecord.predicted_class,
                CandidateRecord.confidence,
                CandidateRecord.false_call_probability,
                CandidateRecord.ground_truth,
            )
            .join(CandidateRecord, CandidateRecord.board_id == Board.id)
            .order_by(Board.id, CandidateRecord.index_on_board)
        ).all()

    pool = [
        {
            "reference": f"{stem}#{index}",
            "model_class": klass,
            "confidence": confidence,
            "ground_truth": truth,
        }
        for stem, index, klass, confidence, false_call, truth in rows
        if false_call < DEFAULT_DISMISS_THRESHOLD
        and not (confidence >= CONFIDENT and klass not in ("open", "false_call"))
        and truth is not None
        and truth != HELD_OUT
    ]
    if len(pool) <= limit:
        return pool
    stride = len(pool) / limit
    return [pool[int(i * stride)] for i in range(limit)]


def rate(correct: int, total: int) -> str:
    return f"{correct}/{total} = {correct / total:.1%}" if total else "—"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--candidates", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument("--raw", type=Path, default=None, help="write per-case JSON")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--from-raw", type=Path, default=None,
        help="rebuild the report from a saved --raw file, without calling the model",
    )
    args = parser.parse_args()

    if args.from_raw:
        results = json.loads(args.from_raw.read_text())
        elapsed = 0.0
        return report(args, results, elapsed)

    sample = investigated(args.candidates)
    if not sample:
        print("no investigated candidates with ground truth", file=sys.stderr)
        return 1

    graph = build_graph(
        OllamaClient(args.model, timeout=EVAL_TIMEOUT_S), InMemorySaver()
    )

    results = []
    started = time.perf_counter()
    for position, case in enumerate(sample, 1):
        state = graph.invoke(
            {"candidate_ref": case["reference"], "trace": [], "timings_ms": {}},
            config={"configurable": {"thread_id": f"eval-{case['reference']}"}},
        )
        record = {
            **case,
            "agent_verdict": state.get("agent_verdict"),
            "escalated": "__interrupt__" in state,
            "rationale": state.get("agent_rationale", ""),
        }
        results.append(record)
        flag = "ESC" if record["escalated"] else "   "
        hit = "ok " if record["agent_verdict"] == case["ground_truth"] else "MISS"
        print(
            f"  [{position:>3}/{len(sample)}] {case['reference']:<16} {flag} "
            f"model {case['model_class']:<10} agent {str(record['agent_verdict']):<10} "
            f"truth {case['ground_truth']:<10} {hit}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    return report(args, results, elapsed)


def report(args, results: list[dict], elapsed: float) -> int:
    kept = [r for r in results if not r["escalated"]]
    sent = [r for r in results if r["escalated"]]

    def correct(rows) -> int:
        return sum(1 for r in rows if r["agent_verdict"] == r["ground_truth"])

    def vision_correct(rows) -> int:
        return sum(1 for r in rows if r["model_class"] == r["ground_truth"])

    # An escape is a real defect dispositioned as a false call. Only the ones
    # the agent kept can escape: an escalated candidate reaches a person.
    escapes = [
        r for r in kept
        if r["agent_verdict"] == "false_call" and r["ground_truth"] != "false_call"
    ]

    lines = [
        "",
        "### Agent layer — does it beat the classifier, and is the escalation calibrated?",
        "",
        f"`{args.model}`, {len(results)} candidates the router sends to investigation, "
        f"sampled by stride across the store. `fragment` ground truth is held out, as "
        f"in training. Ran in {elapsed / 60:.0f} min.",
        "",
        "**What the system dispositions on, against what the LLM would have "
        "dispositioned on.** `decide_node` takes the classifier's class; the "
        "agent column is the counterfactual it replaced.",
        "",
        "| | candidates | system (classifier) | LLM counterfactual |",
        "|---|---|---|---|",
        f"| all investigated | {len(results)} | {rate(vision_correct(results), len(results))} "
        f"| {rate(correct(results), len(results))} |",
        f"| agent kept | {len(kept)} | {rate(vision_correct(kept), len(kept))} "
        f"| {rate(correct(kept), len(kept))} |",
        f"| agent escalated | {len(sent)} | {rate(vision_correct(sent), len(sent))} "
        f"| {rate(correct(sent), len(sent))} |",
        "",
    ]

    kept_acc = correct(kept) / len(kept) if kept else 0.0
    sent_acc = correct(sent) / len(sent) if sent else 0.0
    #: Below this, neither group says anything about calibration. A gap computed
    #: from a handful of cases is noise wearing a percentage sign, and reporting
    #: it as a finding is how a benchmark starts lying.
    MIN_GROUP = 8

    if len(kept) < MIN_GROUP or len(sent) < MIN_GROUP:
        lines += [
            f"**Calibration.** Not assessed: {len(kept)} kept and {len(sent)} "
            f"escalated, and fewer than {MIN_GROUP} in either group cannot "
            "separate a calibrated flag from a coin.",
            "",
        ]
    else:
        margin = kept_acc - sent_acc
        lines += [
            "**Calibration of the hand-off.** The LLM's verdicts were right "
            f"{kept_acc:.1%} of the time on what it kept and {sent_acc:.1%} of the "
            f"time on what it handed over, a gap of {margin:+.1%}. "
            + (
                "The escalations land on the harder cases, which is what the "
                "confidence flag is for."
                if margin >= 0.10
                else "That gap is too small to call the flag informative: the "
                "layer is escalating close to at random, and an operator's time "
                "is being spent without the cases being selected."
            ),
            "",
        ]

    # The sharpest test of whether the LLM's *verdict* is worth anything: on the
    # cases where it overrode the classifier, who turned out to be right. An
    # agent that only ever agrees adds nothing but latency; one that disagrees
    # and loses is actively removing accuracy.
    changed = [r for r in results if r["agent_verdict"] != r["model_class"]]
    agent_won = sum(1 for r in changed if r["agent_verdict"] == r["ground_truth"])
    vision_won = sum(1 for r in changed if r["model_class"] == r["ground_truth"])

    override = (
        "**Where the LLM would have overridden the classifier.** It proposed a "
        "different class on "
        f"{len(changed)} of {len(results)} candidates. The agent was right "
        f"{agent_won} of those times; the classifier had already been right "
        f"{vision_won} times, and {len(changed) - agent_won - vision_won} were "
        "wrong either way."
    )
    if kept and vision_won > agent_won:
        override += (
            " Acting on those proposals would cost accuracy rather than add it, "
            "which is why the flow does not. On the kept set the classifier "
            f"scores {vision_correct(kept)}/{len(kept)} = "
            f"{vision_correct(kept) / len(kept):.1%} against the LLM's "
            f"{correct(kept) / len(kept):.1%}."
        )

    lines += [
        override,
        "",
        f"**Escalation rate.** {len(sent)}/{len(results)} = "
        f"{len(sent) / len(results):.1%} of investigated candidates, which is "
        f"{len(sent) / len(results) * 0.178:.1%} of the whole queue.",
        "",
        f"**Escapes.** {len(escapes)} of {len(kept)} kept candidates were called "
        f"`false_call` while carrying a real defect"
        + (
            "." if not escapes
            else f" — {', '.join(sorted({r['ground_truth'] for r in escapes}))}."
        ),
        "",
        "Distribution of what the agent said, against the truth:",
        "",
        "| truth | n | agent agreed | agent escalated |",
        "|---|---|---|---|",
    ]
    truths = Counter(r["ground_truth"] for r in results)
    for truth, n in truths.most_common():
        rows = [r for r in results if r["ground_truth"] == truth]
        lines.append(
            f"| {truth} | {n} | {correct(rows)} "
            f"| {sum(1 for r in rows if r['escalated'])} |"
        )

    report = "\n".join(lines)
    print("\n" + report)

    if args.raw:
        args.raw.write_text(json.dumps(results, indent=2))
        print(f"\nper-case results -> {args.raw}")
    if not args.dry_run:
        with args.out.open("a") as handle:
            handle.write(report + "\n")
        print(f"appended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
