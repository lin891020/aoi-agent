"""Command line entry point for the re-verification station.

    uv run python -m aoi_agent review 20085293#2
    uv run python -m aoi_agent board 20085293
    uv run python -m aoi_agent corrections
"""

from __future__ import annotations

import argparse
import sys

from langgraph.types import Command

from aoi_agent.graph.flow import DEFAULT_MODEL, build_graph
from aoi_agent.llm.ollama import OllamaClient
from aoi_agent.store.boards import (
    candidates_for_board,
    corrections,
    record_decision,
    sample_board_stems,
)

DISPOSITION_LABEL = {
    "dismissed": "DISMISSED",
    "defect_confirmed": "DEFECT",
    "escalated": "ESCALATED",
}


def _print_result(reference: str, state: dict) -> None:
    label = DISPOSITION_LABEL.get(state.get("disposition", ""), "?")
    print(f"  {reference:<16} {label:<10} {state.get('verdict', '?'):<12} "
          f"by {state.get('decided_by', '?')}")
    if state.get("agent_rationale"):
        print(f"      {state['agent_rationale']}")
    timings = state.get("timings_ms", {})
    if timings:
        parts = " ".join(f"{k} {v:.0f}ms" for k, v in timings.items())
        print(f"      path: {' -> '.join(state.get('trace', []))}")
        print(f"      {parts}")


def _run_one(graph, reference: str, auto_answer: str | None) -> dict:
    config = {"configurable": {"thread_id": reference}}
    state = graph.invoke({"candidate_ref": reference}, config=config)

    if "__interrupt__" in state:
        payload = state["__interrupt__"][0].value
        print(f"\n  escalated: {reference}")
        print(f"    reason: {payload['reason']}")
        print(f"    model said {payload['model_class']} "
              f"({payload['model_confidence']:.3f})")

        if auto_answer:
            verdict, reviewer = auto_answer, "auto"
        else:
            verdict = input(f"    your verdict {payload['options']}: ").strip()
            reviewer = "operator"

        state = graph.invoke(
            Command(resume={"verdict": verdict, "reviewer": reviewer}), config=config
        )
        record_decision(
            reference, verdict, "human", reviewer, payload["reason"]
        )
    else:
        record_decision(
            reference,
            state["verdict"],
            state["decided_by"],
            rationale=state.get("agent_rationale"),
        )
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aoi_agent", description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    sub = parser.add_subparsers(dest="command", required=True)

    review = sub.add_parser("review", help="re-verify one flagged region")
    review.add_argument("candidate_ref")
    review.add_argument("--answer", help="answer escalations automatically (for demos)")

    board = sub.add_parser("board", help="re-verify every region on a board")
    board.add_argument("board")
    board.add_argument("--answer", help="answer escalations automatically")
    board.add_argument("--limit", type=int, default=None)

    sub.add_parser("boards", help="list boards in the store")
    sub.add_parser("corrections", help="show where operators overruled the model")

    args = parser.parse_args(argv)

    if args.command == "boards":
        for stem in sample_board_stems(20):
            print(stem)
        return 0

    if args.command == "corrections":
        rows = corrections()
        if not rows:
            print("no human decisions recorded yet")
            return 0
        overruled = sum(1 for r in rows if r["overruled"])
        print(f"{len(rows)} human decisions, {overruled} overruled the model\n")
        for row in rows[:25]:
            mark = "OVERRULED" if row["overruled"] else "agreed   "
            print(f"  {mark}  {row['reference']:<16} model {row['model_said']:<10} "
                  f"({row['model_confidence']:.2f})  human {row['human_said']}")
        return 0

    graph = build_graph(OllamaClient(args.model))

    if args.command == "review":
        _print_result(args.candidate_ref, _run_one(graph, args.candidate_ref, args.answer))
        return 0

    records = candidates_for_board(args.board)
    if not records:
        print(f"no board {args.board!r} in the store", file=sys.stderr)
        return 1
    if args.limit:
        records = records[: args.limit]

    print(f"board {args.board}: {len(records)} AOI candidates\n")
    for record in records:
        _print_result(
            record["reference"], _run_one(graph, record["reference"], args.answer)
        )
    return 0
