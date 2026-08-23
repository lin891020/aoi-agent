"""Command line entry point for the re-verification station.

    uv run python -m aoi_agent review 20085293#2
    uv run python -m aoi_agent board 20085293 --queue
    uv run python -m aoi_agent queue
    uv run python -m aoi_agent corrections
    uv run python -m aoi_agent explanations
    uv run python -m aoi_agent provenance 20085293
    uv run python -m aoi_agent station

``--queue`` is the normal production shape: the line does not stop to ask a
question. Escalations go on the review station's queue and an operator answers
them when they get to it, which is what the durable checkpointer is for. The
interactive prompt stays because it is the fastest way to walk one case end to
end while working on the flow.
"""

from __future__ import annotations

import argparse
import sys

from aoi_agent.graph.flow import DEFAULT_MODEL, build_graph, explanation_notice
from aoi_agent.llm.ollama import OllamaClient
from aoi_agent.provenance import ReviewerIdentity
from aoi_agent.station import service
from aoi_agent.store import dispositions, escalations
from aoi_agent.store.boards import (
    candidates_for_board,
    corrections,
    explanation_status_counts,
    sample_board_stems,
)

DISPOSITION_LABEL = {
    "dismissed": "DISMISSED",
    "defect_confirmed": "DEFECT",
    "escalated": "ESCALATED",
}


def _print_result(reference: str, state: dict) -> None:
    if "already_pending" in state:
        print(f"  {reference:<16} {'QUEUED':<10} already waiting on an operator")
        return

    label = DISPOSITION_LABEL.get(state.get("disposition", ""), "?")
    print(f"  {reference:<16} {label:<10} {state.get('verdict', '?'):<12} "
          f"by {state.get('decided_by', '?')}")
    if state.get("agent_rationale"):
        print(f"      {state['agent_rationale']}")
    elif explanation_notice(state.get("explanation_status", "")):
        print(f"      {explanation_notice(state['explanation_status'])}")
    timings = state.get("timings_ms", {})
    if timings:
        parts = " ".join(f"{k} {v:.0f}ms" for k, v in timings.items())
        print(f"      path: {' -> '.join(state.get('trace', []))}")
        print(f"      {parts}")


def _run_one(graph, reference: str, auto_answer: str | None, to_queue: bool) -> dict:
    state = service.start_review(graph, reference)

    if "already_pending" in state:
        return state

    if "__interrupt__" not in state:
        return state

    payload = state["__interrupt__"][0].value
    print(f"\n  escalated: {reference}")
    print(f"    reason: {payload['reason']}")
    print(f"    model said {payload['model_class']} ({payload['model_confidence']:.3f})")

    if to_queue:
        print("    left on the review station's queue")
        return {"already_pending": escalations.get(service.thread_for(reference))}

    verdict = (
        auto_answer if auto_answer
        else input(f"    your verdict {payload['options']}: ").strip()
    )
    # The host account, not "operator" and not "auto". Whether the verdict was
    # typed at the prompt or passed on the command line does not change who is
    # answerable for it, and a literal that used to go into this field named
    # nobody at all -- which is the whole reason ``reviewer`` was worthless
    # until the column beside it existed. It is a weaker claim than the
    # station's signed-in operator and it is recorded as a different word.
    return service.resume_review(
        graph, reference, verdict, ReviewerIdentity.host_account()
    )


def _cmd_queue() -> int:
    rows = escalations.pending()
    if rows:
        print(f"{len(rows)} regions waiting on a person\n")
        for row in rows:
            print(f"  {row['reference']:<16} model {row['model_class']:<10} "
                  f"({row['model_confidence']:.2f})  {row['reason'][:70]}")
    else:
        print("queue is empty")

    # Closed entries that no human decision backs. Not part of the queue and
    # not work for anyone; printed here because this is where a person looks at
    # the queue's health, and a disagreement between two tables that nobody
    # asks about is a disagreement that stays.
    orphans = escalations.unattributed_resolutions()
    if orphans:
        print(f"\n  {len(orphans)} closed entries carry no human decision -- "
              "see scripts/mark_unattributed_resolutions.py")
        for row in orphans:
            print(f"    {row['reference']:<16} {row['status']}")
    return 0


def _cmd_explanations() -> int:
    """How many automated dispositions carry no written rationale, and why.

    WI-300 requires this to be answerable, and until 2026-08-23 it was not: an
    explained region and an unexplained one were the same row in every table,
    and the only trace of the difference was a ``ReadTimeout`` sitting where a
    rationale belonged. The LLM's only remaining job is writing these, so the
    rate at which that job fails is the health of the layer.
    """
    decisions = explanation_status_counts()
    queued = escalations.explanation_counts()
    if not decisions and not queued:
        print("no automated dispositions recorded yet")
        return 0

    def _report(label: str, counts: dict[str, int]) -> None:
        total = sum(counts.values())
        explained = counts.get("ok", 0)
        print(f"{label}: {total}")
        if not total:
            return
        print(f"  explained          {explained:>6}  {explained / total:6.1%}")
        for status, count in sorted(counts.items()):
            if status == "ok":
                continue
            print(f"  {status:<18} {count:>6}  {count / total:6.1%}")

    _report("automated dispositions", decisions)
    print()
    _report("waiting on a person", queued)
    return 0


def _cmd_provenance(stem: str) -> int:
    """Who decided this board was fine, when, and on what basis.

    The auditor's question, answerable from a terminal. It was not answerable
    at all before 2026-08-23: the store held 9,140 decisions and not one of
    them named the weights, the thresholds or the code behind it, and nothing
    anywhere recorded what had happened to a *board*.
    """
    rows = dispositions.decision_provenance(stem)
    if not rows:
        print(f"no board {stem!r} in the store", file=sys.stderr)
        return 1

    board_rows = dispositions.history(stem)
    print(f"board {stem}")
    if board_rows:
        for row in board_rows:
            print(f"  {row['disposition'].upper():<9} "
                  f"{(row['decided_at'] or '')[:19].replace('T', ' ')} UTC  "
                  f"by {row['decided_by']}")
            print(f"      {row['basis']}")
            print(f"      model {row['model_digest']}  code {row['code_version']}")
            print(f"      thresholds {row['thresholds_json']}")
    else:
        current = dispositions.assess(stem)
        print(f"  no board-level disposition recorded yet -- "
              f"{current['basis']}")

    print(f"\n  {len(rows)} flagged regions\n")
    for row in rows:
        verdict = row["verdict"] or "-"
        # The name and what stands behind it, together. A reviewer printed
        # alone reads as though somebody was identified, which for every row
        # written before 2026-08-23 is exactly the thing that was not true.
        by = row["reviewer"] or row["source"] or "-"
        if row["reviewer"] and row["reviewer_auth"]:
            by = f"{row['reviewer']}/{row['reviewer_auth']}"
        queue = f"  [{row['queue_status']}]" if row["queue_status"] else ""
        print(f"  {row['reference']:<16} {verdict:<12} by {by:<22} "
              f"{row['model_digest'] or '-':<24} {row['code_version'] or '-'}{queue}")

    absences = {
        digest: count
        for digest, count in dispositions.unattributable().items()
        if digest in ("unrecorded", "unavailable", "null")
    }
    if absences:
        print("\n  store-wide, decisions that name no model: "
              + ", ".join(f"{k} {v}" for k, v in sorted(absences.items())))
    return 0


def _cmd_corrections() -> int:
    rows = corrections()
    if not rows:
        print("no human decisions recorded yet")
        return 0
    overruled = sum(1 for r in rows if r["overruled"])
    print(f"{len(rows)} human decisions, {overruled} overruled the model\n")
    for row in rows[:25]:
        mark = "OVERRULED" if row["overruled"] else "agreed   "
        print(f"  {mark}  {row['reference']:<16} model {row['model_said']:<10} "
              f"({row['model_confidence']:.2f})  human {row['human_said']:<12} "
              f"{row['reviewer'] or '-'}/{row['attribution'] or 'unrecorded'}")

    # Grouped, because "how many of these can the next training round stand
    # behind" is the question the whole column exists for, and it is not
    # answerable from a list of twenty-five.
    by_method: dict[str, int] = {}
    for row in rows:
        by_method[row["attribution"] or "unrecorded"] = (
            by_method.get(row["attribution"] or "unrecorded", 0) + 1
        )
    print("\n  attribution: "
          + ", ".join(f"{k} {v}" for k, v in sorted(by_method.items())))
    return 0


def _cmd_station(host: str, port: int) -> int:
    import uvicorn

    from aoi_agent.station import auth

    print(f"review station on http://{host}:{port}")
    # Said at startup, because the alternative is a supervisor discovering it
    # at the login form with a queue behind it. The station comes up either
    # way: an empty operator file locks it, and locked is the correct state
    # for a station that cannot name whoever would be answering.
    operators = auth.load_operators()
    if operators:
        print(f"  {len(operators)} operator(s) in {auth.operators_path()}")
    else:
        print(f"  no operators in {auth.operators_path()} -- nobody can sign in. "
              "Add one: uv run python scripts/add_operator.py <name>")
    uvicorn.run("aoi_agent.station.app:app", host=host, port=port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aoi_agent", description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    sub = parser.add_subparsers(dest="command", required=True)

    review = sub.add_parser("review", help="re-verify one flagged region")
    review.add_argument("candidate_ref")
    review.add_argument("--answer", help="answer escalations automatically (for demos)")
    review.add_argument("--queue", action="store_true",
                        help="leave escalations for the review station")

    board = sub.add_parser("board", help="re-verify every region on a board")
    board.add_argument("board")
    board.add_argument("--answer", help="answer escalations automatically")
    board.add_argument("--queue", action="store_true",
                       help="leave escalations for the review station")
    board.add_argument("--limit", type=int, default=None)

    sub.add_parser("boards", help="list boards in the store")
    sub.add_parser("queue", help="show what is waiting on a person")
    sub.add_parser("corrections", help="show where operators overruled the model")
    sub.add_parser("explanations",
                   help="how many dispositions carry no written rationale, and why")

    provenance = sub.add_parser(
        "provenance",
        help="who decided this board, when, and under which model",
    )
    provenance.add_argument("board")

    station = sub.add_parser("station", help="serve the review station")
    station.add_argument("--host", default="127.0.0.1")
    station.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.command == "boards":
        for stem in sample_board_stems(20):
            print(stem)
        return 0
    if args.command == "queue":
        return _cmd_queue()
    if args.command == "corrections":
        return _cmd_corrections()
    if args.command == "explanations":
        return _cmd_explanations()
    if args.command == "provenance":
        return _cmd_provenance(args.board)
    if args.command == "station":
        return _cmd_station(args.host, args.port)

    graph = build_graph(OllamaClient(args.model))

    if args.command == "review":
        _print_result(
            args.candidate_ref,
            _run_one(graph, args.candidate_ref, args.answer, args.queue),
        )
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
            record["reference"],
            _run_one(graph, record["reference"], args.answer, args.queue),
        )

    # The regions are decided; now say what happened to the board. A board with
    # regions still on the queue gets no row -- it has not been dispositioned,
    # someone is still looking at it, and recording a verdict on it here would
    # be recording one nobody reached.
    settled = service.settle_board(args.board)
    if settled:
        print(f"\n  board {args.board}: {settled['disposition'].upper()} "
              f"-- {settled['basis']}")
        print(f"  under model {settled['model_digest']}, "
              f"code {settled['code_version']}")
    else:
        print(f"\n  board {args.board}: not dispositioned -- regions are still "
              "waiting on a person")
    return 0
