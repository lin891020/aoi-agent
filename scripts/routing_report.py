"""How much of the queue never reaches the LLM?

The flow only consults a language model for candidates the vision model cannot
settle. That share is an architectural result worth measuring on its own: it
sets the cost of running the station and it does not depend on how fast the
model happens to be on any given day.

Measured directly from the stored candidates, so it needs no GPU.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import select  # noqa: E402

from aoi_agent.graph.flow import CONFIDENT  # noqa: E402
from aoi_agent.store.boards import session_factory  # noqa: E402
from aoi_agent.store.models import Board, CandidateRecord  # noqa: E402
from aoi_agent.vision.inference import DEFAULT_DISMISS_THRESHOLD  # noqa: E402


def route(predicted_class: str, confidence: float, false_call_probability: float) -> str:
    """Mirror of ``flow.route_after_classify``, over stored predictions."""
    if false_call_probability >= DEFAULT_DISMISS_THRESHOLD:
        return "dismiss"
    if confidence >= CONFIDENT and predicted_class not in ("open", "false_call"):
        return "confirm"
    return "investigate"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    args = parser.parse_args()

    with session_factory()() as session:
        rows = session.execute(
            select(
                CandidateRecord.predicted_class,
                CandidateRecord.confidence,
                CandidateRecord.false_call_probability,
                CandidateRecord.ground_truth,
            ).join(Board)
        ).all()

    if not rows:
        print("the store is empty; run scripts/seed_store.py", file=sys.stderr)
        return 1

    routes = Counter()
    escapes_by_route = Counter()
    for predicted, confidence, false_call, truth in rows:
        decision = route(predicted, confidence, false_call)
        routes[decision] += 1
        if decision == "dismiss" and truth not in ("false_call", "fragment"):
            escapes_by_route["dismiss"] += 1

    total = len(rows)
    llm_free = routes["dismiss"] + routes["confirm"]

    lines = [
        "### Routing — how much of the queue reaches the LLM",
        "",
        f"Measured over {total} stored candidates from 500 boards.",
        "",
        "| path | candidates | share | LLM involved |",
        "|---|---|---|---|",
        f"| dismissed by the vision model | {routes['dismiss']} | {routes['dismiss'] / total:.1%} | no |",
        f"| confirmed by the vision model | {routes['confirm']} | {routes['confirm'] / total:.1%} | no |",
        f"| investigated | {routes['investigate']} | {routes['investigate'] / total:.1%} | yes |",
        "",
        f"**{llm_free / total:.1%} of candidates never reach a language model.** They are "
        "dispositioned by the vision model in tens of milliseconds. The LLM is spent "
        "only on the fraction that is genuinely ambiguous, which is what makes a 20B "
        "model affordable at line rate.",
        "",
        f"Escapes on the dismissal path: {escapes_by_route['dismiss']} "
        f"({escapes_by_route['dismiss'] / max(routes['dismiss'], 1):.2%} of dismissals).",
        "",
        "`open` is routed to investigation regardless of confidence, so it never",
        "appears on the confirm path -- WI-201 calls it the class hardest to separate",
        "from a registration artefact.",
        "",
    ]

    report = "\n".join(lines)
    print(report)
    existing = args.out.read_text() if args.out.exists() else "# Benchmarks\n"
    args.out.write_text(existing.rstrip() + "\n\n" + report + "\n")
    print(f"appended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
