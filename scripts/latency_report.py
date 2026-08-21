"""Does the agent layer fit inside WI-300's response budget?

The budget is 10 seconds per escalated region. Whether `gpt-oss:20b` can meet
it is the open question that decides whether the model stays -- WI-300 says a
model that cannot meet the budget is the wrong size for the line, and that the
budget is not to be raised to accommodate it.

Method follows the `measuring-llm-latency` procedure:

* `ollama ps` is captured before and after and printed into the report. A
  number with no residency evidence beside it is not evidence.
* The first call is a warm-up and is discarded. Benchmarking the first call
  benchmarks the loader.
* The headline is **service time** -- Ollama's `total_duration` less
  `load_duration`. `eval_ms` is reported beside it but is not the budget number:
  measured here, `eval_duration` excludes a reasoning model's thinking tokens
  entirely. On `gpt-oss:20b` at `think="low"` that hides more than half the
  time the station actually waits (`think=False` closes the gap to 8ms, which
  is how the cause was identified). Load is excluded because the model is
  resident in production and no operator pays for it.
* Any request with `load_ms > 100` was served after an eviction and is dropped.
* First 60 seconds and steady state are reported separately, because the M5 Air
  is fanless and a single mean hides the throttle.

Real prompts only. The candidates are drawn from the store and pushed through
the actual flow, so the prompt is the one the reason node builds rather than a
synthetic one of the same rough length. The graph reads from the store and
writes nothing, and the checkpointer is in-memory, so a benchmark run leaves no
escalations and no decisions behind.

    uv run python scripts/latency_report.py --candidates 20
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from sqlalchemy import select  # noqa: E402

from aoi_agent.graph.flow import CONFIDENT, DEFAULT_MODEL, build_graph  # noqa: E402
from aoi_agent.llm.ollama import RESPONSE_BUDGET_S, ChatResult, OllamaClient  # noqa: E402
from aoi_agent.store.boards import session_factory  # noqa: E402
from aoi_agent.store.models import Board, CandidateRecord  # noqa: E402
from aoi_agent.vision.inference import DEFAULT_DISMISS_THRESHOLD  # noqa: E402

#: Long enough to observe an over-budget call instead of cutting it short. The
#: point is to find out how long the model takes, which a 10s client timeout
#: would hide behind a ReadTimeout.
BENCH_TIMEOUT_S = 180.0

THERMAL_SPLIT_S = 60.0


class RecordingClient:
    """Delegates to a real client and keeps every ``Timing``."""

    def __init__(self, inner: OllamaClient):
        self.inner = inner
        self.calls: list[tuple[float, ChatResult]] = []
        self.started = time.perf_counter()

    def chat(self, messages, **kwargs) -> ChatResult:
        offset = time.perf_counter() - self.started
        result = self.inner.chat(messages, **kwargs)
        self.calls.append((offset, result))
        return result


def ollama_ps() -> str:
    try:
        return subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"(ollama ps unavailable: {error})"


def investigated_candidates(limit: int) -> list[str]:
    """References that `route_after_classify` sends to the LLM."""
    with session_factory()() as session:
        rows = session.execute(
            select(
                Board.stem,
                CandidateRecord.index_on_board,
                CandidateRecord.predicted_class,
                CandidateRecord.confidence,
                CandidateRecord.false_call_probability,
            ).join(CandidateRecord, CandidateRecord.board_id == Board.id)
        ).all()

    picked = []
    for stem, index, klass, confidence, false_call in rows:
        if false_call >= DEFAULT_DISMISS_THRESHOLD:
            continue
        if confidence >= CONFIDENT and klass not in ("open", "false_call"):
            continue
        picked.append(f"{stem}#{index}")
        if len(picked) >= limit:
            break
    return picked


def summarise(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p90": ordered[max(0, round(0.9 * len(ordered)) - 1)],
        "max": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument("--dry-run", action="store_true", help="print, do not append")
    args = parser.parse_args()

    references = investigated_candidates(args.candidates)
    if not references:
        print("no investigated candidates in the store", file=sys.stderr)
        return 1

    ps_before = ollama_ps()
    print(f"ollama ps before:\n{ps_before}\n")

    inner = OllamaClient(args.model, timeout=BENCH_TIMEOUT_S)
    print("warming up (discarded) ...")
    warm = inner.warm_up()
    print(f"  load {warm.load_ms:.0f}ms, eval {warm.eval_ms:.0f}ms\n")

    client = RecordingClient(inner)
    graph = build_graph(client, InMemorySaver())

    for position, reference in enumerate(references, 1):
        started = time.perf_counter()
        graph.invoke(
            {"candidate_ref": reference, "trace": [], "timings_ms": {}},
            config={"configurable": {"thread_id": f"bench-{reference}"}},
        )
        elapsed = (time.perf_counter() - started) * 1000
        print(f"  [{position:>3}/{len(references)}] {reference:<16} {elapsed:>8.0f}ms wall")

    ps_after = ollama_ps()
    print(f"\nollama ps after:\n{ps_after}")

    kept = [(offset, r) for offset, r in client.calls if not r.timing.was_reloaded]
    evicted = len(client.calls) - len(kept)

    def service_ms(result) -> float:
        """What the station waits for: everything but loading the model.

        `eval_duration` is not it. Ollama does not attribute a reasoning model's
        thinking tokens to `eval_duration`, so on `think="low"` it reports under
        half the real figure. `total_duration` covers the thinking; subtracting
        `load_duration` removes the one component production never pays,
        because `keep_alive` holds the model resident.
        """
        body = result.raw
        return (body.get("total_duration", 0) - body.get("load_duration", 0)) / 1e6

    early = [service_ms(r) for offset, r in kept if offset < THERMAL_SPLIT_S]
    late = [service_ms(r) for offset, r in kept if offset >= THERMAL_SPLIT_S]
    every = [service_ms(r) for _, r in kept]
    prompts = [r.timing.prompt_eval_ms for _, r in kept]
    evals = [r.timing.eval_ms for _, r in kept]
    hidden = [service_ms(r) - r.timing.eval_ms - r.timing.prompt_eval_ms
              for _, r in kept]

    # The contention test compares the client's round trip against Ollama's own
    # total_duration. Everything Ollama reports is work it did; anything beyond
    # it is time the request spent queued or in transit. Comparing wall against
    # eval instead flags a healthy run, because eval excludes both prompt
    # ingestion and thinking.
    queueing = [
        r.timing.wall_ms - r.raw.get("total_duration", 0) / 1e6 for _, r in kept
    ]

    if not every:
        print("every request was served after an eviction; the run is invalid",
              file=sys.stderr)
        return 1

    overall = summarise(every)
    over_budget = sum(1 for ms in every if ms > RESPONSE_BUDGET_S * 1000)
    queued_share = statistics.fmean(queueing) / statistics.fmean(
        [r.timing.wall_ms for _, r in kept]
    )

    lines = [
        "",
        "### Agent-layer latency — does the reason node fit the response budget?",
        "",
        f"`{args.model}` at `think=\"low\"`, {len(kept)} real reason-node calls "
        f"over candidates the router sends to the LLM. Budget is WI-300's "
        f"{RESPONSE_BUDGET_S:.0f}s.",
        "",
        "Latency here is **service time**: Ollama's `total_duration` less "
        "`load_duration`. It is not `eval_ms`. Measured on this model, "
        "`eval_duration` does not account for thinking tokens at all, and "
        "reports under half the time the station waits.",
        "",
        "```",
        "ollama ps before the run",
        ps_before,
        "",
        "ollama ps after the run",
        ps_after,
        "```",
        "",
        "| | calls | median | mean | p90 | max |",
        "|---|---|---|---|---|---|",
    ]

    for label, values in (("first 60s", early), ("steady state", late), ("all", every)):
        if not values:
            lines.append(f"| {label} | 0 | — | — | — | — |")
            continue
        s = summarise(values)
        lines.append(
            f"| {label} | {s['n']} | {s['median'] / 1000:.1f}s | "
            f"{s['mean'] / 1000:.1f}s | {s['p90'] / 1000:.1f}s | {s['max'] / 1000:.1f}s |"
        )

    verdict = (
        f"**Within budget.** p90 is {overall['p90'] / 1000:.1f}s against a "
        f"{RESPONSE_BUDGET_S:.0f}s budget."
        if overall["p90"] <= RESPONSE_BUDGET_S * 1000
        else f"**Over budget.** p90 is {overall['p90'] / 1000:.1f}s against a "
             f"{RESPONSE_BUDGET_S:.0f}s budget, and {over_budget} of {len(every)} "
             f"calls exceeded it. WI-300 says the model is the wrong size for the "
             f"line, not that the budget moves."
    )

    lines += [
        "",
        verdict,
        "",
        f"Of that service time, `eval_duration` accounts for "
        f"{statistics.fmean(evals) / 1000:.1f}s and prompt ingestion for "
        f"{statistics.fmean(prompts) / 1000:.1f}s on average. The remaining "
        f"{statistics.fmean(hidden) / 1000:.1f}s is thinking tokens, which Ollama "
        f"generates and bills to nobody. Reporting `eval_ms` as the latency would "
        f"have understated this run by {statistics.fmean(hidden) / statistics.fmean(every):.0%}.",
        "",
        f"Queueing check: {queued_share:.1%} of mean wall time is not load, prompt "
        f"or generation"
        + (
            " — the request went straight to the GPU, so the run is not contended."
            if queued_share < 0.15
            else " — that is time spent waiting for the GPU. Something else held it;"
                 " treat these numbers as invalid and check `ollama ps`."
        ),
        "",
        f"{evicted} request(s) dropped for `load_ms > 100` (model evicted mid-run)."
        if evicted
        else "No request was served after an eviction.",
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
