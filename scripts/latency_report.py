"""How long does the reason node take, and does it fit the deadline it runs under?

Two numbers, and until 2026-08-23 this script conflated them because the code
did. WI-300's **response budget** is 10 seconds and covers the *verdict* -- the
disposition that holds or releases a part. The verdict is `classify_node`'s,
measured at 2.5ms per candidate, so the budget is not in question and a
reason-node service time is not the thing to compare against it. What the reason
node runs under is the **explanation deadline**: the client's own bound on
waiting for prose nothing blocks on. That is what this script reports against.

The deadline is where the answer to "is this model the right size" now lives.
`gpt-oss:20b` misses the response budget only if you ask it to produce a verdict,
which it does not.

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
* **The run uses the deadline the station uses.** It used to override it to
  180s, which measured a configuration nothing runs. A call that exceeds the
  deadline is now counted as a call that produced no explanation, which is what
  it is in production, and the distribution below is reported as censored at
  that point rather than quietly completed past it.

Real prompts only. The candidates are drawn from the store and pushed through
the actual flow, so the prompt is the one the reason node builds rather than a
synthetic one of the same rough length. The graph reads from the store and
writes nothing, and the checkpointer is in-memory, so a benchmark run leaves no
escalations and no decisions behind.

    uv run python scripts/latency_report.py --candidates 20
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from sqlalchemy import select  # noqa: E402

from aoi_agent.i18n import line_language  # noqa: E402
from aoi_agent.graph.flow import (  # noqa: E402
    CONFIDENT,
    DEFAULT_MODEL,
    RESPONSE_BUDGET_S,
    build_graph,
)
from aoi_agent.llm.ollama import (  # noqa: E402
    EXPLANATION_DEADLINE_S,
    ChatResult,
    OllamaClient,
)
from aoi_agent.store.boards import session_factory  # noqa: E402
from aoi_agent.store.models import Board, CandidateRecord  # noqa: E402
from aoi_agent.vision.inference import DEFAULT_DISMISS_THRESHOLD  # noqa: E402

# The contention sweep lives with the re-verifier benchmark and is imported
# rather than copied. `ollama ps` alone is not a machine check: it reports
# Ollama's own resident models and is blind to a torch job on the same silicon
# or four ffmpeg transcodes on the same cores, both of which have already cost
# this project a published number.
from reverifier_latency import (  # noqa: E402
    competing_processes,
    process_table,
    resident_models,
)


THERMAL_SPLIT_S = 60.0


class RecordingClient:
    """Delegates to a real client and keeps every ``Timing``.

    Failures are recorded too, and separately. A call that hits the deadline
    produces no ``Timing`` at all, so summarising only what came back would
    report the surviving calls as the distribution and hide the censoring -- the
    exact shape of the defect this script was rewritten for.
    """

    def __init__(self, inner: OllamaClient):
        self.inner = inner
        self.calls: list[tuple[float, ChatResult]] = []
        self.failures: list[str] = []
        self.started = time.perf_counter()

    def chat(self, messages, **kwargs) -> ChatResult:
        offset = time.perf_counter() - self.started
        try:
            result = self.inner.chat(messages, **kwargs)
        except Exception as error:
            self.failures.append(type(error).__name__)
            raise
        self.calls.append((offset, result))
        return result


def ollama_ps() -> str:
    try:
        return subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"(ollama ps unavailable: {error})"


def contention(model: str) -> list[str]:
    """Everything sharing this machine that is not the model being measured.

    A resident model other than the one under test means Ollama is about to
    evict something; a busy torch or ffmpeg process means the silicon is shared
    whatever `ollama ps` says.
    """
    others = [
        name for name in resident_models(ollama_ps())
        if not name.startswith(model.split(":")[0])
    ]
    busy = [
        line for line in competing_processes(process_table(), os.getpid())
        if not is_ollama_runner(line)
    ]
    return [f"resident model: {name}" for name in others] + busy


#: The path Ollama launches its own model runner under. That process *is* the
#: model being measured -- it was seen at 6% CPU in the sweep taken right
#: after a run, still releasing the last call's buffers, and two English
#: measurements on 2026-08-30 were refused for it. A runner serving some
#: *other* model is the residency check's job, one line above, and is still
#: refused there.
OLLAMA_RUNNER_MARKER = "/ollama/llama-server"


def is_ollama_runner(process_line: str) -> bool:
    """Is this sweep line Ollama's own runner, rather than a competitor?"""
    return OLLAMA_RUNNER_MARKER in process_line or " ollama runner " in process_line


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

    busy_before = contention(args.model)
    print("busy processes before:\n  " + ("\n  ".join(busy_before) or "(none)") + "\n")
    if busy_before and not args.dry_run:
        print(
            "something else is holding this machine; a contended run is to be "
            "discarded, not published. Re-run when it is quiet, or use --dry-run.",
            file=sys.stderr,
        )
        return 1

    inner = OllamaClient(args.model, timeout=EXPLANATION_DEADLINE_S)
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
    busy_after = contention(args.model)
    print(f"\nollama ps after:\n{ps_after}")
    print("busy processes after:\n  " + ("\n  ".join(busy_after) or "(none)"))
    if busy_after and not args.dry_run:
        print(
            "something joined this machine mid-run; the numbers are not this "
            "machine idle and are not published.",
            file=sys.stderr,
        )
        return 1

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
    over_deadline = len(client.failures)
    over_budget = sum(1 for ms in every if ms > RESPONSE_BUDGET_S * 1000)
    queued_share = statistics.fmean(queueing) / statistics.fmean(
        [r.timing.wall_ms for _, r in kept]
    )

    lines = [
        "",
        "### Agent-layer latency — does the reason node fit the explanation deadline?",
        "",
        f"`{args.model}` at `think=\"low\"`, {len(kept)} real reason-node calls "
        f"over candidates the router sends to the LLM. The deadline is "
        f"`EXPLANATION_DEADLINE_S`, {EXPLANATION_DEADLINE_S:.0f}s, and the run "
        f"used it rather than overriding it — a call that misses it here is a "
        f"call that produces no explanation in production. Explanations were "
        f"written in `{line_language()}` (`AOI_LINE_LANGUAGE`); a figure "
        f"taken in one language says nothing about the other, since the same "
        f"content is more tokens in Chinese than in English.",
        "",
        f"**This is not WI-300's {RESPONSE_BUDGET_S:.0f}s response budget, and "
        f"comparing it against that budget is the error this script used to "
        f"make.** The budget covers the verdict, which is `classify_node`'s at "
        f"2.5ms per candidate. The LLM writes the operator's explanation and "
        f"dispositions nothing, so what bounds it is a resource limit, not a "
        f"promise.",
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
        "busy processes before the run",
        "\n".join(busy_before) or "(none)",
        "",
        "ollama ps after the run",
        ps_after,
        "",
        "busy processes after the run",
        "\n".join(busy_after) or "(none)",
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

    attempted = len(every) + over_deadline
    verdict = (
        f"**Inside the deadline.** p90 is {overall['p90'] / 1000:.1f}s against "
        f"{EXPLANATION_DEADLINE_S:.0f}s, and {over_deadline} of {attempted} "
        f"calls produced no explanation."
        if over_deadline == 0
        else f"**Censored.** {over_deadline} of {attempted} calls hit the "
             f"{EXPLANATION_DEADLINE_S:.0f}s deadline and wrote no explanation, "
             f"so every figure in the table above is the distribution of the "
             f"calls that survived. p90 of those is "
             f"{overall['p90'] / 1000:.1f}s. A deadline this model routinely "
             f"misses is the signal WI-300 means: change the model, not the "
             f"number."
    )
    against_budget = (
        f"Against WI-300's {RESPONSE_BUDGET_S:.0f}s response budget, for "
        f"reference and not as the verdict: {over_budget} of {len(every)} "
        f"explanations took longer than the budget allows a *verdict* to take. "
        f"No verdict waited on any of them — `classify_node` had already "
        f"produced the disposition before the reason node was entered."
    )

    lines += [
        "",
        verdict,
        "",
        against_budget,
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
