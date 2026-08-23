---
name: measuring-llm-latency
description: Use when reporting, comparing, or reasoning about any agent-layer timing number in this repo — LLM latency, tokens/sec, tool-call round trips, graph step duration, or "is this fast enough for the line" claims.
---

# Measuring LLM Latency on This Machine

## Overview

On this machine a latency number is guilty until proven innocent. Two effects
corrupt it silently, and both produce plausible-looking figures:

- **Model eviction.** A neighbouring job (`~/Projects/video_transfer`) periodically
  holds `gpt-oss:20b` and 12GB of GPU. Ollama evicts and reloads models between
  requests, so almost all of the measured wall time is load time. This already
  produced a bogus "85 s cold / 145 s warm for a 132-token tool call" — see the
  module docstring in `src/aoi_agent/llm/ollama.py`.
- **Thermal throttling.** The M5 Air is fanless. Sustained load degrades
  throughput partway through a run.

Neither one raises an error. You only catch them by checking.

There is now a third thing to know, and this one *does* raise an error:
`EXPLANATION_DEADLINE_S` caps every call at 60 s. Under contention a call no
longer produces a slow-but-usable number — it times out, and the region is
dispositioned with no written explanation. **A timeout under contention is
correct behaviour, not a failed measurement.** Do not raise the deadline to make
a benchmark complete: it is sized from the measured distribution, so raising it
to fit a run is the run rewriting its own ruler.

That constant is **not** WI-300's response budget, and it was until 2026-08-23.
Two names now, because they are two things:

| | what it is | may it follow the model? |
|---|---|---|
| `RESPONSE_BUDGET_S`, 10 s, `graph/flow.py` | WI-300's promise about when a *verdict* reaches the record. Met by `classify_node` at 2.5 ms | **no** — WI-300 forbids it |
| `EXPLANATION_DEADLINE_S`, 60 s, `llm/ollama.py` | the httpx client's bound on waiting for prose nobody blocks on | yes — it is sized from measurement |

Quoting `gpt-oss:20b`'s 12.5 s median against "the 10 s budget" is now the wrong
comparison and was always the wrong conclusion: the budget is about a verdict
the LLM does not produce. Compare a reason-node service time against
`EXPLANATION_DEADLINE_S`.

## Procedure

1. **`ollama ps` before the run.** Record the output verbatim. If a model you
   are not benchmarking is resident, stop — you are measuring contention, not
   inference. Wait for `UNTIL` to expire or accept that the run is invalid.
2. **Warm up and discard.** Call `OllamaClient.warm_up()` first and throw the
   result away. Benchmarking the first call benchmarks the loader.
3. **Group by model.** Never interleave requests across models. `KEEP_ALIVE` is
   30m and set explicitly on every request; interleaving defeats it.
4. **Report service time, not `eval_ms`.** Service time is `total_duration`
   less `load_duration`: prompt ingestion plus thinking plus generation, minus
   the one component production never pays, because `KEEP_ALIVE` holds the model
   resident. `scripts/latency_report.py` computes it. Report `eval_ms` alongside,
   never as the headline — see the reasoning token trap below.
5. **Check `Timing.was_reloaded` is False.** Use the codebase's own test —
   `load_ms > Timing.RELOAD_MS` in `src/aoi_agent/llm/ollama.py` — rather than
   inventing a threshold here. True means the model was evicted mid-run; discard
   that measurement. Note the threshold is not zero: measured, a warm resident
   `gpt-oss:20b` reports a steady ~168ms of `load_duration`, so a gate at zero
   discards every healthy request and the run silently reports nothing.
6. **Split thermal phases.** Report "first 60s" and "steady state" as separate
   numbers. A single mean hides the throttle.
7. **`ollama ps` after the run.** If `NAME` or `UNTIL` changed unexpectedly,
   something else touched the GPU. The run is invalid.
8. **Append to `docs/benchmarks.md`**, newest last, with the `ollama ps` output
   included as evidence.

## Quick Reference

| Field | Meaning | Use it as latency? |
|---|---|---|
| `total_duration` − `load_duration` | service time | **yes — this is the number** |
| `eval_ms` | *visible* token generation | no — excludes thinking, see below |
| `prompt_eval_ms` | prompt ingestion | yes, reported separately |
| `load_ms` | model load | no — read it through `was_reloaded`, then discard the run |
| `wall_ms` | client-side round trip | no — but compare against `total_duration` to detect queueing |
| `total_duration` alone | Ollama's own total | no — still includes load |

## The reasoning token trap

`eval_duration` does not account for a reasoning model's thinking tokens.
Measured on `gpt-oss:20b`, the same request twice:

| | `total_duration` | `eval_ms` | unaccounted |
|---|---|---|---|
| `think="low"` | 10722ms | 4671ms | **5435ms** |
| `think=False` | 3254ms | 1921ms | 8ms |

The gap is thinking, and it closes to nothing when thinking is off. The flow's
reason node runs at `think="low"`, so `eval_ms` can understate what the station
waits for by around half — enough to move a call from one side of any deadline
to the other. That is not hypothetical: it is what the first run of
`scripts/latency_report.py` concluded before the method was corrected. (The
thinking share is not stable: the 2026-08-23 run put it at 13% of service time,
against 47% on 2026-08-22. Measure it, do not assume a factor.)

Any model emitting reasoning tokens has this. Check `len(result.thinking)`
before trusting `eval_ms` as a whole-request figure.

## Red Flags — the number is invalid

- No `ollama ps` output recorded alongside it
- `Timing.was_reloaded` True on a steady-state measurement
- Reported as a single mean with no first-60s / steady-state split
- `eval_ms` quoted as the latency of a request that produced thinking tokens
- Raw `wall_ms` or raw `total_duration` quoted as "latency"
- Two models benchmarked in an interleaved loop
- Wall time exceeds `total_duration` by more than a few percent — that gap is
  the contention tell. The `wall_ms`/`eval_ms` ratio is **not**: it sits near 2x
  on a perfectly healthy reasoning-model run.
- A raised `EXPLANATION_DEADLINE_S`, or a per-call `timeout=` override, used to
  let a slow benchmark finish
- A reason-node latency compared against `RESPONSE_BUDGET_S`. That budget covers
  the verdict, which is the classifier's, not the LLM's

## Common Mistakes

- **Re-running to "check" a suspicious number.** The second run reloads the
  model and looks better for the wrong reason. Check `ollama ps` instead.
- **Treating `wall_ms` as user-visible latency.** It is, for a single user on an
  idle box — but this box is not idle, so it is not a property of the model.
- **Benchmarking during a training run.** `scripts/train.py` holds MPS.

## What This Skill Reads From the Code

This skill states facts that live in `src/aoi_agent/llm/ollama.py`, and one that
lives in `src/aoi_agent/graph/flow.py`. If any of them change, this file is
wrong until someone updates it — and nothing will
raise an error. `scripts/check_skill_freshness.py` asserts every row below.

| Symbol | Stated here as |
|---|---|
| `KEEP_ALIVE` | `"30m"` |
| `EXPLANATION_DEADLINE_S` | `60.0` |
| `Timing.was_reloaded` | exists; tests `load_ms > Timing.RELOAD_MS` |
| `Timing.RELOAD_MS` | `2000.0` |
| `Timing.eval_ms` | exists |
| `Timing.load_ms` | exists |
| `Timing.wall_ms` | exists |
| `OllamaClient.warm_up` | exists |
| `OllamaClient.resident_models` | exists |
| `RESPONSE_BUDGET_S` (in `graph/flow.py`) | `10.0`, and absent from `llm/ollama.py` |

This skill is a *reference*, not a discipline rule. Its failure mode is going
stale, not being argued around — so it is verified by the freshness check, not
by pressure-testing an agent against it.
