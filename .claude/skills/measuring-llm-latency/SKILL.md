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
`RESPONSE_BUDGET_S` caps every call at 10 s (it was 600 s). Under contention a
call no longer produces a slow-but-usable number — it times out and the region
escalates. **A timeout under contention is correct behaviour, not a failed
measurement.** Do not raise the budget to make a benchmark complete; WI-300 says
to change the model, not the budget.

## Procedure

1. **`ollama ps` before the run.** Record the output verbatim. If a model you
   are not benchmarking is resident, stop — you are measuring contention, not
   inference. Wait for `UNTIL` to expire or accept that the run is invalid.
2. **Warm up and discard.** Call `OllamaClient.warm_up()` first and throw the
   result away. Benchmarking the first call benchmarks the loader.
3. **Group by model.** Never interleave requests across models. `KEEP_ALIVE` is
   30m and set explicitly on every request; interleaving defeats it.
4. **Report `eval_ms`.** From `Timing` in `src/aoi_agent/llm/ollama.py`.
5. **Check `Timing.was_reloaded` is False.** Use the codebase's own test —
   `load_ms > 100` in `src/aoi_agent/llm/ollama.py` — rather than inventing a
   threshold here. True means the model was evicted mid-run; discard that
   measurement.
6. **Split thermal phases.** Report "first 60s" and "steady state" as separate
   numbers. A single mean hides the throttle.
7. **`ollama ps` after the run.** If `NAME` or `UNTIL` changed unexpectedly,
   something else touched the GPU. The run is invalid.
8. **Append to `docs/benchmarks.md`**, newest last, with the `ollama ps` output
   included as evidence.

## Quick Reference

| Field | Meaning | Use it as latency? |
|---|---|---|
| `eval_ms` | token generation | **yes — this is the number** |
| `prompt_eval_ms` | prompt ingestion | yes, reported separately |
| `load_ms` | model load | no — read it through `was_reloaded` (`> 100`), then discard the run |
| `wall_ms` | client-side round trip | no — includes queueing and load |
| `total_duration` (raw) | Ollama's own total | never |

## Red Flags — the number is invalid

- No `ollama ps` output recorded alongside it
- `Timing.was_reloaded` True on a steady-state measurement
- Reported as a single mean with no first-60s / steady-state split
- `wall_ms` or `total_duration` quoted as "latency"
- Two models benchmarked in an interleaved loop
- Wall time is a large multiple of `eval_ms` — that ratio is the contention tell
- A raised `RESPONSE_BUDGET_S`, or a per-call `timeout=` override, used to let a
  slow benchmark finish

## Common Mistakes

- **Re-running to "check" a suspicious number.** The second run reloads the
  model and looks better for the wrong reason. Check `ollama ps` instead.
- **Treating `wall_ms` as user-visible latency.** It is, for a single user on an
  idle box — but this box is not idle, so it is not a property of the model.
- **Benchmarking during a training run.** `scripts/train.py` holds MPS.

## What This Skill Reads From the Code

This skill states facts that live in `src/aoi_agent/llm/ollama.py`. If any of
them change, this file is wrong until someone updates it — and nothing will
raise an error. `scripts/check_skill_freshness.py` asserts every row below.

| Symbol | Stated here as |
|---|---|
| `KEEP_ALIVE` | `"30m"` |
| `RESPONSE_BUDGET_S` | `10.0` |
| `Timing.was_reloaded` | exists; tests `load_ms > 100` |
| `Timing.eval_ms` | exists |
| `Timing.load_ms` | exists |
| `Timing.wall_ms` | exists |
| `OllamaClient.warm_up` | exists |
| `OllamaClient.resident_models` | exists |

This skill is a *reference*, not a discipline rule. Its failure mode is going
stale, not being argued around — so it is verified by the freshness check, not
by pressure-testing an agent against it.
