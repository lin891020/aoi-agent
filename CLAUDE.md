# AOI-Agent

A PCB AOI re-verification system. Production optical inspection is tuned for
recall and over-flags; every flagged region goes to a human today. This puts a
vision model in front of that queue and an agent behind it for the cases the
model cannot settle.

Started 2026-08-22. Portfolio project aimed at smart-manufacturing AI roles, so
the engineering has to hold up to an interviewer's questions, not just run.

Over the 60-line cap in ~/Projects/CLAUDE.md on purpose (Mike, 2026-09-25): the
rule titles stay here so every session sees them; the rest moved out verbatim.

## Where things are

- [docs/architecture.md](docs/architecture.md) -- layers, thresholds, and the source layout (end of file)
- [docs/commands.md](docs/commands.md) -- every script and what it measures
- [docs/invariants.md](docs/invariants.md) -- the rules below in full, parsed by `scripts/invariant_audit.py`
- [docs/still-open.md](docs/still-open.md) -- open questions and the dated log of what closed them
- [docs/benchmarks.md](docs/benchmarks.md) -- every measurement, append-only, newest last
- [docs/experiments/](docs/experiments/README.md) -- the index of every experiment and its result

Invoke the project skills; do not just read past them. `retraining-the-reverifier`
before touching the training pipeline or regenerating a benchmark, and
`measuring-llm-latency` before quoting any timing number. Both encode a failure
that is silent -- a stale threshold, a contended GPU -- and neither shows up as
an error.

```bash
uv run pytest                                    # no GPU needed, no model called
uv run python -m aoi_agent station               # review station on :8110
uv run python scripts/invariant_audit.py         # which rules below would fail a test if broken
```

## The twenty rules -- do not quietly change these

Argued, measured and dated in [docs/invariants.md](docs/invariants.md) -- read the entry first.

- Report an operating-point curve, never bare accuracy
- The dismissal threshold is never chosen on the split it is reported against
- A published figure names the run it came from
- The LLM explains; it does not decide
- On the disposition path, every failure escalates to a human, except the LLM's own
- On the disposition path, an escalation must outlive the process
- The review station never shows `ground_truth`
- Free-form SQL reaches the store only through the read-only guard, and whether it stays is a measurement
- The fan-out is the shape of the work, not a latency optimisation
- Criteria retrieved about a class come from that class's document
- Thresholds come from the sweep or the work instructions
- Registration recovers a translation and nothing else, and it must be able to refuse
- The escape budget is one number over six classes the work instructions do not treat alike
- Say what the prevalence is, and what it costs
- Use the official DeepPCB split
- Split train/val by image, never by patch
- An automated disposition names what produced it
- A human disposition names who made it
- Language is a rendering, not a record
- Say what is simulated

## Environment gotchas

- MacBook Air M5, 32GB, **fanless** -- sustained load throttles. Report "first
  60s" and "steady state" separately.
- **Ollama contention is the big one.** A translation job in
  `~/Projects/video_transfer` periodically holds `gpt-oss:20b` and 12GB of GPU.
  When it runs, an LLM call's wall time can be 25x its inference time. Always
  check `ollama ps` before believing a latency number, and report
  `eval_duration`, never `total_duration`.
- **`ollama ps` is not a GPU check.** It reports Ollama's own resident models
  and nothing else, so a torch/MPS job in another shell saturates the same
  silicon while that check comes back clean. It has already cost one run: an
  MPS benchmark taken beside a concurrent detector training job, with a clean
  residency check the whole way. Sweep the process table too --
  `scripts/reverifier_latency.py` does both and refuses to publish when either
  fires.
- MCP SDK is **2.0**: `MCPServer` not `FastMCP`, `server_info` not `serverInfo`.
- torch runs on **MPS**. Python is pinned to 3.12 for torch compatibility.
- Dataset (231MB), patches, models, the SQLite db and the Chroma index are all
  gitignored and rebuilt by scripts.

