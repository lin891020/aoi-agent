# AOI-Agent

A PCB AOI re-verification system. Production optical inspection is tuned for
recall and over-flags; every flagged region goes to a human today. This puts a
vision model in front of that queue and an agent behind it for the cases the
model cannot settle.

Started 2026-08-22. Portfolio project aimed at smart-manufacturing AI roles, so
the engineering has to hold up to an interviewer's questions, not just run.

## Layout

```
src/aoi_agent/
    data/deeppcb.py         dataset access, official splits
    aoi/simulator.py        template differencing -- the "AOI"
    aoi/matching.py         label candidates against ground truth
    vision/                 patches, dataset, ResNet-18, inference, operating point
    store/                  SQLAlchemy models, seeding, standards retrieval
    mcp_servers/            three MCP servers (classify, production, standards)
    graph/                  LangGraph flow with the human escalation,
                            durable SQLite checkpointer
    station/                the review station -- FastAPI + Jinja, the
                            escalation queue, and the service layer the
                            CLI shares with it
    cli.py
scripts/                    gate_check, build_patches, train, report, seed_store, ...
tests/                      77 tests; dataset-dependent ones behind `-m dataset`
docs/benchmarks.md          every measurement run, newest last
docs/architecture.md        layers, thresholds and where they come from
.claude/skills/             project skills -- procedures with gates, not notes
```

Invoke the project skills; do not just read past them. `retraining-the-reverifier`
before touching the training pipeline or regenerating a benchmark, and
`measuring-llm-latency` before quoting any timing number. Both encode a failure
that is silent -- a stale threshold, a contended GPU -- and neither shows up as
an error.

## Commands

```bash
uv run pytest                                    # 77 tests, no GPU needed
uv run python scripts/gate_check.py              # S0: does differencing make false calls?
uv run python scripts/train.py                   # ~4 min on the M5 Air (MPS)
uv run python scripts/report.py                  # operating-point table -> docs/benchmarks.md
uv run python scripts/routing_report.py          # how much never reaches the LLM
uv run python scripts/check_mcp_servers.py       # servers start and advertise tools
uv run python -m aoi_agent board 20085294 --queue  # run a board, queue what it cannot settle
uv run python -m aoi_agent station               # review station on :8000
uv run python -m aoi_agent queue                 # what is waiting on a person
```

## Invariants — do not quietly change these

- **Report an operating-point curve, never bare accuracy.** An escape ships a
  bad board; a false call costs seconds. Headline is "review removed at an
  escape budget".
- **Every failure path escalates to a human.** Unparseable verdict, LLM
  unreachable, `open` at any confidence -- all route to a person. Never guess to
  avoid escalating.
- **An escalation must outlive the process.** The checkpointer is a SQLite file,
  not `InMemorySaver`. An escalation that dies with the CLI run is a prompt
  wearing a graph's clothes.
- **The review station never shows `ground_truth`.** The operator's answer is
  the next training round's label; showing them the answer key first collects an
  echo, not a judgement. Enforced at the dict boundary, not by grepping HTML.
- **No free-form text-to-SQL.** Typed parameters over a fixed query set. A valid
  but semantically wrong query returns a plausible number and gets acted on.
- **Thresholds come from the sweep or the work instructions**, not from hand
  tuning against the test set. See docs/architecture.md.
- **Use the official DeepPCB split.** Do not re-split; comparability matters.
- Split train/val **by image**, never by patch -- patches from one board leak.
- **Say what is simulated.** Production metadata is generated with one planted
  signal; acceptance criteria are original documents, not IPC-A-610 (copyrighted,
  must stay out).

## Environment gotchas

- MacBook Air M5, 32GB, **fanless** -- sustained load throttles. Report "first
  60s" and "steady state" separately.
- **Ollama contention is the big one.** A translation job in
  `~/Projects/video_transfer` periodically holds `gpt-oss:20b` and 12GB of GPU.
  When it runs, an LLM call's wall time can be 25x its inference time. Always
  check `ollama ps` before believing a latency number, and report
  `eval_duration`, never `total_duration`.
- MCP SDK is **2.0**: `MCPServer` not `FastMCP`, `server_info` not `serverInfo`.
- torch runs on **MPS**. Python is pinned to 3.12 for torch compatibility.
- Dataset (231MB), patches, models, the SQLite db and the Chroma index are all
  gitignored and rebuilt by scripts.

## Still open

Agent-layer latency benchmarks -- specifically, whether `gpt-oss:20b` answers
inside WI-300's 10s response budget on a quiet machine. If it does not, WI-300
says change the model, not the budget. Also: retraining from operator
corrections, INT8/ONNX quantisation, demo video.

On the station itself:

- **Split escalations by kind.** `agent_uncertain` vs `infrastructure`. With a
  10s budget, an Ollama outage escalates *every* candidate and buries the
  genuinely ambiguous ones under a page of "the model did not answer". The
  behaviour is right -- fail towards a person -- but the queue stops being
  triageable, and the infrastructure batch could be re-run on recovery instead
  of spending operator time. One column on `Escalation`, plus a filter.
- Corrections review page (the CLI command exists, the page does not).
- Board browser, so the 82% the agent settled is visible and not just the queue.
- Operator authentication. `reviewer` is a free-text field, so the corrections
  that feed retraining carry no trustworthy identity.
