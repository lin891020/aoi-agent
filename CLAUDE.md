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
    store/                  SQLAlchemy models, seeding, standards retrieval,
                            the analysis_runs log
    mcp_servers/            three MCP servers (classify, production, standards)
    graph/                  LangGraph flow with the human escalation,
                            durable SQLite checkpointer
    analysis/               the /ask flow -- typed plan, validator, Send
                            fan-out, chart derived from the result shape
    station/                the review station -- FastAPI + Jinja, the
                            escalation queue, /ask and its SSE progress
                            stream, and the service layer behind both.
                            result_view.py is the ground_truth boundary;
                            chart_svg.py renders a chart spec server-side
    cli.py
scripts/                    gate_check, build_patches, train, report, seed_store,
                            analysis_eval, ...
tests/                      529 tests; dataset-dependent ones behind `-m dataset`
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
uv run pytest                                    # 529 tests, no GPU needed, no model called
uv run python scripts/gate_check.py              # S0: does differencing make false calls?
uv run python scripts/train.py                   # ~4 min on the M5 Air (MPS)
uv run python scripts/report.py                  # operating-point table -> docs/benchmarks.md
uv run python scripts/routing_report.py          # how much never reaches the LLM
uv run python scripts/latency_report.py          # does the reason node fit the response budget?
uv run python scripts/agent_eval.py              # does the agent layer beat the classifier? (~9 min)
uv run python scripts/analysis_eval.py           # does the planner plan the right lookups, and refuse the rest?
uv run python scripts/analysis_eval.py --plan-only  # the same score, without the tools and the prose nobody scores
uv run python scripts/check_mcp_servers.py       # servers start and advertise tools
uv run python -m aoi_agent board 20085294 --queue  # run a board, queue what it cannot settle
uv run python -m aoi_agent station               # review station on :8000 -- the queue, and /ask
uv run python -m aoi_agent queue                 # what is waiting on a person
```

## Invariants — do not quietly change these

Three of these are scoped to the **disposition path** -- the flow that ends in a
board being dismissed, confirmed or held. That scope is not a loophole, it is
what the invariants are about: they all buy the same thing, which is that a
wrong call cannot ship a board or corrupt the next training round. The `/ask`
flow dispositions nothing. Nothing on that page holds or releases a part, no
label it produces reaches `review_decisions`, and its worst failure is a
supervisor reading a wrong sentence with the raw figures printed beside it. So
it has no checkpointer and no escalation queue, deliberately -- see
docs/architecture.md, "The analysis path". Anything that ever does disposition a
board is back under the unscoped reading.

- **Report an operating-point curve, never bare accuracy.** An escape ships a
  bad board; a false call costs seconds. Headline is "review removed at an
  escape budget".
- **The LLM explains; it does not decide.** Measured, its verdict was worse
  than the classifier's (12 overrides, 1 right) and its `confident` flag was
  worse at selecting who needs a person than a plain threshold on the
  classifier's own number. `route_after_reason` routes on `ESCALATE_BELOW`,
  `decide_node` takes `model_class`, and what the LLM writes is what the
  operator reads. Do not put it back on the decision path without a measurement
  that says to. On the analysis path the same rule holds in its own terms: the
  planner chooses which *lookups* to make, every one of them is validated
  against the real signatures and value domains before it runs, and the chart is
  derived from the result shape rather than chosen by the model. What the LLM
  contributes at either end is language.
- **On the disposition path, every failure escalates to a human, except an LLM
  outage.** Unparseable verdict, `open` below the threshold, anything under
  `ESCALATE_BELOW` -- all route to a person. An unreachable LLM no longer does,
  because the decision no longer depends on it: the operator loses an
  explanation, not a verdict, and the queue does not fill with every candidate
  on the line. The analysis flow's failures terminate in a message on the page
  instead -- an unplanned question, a rejected plan, a branch whose tool raised.
  There is no disposition waiting on any of them and nothing for a person to
  answer, so a queue entry would be a task nobody can close.
- **On the disposition path, an escalation must outlive the process.** The
  checkpointer is a SQLite file, not `InMemorySaver`. An escalation that dies
  with the CLI run is a prompt wearing a graph's clothes. The analysis flow has
  no checkpointer because nothing in it suspends -- adopting the feature anyway
  is what this project spent a day removing from the other graph. What that
  flow needs instead is reproducibility, and the `analysis_runs` table serves
  it: a chart is redrawn from stored data rather than by re-running a plan the
  model may not reproduce.
- **The review station never shows `ground_truth`.** The operator's answer is
  the next training round's label; showing them the answer key first collects an
  echo, not a judgement. Enforced at the dict boundary, not by grepping HTML --
  `store.boards.resolve_candidate` for the queue, and on the analysis page both
  routes out of a tool's payload: `result_view.readable_rows` for the table and
  `result_view.strip_hidden` for what the synthesis prompt is shown. A boundary
  with a second door is not a boundary.
- **No free-form text-to-SQL.** Typed parameters over a fixed query set. A valid
  but semantically wrong query returns a plausible number and gets acted on.
  This is why `/ask` validates a plan's argument *values* against the store's
  real domains and refuses rather than retrying: `line_id="L4"` raises nothing
  and returns nothing, and the missing series reads as a finding.
- **The fan-out is the shape of the work, not a latency optimisation.** The plan
  expands into `Send` branches because the facts are independent; four real
  tools take 183ms in parallel against 462ms in sequence, either side of two
  model calls costing around 25 seconds. Nothing in the code, the docs or the
  page may present it as a speed-up.
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

Retraining from operator corrections, INT8/ONNX quantisation, demo video.

`gpt-oss:20b` misses WI-300's 10s response budget -- p90 service time 15.6s on a
quiet machine, 20 of 24 calls over. It no longer gates anything, because the LLM
is off the decision path and an operator waits for a verdict, not for an
explanation. Worth revisiting only if the LLM is ever put back on that path, or
if the station starts blocking on the explanation to render.

On the station itself:

- Board browser, so the 82% the agent settled is visible and not just the queue.
- **Timestamps are stored UTC and displayed UTC**, unlabelled. On a quality
  record read at UTC+8 that is an eight-hour lie. Store UTC, render local, say
  which -- pairs with the operator-identity gap below.
- **Authentication, which `/ask` turned from a backlog item into a
  precondition.** There is none, on either page. Two separate costs now. The
  one that was always here: `reviewer` is a free-text field, so the corrections
  that feed retraining carry no trustworthy identity -- demonstrated the hard
  way, when five regions were clicked through without domain knowledge, four of
  them wrong, and nothing in the system could tell those labels from an
  expert's. They had to be deleted by hand. The one this branch added: an
  unauthenticated visitor to the queue sees the regions on one line, and the
  same visitor at `/ask` can pull production statistics for the whole plant.
  That is a change of kind, and it is the item on this list that should block
  the station running anywhere but a laptop.
- **The criteria answer the wrong question for the operator.** For `open` the
  retrieved passage says any confirmed open is critical -- how to *disposition*
  one. It never says how to *confirm* one, which is what the person looking at
  the images actually needs.
