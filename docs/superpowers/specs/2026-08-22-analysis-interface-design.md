# Natural-language analysis interface — design

Status: approved for planning. 2026-08-22.

## Why

Taking the LLM off the disposition path left the three MCP servers without a
consumer: `query_defect_history`, `query_machine_stats`, `query_board_context`
and `search_standards` are now reachable only from a node that no longer decides
anything with them. Meanwhile the review station answers one question well --
"what should I do with this region" -- and cannot answer the question a shift
supervisor actually walks up with: "is M22 drifting, and does that matter?"

The user is the reason this is natural language rather than a query form. A
shift supervisor does not write SQL, and will not learn to. That justification
cuts both ways and shapes the design below: someone who cannot write a query
also cannot check one, so the assumptions display and the refusal path are not
polish, they are the direct consequence of who is asking.

It also bounds the claim. If most of what a supervisor asks is the same ten
questions, a dashboard beats a language model on every axis -- faster, free,
deterministic. What a language model earns is the long tail: the question nobody
built a tile for. So the page carries both, five common questions as one click
beside the free-form box, and the design does not pretend to replace the
dashboard it should be sitting next to.

The measured lesson from the disposition path applies directly. The LLM lost
where the task was a quantitative judgement over structured data, and it is
strong where the input or output is natural language. Turning a supervisor's
question into a set of typed tool calls, and turning the results back into a
sentence, is that second kind of task on both ends.

## Scope

**In.** Questions that genuinely need several *different* tools, because they
span independent facts. Those are parallel by the shape of the question, not by
slicing one tool artificially -- a distinction that matters, since a
finer-grained tool would collapse the second kind of parallelism and expose it
as decoration.

**Out.** Multi-period comparison, until the seeded span supports it (below).
Free-form SQL, permanently: the project's existing invariant holds here for the
same reason, and typed parameters over a fixed query set are what the MCP
servers already expose. Re-planning after results (approach C below).

## Approach

Three were considered.

**A — plan, then dynamic fan-out.** One LLM call produces a typed plan;
validation runs before any tool does; `Send` expands the plan into N parallel
branches; a reducer accumulates their results; a deterministic node builds the
chart and a second LLM call writes the prose. **Chosen.**

**B — fixed parallel branches.** One branch per tool, with the plan switching
them on and off. Simpler and easier to test, but fan-out width is bounded by the
tool count and `Send` is never needed, so the dynamic scheduling this exists to
exercise never happens.

**C — A plus a re-planning loop.** After results return, the LLM decides whether
it has enough and can fan out again. This is the more capable shape and the
`operator.add` reducer already supports it -- a second round's results stack on
the first with no extra merge logic. Deferred, for three reasons: it needs a
round cap to terminate, it multiplies a 10-15s model call by the round count,
and "do I have enough" is itself an LLM judgement, which this project does not
adopt without measuring it against a simpler rule.

## 1. The plan, and validating it before anything runs

```python
class ToolCall(TypedDict):
    tool: Literal["query_defect_history", "query_machine_stats",
                  "query_board_context", "search_standards", "list_candidates"]
    args: dict
    why: str

class Plan(TypedDict):
    interpretation: str        # how the question was read
    assumptions: list[str]     # baselines and defaults chosen, in plain language
    calls: list[ToolCall]
```

Validation runs in three layers before any tool executes:

1. **Tool name** against the whitelist -- the schema's `Literal` catches most.
2. **Parameter names** against the tool's real signature, via `inspect.signature`.
   Unknown and missing names both fail.
3. **Parameter values** against their domains: `line_id` in L1/L2/L3,
   `machine_id` in the six that exist, `defect_type` in the seven classes, `days`
   within the seeded span.

Layer 3 carries the weight, and it is the same argument as the no-SQL invariant.
`line_id="L4"` raises nothing and returns nothing; the chart simply comes back
with one fewer line and no one notices. A syntactically valid query with the
wrong semantics is the failure this system is built to prevent.

A failed validation does not retry. The plan and the errors are shown to the
user, and nothing runs. The station's principle transfers: better to show a
person "I cannot answer this" than a plausible wrong answer.

`classify_defect` is excluded from the planable set. It loads a torch model onto
MPS, so a plan fanning out to ten of them is ten GPU contentions.

## 2. The graph

```python
class AnalysisState(TypedDict):
    question: str
    plan: Plan | None
    plan_errors: list[str]
    results: Annotated[list[ToolResult], operator.add]
    timings_ms: Annotated[dict[str, float], operator.or_]
    chart_spec: dict | None    # serialisable; the page renders it, see below
    answer: str
```

`chart_spec` is data, not an image: the axis, the series and their labels.
The page renders it. That is what lets a stored answer be redrawn later
without re-running anything, and it keeps chart choice inspectable rather
than baked into a PNG.

```
START → plan → validate ─┬─ invalid → refuse → END
                         └─ valid → Send(run_tool, call) × N
                                        │
                                   ┌────┼────┐
                                   ▼    ▼    ▼
                                 run  run  run     each returns {"results": [one]}
                                   └────┼────┘     the reducer accumulates
                                        │
                                     chart → synthesise → END
```

`Send` is what makes this a scheduler rather than a switch: the number of
branches comes from the plan, not from the graph's shape.

**A failed branch is data, not an exception.** `run_tool` catches its own
failures and returns `ToolResult(ok=False, error=...)`, so the join sees "three
succeeded, one failed" and the answer says which. The branches are independent
by construction; letting one abort the others would be a defect, not safety.

Three deliberate omissions:

- **A separate graph from the disposition flow.** Different entry point,
  different state, different user. Merging them would make both harder to read.
- **No checkpointer.** Nobody is in the loop, nothing suspends, nothing resumes
  days later. The disposition flow needs one because of `interrupt`; this does
  not, and adopting an unneeded framework feature is what this project just spent
  a day removing. Reproducibility is served instead by an
  `analysis_runs` table -- question, plan, raw results, `chart_spec`, answer,
  timings -- so a chart is rebuilt from stored data rather than by re-running
  a plan that may not regenerate identically. It is also the log the eval
  script and any later live view read from.
- **The chart type is derived from the result shape**, not chosen by the model.
  Time series to a line, cross-entity comparison to bars. The LLM explains; it
  does not decide -- including about charts.

## 3. What the user sees

A new page on the review station. Above the box, three example questions and the
data's coverage stated plainly. The examples are few-shot for the *user*: people
ask in the shape they are shown, and it is the cheapest available fix for the
class of problems that come from a badly formed question.

Every answer shows five things:

1. **Interpretation** -- how the question was read
2. **The calls** -- which tools ran, with what arguments and why
3. **Assumptions** -- the baseline chosen, stated in plain language
4. **Timing** -- per tool, plus `4 tools · 1.2s parallel / 5.8s sequential`
5. **Chart and prose**, with failed branches named

Items 2 and 3 exist for trust, not debugging. A supervisor can only judge
"open is up 30%" if they can see it was measured against the fleet average
rather than against last week.

Item 4 is the only evidence on screen that the fan-out is real.

### Progress, in the first version

Two model calls at 10-15s each put an answer around 20-25s away. An
indeterminate spinner over that wait makes it feel longer than it is, and shows
nothing. The same seconds spent watching tools tick off read as work:

```
Planning…                              ✓  2.1s
Running 3 tools
  ✓ open rate by machine                  0.3s
  ✓ acceptance criteria for open          0.2s
  ⟳ defect history, L2, 30d
```

This was going to be deferred as a live view of the running graph. It is
promoted into the first version because it does two jobs at once -- it makes the
wait tolerable, and it is the only thing that shows a viewer the fan-out is
concurrent. The data is already being recorded for item 4; what is added is
streaming it, which LangGraph supports natively.

The richer animated graph view stays deferred. Recording node entry and exit now
keeps it a rendering problem rather than an architectural one.

### Few-shot in the prompt: five examples, all boundaries

Diversity matters more than count, and the useful examples show edges rather
than the happy path:

| # | Shape | Teaches |
|---|---|---|
| 1 | Standard cross-tool question | What a normal expansion looks like |
| 2 | Comparison with no stated baseline | State the baseline in `assumptions` |
| 3 | A causal "why" question | Fetch what exists; say causality is unanswerable |
| 4 | Outside the data range | Refuse, and state the coverage |
| 5 | Too vague ("how's the line?") | Refuse and ask which aspect |

## 4. Data

The 8-day span is a property of the seed generator, not of DeepPCB, which ships
no timestamps at all: `START + timedelta(hours=position * 0.4)` over 500 boards.
Only the test split is seeded; 1000 trainval boards are unused.

Seeding all 1500 and widening the spread is a small change, but the trade-off is
real and capped: 1500 boards over two months is 25 boards a day, over six months
is 8. Thin for a real line. The metadata is already declared simulated, so a
thin-but-honest span is preferred to fabricating three years of dense data.

Multi-period questions stay out of scope until this is decided.

## 5. How we know it works

The tools are deterministic, so a correct plan yields correct data by
construction. The errors live in the plan. That means the evaluation needs
ground truth for *plans*, not for answers -- which is hand-buildable, where
ground truth for answers is not.

`scripts/analysis_eval.py` measures three things:

| Measure | Method |
|---|---|
| Plan accuracy | ~20 hand-written questions with expected tools and arguments |
| Refusal accuracy | A set that *should* be refused: out of range, causal, vague |
| Determinism | The same question five times; is the plan stable? |

Refusal accuracy carries more weight than it looks. A system that answers
everything is more dangerous on a factory floor than one that says it does not
know.

**A known weakness in the method:** the same person writes the few-shot examples
and the evaluation questions, so the questions will tend to be the shapes the
prompt was designed for. Unresolved. Partial mitigations are to write the
questions before reading the examples, or to take them from someone else.

**Not covered:** synthesis can describe correct data incorrectly. The mitigation
is that the raw results sit beside the prose, so a reader can check. This is a
known gap, not a solved problem.

## Sequencing

The data decision in section 4 is independent of everything else and gates
only the multi-period questions that are already out of scope. It can be
taken before, during or after; the plan should not block on it.

## Open questions

- The station has no authentication. Today an unauthenticated visitor sees a
  queue; with this page they can pull production statistics for the whole plant.
  Operator authentication moves from a backlog item to a precondition.
- Two LLM calls per question at 10-15s each puts an answer around 20-25s.
  Streamed progress is the mitigation, not a fix. Whether a supervisor waits that
  long is unmeasured, and worth measuring early rather than at the end: if they
  will not, the bottleneck is those two calls and no amount of fan-out helps.
