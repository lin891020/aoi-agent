# Architecture

## Layers

| layer | job | what it must not do |
|---|---|---|
| AOI simulator | flag every region that differs from the golden template | decide what a difference means |
| Re-verifier (PyTorch) | say what one flagged region is, with a calibrated probability | decide what happens to the board |
| MCP tools | expose the model, the production store and the criteria as callable functions | choose which of themselves to call, or answer about one defect class out of another's document |
| LangGraph flow | route on confidence, gather evidence, hand over to a person | contain domain rules that belong in the work instructions |
| Review station | show a person the evidence the agent had, and take their answer | re-run the flow to render a page, or show the ground truth |
| Local LLM | explain to an operator why a region reached them | decide anything -- measured worse than the classifier at both the class and the hand-off |
| Analysis planner (LLM) | turn a supervisor's question into a typed plan of tool calls, or refuse it | run anything, choose a chart, or answer from its own knowledge |
| Plan validator | reject a tool, argument or value that does not exist, before any of it runs | retry, or repair a plan on the model's behalf |
| Analysis flow | fan out over the plan, derive a chart from the result shape, then have the LLM write the prose | disposition anything -- it answers questions about boards, it does not decide about one |

## The decision path

```
                          AOI candidate
                                │
                          classify_defect
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
     P(false call) ≥ .915   conf ≥ .95        everything else
       and not `open`      and not `open`            │
              │                 │                    ▼
           dismiss           confirm         query_board_context
              │                 │            query_machine_stats
              │                 │            search_standards
              │                 │                    │
              │                 │                    ▼
              │                 │              LLM weighs it
              │                 │                    │
              │                 │        ┌───────────┴──────────┐
              │                 │        ▼                      ▼
              │                 │     verdict            interrupt():
              │                 │                        ask an operator
              └─────────────────┴────────────┬───────────────┘
                                             ▼
                                     review_decisions
                                             │
                                             ▼
                                    next training round
```

## Where an escalation waits

``interrupt()`` suspends the run and checkpoints it. The operator may answer in
a second or in two days, and almost certainly in a different process from the
one that raised it -- the line does not stop to ask a question. So:

```
  flow run                    review station
     │                              │
  interrupt()                       │
     │                              │
     ├──> checkpoints.db  ──────────┤  graph state, resumed verbatim
     │    (SqliteSaver)             │
     │                              │
     └──> escalations table ────────┤  the queue: who is still waiting
          (status, thread_id)       │
                                    ▼
                            operator answers
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            review_decisions                Command(resume=...)
            (recorded first)                (run finishes)
```

Two stores, one question each. The checkpointer answers "what was this run's
state"; the `escalations` table answers "is anyone still waiting on it". Putting
the graph state in the table as well would give two answers to the same
question, and they would drift.

The decision is written *before* the queue entry is closed. If the process dies
between the two, the region stays on the queue and gets looked at again; the
other order drops an operator's verdict silently.

The station reads the suspended state rather than re-running the flow. Re-running
would spend another 20B-model inference and could hand back a different rationale
than the one on the operator's screen.

## The analysis path

A second entrance, for a different person. The queue answers "what do I do with
this region"; `/ask` answers the question a shift supervisor walks up with --
"is M22 drifting, and does that matter". It reads the same MCP tools the
disposition path uses and it dispositions nothing.

```
                  supervisor's question
                            │
                          plan            one LLM call, typed output
                            │
                     validate_plan
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
        invalid: show the plan          valid
        and every error; run            │
        nothing                    Send(run_tool, call) × N
                                        │
                                   ┌────┼────┐
                                   ▼    ▼    ▼
                                 run  run  run    a failed branch is data,
                                   └────┼────┘    not an exception
                                        │
                                     collect      chart derived from the
                                        │         result shape
                                        ▼
                                   synthesise     second LLM call: prose
                                        │
                                        ▼
                                      /ask
```

Validation runs in three layers before any tool executes -- the tool name, the
argument names against the real signature, and the argument *values* against the
domains the store actually holds. The third carries the weight, and it is the
no-SQL invariant one level up: `line_id="L4"` raises nothing and returns
nothing, so the chart comes back with one fewer line and the gap reads as a
finding. A plan that fails is shown to the person, not retried.

`Send` is what makes this a scheduler rather than a switch: the branch count
comes from the plan, not from the graph's shape. It is not a latency
optimisation and the page does not present it as one -- four real tools take
183ms in parallel against 462ms in sequence, either side of two model calls
costing around 25 seconds. It is the correct structure for independent work,
and it scales as tools multiply.

Two absences that are deliberate, and that the disposition path's invariants
would otherwise appear to forbid:

- **No checkpointer.** Nothing suspends, nobody is in the loop, and a question
  is answered in one invocation. Reproducibility is served by the
  `analysis_runs` table instead -- question, plan, raw results, chart spec,
  answer, timings -- so a chart is redrawn from stored data rather than by
  re-running a plan the model may not reproduce.
- **No escalation queue.** A failure here has nothing to hand a person: no
  board is held or shipped by anything on this page. Both invariants in
  CLAUDE.md are scoped to the disposition path for that reason.

`/ask` has no authentication in front of it, and neither does the queue. An
unauthenticated visitor to the queue sees the regions on one line; the same
visitor here can pull statistics for the whole plant.

## Thresholds and where they come from

Every row cites something a reader can open: a script they can run, or a line in
a document that states the number. `tests/test_threshold_citations.py` parses
this table and fails when a value drifts from the constant, when a source stops
resolving, or when a threshold constant is added to the code and not to the
table.

| constant | value | source |
|---|---|---|
| `DEFAULT_DISMISS_THRESHOLD` | 0.915 | the operating-point sweep at the ≤0.5% escape budget -- `scripts/report.py` |
| `ESCALATE_BELOW` | 0.915 | `= DEFAULT_DISMISS_THRESHOLD`, which empties the only band in which the agent branch could dismiss -- `scripts/threshold_sweep.py` |
| `CONFIDENT` | 0.95 | a cost gate: changes no disposition at or above `ESCALATE_BELOW`, and must not drop below it -- `scripts/threshold_sweep.py` |
| `LOW_CONFIDENCE` | 0.70 | WI-300 escalation triggers, stated there as a floor -- `data/standards/reverification-procedure.md` |
| `RESPONSE_BUDGET_S` | 10.0 | WI-300 response budget, derived from QP-110 -- `data/standards/reverification-procedure.md` |
| `IOU_THRESHOLD` | 0.33 | DeepPCB's own benchmark convention -- `data/DeepPCB/README.md` |
| `FRAGMENT_GAP_PX` | 20 | measured: 6.1% of unmatched candidates touch a real defect -- `scripts/diagnose_false_calls.py` |

None of these are tuned by hand against the test set. The dismissal threshold
comes out of the sweep; the floor and the response budget are written down in
the work instructions and the code reads them from there rather than the
reverse. `tests/test_response_budget.py` fails if the budget and WI-300 stop
agreeing.

Two rows were wrong until 2026-08-23, and how they were wrong is worth keeping.
`CONFIDENT` was cited to "WI-300 decision authority", which states no such
number. `ESCALATE_BELOW` was cited as "the lowest threshold adding no escape to
the budget, see docs/benchmarks.md", and no sweep of it had ever been run --
`scripts/threshold_sweep.py` is that sweep, written afterwards. It found the
claim false in both directions: the lowest zero-escape threshold on the test
split is 0.875, not the 0.90 the code carried, and 0.90 was not derived from
any sweep at all. It was a round number that happened to be conservative.

Neither number is the one now in the code, because both are read off one split.
0.875 clears the highest-confidence real defect this branch would dismiss by
0.003; the next lot's tail sits on top of that. The value that needs no split is
the dismissal threshold itself. The agent branch can only dismiss when the
classifier's class is `false_call`, and for that class its confidence *is*
`P(false call)`, so the region must lie in the band [`ESCALATE_BELOW`,
`DEFAULT_DISMISS_THRESHOLD`). Setting them equal empties that band by
construction and keeps it empty through a retrain. **The agent branch may
confirm a defect; it may never dismiss one** -- WI-300 §1 now says so, and
`tests/test_graph.py::test_the_agent_branch_cannot_dismiss` holds it
independently of what the numbers are. It costs 47 more escalations out of 8143
candidates, 0.6% of the queue.

`CONFIDENT` did not move, and the sweep is why: at or above `ESCALATE_BELOW` it
changes zero dispositions, because `confirm_node` and `decide_node` write the
same verdict. It decides who gets an LLM call and a written rationale, not what
happens to the board. What it must not do is fall below `ESCALATE_BELOW`, where
it starts confirming unreviewed regions the flow would have escalated -- 66 of
them at 0.70. The constraint is the citation; the value inside it is a dial.

The response budget is the one that most invites being quietly raised, so
WI-300 states the direction explicitly: if the configured model cannot answer
within it, the model is the wrong size for the line. Whether `gpt-oss:20b` fits
inside 10s is an open measurement -- see docs/benchmarks.md.

## Failure directions

Every branch that can fail fails towards a person:

- confidence below `ESCALATE_BELOW` → escalate
- `open` at any confidence → investigate, never short-circuit
- a candidate that fragments a real defect → held out of training, not labelled spurious
- the process holding a suspended run dies → the run is on disk, the queue still
  lists it, and a second process finishes it --
  `tests/test_checkpoint_durability.py` raises the escalation in an interpreter
  that then exits

The exceptions are the LLM's own two failures: an unreachable model, and a
response that will not parse. Both used to escalate and neither does now,
because the decision no longer depends on the LLM -- the run falls back to the
classifier's own class and confidence and routes on those. Above
`ESCALATE_BELOW` that means an unparseable verdict is dispositioned, not
escalated; the parse failure is recorded in the rationale the operator would
read. This list said "an unparseable verdict → escalate" until 2026-08-23,
which was left over from when the LLM decided. The operator loses an
explanation, not a verdict, and the queue does not fill with every candidate on
the line.

An escalation costs an operator a few seconds. The alternatives cost a shipped
board or a silently mislabelled training set.

On the analysis path a failure terminates in a message on the page, not in a
queue, because there is no disposition waiting on it and nothing for a person to
answer:

- the planner unreachable, or a response that will not parse → "no plan was
  produced", and nothing runs
- a plan that fails validation → the plan and every error, and nothing runs
- a question the data cannot answer → a refusal that states the coverage
- one branch's tool raising → that branch returns `ok=False`; its siblings
  finish and the answer names the one that did not
- synthesis unreachable → the results are already correct and already on
  screen; the prose is what is lost

The direction is the same one the station takes: better to show a person "I
cannot answer this" than a plausible wrong answer they have no way to check.

## What is real and what is simulated

| | real | simulated |
|---|---|---|
| board images | ✓ DeepPCB line-scan CCD | |
| defect locations and classes | ✓ annotated | partly augmented by the dataset authors |
| AOI candidates | ✓ produced by differencing | |
| false calls | ✓ produced by the same algorithm | |
| lot / line / machine / shift | | ✓ generated, with one planted signal |
| acceptance criteria | | ✓ original documents, not a real standard |
| operator identity | | ✓ a free-text field; the station has no auth yet |
| the analysis page's figures | ✓ computed from the store, by fixed queries | ✓ over the generated line metadata above |
