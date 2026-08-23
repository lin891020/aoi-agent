# Architecture

## Layers

| layer | job | what it must not do |
|---|---|---|
| AOI simulator | flag every region that differs from the golden template | decide what a difference means |
| Re-verifier (PyTorch) | say what one flagged region is, with a calibrated probability | decide what happens to the board |
| MCP tools | expose the model, the production store and the criteria as callable functions | choose which of themselves to call, or answer about one defect class out of another's document |
| LangGraph flow | route on confidence, gather evidence, hand over to a person | contain domain rules that belong in the work instructions |
| Review station | show a person the evidence the agent had, and take their answer | re-run the flow to render a page, or show the ground truth |
| The store | be the record: what was decided, by what, under which weights and thresholds | accept an automated decision that names none of them, or let one absence stand for two |
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

## What a decision names

A customer returns a batch and the auditor asks three questions: who decided
this board was fine, when, and on what basis. Until 2026-08-23 this store could
answer none of them. It held 9,140 decisions, each naming a verdict and a
source -- `model`, `agent` or `human` -- and nothing about the weights, the
operating point or the code that produced it. There was also no row anywhere
about a *board*, which is the thing a line ships.

That is not only an audit problem. A decision you cannot attribute to a model
version and a threshold is a decision you cannot revisit, and "which model
produced these dispositions" is the first question asked when a metric moves.

Three things are recorded, and all three are **derived rather than declared**:

| what | where it comes from | why not the obvious thing |
|---|---|---|
| `model_digest` | SHA-256 of the checkpoint file, first 16 hex | not the path: `models/reverifier.pt` is a slot every training run overwrites. Not a `model_version` string: a field somebody must remember to bump is worse than no field, because the release it is forgotten on is the release where it is wrong and looks right |
| `thresholds_json` | the constants the flow routed on | the same weights disposition differently at a different operating point, and the operating point is the half that moves -- `ESCALATE_BELOW` moved on 2026-08-23 |
| `code_version` | `git rev-parse`, `+dirty` where the tree has changes | read from the tree, not kept in a constant. `AOI_AGENT_CODE_VERSION` covers a container that ships no `.git`; failing both, `unknown` |

The digest is computed once, when `ReVerifier` loads the checkpoint, and travels
out of `classify_defect` with the reading. It goes into the **graph state**, so
it is checkpointed with the run and comes back with it: an escalation answered a
week after a retrain is attributed to the weights that raised it, not to
whatever is on disk that morning.

The columns are storage. The mechanism is `store.boards.record_decision`, which
raises on a `model` or `agent` row that names no model. An operator's answer is
never refused for want of it -- a run checkpointed before this existed resumes
carrying none, and their label is the scarcest thing in the system.

Three absences, three words, and none of them is `NULL`:

| value | means |
|---|---|
| `unrecorded` | the row predates the columns. Stamped by the migration, at the one moment "written before provenance existed" is knowable without guessing |
| `unavailable` | written after the columns existed and still could not name the model -- an operator's answer resumed from an older checkpoint |
| `unknown` | the code version could not be read: no git, no build stamp |

`NULL` therefore means nothing at all, and `tests/test_provenance.py` says so.
Folding these into one word, or into `NULL`, is the defect
`explanation_status` was fixed for in miniature: an absence that reads like a
value gets counted as one.

## What happened to the board

Every other table records a judgement about a *region*. `board_dispositions` is
the one that answers for a board, and it holds the smallest thing that answers
the auditor: released or held, when, on whose authority, under which model, and
the counts it was computed from.

```
  candidates ── latest decision each ──┐
  escalations (pending) ───────────────┤
                                       ▼
                             released | held
                                       │
                            board_dispositions
```

The rule is deliberately dull. A region's fate is its **latest** decision,
because decisions accumulate and an operator's correction is a second row
rather than an edit of the first. A region with no decision, or one still on the
queue, is pending. A region whose latest verdict is anything but `false_call` is
confirmed. A board with any confirmed or any pending region is **held**;
otherwise it is **released**.

Both reasons to hold are one state because the line does the same thing with
them, and the counts on the row say which it was. The pending half is the one
that matters: aggregating over decisions alone comes back clean while a person
is still holding the region that would have stopped the board. "Nobody has
looked at it yet" is not a release.

Rows accumulate the way decisions do -- held on Monday, released on Tuesday
after an operator answered, is two rows and the pair is the record. Written when
a board reaches a settled state: the end of a board run, and the moment an
operator answers the last region on it, which is also where "on whose
authority" gets a name rather than "automated". Since 2026-08-23 that name is
the operator's signed-in one, taken from the session rather than from a form
field, and the region-level rows beside it record how it was established.

Where the board's decisions do not agree on one model -- part-decided before a
retrain and part after -- the row says `mixed`. It was not released under a
model; naming one of the two would be the comfortable lie.

Read at `/board/<stem>` on the station and `uv run python -m aoi_agent
provenance <board>` at a terminal. Neither shows `ground_truth`, held at the
dict boundary like every other route out of this store.

## Two tables that disagreed

`escalations` and `review_decisions` are meant to agree: `resume_review` writes
the verdict and *then* closes the queue entry, so a crash between the two costs
a second look rather than an operator's judgement. Five rows were closed with no
human decision beneath them -- a combination the station cannot produce.

They are the residue of the incident in CLAUDE.md: five regions clicked through
without the domain knowledge to judge them, four of the five labels wrong, and
the labels deleted by hand on 2026-08-22. The queue rows were left closed, so
the two tables spent three days disagreeing about what happened to those
regions.

They are marked, not repaired, on the precedent
`quarantine_fabricated_criteria.py` set the day before. Deleting the queue rows
repeats the mistake that caused this. Re-opening them asserts the first review
never happened, days after the boards left the line. Back-filling a synthetic
human decision invents the label that was deleted, and it would be
indistinguishable from a judgement in the next training round.

So `resolved_unattributed` is a third status -- one the station cannot write,
and one no query treats as pending -- and the reason carries a banner over the
original text, kept verbatim because it is what the operator was shown.
`scripts/mark_unattributed_resolutions.py` applies it, is idempotent, and
`uv run python -m aoi_agent queue` prints the disagreement under the queue.

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

Both pages are behind a sign-in since 2026-08-23. `/ask` is what made that a
precondition rather than a backlog item -- an unauthenticated visitor to the
queue saw the regions on one line, while the same visitor here could pull
statistics for the whole plant -- but the reason the mechanism exists is the
queue's: an operator's answer is the next training round's label, and
`store.boards.record_decision` refuses one that cannot name who made it. See
`station/auth.py` for what the scheme does not protect against.

### Language

The station reads in Traditional Chinese or English. What moves when the switch
moves is chrome: headings, column names, axis labels, the words the progress
panel says. Two string tables in `i18n.py`, chosen by a cookie rather than the
session -- how a person reads a screen is not a claim about who they are, and on
a shared terminal it should outlive the next sign-out.

What does *not* move is the record. The question a supervisor typed stays as
they typed it: translating it would put a question nobody asked beside the name
of the person who asked, and `asked_by` is most of what makes that row worth
keeping. Everything the planning call wrote -- `interpretation`, the
assumptions, each call's `why` -- stays too, because the planning call is never
made again and so what it wrote is a record of how the question was read on the
day. Those sections carry a badge saying so, shown only when the run was asked
in another language.

The synthesised answer is the exception, and the *way* it is the exception is
the design:

```
   stored run            switch to `en`
   ─────────                  │
   question      (frozen)     │
   plan_json     (frozen)     │
   results_json  ─────────────┴──► synthesise(..., lang="en") ──► answers_json
                    the same payload,        one model call,        one more key
                    never re-planned         ~18s                   never overwritten
```

It is **written again, not translated**. A translation is a third artefact,
produced from prose rather than from results, and nothing in this project
measures one -- `synthesis_eval.py` checks a figure in a sentence against the
payload it came from, and a translation has no payload of its own. Re-writing
from `results_json` goes down the identical path the first answer did, one
sentence of the prompt apart.

Re-planning is not an option either, for the reason the `analysis_runs` table
exists: a model asked the same question twice does not produce the same plan,
and a page that redraws differently every time it is opened is not a record.

That is why adding a language makes the measurement *stronger* rather than
diluting it. Two write-ups of one payload are a cross-check a single-language
system cannot perform, so `scripts/synthesis_eval.py --lang both` scores both
surfaces and refuses to publish one alone. The cross-language comparison itself
is a signal for adjudication rather than a gate -- two languages legitimately
quote different subsets of a payload -- and the hard gate stays per language:
every figure in every answer rendering from the results it was written from.

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
| `EXPLANATION_DEADLINE_S` | 60.0 | how long the client waits for a rationale nobody blocks on; 2.8x the slowest of 24 measured calls -- `scripts/latency_report.py` |
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

The last two rows look like one number split in half and are not. Until
2026-08-23 they *were* one number: `RESPONSE_BUDGET_S` was both WI-300's promise
to the operator and the httpx client's timeout, and the two roles pull opposite
ways. The budget is a promise and must not follow the model; a client timeout is
a resource bound and must follow the measurement. Held together at 10s, against
a model whose measured service time has a median of 12.5s, more than half of the
station's LLM calls failed by construction -- and after the LLM came off the
decision path, writing the operator's explanation was the only job it had left.
The queue held an escalation whose entire content was `the model did not answer
(ReadTimeout)`, and nothing counted how many others there were.

So each role now has its own name and its own justification:

- **`RESPONSE_BUDGET_S`** is unchanged at 10s and still read from WI-300, which
  still forbids raising it. It bounds the **verdict** -- the disposition that
  holds or releases a part. That is `classify_node`, measured at 2.5ms per
  candidate on CPU, so the budget is met with three orders of magnitude to
  spare. It lives in `graph/flow.py` now, on the path it describes.
- **`EXPLANATION_DEADLINE_S`** is the client's own bound on waiting for prose.
  Nothing blocks on it: the disposition is decided from `model_class` and
  `model_confidence`, both of which exist before the reason node is entered. It
  is sized from the measured distribution -- 60s is 2.8x the slowest of 24 calls
  on a verified-quiet machine -- and bounded above by the fact that a contended
  GPU can multiply an LLM call's wall time 25x, which no deadline should absorb.
  The previous value in that role, before 10s, was 600s: a busy GPU then meant a
  ten-minute blocked workstation.

WI-300 was corrected to say which of the two it governs. The correction is not
"the station was slower than the document" -- that is the failure this project
spent 2026-08-23 curing, and the budget did not move. It is that WI-300's §1 and
§2 had already moved decision authority to the re-verification model's
calibrated threshold when the agent layer was measured, and the Response budget
section was not revisited, so the two sections described different stations. The
budget bounds the verdict; the rationale is a record item with its own deadline,
and WI-300 now requires its absence to be recorded and counted rather than
substituted.

## Failure directions

Every branch that can fail fails towards a person:

- confidence below `ESCALATE_BELOW` → escalate
- the agent branch reaching a `false_call` it might dismiss → it cannot;
  `ESCALATE_BELOW` equals the dismissal threshold, so that region was
  dismissed upstream or it escalates. Only the calibrated threshold spends
  the escape budget
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
escalated. What is lost is the explanation, and the loss is recorded rather
than papered over: `explanation_status` carries `timed_out`, `unreachable` or
`unparsed`, the station renders a notice from it, and
`uv run python -m aoi_agent explanations` counts them. Until 2026-08-23 the
rationale field carried an exception class name instead -- the queue held an
escalation whose entire content was `the model did not answer (ReadTimeout)` --
and nothing counted how many there had been.
This list said "an unparseable verdict → escalate" until 2026-08-23,
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
| operator identity | ✓ a signed-in name, and how it was established | ✓ the operators themselves, on a demo store |
| the analysis page's figures | ✓ computed from the store, by fixed queries | ✓ over the generated line metadata above |
