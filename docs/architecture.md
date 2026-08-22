# Architecture

## Layers

| layer | job | what it must not do |
|---|---|---|
| AOI simulator | flag every region that differs from the golden template | decide what a difference means |
| Re-verifier (PyTorch) | say what one flagged region is, with a calibrated probability | decide what happens to the board |
| MCP tools | expose the model, the production store and the criteria as callable functions | choose which of themselves to call |
| LangGraph flow | route on confidence, gather evidence, hand over to a person | contain domain rules that belong in the work instructions |
| Review station | show a person the evidence the agent had, and take their answer | re-run the flow to render a page, or show the ground truth |
| Local LLM | explain to an operator why a region reached them | decide anything -- measured worse than the classifier at both the class and the hand-off |

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

## Thresholds and where they come from

| constant | value | source |
|---|---|---|
| `DEFAULT_DISMISS_THRESHOLD` | 0.915 | the operating-point sweep, at the ≤0.5% escape budget |
| `CONFIDENT` | 0.95 | WI-300 decision authority |
| `ESCALATE_BELOW` | 0.90 | lowest threshold adding no escape to the budget; see docs/benchmarks.md |
| `LOW_CONFIDENCE` | 0.70 | WI-300 escalation triggers |
| `RESPONSE_BUDGET_S` | 10.0 | WI-300 response budget, derived from QP-110 |
| `IOU_THRESHOLD` | 0.33 | DeepPCB's own benchmark convention |
| `FRAGMENT_GAP_PX` | 20 | measured: 6.1% of unmatched candidates touch a real defect |

None of these are tuned by hand against the test set. The dismissal threshold
comes out of the sweep; the confidence thresholds and the response budget are
written down in the work instructions and the code reads them from there rather
than the reverse. `tests/test_response_budget.py` fails if the constant and
WI-300 stop agreeing.

The response budget is the one that most invites being quietly raised, so
WI-300 states the direction explicitly: if the configured model cannot answer
within it, the model is the wrong size for the line. Whether `gpt-oss:20b` fits
inside 10s is an open measurement -- see docs/benchmarks.md.

## Failure directions

Every branch that can fail fails towards a person:

- an unparseable verdict → escalate
- the LLM unreachable or too slow → escalate
- `open` at any confidence → investigate, never short-circuit
- a candidate that fragments a real defect → held out of training, not labelled spurious
- the process holding a suspended run dies → the run is on disk, the queue still lists it

An escalation costs an operator a few seconds. The alternatives cost a shipped
board or a silently mislabelled training set.

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
