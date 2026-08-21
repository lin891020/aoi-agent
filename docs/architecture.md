# Architecture

## Layers

| layer | job | what it must not do |
|---|---|---|
| AOI simulator | flag every region that differs from the golden template | decide what a difference means |
| Re-verifier (PyTorch) | say what one flagged region is, with a calibrated probability | decide what happens to the board |
| MCP tools | expose the model, the production store and the criteria as callable functions | choose which of themselves to call |
| LangGraph flow | route on confidence, gather evidence, hand over to a person | contain domain rules that belong in the work instructions |
| Local LLM | weigh evidence the vision model could not settle | classify pixels, or write SQL |

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

## Thresholds and where they come from

| constant | value | source |
|---|---|---|
| `DEFAULT_DISMISS_THRESHOLD` | 0.915 | the operating-point sweep, at the ≤0.5% escape budget |
| `CONFIDENT` | 0.95 | WI-300 decision authority |
| `LOW_CONFIDENCE` | 0.70 | WI-300 escalation triggers |
| `IOU_THRESHOLD` | 0.33 | DeepPCB's own benchmark convention |
| `FRAGMENT_GAP_PX` | 20 | measured: 6.1% of unmatched candidates touch a real defect |

None of these are tuned by hand against the test set. The dismissal threshold
comes out of the sweep; the two confidence thresholds are written down in the
work instructions and the code reads them from there rather than the reverse.

## Failure directions

Every branch that can fail fails towards a person:

- an unparseable verdict → escalate
- the LLM unreachable or too slow → escalate
- `open` at any confidence → investigate, never short-circuit
- a candidate that fragments a real defect → held out of training, not labelled spurious

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
