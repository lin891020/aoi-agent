# Benchmarks

Every measurement run, newest last. Numbers are on the official DeepPCB test
split unless stated otherwise.

## 2026-08-22 · commit uncommitted

Model: ResNet-18, 3x64x64 (template / test / difference), 10 epochs
Hardware: MacBook Air M5, 32GB, MPS
Test split: 8143 AOI candidates from 499 unseen boards (2997 real defects, 5146 false calls)

### Operating points

Every candidate goes to a human today. The model dismisses the ones it is
confident are false calls; the rest still go to a human.

| escape budget | achieved escape rate | manual review removed | escapes | false calls dismissed |
|---|---|---|---|---|
| ≤0.10% | 0.07% | **26.9%** | 2/2997 | 2186/5146 (42.5%) |
| ≤0.25% | 0.23% | **50.2%** | 7/2997 | 4077/5146 (79.2%) |
| ≤0.50% | 0.47% | **56.2%** | 14/2997 | 4561/5146 (88.6%) |
| ≤1.00% | 0.97% | **60.6%** | 29/2997 | 4907/5146 (95.4%) |
| ≤2.00% | 1.97% | **63.1%** | 59/2997 | 5078/5146 (98.7%) |
| ≤5.00% | 4.97% | **64.7%** | 149/2997 | 5120/5146 (99.5%) |

Overall classification accuracy: 96.5% (reported for reference only — it weighs an escape the same as a false call)

### Whole-line escape rate

The model only sees what the AOI stage flagged. Defects the AOI never
caught are already gone and no threshold recovers them.

- AOI stage misses: **5.0%** of defects
- re-verification adds: 0.47% of what reached it
- **whole line: 5.4%**

The AOI stage dominates. Tuning the model past this point buys nothing
until the detector's recall improves.

### Where the escapes are

At the ≤0.5% budget (threshold 0.915):

| defect class | in test set | escaped | escape rate |
|---|---|---|---|
| open | 594 | 8 | 1.35% |
| short | 451 | 2 | 0.44% |
| mousebite | 550 | 1 | 0.18% |
| spur | 474 | 0 | 0.00% |
| copper | 464 | 2 | 0.43% |
| pin-hole | 464 | 1 | 0.22% |

### Routing — how much of the queue reaches the LLM

Measured over 9070 stored candidates from 500 boards.

| path | candidates | share | LLM involved |
|---|---|---|---|
| dismissed by the vision model | 5062 | 55.8% | no |
| confirmed by the vision model | 2398 | 26.4% | no |
| investigated | 1610 | 17.8% | yes |

**82.2% of candidates never reach a language model.** They are dispositioned by the vision model in tens of milliseconds. The LLM is spent only on the fraction that is genuinely ambiguous, which is what makes a 20B model affordable at line rate.

Escapes on the dismissal path: 15 (0.30% of dismissals).

`open` is routed to investigation regardless of confidence, so it never
appears on the confirm path -- WI-201 calls it the class hardest to separate
from a registration artefact.


### Agent-layer latency — does the reason node fit the response budget?

`gpt-oss:20b` at `think="low"`, 24 real reason-node calls over candidates the router sends to the LLM. Budget is WI-300's 10s.

Latency here is **service time**: Ollama's `total_duration` less `load_duration`. It is not `eval_ms`. Measured on this model, `eval_duration` does not account for thinking tokens at all, and reports under half the time the station waits.

```
ollama ps before the run
NAME           ID              SIZE     PROCESSOR    CONTEXT    UNTIL               
gpt-oss:20b    17052f91a42e    12 GB    100% GPU     32768      28 minutes from now

ollama ps after the run
NAME           ID              SIZE     PROCESSOR    CONTEXT    UNTIL               
gpt-oss:20b    17052f91a42e    12 GB    100% GPU     32768      29 minutes from now
```

| | calls | median | mean | p90 | max |
|---|---|---|---|---|---|
| first 60s | 6 | 9.3s | 9.1s | 10.7s | 10.7s |
| steady state | 18 | 12.5s | 13.2s | 15.6s | 21.1s |
| all | 24 | 11.2s | 12.2s | 15.6s | 21.1s |

**Over budget.** p90 is 15.6s against a 10s budget, and 20 of 24 calls exceeded it. WI-300 says the model is the wrong size for the line, not that the budget moves.

Of that service time, `eval_duration` accounts for 6.4s and prompt ingestion for 0.1s on average. The remaining 5.7s is thinking tokens, which Ollama generates and bills to nobody. Reporting `eval_ms` as the latency would have understated this run by 47%.

Queueing check: 0.0% of mean wall time is not load, prompt or generation — the request went straight to the GPU, so the run is not contended.

No request was served after an eviction.

### Agent layer — does it beat the classifier, and is the escalation calibrated?

`gpt-oss:20b`, 60 candidates the router sends to investigation, sampled by stride across the store. `fragment` ground truth is held out, as in training. Ran in 0 min.

**Accuracy against the classifier it second-guesses**

| | candidates | vision model | agent |
|---|---|---|---|
| all investigated | 60 | 51/60 = 85.0% | 43/60 = 71.7% |
| agent kept | 23 | 21/23 = 91.3% | 20/23 = 87.0% |
| agent escalated | 37 | 30/37 = 81.1% | 23/37 = 62.2% |

**Calibration.** The agent's verdicts were right 87.0% of the time on what it kept and 62.2% of the time on what it handed over, a gap of +24.8%. The escalations land on the harder cases, which is what the confidence flag is for.

**Where the agent overrode the classifier.** It changed the class on 12 of 60 candidates. The agent was right 1 of those times; the classifier had already been right 9 times, and 2 were wrong either way. Re-classification is costing accuracy rather than adding it. The layer's value is the escalation flag, and taking the classifier's class whenever the agent does not escalate would score 21/23 = 91.3% on the kept set, against the agent's 87.0%.

**Escalation rate.** 37/60 = 61.7% of investigated candidates, which is 11.0% of the whole queue.

**Escapes.** 0 of 23 kept candidates were called `false_call` while carrying a real defect.

Distribution of what the agent said, against the truth:

| truth | n | agent agreed | agent escalated |
|---|---|---|---|
| open | 27 | 27 | 15 |
| false_call | 27 | 14 | 19 |
| short | 3 | 1 | 1 |
| mousebite | 1 | 0 | 1 |
| spur | 1 | 1 | 0 |
| copper | 1 | 0 | 1 |

### Agent layer after the change — routing on the classifier's confidence

`ESCALATE_BELOW = 0.90` replaced the LLM's `confident` flag, and `decide_node` now takes the classifier's class. Same procedure as the run above.


`gpt-oss:20b`, 30 candidates the router sends to investigation, sampled by stride across the store. `fragment` ground truth is held out, as in training. Ran in 0 min.

**What the system dispositions on, against what the LLM would have dispositioned on.** `decide_node` takes the classifier's class; the agent column is the counterfactual it replaced.

| | candidates | system (classifier) | LLM counterfactual |
|---|---|---|---|
| all investigated | 30 | 27/30 = 90.0% | 22/30 = 73.3% |
| agent kept | 15 | 15/15 = 100.0% | 15/15 = 100.0% |
| agent escalated | 15 | 12/15 = 80.0% | 7/15 = 46.7% |

**Calibration of the hand-off.** The LLM's verdicts were right 100.0% of the time on what it kept and 46.7% of the time on what it handed over, a gap of +53.3%. The escalations land on the harder cases, which is what the confidence flag is for.

**Where the LLM would have overridden the classifier.** It proposed a different class on 5 of 30 candidates. The agent was right 0 of those times; the classifier had already been right 5 times, and 0 were wrong either way. Acting on those proposals would cost accuracy rather than add it, which is why the flow does not. On the kept set the classifier scores 15/15 = 100.0% against the LLM's 100.0%.

**Escalation rate.** 15/30 = 50.0% of investigated candidates, which is 8.9% of the whole queue.

**Escapes.** 0 of 15 kept candidates were called `false_call` while carrying a real defect.

Distribution of what the agent said, against the truth:

| truth | n | agent agreed | agent escalated |
|---|---|---|---|
| open | 14 | 14 | 1 |
| false_call | 12 | 6 | 12 |
| short | 2 | 1 | 1 |
| mousebite | 1 | 0 | 1 |
| spur | 1 | 1 | 0 |

### Analysis planner — does it plan the right lookups, and refuse the rest?

`gpt-oss:20b`, 20 hand-written questions, each asked 3 times. Plans are scored, not answers: the tools are deterministic, so a correct plan yields correct data by construction and the errors live in the plan. The store held 9 days at the time of the run.

Re-run after the scorer changed: it now checks the argument values a question names (`defect_type`, `line_id`, `machine_id`, `board`) and not tool names alone, and the `哪一台機器的缺陷率最高？` expectation was corrected — `query_machine_stats` compares one defect class at a time and cannot rank machines by their overall rate. The earlier 20/20 figures are not carried forward: they came from a scorer that would have passed a plan querying the wrong defect class, so the two are not comparable.

| | questions | correct |
|---|---|---|
| should answer | 13 | 12/13 = 92% |
| should refuse | 7 | 7/7 = 100% |
| determinism | 20 | 20/20 = 100% planned the same tools across 3 runs |

**Held out from the prompt.** 5 of the 20 questions are few-shot examples verbatim or near-paraphrases, so on those the model is reciting rather than planning. On the remaining 15 it scored 14/15 = 93%, with 15/15 = 100% stable. That is the number to read; the headline above is the optimistic one.

**Plans `validate_plan` threw out.** 0 of 20 did not validate, 0 of which had scored a hit on tools and arguments and so would be counted correct above while running nothing. The usual cause is a `days` beyond the 9 the store holds. Scoring counts the plan, so these are reported here rather than folded into the table.

**Planner failures.** 0 question(s) produced no plan at all (model unreachable, or a response that would not parse). These score as misses, not as refusals: a timeout that counted as a refusal would make a contended machine look well-calibrated.

Misses:

- 哪一台機器的缺陷率最高？ — never called ['query_defect_history']

Refusal accuracy carries more weight than the count suggests. A planner that answers everything is more dangerous on a line than one that says it cannot, and nothing else in this project measures that.

**What "correct" means here, exactly.** A hit is: the plan called every tool the question needs, and passed the argument values the question named for `defect_type`, `line_id`, `machine_id` and `board`. Three things that leaves open. It is recall without precision — extra tools are free, so a planner that called all five tools on every answerable question would score full marks on the first row. `days` and `top_k` go unscored, since no question pins a window and `validate_plan` already bounds `days`. And a refusal is read off an empty `calls` list and nothing else, so "refused because it is a write against the database" and "refused as too vague" are the same event to this scorer — which is most of what makes the `把 candidates 資料表刪掉` row less reassuring than it looks.

What this does not establish: the expected plans and the few-shot examples have the same author, so this is agreement with one opinion of the right plan and not an independent ground truth. It is a single point, not an operating-point curve, and it says nothing about whether the prose written over correct data is correct. Both are recorded in the design rather than solved.
