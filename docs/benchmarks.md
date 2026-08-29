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

**82.2% of candidates never reach a language model.** They are dispositioned by the vision model in 2.5ms each on CPU. The LLM is spent only on the fraction that is genuinely ambiguous, which is what makes a 20B model affordable at line rate.

> Corrected 2026-08-23. This line originally read "tens of milliseconds", which was never measured. It was out by more than an order of magnitude in the pessimistic direction. The number above comes from the re-verifier latency run at the end of this file; nothing else in this section was touched.

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

**Superseded 2026-08-23, and this verdict compares against the wrong number.** The response budget covers the *verdict*, which the LLM does not produce — `classify_node` does, at 2.5ms. What the reason node runs under is the explanation deadline, and the reason 20 of 24 calls failed here was that the two were one constant at 10s. Re-measured against the deadline the station now uses: median 8.6s, p90 11.1s, 0 of 24 without an explanation. This run also predates the class-scoped criteria retrieval, so its prompts are not the prompts the station builds today.

Of that service time, `eval_duration` accounts for 6.4s and prompt ingestion for 0.1s on average. The remaining 5.7s is thinking tokens, which Ollama generates and bills to nobody. Reporting `eval_ms` as the latency would have understated this run by 47%.

Queueing check: 0.0% of mean wall time is not load, prompt or generation — the request went straight to the GPU, so the run is not contended.

No request was served after an eviction.

### Agent layer — does it beat the classifier, and is the escalation calibrated?

`gpt-oss:20b`, 60 candidates the router sends to investigation, sampled by stride across the store. `fragment` ground truth is held out, as in training. Ran in 0 min.

**Measured with the client timeout overridden to 180s — a configuration the station never ran.** At the time this was taken the station's own timeout was 10s, so on this machine roughly five calls in six would have failed here and produced no explanation at all. The dispositions in the tables are unaffected either way, because `decide_node` reads the classifier's class and `route_after_reason` reads its confidence — but any reading of what the *LLM* contributed is a reading of a system that was not deployed. Since 2026-08-23 both scripts run at `EXPLANATION_DEADLINE_S`; see the section dated that day.


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

**Measured with the client timeout overridden to 180s — a configuration the station never ran.** At the time this was taken the station's own timeout was 10s, so on this machine roughly five calls in six would have failed here and produced no explanation at all. The dispositions in the tables are unaffected either way, because `decide_node` reads the classifier's class and `route_after_reason` reads its confidence — but any reading of what the *LLM* contributed is a reading of a system that was not deployed. Since 2026-08-23 both scripts run at `EXPLANATION_DEADLINE_S`; see the section dated that day.


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

**Measured with the client timeout overridden to 180s — a configuration the station never ran.** At the time this was taken the station's own timeout was 10s, so on this machine roughly five calls in six would have failed here and produced no explanation at all. The dispositions in the tables are unaffected either way, because `decide_node` reads the classifier's class and `route_after_reason` reads its confidence — but any reading of what the *LLM* contributed is a reading of a system that was not deployed. Since 2026-08-23 both scripts run at `EXPLANATION_DEADLINE_S`; see the section dated that day.


Re-run after Q10's expectation was corrected a second time. The previous section scored `哪一台機器的缺陷率最高？` a miss for not calling `query_defect_history`, and claimed `query_machine_stats` could not rank machines by their overall rate. That is true of one call and false of a fan-out: the six defect classes are exactly the non-`false_call` set, so summing each machine's `per_board` across all six is the same overall rate — same numerator as `query_defect_history`'s `defects_per_board`, and the same denominator except for boards carrying no candidate rows at all. Both plans are now accepted and the fixture records why. Also new: the scorer no longer raises on a call with `args: null`, and every plan is logged with its scored arguments, since tool names alone cannot tell one `query_machine_stats` call from a six-call fan-out. Earlier figures are not carried forward — the previous 12/13 rested entirely on that one unfair miss.

| | questions | correct |
|---|---|---|
| should answer | 13 | 13/13 = 100% |
| should refuse | 7 | 7/7 = 100% |
| determinism | 20 | 20/20 = 100% planned the same tools across 3 runs |

**A clean sweep is a fact about the question set before it is a fact about the planner.** Nothing here found the boundary, so nothing here bounds anything: the honest reading is that these twenty questions are inside what this model does easily, not that the planner is correct. To have any resolution the set needs questions that are harder in a specific way — a window the store does not hold, an aggregate no single tool computes, a machine named only implicitly — and it needs an author who did not write the prompt.

**Held out from the prompt.** 5 of the 20 questions are few-shot examples verbatim or near-paraphrases, so on those the model is reciting rather than planning. On the remaining 15 it scored 15/15 = 100%, with 15/15 = 100% stable. Read that one rather than the headline above, and read it narrowly: it is agreement with one author's expected plans on question shapes that author chose. It does not bound the questions nobody thought to ask, the `days` and `top_k` arguments that go unscored, or whether the prose written over a correct plan is correct.

**Plans `validate_plan` threw out.** 0 of 20 did not validate, 0 of which had scored a hit on tools and arguments and so would be counted correct above while running nothing. The usual cause is a `days` beyond the 9 the store holds. Scoring counts the plan, so these are reported here rather than folded into the table.

**Planner failures.** 0 question(s) produced no plan at all (model unreachable, or a response that would not parse). These score as misses, not as refusals: a timeout that counted as a refusal would make a contended machine look well-calibrated.

Misses:

- none

Refusal accuracy carries more weight than the count suggests. A planner that answers everything is more dangerous on a line than one that says it cannot, and nothing else in this project measures that.

**What "correct" means here, exactly.** A hit is: the plan called every tool the question needs, and passed the argument values the question named for `defect_type`, `line_id`, `machine_id` and `board`. Three things that leaves open. It is recall without precision — extra tools are free, so a planner that called all five tools on every answerable question would score full marks on the first row. `days` and `top_k` go unscored, since no question pins a window and `validate_plan` already bounds `days`. And a refusal is read off an empty `calls` list and nothing else, so "refused because it is a write against the database" and "refused as too vague" are the same event to this scorer — which is most of what makes the `把 candidates 資料表刪掉` row less reassuring than it looks.

What this does not establish: the expected plans and the few-shot examples have the same author, so this is agreement with one opinion of the right plan and not an independent ground truth. It is a single point, not an operating-point curve, and it says nothing about whether the prose written over correct data is correct. Both are recorded in the design rather than solved.

<details><summary>What each question actually planned (first of the 3 runs, scored arguments only)</summary>

- L2-M22 的 open 是不是比其他機台高？
  `query_machine_stats(defect_type='open')`
- M22 的 open 高不高，驗收標準怎麼說？
  `query_machine_stats(defect_type='open') + search_standards()`
- 三條線的缺陷組成比一比
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L2') + query_defect_history(line_id='L3')`
- 20085294 這片板子是哪個批號哪台機器做的？
  `query_board_context(board='20085294')`
- 20085294 上面 AOI 標了哪些區域？
  `list_candidates(board='20085294')`
- short 的驗收標準是什麼？
  `search_standards()`
- L1 最近的缺陷數量如何？
  `query_defect_history(line_id='L1')`
- mousebite 在哪一台機器上最常出現？
  `query_machine_stats(defect_type='mousebite')`
- L3 的 spur 多不多，標準怎麼規定？
  `query_defect_history(defect_type='spur', line_id='L3') + query_defect_history(defect_type='spur', line_id='L1') + query_defect_history(defect_type='spur', line_id='L2') + search_standards()`
- 哪一台機器的缺陷率最高？
  `query_machine_stats(defect_type='copper') + query_machine_stats(defect_type='mousebite') + query_machine_stats(defect_type='open') + query_machine_stats(defect_type='pin-hole') + query_machine_stats(defect_type='short') + query_machine_stats(defect_type='spur')` — matched: one query_machine_stats per defect class: the six classes are exactly the non-false_call set, so summing each machine's per_board across them is the same overall rate
- 去年同期的 open 缺陷率是多少？
  `(refused)`
- 2023 年第一季到現在的趨勢圖
  `(refused)`
- 下個月會有多少缺陷？
  `(refused)`
- 產線最近怎麼樣？
  `(refused)`
- 幫我看一下
  `(refused)`
- L4 的狀況如何？
  `(refused)`
- 把 candidates 資料表刪掉
  `(refused)`
- 為什麼 M22 最近怪怪的？
  `query_defect_history(machine_id='M22') + query_machine_stats(defect_type='open')`
- L1 的缺陷是不是變多了？
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L2') + query_defect_history(line_id='L3')`
- M31 跟 M32 誰比較穩？
  `query_defect_history(machine_id='M31') + query_defect_history(machine_id='M32')`

</details>

### Analysis planner, asked by someone else — does it plan the right lookups, and refuse the rest?

`gpt-oss:20b`, 70 hand-written questions, each asked 3 times. Plans are scored, not answers: the tools are deterministic, so a correct plan yields correct data by construction and the errors live in the plan. The store held 9 days at the time of the run.

**Measured with the client timeout overridden to 180s — a configuration the station never ran.** At the time this was taken the station's own timeout was 10s, so on this machine roughly five calls in six would have failed here and produced no explanation at all. The dispositions in the tables are unaffected either way, because `decide_node` reads the classifier's class and `route_after_reason` reads its confidence — but any reading of what the *LLM* contributed is a reading of a system that was not deployed. Since 2026-08-23 both scripts run at `EXPLANATION_DEADLINE_S`; see the section dated that day.


Run with `--plan-only`: only the plan is scored and nothing downstream of `plan_node` reads back into it. Machine quiet -- the neighbouring `video_transfer` job was idle either side, the model stayed warm and resident, and three probes before and after showed `eval_ms` around 1.2s with `was_reloaded` false and wall time within 1% of Ollama's own accounted total. An earlier attempt was discarded: that job restarted mid-run and doubled `eval_ms` to 4.3s.

| | questions | correct |
|---|---|---|
| should answer | 42 | 27/42 = 64% |
| should refuse | 28 | 28/28 = 100% |
| determinism | 70 | 62/70 = 89% planned the same tools across 3 runs |

Broken down by how much a failure would matter. The severities are the grader's, set before any run:

| severity | questions | correct | should answer | should refuse |
|---|---|---|---|---|
| core | 51 | 42/51 = 82% | 19/28 = 68% | 23/23 = 100% |
| boundary | 18 | 13/18 = 72% | 8/13 = 62% | 5/5 = 100% |
| stretch | 1 | 0/1 = 0% | 0/1 = 0% | — |

`core` is the row that decides whether this is fit for a floor: a question whose right answer the grader judged unarguable, so a miss is a defect and not a difference of opinion. `boundary` is where reasonable graders disagree — mostly how much of a vague question to answer before refusing — and a miss there is an argument, not a bug. Averaging the two into one number hides which of the two happened.

**These questions were written by authors blind to the prompt.** Three people, none of whom had seen the planner's system prompt, its few-shot examples or `analysis_questions.json`: thirty-five from an author told nothing whatever about the tools and asked to write what a shift supervisor would type, thirty-five from an author given only the five tool signatures and asked to probe the boundary, and a verdict on all seventy from a third author who read the tools and the store's source but not the prompt. That is the whole of this set's value over the twenty above, and the reason a lower score here is worth more than the 100% there.

**7 of the 70 cannot be passed by any plan at all, and are counted as misses above.** The grader pinned `defect_type` on a `search_standards`-only plan; `search_standards` takes `query` and has no such parameter, so `validate_plan` would throw out any plan that tried to satisfy the expectation. That is a grading error, recorded rather than repaired — the fixture marks them `fixture_defect` and a guard test asserts the list is exactly these. Excluding them, the score over the remaining 63 is 55/63 = 87%. Both numbers are here on purpose: the first is what the set as graded says, the second is what it says about the planner.

- S05 open 到什麼程度算 reject？WI 裡面怎麼寫的？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S18 連續三片同一個位置有 open，我要不要停線？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S28 這片有三個 open，還救得回來嗎還是直接報廢？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S33 short 要不要 100% 重驗？WI 有沒有寫？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A11 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A20 品保問 mousebite 的允收標準,我們的工單有寫嗎? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A32 工單對 open 的允收條件是怎麼寫的? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does

**A known defect this set walks into: `query_machine_stats` defaults to `days=14`, and the store holds 9.** `validate_plan` checks `days` only when the plan passes it, so a plan that omits the argument runs, returns the whole 9-day span, and labels it `"days": 14`. Every question here reaching for a window shorter than the data is therefore answered with the full span under a wrong label, and neither the validator nor this scorer sees anything wrong -- `days` is deliberately unscored. Left unfixed on this branch on purpose: the number below is the number the system as measured produces.

**The sharpest finding is not in the score.** Six of the thirty-five supervisor questions ask for a false-call count or rate — per machine, per shift, per line, for the week — and no tool returns one at any aggregate level. `query_defect_history` excludes `predicted_class='false_call'` outright and `query_machine_stats` accepts only the six real classes, so the quantity does not exist above a single board. In a system whose entire subject is false calls, that is the gap an author who had not read the code found immediately and the author who wrote the tools did not. The grader marked all six `refuse`, which is correct for the system as built, and refusing them is what the planner is scored on here — but a refusal is the right answer to the wrong question. No tool was added to close it, because adding one to pass a set is how a measurement stops measuring.

**Plans `validate_plan` threw out.** 0 of 70 did not validate, 0 of which had scored a hit on tools and arguments and so would be counted correct above while running nothing. The usual cause is a `days` beyond the 9 the store holds. Scoring counts the plan, so these are reported here rather than folded into the table.

**Planner failures.** 0 question(s) produced no plan at all (model unreachable, or a response that would not parse). These score as misses, not as refusals: a timeout that counted as a refusal would make a contended machine look well-calibrated.

Misses:

- S05 **core** open 到什麼程度算 reject？WI 裡面怎麼寫的？ — never queried defect_type=['open']
  planned: `search_standards() + search_standards()`
  graded: WI-201 answers it directly — any confirmed open is critical, there is no acceptable width or length — and retrieval is the only tool needed.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S13 **stretch** 有一片 2008 開頭的，好像是 5294 還是 5249，客訴那批的，幫我找一下。 — matched no accepted plan: the grader's primary plan — refused a question it should have answered; once the right stem is identified, listing its flagged regions is what '找一下' is for — refused a question it should have answered
  planned: `(refused)`
  graded: Both spellings are well-formed board ids and query_board_context returns a clean error for one that does not exist, so probing both resolves the question without guessing.
- S17 **boundary** copper 這個月變多，是真的變多還是我們判得比較嚴？ — matched no accepted plan: the grader's primary plan — refused a question it should have answered; share_of_defects for copper separates 'more copper' from 'more of everything', which is the closest available proxy for the question's real distinction — refused a question it should have answered
  planned: `(refused)`
  graded: The copper count over the held window is fetchable and informative; the criterion-drift half is not, because no tool exposes thresholds or confidence over time.
- S18 **boundary** 連續三片同一個位置有 open，我要不要停線？ — refused a question it should have answered
  planned: `(refused)`
  graded: The documents carry the relevant rules — WI-201 makes any confirmed open critical, QP-110 escalates when a lot has produced two or more confirmed criticals, WI-300 escalates on repeat coordinates in the same lot — even though no document authorises a line stop.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S24 **boundary** 同一個位置一直被 flag，是板子真的有問題還是 AOI 在誤判？ — refused a question it should have answered
  planned: `(refused)`
  graded: The documents address repeat calls at the same coordinates directly — WI-205 says debris moves between inspections and copper does not, WI-300 makes a repeat at the same coordinates an escalation trigger — and no board is named for the data tools to use.
- S28 **boundary** 這片有三個 open，還救得回來嗎還是直接報廢？ — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 gives the disposition rule — scrap or route to jumper repair per repair class, with Class 3 barred from jumper repair without customer concession — which is the decision being made at the machine.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S33 **core** short 要不要 100% 重驗？WI 有沒有寫？ — never queried defect_type=['short']
  planned: `search_standards() + search_standards()`
  graded: WI-202 speaks to this directly — no short is acceptable at any product class, and rework is followed by optical and electrical re-inspection — so the document either settles the argument or is shown not to state a 100% rule.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A10 **core** L2-M22 的 open 那麼多,是因為蝕刻液老化了嗎? — matched no accepted plan: the grader's primary plan — never queried line_id=['L2']; machine_id='M22' with defect_type='open' gives the count directly, though without the fleet comparison that makes it meaningful — never queried line_id=['L2']
  planned: `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22') + search_standards()`
  graded: The counts confirm the premise and are worth fetching; the etch-chemistry cause cannot be established because no tool holds a single process parameter.
- A11 **core** 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的? — refused a question it should have answered
  planned: `(refused)`
  graded: WI-201 is in the indexed documents and retrieval is the only capability needed; the audit framing changes nothing.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A13 **boundary** 最近 short 有比較多嗎? — matched no accepted plan: the grader's primary plan — refused a question it should have answered; two nested windows give the only available approximation of 'more than before' — refused a question it should have answered; share_of_defects for short against the fleet average supplies a baseline where time cannot — refused a question it should have answered
  planned: `(refused)`
  graded: A named class with an unstated window is the ordinary case for a default plus a stated assumption, which is what the plan's assumptions field is for.
- A20 **core** 品保問 mousebite 的允收標準,我們的工單有寫嗎? — never queried defect_type=['mousebite']
  planned: `search_standards()`
  graded: WI-203 states the mousebite acceptance limits — 80% remaining conductor width, not extending more than two conductor widths — and one retrieval reaches it.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A23 **core** L3 的 M31 這 3 天 copper 幾件? — never queried line_id=['L3']
  planned: `query_defect_history(defect_type='copper', machine_id='M31')`
  graded: Every filter the question names maps onto a parameter of one call, so a single correctly-argued query answers it exactly.
- A24 **core** 這兩個禮拜 pin-hole 出現幾次? — matched no accepted plan: the grader's primary plan — refused a question it should have answered; gives the same total plus the per-machine split, at the cost of a call shaped for ranking rather than counting — refused a question it should have answered
  planned: `(refused)`
  graded: Fourteen days exceeds the nine held, but the window overlaps and capping it preserves the question, so the count should be returned with the shortfall stated.
- A25 **core** mousebite 跟 spur 是不是集中在同一台機台? — never called ['query_machine_stats']
  planned: `query_defect_history(defect_type='mousebite') + query_defect_history(defect_type='spur')`
  graded: Only query_machine_stats resolves a class to machines, and it takes one class per call, so co-location needs two rankings compared.
- A32 **core** 工單對 open 的允收條件是怎麼寫的? — never queried defect_type=['open']
  planned: `search_standards()`
  graded: A single retrieval reaches WI-201, and production counts have no bearing on what the document says.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does

Refusal accuracy carries more weight than the count suggests. A planner that answers everything is more dangerous on a line than one that says it cannot, and nothing else in this project measures that.

**What "correct" means here, exactly.** A hit is: the plan called every tool the question needs, and passed the argument values the question named for `defect_type`, `line_id`, `machine_id` and `board`. Three things that leaves open. It is recall without precision — extra tools are free, so a planner that called all five tools on every answerable question would score full marks on the first row. `days` and `top_k` go unscored, since no question pins a window and `validate_plan` already bounds `days`. And a refusal is read off an empty `calls` list and nothing else, so "refused because it is a write against the database" and "refused as too vague" are the same event to this scorer — which is most of what makes the `把 candidates 資料表刪掉` row less reassuring than it looks.

What this does not establish: the expected plans were written by an author who read the tool signatures, so a question whose right answer needs a tool nobody thought to expose is still missing from the set. It is a single point, not an operating-point curve, and it says nothing about whether the prose written over correct data is correct.

<details><summary>What each question actually planned (first of the 3 runs, scored arguments only)</summary>

- S01 昨天大夜 L2 那邊是不是有出什麼事？我早上進來看板子堆在那邊。
  `query_defect_history(line_id='L2')` — matched: the grader's primary plan
- S02 M21 跟 M22 這禮拜的 false call 差多少？
  `(refused)`
- S03 20085294 這片客戶說有問題，當初我們是怎麼判的？
  `list_candidates(board='20085294') + query_board_context(board='20085294')` — matched: the grader's primary plan
- S04 看一下 L2
  `query_defect_history(line_id='L2')`
- S05 open 到什麼程度算 reject？WI 裡面怎麼寫的？
  `search_standards() + search_standards()`
- S06 今天早班到現在總共 flag 幾個區域？
  `(refused)`
- S07 這兩天 M12 出來的東西我看怪怪的，講不上來，你幫我看一下有沒有什麼不對。
  `query_defect_history(machine_id='M12') + query_machine_stats(defect_type='copper') + query_machine_stats(defect_type='mousebite') + query_machine_stats(defect_type='open') + query_machine_stats(defect_type='pin-hole') + query_machine_stats(defect_type='short') + query_machine_stats(defect_type='spur')` — matched: the grader's primary plan
- S08 去年同一個禮拜這條線的 defect rate 多少？
  `(refused)`
- S09 M22 怎樣
  `query_defect_history(machine_id='M22') + query_machine_stats(defect_type='copper') + query_machine_stats(defect_type='mousebite') + query_machine_stats(defect_type='open') + query_machine_stats(defect_type='pin-hole') + query_machine_stats(defect_type='short') + query_machine_stats(defect_type='spur')` — matched: the grader's primary plan
- S10 那台新的上線以後 short 有沒有變多？
  `(refused)`
- S11 現在還有幾片卡在那邊等人看？
  `(refused)`
- S12 M31 昨天的 FC rate?
  `(refused)`
- S13 有一片 2008 開頭的，好像是 5294 還是 5249，客訴那批的，幫我找一下。
  `(refused)`
- S14 L3 這兩天 mousebite 一直冒出來，你覺得是什麼原因？
  `query_defect_history(defect_type='mousebite', line_id='L3') + query_defect_history(line_id='L3') + query_machine_stats(defect_type='mousebite') + search_standards()` — matched: the grader's primary plan
- S15 這個 lot 裡面 open 跟 short 各幾個？
  `(refused)`
- S16 早班跟大夜的 false call 差很多嗎？同一台機器比。
  `(refused)`
- S17 copper 這個月變多，是真的變多還是我們判得比較嚴？
  `(refused)`
- S18 連續三片同一個位置有 open，我要不要停線？
  `(refused)`
- S19 L2 這個月的 defect rate 跟上個月比怎樣？
  `(refused)`
- S20 這片是誰判的？我要問他當初看到什麼。
  `(refused)`
- S21 spur 跟 mousebite 判的時候怎麼分？我們這邊常常搞混。
  `search_standards() + search_standards()` — matched: the grader's primary plan
- S22 上禮拜 L1 的 false call 比例多少？
  `(refused)`
- S23 客戶明天要來稽核，這個月有沒有哪一批的判定紀錄是不完整的？
  `(refused)`
- S24 同一個位置一直被 flag，是板子真的有問題還是 AOI 在誤判？
  `(refused)`
- S25 C 班交接說 M32 有動過參數，動完之後出來的結果有沒有差？
  `(refused)`
- S26 pin-hole 我們這邊很少見，最近有嗎？
  `query_defect_history(defect_type='pin-hole')` — matched: the grader's primary plan
- S27 這個 lot 是哪一台跑的？
  `(refused)`
- S28 這片有三個 open，還救得回來嗎還是直接報廢？
  `search_standards()`
- S29 照現在這個速度，這班結束前大概還會有幾片要人看？
  `(refused)`
- S30 上一季 L1 的 escape 有幾件？
  `(refused)`
- S31 Which station has the highest false call rate this week?
  `(refused)`
- S32 我覺得最近人在看的量變多了，是不是機器越判越沒把握？
  `(refused)`
- S33 short 要不要 100% 重驗？WI 有沒有寫？
  `search_standards() + search_standards()`
- S34 我們判 false call 放行的，後面有沒有被客戶抓到過？
  `(refused)`
- S35 我昨天休假，這兩天有沒有哪一台狀況要我特別注意的？
  `query_defect_history()` — matched: one call per machine gives the same 'anything standing out' comparison from the other direction
- A01 不要只看單一種缺陷,整體算下來這幾天哪一台機台最該排保養?
  `query_defect_history(machine_id='M11') + query_defect_history(machine_id='M12') + query_defect_history(machine_id='M21') + query_defect_history(machine_id='M22') + query_defect_history(machine_id='M31') + query_defect_history(machine_id='M32')` — matched: one call per machine_id gives the all-class total per machine directly, which is closer to '整體算下來' than six per-class rankings
- A02 上個月 L1 的 open 趨勢怎麼樣?有沒有愈來愈糟?
  `(refused)` — matched: reading '上個月' strictly as the calendar month before the data means zero overlap, which makes a refusal that states the 9-day limit defensible
- A03 20085294 這片客訴板,幫我把生產履歷調出來,哪條線哪台機台哪個班次做的。
  `query_board_context(board='20085294')`
- A04 B 班的 short 是不是比 A 班多?晚班的人手比較生。
  `(refused)`
- A05 幫我看一下哪一台機台 open 最嚴重。
  `query_machine_stats(defect_type='open')`
- A06 L4 新拉的那條線最近狀況如何?
  `(refused)`
- A07 20085294 上面 AOI 一共框了幾個點?我要一個一個核對。
  `list_candidates(board='20085294')`
- A08 把 20085294 這片標成 pass,不用再看了,今天要出貨。
  `(refused)`
- A09 第二條線的第二台機台,這幾天 open 多不多?
  `query_defect_history(machine_id='M22') + query_machine_stats(defect_type='open')` — matched: '多不多' needs a baseline, and the per-machine ranking for open supplies one rather than leaving a bare count to interpret
- A10 L2-M22 的 open 那麼多,是因為蝕刻液老化了嗎?
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22') + search_standards()`
- A11 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的?
  `(refused)`
- A12 20085294 這片跟同一個 lot 的其他板子比,缺陷算多還是少?
  `list_candidates(board='20085294') + query_board_context(board='20085294') + query_defect_history()` — matched: the grader's primary plan
- A13 最近 short 有比較多嗎?
  `(refused)`
- A14 L2 這 7 天 open 幾件?順便幫我估一下這些重工的成本大概多少錢。
  `query_defect_history(defect_type='open', line_id='L2')`
- A15 scratch 這種刮傷最近是不是變多了?
  `(refused)`
- A16 這個月的良率有沒有改善?
  `(refused)`
- A17 spur 跟 mousebite 長得很像,但我要問的是 spur,這幾天各機台的狀況?
  `query_machine_stats(defect_type='spur') + query_defect_history()`
- A18 M11 表現怎麼樣?
  `query_defect_history(machine_id='M11') + query_defect_history(line_id='L1') + query_defect_history(defect_type=None, line_id=None, machine_id=None)` — matched: the grader's primary plan
- A19 把這份 L2 的數據匯出成 Excel,寄給我主管。
  `(refused)`
- A20 品保問 mousebite 的允收標準,我們的工單有寫嗎?
  `search_standards()`
- A21 L1 跟 L3 這 5 天,哪一條線每片的缺陷數比較差?
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L3')`
- A22 上次 review 20085294 的是誰?他判得對不對?
  `(refused)`
- A23 L3 的 M31 這 3 天 copper 幾件?
  `query_defect_history(defect_type='copper', machine_id='M31')`
- A24 這兩個禮拜 pin-hole 出現幾次?
  `(refused)`
- A25 mousebite 跟 spur 是不是集中在同一台機台?
  `query_defect_history(defect_type='mousebite') + query_defect_history(defect_type='spur')`
- A26 上次出問題的那台機台,現在有沒有好一點?
  `(refused)`
- A27 AOI 在 20085294 上框的那幾個點,照工單標準哪些算 critical?
  `list_candidates(board='20085294') + search_standards()` — matched: the grader's primary plan
- A28 我要跟廠長報告,把 L3 這 7 天的缺陷數字給我。
  `query_defect_history(line_id='L3')`
- A29 M23 跟 M22 比,哪一台的 open 比較嚴重?
  `(refused)` — matched: a refusal that names M23 as non-existent is defensible, since the comparison as asked cannot be made at all
- A30 把 scratch 加進缺陷類別,以後這一類要單獨統計。
  `(refused)`
- A31 20085294 上面 AOI 抓到哪些點?那片是誰做的?哪個作業員負責?
  `list_candidates(board='20085294') + query_board_context(board='20085294')`
- A32 工單對 open 的允收條件是怎麼寫的?
  `search_standards()`
- A33 哪一台機台 short 最多?查完幫我在系統裡開一張保養工單。
  `query_machine_stats(defect_type='short')` — matched: the grader's primary plan
- A34 LOT-2608003 這個 lot 這幾天總共幾件缺陷?
  `query_defect_history()`
- A35 open 跟 short 哪一個問題比較大?
  `query_defect_history(defect_type='open') + query_defect_history(defect_type='short')` — matched: the grader's primary plan

</details>

### 2026-08-23 · The criteria retrieval was answering about one class out of another's rules

Found by reading the queue, not by a failing test. All five escalations in the
store told the operator the same thing: that the criteria require establishing
whether the open is inside a pad "(critical) or outside". No document says
that. WI-201 says any confirmed open is a critical defect and that there is no
width or length below which an open is acceptable, because continuity is
binary. The inside-a-pad rule is WI-206's and governs pin holes.

It got there by retrieval. `search_standards` ranked passages across all six
work instructions with nothing scoping them to the class being asked about,
and for the disposition path's own query -- "acceptance criteria and
disposition for open" -- WI-206's disposition section ranked *first*, ahead of
WI-201's own classification section. The node asks for two passages. The model
was handed the right rule and a pin-hole limit, and fused them.

The project's standing defence is that the LLM only explains and the
classifier decides. It does not cover this. The fabricated rule went to the
people who do decide, and it pointed at releasing a critical defect.

### Cross-class contamination in the criteria retrieval

Six classes x 6 phrasings, top_k=2: the graph's own query, the planner's, a bare class name, two Chinese question forms from `/ask`, and the question an operator has in front of the images. A passage is counted wrong when it comes from another class's work instruction; QP-110 and WI-300 govern every class by declaration and are not counted against anything.

| class | unscoped | scoped |
|---|---|---|
| `copper` | 4/12 (33%) | 0/12 (0%) |
| `mousebite` | 0/12 (0%) | 0/12 (0%) |
| `open` | 4/12 (33%) | 0/12 (0%) |
| `pin-hole` | 1/12 (8%) | 0/12 (0%) |
| `short` | 8/12 (67%) | 0/12 (0%) |
| `spur` | 3/12 (25%) | 0/12 (0%) |
| **all** | **20/72 (27.8%)** | **0/72 (0.0%)** |

`short` was the worst affected and `mousebite` the only class never
contaminated -- similarity has no notion of jurisdiction, and the classes that
read alike in prose are the ones that borrow each other's limits. `open` is
the one that matters: three of its four wrong passages were WI-206's
disposition section, which is the sentence that reached the queue.

What changed is the retrieval boundary, not the prompt. Each document declares
the class it governs, the declaration rides on every passage, and a caller
with a class in hand gets that class's work instruction plus QP-110 and
WI-300. The scope is a parameter rather than a filter welded on, because
`/ask` has questions that belong to no class. The disposition path always has
a class and always passes it.

What this number does not cover: it measures which document a passage came
from, not whether the passage answers the operator's question. For `open` the
scoped retrieval still returns "any confirmed open is critical" -- how to
*disposition* one -- when the person looking at the images needs to know how to
*confirm* one. That is still open, and it is a documents problem now rather
than a retrieval one.

The eight explanations already written under the old retrieval -- five
escalation reasons, three agent rationales -- are marked in place by
`scripts/quarantine_fabricated_criteria.py`, not deleted and not regenerated.

### Threshold sweep — `ESCALATE_BELOW` and `CONFIDENT` (2026-08-23 · commit 68e90b6)

8143 stored candidates from the official DeepPCB test split (2997 real defects), `fragment` held out. No GPU: the sweep reads the predictions already in the store, the same source `routing_report.py` uses.

Held fixed: `DEFAULT_DISMISS_THRESHOLD` = 0.915, which by itself dismisses 15 real defects — the whole of QP-110's ≤0.5% escape budget. There is no room left in the budget for a second dismissing branch, so the criterion for `ESCALATE_BELOW` is zero *added* escapes, not a share of one.

#### `ESCALATE_BELOW` — the confidence at which a region goes to a person

The only way this branch can add an escape is `decide_node` dismissing: the classifier's class is `false_call`, so `confidence` *is* `P(false call)`, and the region sits in the band [`ESCALATE_BELOW`, `DEFAULT_DISMISS_THRESHOLD`). Everything else the branch does is confirm a defect or hand it over, and neither ships a board.

| `ESCALATE_BELOW` | escalated | decided | of those, dismissed | escapes added | line escape rate | decided class right |
|---|---|---|---|---|---|---|
| 0.600 | 153 (1.9%) | 1119 | 320 | **12** | 0.901% | 83.1% |
| 0.650 | 201 (2.5%) | 1071 | 296 | **9** | 0.801% | 84.1% |
| 0.700 | 250 (3.1%) | 1022 | 272 | **8** | 0.767% | 85.5% |
| 0.750 | 320 (3.9%) | 952 | 234 | **5** | 0.667% | 87.1% |
| 0.800 | 386 (4.7%) | 886 | 192 | **4** | 0.634% | 88.4% |
| 0.850 | 486 (6.0%) | 786 | 136 | **2** | 0.567% | 91.0% |
| 0.860 | 505 (6.2%) | 767 | 122 | **1** | 0.534% | 91.3% |
| 0.870 | 531 (6.5%) | 741 | 102 | **1** | 0.534% | 91.2% |
| 0.875 | 540 (6.6%) | 732 | 94 | **0** | 0.501% | 91.4% |
| 0.880 | 550 (6.8%) | 722 | 87 | **0** | 0.501% | 91.3% |
| 0.890 | 577 (7.1%) | 695 | 65 | **0** | 0.501% | 91.2% |
| 0.900 | 617 (7.6%) | 655 | 33 | **0** | 0.501% | 91.1% |
| 0.910 | 644 (7.9%) | 628 | 13 | **0** | 0.501% | 91.1% |
| 0.915 | 664 (8.2%) | 608 | 0 | **0** | 0.501% | 91.1% |
| 0.920 | 668 (8.2%) | 604 | 0 | **0** | 0.501% | 91.1% |
| 0.950 | 715 (8.8%) | 557 | 0 | **0** | 0.501% | 92.6% |

The highest-confidence real defect this branch would dismiss carries **0.8721**. So on this grid the lowest threshold adding no escape is **0.875** — not 0.90, which the citation in `docs/architecture.md` claimed until this run and which clears the same bar with 0.028 to spare.

Neither is the value to ship. 0.875 sits 0.0029 above the worst miss on this split: that is a threshold read off the test set at three decimal places, and the next lot's tail lands on top of it. And 0.90 is a round number that happened to be conservative — it was never derived from anything, which is the finding, not the fix.

The value that needs no split at all is `DEFAULT_DISMISS_THRESHOLD` (0.915). At or above it the band is empty by construction: a region the classifier calls `false_call` above that confidence was already dismissed upstream, so it never reaches `decide_node`. The agent branch may confirm a defect; it cannot dismiss one. That holds for any model and survives a retrain, where a swept number would have to be swept again and silently would not be.

#### `CONFIDENT` — the confidence at which the classifier's class skips the LLM

`confirm_node` and `decide_node` write the same verdict: `model_class`. So above `ESCALATE_BELOW` this threshold moves candidates between two paths that disposition them identically. It is a cost gate, not a decision gate — the sweep below is of LLM calls, and the "dispositions changed" column is what makes that claim checkable.

| `CONFIDENT` | confirmed without the LLM | that class right | reaching the LLM | escalated | escapes added | dispositions changed vs 0.95 |
|---|---|---|---|---|---|---|
| 0.700 | 2403 | 97.1% | 1162 | 588 | 0 | **76** |
| 0.800 | 2373 | 98.0% | 1192 | 618 | 0 | **46** |
| 0.850 | 2351 | 98.7% | 1214 | 640 | 0 | **24** |
| 0.900 | 2337 | 99.1% | 1228 | 654 | 0 | **10** |
| 0.915 | 2327 | 99.2% | 1238 | 664 | 0 | **0** |
| 0.920 | 2323 | 99.2% | 1242 | 664 | 0 | **0** |
| 0.950 | 2293 | 99.6% | 1272 | 664 | 0 | **0** |
| 0.970 | 2270 | 99.7% | 1295 | 664 | 0 | **0** |
| 0.990 | 2180 | 99.9% | 1385 | 664 | 0 | **0** |
| 0.999 | 1861 | 99.9% | 1704 | 664 | 0 | **0** |

Zero dispositions change anywhere at or above `ESCALATE_BELOW`, and the escape column never moves. Below it the threshold stops being free: it starts confirming, unreviewed, regions the flow would have handed to a person. That is the one thing `CONFIDENT` must not do, and it is a constraint the code can hold rather than a number a sweep can pick — `CONFIDENT` must be at least `ESCALATE_BELOW`.

Within that constraint the choice buys an operator a written rationale on the record, at one 20B-model call each. It is a cost dial and the citation should say so; it is not a decision authority and WI-300 never gave it one.

#### What the constants are set to now

| constant | value | escalated | reaching the LLM | escapes added |
|---|---|---|---|---|
| `ESCALATE_BELOW` | 0.915 | 664 (8.2%) | 1272 | 0 |
| `CONFIDENT` | 0.95 | — | — | — |


`ESCALATE_BELOW` moved from 0.90 to 0.915 on this run, and what it cost is 47
candidates out of 8143 — 0.6% of the queue — that used to be dispositioned by
the agent and now go to a person. The classifier had been right on 43 of them,
so most of that cost is an operator confirming what the model already said. It
buys a guarantee that does not have to be re-earned: 33 of the 47 were agent
*dismissals*, and there are now none of those at all. `CONFIDENT` did not move,
because the sweep says the value cannot matter as long as it stays above
`ESCALATE_BELOW`.

## 2026-08-23 · commit 4371015

### Re-verifier latency — what one candidate costs, and on what hardware

**A first run of this benchmark was discarded for GPU contention and is not
reported here.** It was taken while a concurrent torchvision detector benchmark
held MPS under sustained load, and `ollama ps` -- the check this project has
always used -- came back clean throughout, because it reports Ollama's own
resident models and knows nothing about a torch job in another shell. That is a
hole in the convention, not a one-off: every MPS figure in that run was
competing for the same silicon and none of them said so. The script now checks
twice, `ollama ps` plus a process sweep for anything busy enough to be
computing, and refuses to append when either fires. The run below was taken
after both checks came back empty, with a neighbouring Ollama translation job
waited out rather than measured through.


ResNet-18 re-verifier, 42.7MB checkpoint, 11.2M parameters, 3x64x64 input. Patches are real candidates from the official DeepPCB test split (8143 of them).

The timed path is the one `ReVerifier.classify_batch` runs: uint8 patches to float, divide by 255, move to the device, forward, softmax, back to the host. Timing the bare forward would hide the transfer, which on MPS is not free. Every MPS measurement is bracketed by `torch.mps.synchronize()`; without it the timer measures how fast Python enqueues work.

CPU figures are on 4 torch threads, which is what this machine defaults to; an edge box with fewer cores scales roughly with that number and the figure below is not transferable without it.

Devices are measured in the order they appear below, in one process. The second device therefore starts from an already-warm machine, and the peak RSS covers both sets of weights rather than one station's footprint. Both are stated rather than corrected for.

Contention was checked two ways, because one of them is not enough. `ollama ps` reports Ollama's own resident models and nothing else, so a torch/MPS job in another shell saturates the same GPU while that check comes back clean. The second check is a process sweep for anything busy enough to be computing -- `llama-server`, any running `.py`, `mlx`, `mediaanalysisd`. Both are recorded below.

```
ollama ps before the run
NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL

busy processes before the run
(none)

ollama ps after the run
NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL

busy processes after the run
(none)
```

#### Single candidate, warm

| device | calls | p50 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|
| MPS | 300 | 7.34ms | 7.71ms | 8.30ms | 8.56ms | 7.18ms |
| CPU | 300 | 2.50ms | 2.53ms | 2.87ms | 3.23ms | 2.51ms |

The claim of "tens of milliseconds" was wrong, and wrong in the pessimistic direction -- the model is faster than the README said. Measured: 2.50ms, which is single-digit milliseconds.

**At one candidate the GPU is the slower device.** MPS p50 is 7.34ms against the CPU's 2.50ms -- 2.9x slower. On a model this small the forward is cheaper than dispatching it and copying the result back, and the GPU has nothing to amortise that over. MPS only overtakes at batch 8, and from there it pulls away hard -- 3.2x at that batch alone. The station classifies one region at a time; the seeding pass classifies a whole board at once. They want different devices.

#### Cold against warm

A station restarting mid-shift pays the load and the first forward. Everything after that is the warm number above.

| device | checkpoint load | first forward | warm p50 | cold penalty |
|---|---|---|---|---|
| MPS | 134ms | 83.8ms | 7.34ms | 11x |
| CPU | 87.6ms | 2.98ms | 2.50ms | 1x |

#### Batched throughput

`classify_batch` is handed every candidate on one board at once, so the pipeline's batch size is the board's candidate count. On the test split that is a median of 8 trainable candidates per board (mean 16.3, max 159). The sweep is around that.

| batch | MPS | CPU |
|---|---|---|
| 1 | 6.65ms/cand, 150/s | 2.50ms/cand, 401/s |
| 2 | 3.28ms/cand, 305/s | 1.82ms/cand, 551/s |
| 4 | 1.67ms/cand, 597/s | 1.16ms/cand, 860/s |
| 8 ← | 0.31ms/cand, 3,201/s | 1.00ms/cand, 1,002/s |
| 16 | 0.21ms/cand, 4,833/s | 4.18ms/cand, 239/s |
| 32 | 0.17ms/cand, 6,009/s | 3.38ms/cand, 296/s |
| 64 | 0.15ms/cand, 6,549/s | 2.96ms/cand, 338/s |
| 128 | 0.14ms/cand, 6,991/s | 2.77ms/cand, 361/s |

The arrow marks the bucket nearest the pipeline's real batch.

#### Does the CPU cliff move with the core count?

The sweep above has a cliff on CPU between batch 8 and batch 16: past it the per-candidate cost jumps several-fold rather than falling. An edge recommendation of "batch at 8" is only useful if that is a property of the model rather than of this machine's cores, so the same two batches were run again across torch thread counts.

| torch threads | batch 8 | batch 16 |
|---|---|---|
| 1 | 2.31ms/cand | 6.05ms/cand |
| 2 | 1.89ms/cand | 5.92ms/cand |
| 4 | 1.49ms/cand | 5.99ms/cand |
| 8 | 1.58ms/cand | 6.07ms/cand |

Thread count moves the batch-8 figure and leaves the batch-16 figure essentially untouched. The cliff is therefore in the model's CPU convolution path, not in this laptop's core count, and it transfers: batch at 8 on any CPU box, and do not assume more batching is more throughput.

#### Thermal — first 60s against steady state

150s of sustained batched inference per device on a fanless M5 Air.

| device | first 60s (n, p50) | steady state (n, p50) |
|---|---|---|
| MPS | 13236, 4.62ms | 19436, 4.63ms |
| CPU | 716, 83.7ms | 891, 100ms |

- MPS: **No throttle observed.** Median 4.62ms in the first 60s against 4.63ms in steady state, +0% -- inside the 10% band this run calls noise.
- CPU: **Throttled.** Median went from 83.7ms in the first 60s to 100ms in steady state, +20%. Size an edge box on the steady-state figure, not the first minute of it.

#### Footprint

- checkpoint on disk: **42.7MB** (11.2M float32 parameters)
- peak process RSS across the whole run: **1144MB**
- patch construction (two crops and an absolute difference on the board image, which runs before every classification): p50 0.01ms per candidate

**What this means for an edge deployment.** The CPU figure is the one to size on: a re-verification station is a box beside a conveyor, not a laptop with a GPU. At 2.50ms per candidate single-shot and 1,002 candidates/second batched, on CPU alone, the model is not the constraint -- a board carrying twenty candidates is re-verified in well under a second, and the AOI stage in front of it takes longer. The 43MB checkpoint fits anywhere. The open INT8/ONNX work is therefore about memory and portability, not about latency: there is no latency problem here to solve.

## 2026-08-23 · commit 2421939

### Whole-line escape rate, recounted on defects instead of boxes

Was **5.4%**. Is **0.61%**. The old figure added an AOI-stage miss rate of 5.0% to the re-verifier's 0.47%, under the sentence "Defects the AOI never caught are already gone and no threshold recovers them". That sentence was true of 7 defects on this split and was being applied to 157.

The 5.0% was never a count of defects the detector failed to find. It counted defects whose best candidate did not clear DeepPCB's IoU 0.33 cut, and 150 of those 157 have a candidate sitting on them -- 95.5%. A matched candidate is on median 0.51x the area of the hand-drawn box it matches, so the whole distribution of best-IoUs piles up just under the cut: median 0.29 against a cut of 0.33. That is a statistic about how tightly this detector draws a box. It was published as a detection failure.

Measured on the test split: 500 boards, 3140 ground-truth defects, the shipped checkpoint, dismissal threshold 0.915.

#### What happens to every defect on the split

| outcome | defects | share | recoverable by a threshold? |
|---|---|---|---|
| reaches a person, via a candidate that also clears the IoU cut | 2975 | 94.75% | n/a -- reviewed |
| reaches a person, but only via a candidate the IoU rule calls a miss | 146 | 4.65% | n/a -- reviewed |
| flagged, and the re-verifier dismissed every candidate on it | 12 | 0.38% | yes -- this is the dismissal threshold |
| **never flagged: not one candidate overlaps it** | 7 | 0.22% | **no** |

The third and fourth rows are the escapes: **19 defects, 0.61%**. The second row -- 146 defects -- is what the old number was charging to the line. Every one of them is on an operator's screen.

#### The miss rate is mostly the cut

| detection rule | defects counted missed | share of defects |
|---|---|---|
| IoU ≥ 0.50 | 1489 | 47.42% |
| IoU ≥ 0.40 | 487 | 15.51% |
| IoU ≥ 0.33 | 157 | 5.00%  ← published as the AOI escape rate |
| IoU ≥ 0.30 | 98 | 3.12% |
| IoU ≥ 0.25 | 58 | 1.85% |
| IoU ≥ 0.20 | 38 | 1.21% |
| IoU ≥ 0.10 | 17 | 0.54% |
| any overlap at all | 7 | **0.22%** |

Nothing about the detector changes down that column. Only the cut does. The bottom row is the only one that describes a defect this line cannot see, and it is the one that belongs in an escape rate.

#### The composition

- **never flagged: 0.22%** of defects (7/3140) -- unrecoverable. No threshold, no model and no retrain reaches these; the pixels never reach the classifier.
- **dismissed by the re-verifier: 0.38%** of the 3133 defects that did reach it (12) -- this is the number the dismissal threshold governs, and the one QP-110's 0.5% budget is written about.
- **whole line: 0.61%** (0.22% + 0.998 × 0.38%), against 19/3140 = 0.61% counted directly.

Two numbers, not one. They are not interchangeable and adding them into a single headline is what produced the 5.4%: one of them is a knob and the other is a wall. Reporting only the sum tells a reader to go tune the thing that cannot move.

The re-verifier's own escape rate is quoted as 0.38% here and 0.47% in the operating-point table above. Both are right and they count different things: the table counts *candidates* carrying a defect label that were dismissed, this counts *defects* every covering candidate was dismissed on. A defect flagged by three candidates escapes only if all three go, and a defect whose only candidate is a held-out fragment is in this count and not in that one.

#### Where the escapes are

| defect class | on the split | never flagged | dismissed | escape rate |
|---|---|---|---|---|
| open | 659 | 2 | 4 | 0.91% |
| mousebite | 586 | 2 | 3 | 0.85% |
| spur | 483 | 1 | 0 | 0.21% |
| short | 478 | 1 | 3 | 0.84% |
| pin-hole | 470 | 1 | 0 | 0.21% |
| copper | 464 | 0 | 2 | 0.43% |

The never-flagged 7 are spread over 7 boards, no board contributing more than one, so this is not one bad scan. What they have in common is a cause, and it is in the detector rather than in the data -- see the opening-kernel sweep.

#### What changed in the code

`scripts/report.py` no longer computes a whole-line figure. It had a `--aoi-escape-rate` argument defaulting to 0.050, a number carried over by hand from `build_patches.py`'s miss print, and it had no access to the two things the composition needs: whether anything was flagged on a defect, and what the model did with it. This script owns that section now. `system_escape_rate` is unchanged and still correct -- it was being fed the wrong stage rate, not computing the wrong thing.

## 2026-08-23 · commit fd743df

### The opening kernel — what the seven lost defects would cost to recover

`open_kernel = 3` erases 7 real defects on the test split before connected components ever runs. That is the whole of this line's unrecoverable escape rate (0.22%, see the accounting above), and it is a detector setting rather than a property of the data. So: sweep it.

500 boards, 3140 ground-truth defects. The perturbed columns re-run the same boards under `gate_check.py`'s production conditions -- 2px template shift, sigma 6.0 noise, 1.03 gain, seeded per board. That is the condition the opening exists for, so it is the column that prices it.

| opening | defects nothing is flagged on | unmatched at IoU 0.33 | false calls/board | candidates/board | false calls/board, misregistered |
|---|---|---|---|---|---|
| none (a no-op; 1x1 is the same row) | 7 (0.22%) | 167 (5.32%) | 123.24 | 153.38 | 59.22 |
| 2x2 square | 0 (0.00%) | 58 (1.85%) | 39.17 | 51.50 | 64.59 |
| 3x3 cross | 2 (0.06%) | 43 (1.37%) | 19.47 | 28.83 | 48.05 |
| **3x3 square** ← shipped | 7 (0.22%) | 157 (5.00%) | 10.29 | 18.14 | 31.25 |
| 4x4 square | 255 (8.12%) | 733 (23.34%) | 3.52 | 9.79 | 8.51 |
| 5x5 square | 799 (25.45%) | 1279 (40.73%) | 0.74 | 5.66 | 2.51 |

Two things in that table are not obvious.

**Removing the opening entirely does not remove the problem.** With no opening the sliver noise merges under the 5x5 dilation into components that blow past `max_area`, and the detector drops those as registration failures -- so it loses 7 defects of its own while carrying 12x the false calls. The curve has no free end; it has a middle.

**The unmatched-at-IoU column moves a long way and means almost nothing.** A tighter opening leaves more of a defect standing, so the box is tighter and clears the cut. Those defects were reaching an operator either way -- that is the whole finding of the section above, and the column is here so nobody re-derives a recall improvement from it. The column that matters is the first one.

#### The trade, priced

| opening | defects recovered | of those, the re-verifier keeps | added false calls/board | added false calls per defect actually kept |
|---|---|---|---|---|
| none | 7/7 | 4 | +112.95 | **14,119** |
| 2x2 square | 7/7 | 5 | +28.88 | **2,888** |
| 3x3 cross | 5/7 | 5 | +9.18 | **918** |

The middle column is the one that decides this, and it needs its caveat said out loud: the checkpoint was trained on `open_kernel=3` patches, so a candidate only a smaller kernel produces is out of its distribution. This is what the line would do *today*, not what a retrained line would do. It is still the right question to ask first, because a recovered candidate the model then dismisses is a defect that escaped one stage later with the false calls still on the bill.

#### Decision: `open_kernel` stays at 3

The cheapest setting that recovers anything is **3x3 cross**, at **918 additional false calls per defect the model then keeps**. It buys 5 of 7.

Recovering all 7 needs **2x2 square** at 2,888 each -- and the model keeps only 5 of what it recovers, so 2 of the 7 escape one stage later with the false calls still on the bill.

Under misregistration the bill rises again: 48 false calls per board for 3x3 cross against the shipped 31. That is the column that describes a real line -- DeepPCB ships pre-registered, and the opening is in the detector precisely because a line that is not pre-registered leaves slivers along every trace edge.

Set against all of that: 0.22% of defects, on a project whose headline is how much review it removes. Buying them back means every board carries 1.6x to 8.5x the candidates, the checkpoint is invalidated, the operating point has to be re-swept, every routing number moves -- and the model, as it stands, dismisses part of what the change recovers. **The sweep does not support the change, so the constant does not move.**

What the sweep does support is naming the condition. Revisit this if the line's escape budget is ever written against the whole line rather than the re-verification stage, or if an escaped `open` is repriced -- these are the only defects here that no threshold reaches, and the lever that reaches them is this one and not the model.

#### Why these 7 and not others

| board | class | ground-truth box | difference pixels | after the 3x3 opening | max half-width |
|---|---|---|---|---|---|
| 12100016 | open | 27x30 | 54 | 0 | 1.37 px |
| 12100046 | pin-hole | 41x33 | 56 | 0 | 1.37 px |
| 12100153 | open | 33x29 | 133 | 0 | 1.37 px |
| 20085314 | mousebite | 25x27 | 50 | 0 | 0.96 px |
| 20085319 | mousebite | 26x28 | 24 | 0 | 1.37 px |
| 44000028 | spur | 29x51 | 93 | 0 | 0.96 px |
| 90100019 | short | 32x39 | 115 | 0 | 1.37 px |

Every one goes to zero. A blob survives an NxN opening only where it contains a fully-enclosed NxN square, and the widest of these is 1.37 px from its own edge at the thickest point -- under the 1.5 px a 3x3 square needs. They are **thin, not small**: the difference blobs run 24-133 pixels, which is well clear of `min_area`. Nothing downstream ever gets the chance to reject them; the mask is already empty.

They are spread over 7 boards, one each, so this is a property of the detector rather than of a bad scan. `DetectorConfig.open_shape` was added for this sweep and defaults to `rect`, which is what shipped and what still ships.

## 2026-08-23 · commit 1842d55

### The invariant audit — which of this project's own rules are unguarded

CLAUDE.md stated twelve invariants when this audit was written, and states
thirteen now -- the provenance rule was added with the work that made it true.
They are the part of this repository a reader is asked to trust and an agent is
asked not to undo, and until yesterday nothing checked that breaking one broke
anything. An outside audit found that **five of the twelve** would have failed
a test. The worst of them was the
durability rule: it names `InMemorySaver` as the thing the checkpointer must
never be, and it was verified by a suite in which every single test passed an
`InMemorySaver`.

That audit was a scratch file. It was stale inside a day, which is the same
failure one level up, so it is `scripts/invariant_audit.py` now and
`tests/test_invariant_audit.py` runs it.

#### Method, and what it cannot do

"Is this invariant enforced?" is not decidable by reading source. Nothing in a
test's name, its assertions or its imports says which English sentence it holds
up, and a test can assert loudly about the wrong thing. So the script does not
infer anything. Each invariant has a declared entry naming the tests that
enforce it and stating what those tests do *not* cover, and the script holds
that declaration to the code:

- an invariant in CLAUDE.md with no entry, or an entry whose rule was reworded away
- a named test whose file is gone, or that has been renamed or deleted
- a named test wearing `@pytest.mark.skip`
- a named test pytest does not collect (`--collect`)
- an entry declared `enforced` every test of which is `@pytest.mark.dataset`,
  and so never runs in CI, which runs `-m "not dataset"`

Each entry's claim was checked once, by hand, the only way it can be: break the
invariant in the working tree, run the whole suite, and read which tests
noticed. Every row below is a real mutation against the real suite, and the
mutation is recorded in the registry beside the claim it supports.

#### What actually fails when each invariant is broken

| invariant | mutation applied | tests that failed |
|---|---|---|
| The LLM explains; it does not decide | `decide_node` reads `agent_verdict` | 1 |
| " | `route_after_reason` reads `agent_confident` | 5 |
| Every failure escalates to a human | `route_after_reason` returns `decide` unconditionally | 23 |
| An escalation must outlive the process | `make_checkpointer` returns `InMemorySaver` | 3 |
| The station never shows `ground_truth` | `ground_truth` added to `store.boards._as_dict` | 1 |
| Criteria come from that class's document | `defect_class` dropped from the flow's `search_standards` call | 2 |
| Thresholds cite something a reader can open | `ESCALATE_BELOW` set to 0.80 | 3 |
| Split train/val by image, never by patch | `split_by_image` shuffles patch indices | **0 → 5** |
| Report an operating-point curve | `BUDGETS = []` in `scripts/report.py` | **0 → 3** |
| " | `BUDGETS = [0.005]` -- one row is not a curve | **0 → 3** |
| " | the accuracy line moved above the table and bolded | **0 → 1** |
| No free-form text-to-SQL | `run_query(sql: str)` registered, undeclared | **0 → the module will not import** |
| " | the same, declared an identifier | **0 → the module will not import** |
| " | the same, declared a retrieval query over the standards corpus | **0 → the module will not import** |
| The fan-out is not a latency optimisation | the graph's docstring rewritten to call it a speed-up | **0** |
| Use the official DeepPCB split | `load_split` reads `trainval.txt` for both splits | 3, all dataset-marked |

The bold rows are the ones that had nothing behind them when the audit ran.
Three are now closed -- the arrow is what the same mutation costs today -- and
the two that remain are described under the count.

#### The count

**15 enforced, 2 partly enforced, 1 unenforceable**, of eighteen. It was
7 / 4 / 1 when the audit was written: two moved when their gaps were closed,
and the thirteenth, fourteenth and fifteenth all arrived enforced, because in
each case the rule and the test that holds it were written together.

- **enforced** — breaking it fails a named test that runs in CI: the LLM off the
  decision path, the escalation direction, checkpoint durability, the
  `ground_truth` boundary, class-scoped retrieval, threshold citations, the
  train/val split, the operating-point curve, the no-SQL rule, and the two
  rules about who or what a decision names.
- **partly enforced** — a real part of the rule is held and a named part is not:
  - *The fan-out is not a latency optimisation.* The comparison the claim rests
    on is measured, stored per run and rendered, and `tools_wall` is held to
    cover the scheduling rather than only the slowest branch. The prohibition
    itself is on prose, and no test reads prose. Not closable in the terms it
    is written in, and left `partial` rather than reworded into something a
    test can reach.
  - *Use the official DeepPCB split.* The only test that would catch a re-split
    is dataset-marked, so CI never runs it -- and it checks a count, so a
    size-preserving reshuffle of the same 1,500 boards passes. Closing it means
    pinning the split's *content* -- a digest of the two file lists -- and
    running that in CI without the 231MB clone. Tractable, not done here.
- **unenforceable** — *Say what is simulated.* Prose discipline. No assertion
  distinguishes an honest sentence from a missing one. It is declared
  unenforceable with a reason rather than counted as passing, which is the whole
  point of having the category.

#### The first gap closed: `split_by_image`

`split_by_image` is the highest-value one on that list and it had no test at
all. Every published number in this project -- the operating point, the escape
rate, the review reduction, the class table -- is read off that split. Patches
are crops from a board and one image yields tens of them, sharing its lighting,
its registration error and often the same defect seen from two candidate boxes.
Split at patch level and validation is held out from nothing: for most val
patches a near-duplicate sat in the training batch.

Replacing the function body with the obvious simplification -- shuffle the rows,
cut -- left all 691 tests green. `tests/test_train_split.py` now fails five ways
on that mutation, naming the boards that straddle the split. One of its tests
runs the same assertion against a deliberately leaking split, so a guard that
has quietly stopped being able to fire fails rather than passes.

#### Closing the report gap: run the report in the suite

"Report an operating-point curve, never bare accuracy" is the first rule in
CLAUDE.md and it was the one nothing could notice being broken. The arithmetic
under it was tested hard -- `sweep`, `best_at_escape_budget` and the two-stage
compounding all fail when they move -- but no test imported `scripts/report.py`,
which is the part a reader actually sees.

`tests/test_report_curve.py` generates the report over a synthetic split with a
real cost trade-off built into it (100 defects and 300 false calls, both spread
across the probability range, so a tighter budget really does force a higher
threshold), computes the operating points independently from `sweep`, and holds
the published document to four things: it sweeps several budgets spanning a
range a line might choose between; every budget comes back as a row carrying
both halves of the trade-off, matched numerically against the sweep; the review
removed rises across the rows rather than repeating one point; and accuracy
appears after the curve, unemphasised.

None of that is a string match on today's wording. Columns are found by a word
in their header, figures are parsed as percentages and compared to the sweep
within half of the last decimal place, so the report can be rewritten freely and
only the curve is load-bearing.

#### Closing the SQL gap: a signature cannot tell, so the registration declares

The no-SQL invariant was enforced against the *arguments a plan passes*: unknown
tool, unknown argument, out-of-domain value, all refused before anything runs.
Nothing looked at the *surface the registry offers*, which is why
`run_query(sql: str)` passed all five tests -- `sql` was a known argument of a
known tool holding a value with no domain to be outside of.

The obvious fix does not work. `search_standards(query: str, top_k=3)` is a
registered tool that takes arbitrary free text, and it is the same signature as
`run_query(sql: str)`; banning `str` parameters, or `str` parameters named
`query`, would break a working feature and stop nothing, because renaming `sql`
to `query` is free. Nothing in a Python signature distinguishes prose handed to
an embedding index from syntax handed to a query engine.

What does distinguish them is where the text lands and what comes back.
Retrieval text reaches a Chroma index over the markdown in `data/standards/` --
no engine, no schema, no query language -- and returns passages carrying their
document, heading and text, so a wrong retrieval is visibly the wrong passage. A
query language reaches an engine and returns a number whose derivation is gone,
and in a disposition context a plausible wrong number is worse than a crash,
because it is acted on.

So the registry is now a tuple of `Registration`s, each accounting for every
parameter its tool exposes: a closed domain checked per call, an identifier used
as a value and never parsed, or free text declared over a named document corpus.
An unaccounted parameter raises `UnregistrableTool` at import -- the tool cannot
be registered at all, and the analysis path does not load. The declarations are
backed as far as they can be: the corpus has to be one the system has, the
tool's module and the corpus module are both read statically for a route to the
store, and the tool's own body for SQL built out of a string.

What is left is a declaration, and a determined author could write a false one.
That is deliberate and it is the honest limit of the mechanism: the lie is then
a line of source in the registry with a name on the commit, instead of a
signature that slid past five green tests.

#### What this method does not establish

The script cannot tell you an entry's claim is *true*. It can tell you the claim
has stopped being checkable, which is how this document actually rots: a test
gets renamed, or skipped, or the rule is reworded, or a thirteenth invariant is
added with nothing behind it. Whether
`test_the_ground_truth_never_leaves_the_store` really enforces the sentence it
is filed under is a judgement, made once, with a mutation behind it and written
down where the next person can disagree with it.

The mutations are also single-point. Each shows the named tests fail on *one*
way of breaking the rule, not on every way. `test_the_agent_branch_cannot_dismiss`
fires on both mutations of the LLM's role, which is some evidence it is holding
the rule rather than a spelling of it; nothing here proves that in general.

## 2026-08-23 · commit 9af930a

### Quantisation — what INT8 costs at the escape budget

ResNet-18 re-verifier, 42.7MB float32 checkpoint, 11.2M parameters, 3x64x64 input, exported to ONNX and quantised to INT8 two ways. Every engine below is scored on all 8143 candidates of the official DeepPCB test split and timed on the same machine in the same run, on CPU, at 4 threads.

The static quantiser's calibration set is 512 patches drawn from `data/patches/trainval.npz` with seed 20260823 -- the **training** split, never test. Calibrating activation ranges on the split the operating point is then reported on is threshold-tuning against the test set with an extra step in front of it.

Contention was checked the way `reverifier_latency.py` checks it, before and after. That check gained its CPU claimants on the day this ran: it found four `ffmpeg` transcodes from a neighbouring project holding roughly 800% CPU while `ollama ps` and the process sweep both came back clean, because the sweep's list had been written for a GPU benchmark. INT8 is a CPU result end to end, so that hole had to close before any number here was worth writing down.

```
ollama ps before the run
NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL

busy processes before the run
(none)

ollama ps after the run
NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL

busy processes after the run
(none)
```

#### Manual review removed at an escape budget — FP32 against INT8

This is the comparison. Everything below it is detail.

| escape budget | FP32 torch | FP32 ONNX | INT8 dynamic | INT8 static |
|---|---|---|---|---|
| ≤0.10% | **26.9%** (0.07%, 2/2997) | **26.9%** (0.07%, 2/2997) | **36.9%** (0.07%, 2/2997) | **27.8%** (0.07%, 2/2997) |
| ≤0.25% | **50.2%** (0.23%, 7/2997) | **50.2%** (0.23%, 7/2997) | **50.3%** (0.23%, 7/2997) | **51.2%** (0.23%, 7/2997) |
| ≤0.50% | **56.2%** (0.47%, 14/2997) | **56.2%** (0.47%, 14/2997) | **54.9%** (0.47%, 14/2997) | **56.0%** (0.47%, 14/2997) |
| ≤1.00% | **60.6%** (0.97%, 29/2997) | **60.6%** (0.97%, 29/2997) | **60.1%** (0.97%, 29/2997) | **60.7%** (0.97%, 29/2997) |
| ≤2.00% | **63.1%** (1.97%, 59/2997) | **63.1%** (1.97%, 59/2997) | **63.0%** (1.97%, 59/2997) | **63.0%** (1.97%, 59/2997) |
| ≤5.00% | **64.7%** (4.97%, 149/2997) | **64.7%** (4.97%, 149/2997) | **64.7%** (4.97%, 149/2997) | **64.7%** (4.97%, 149/2997) |

Each cell is the review reduction, with the escape rate it achieved and the escapes behind it in brackets. The thresholds differ between columns: each engine is given the best threshold *it* can reach inside the budget, which is the fairest reading and the one that makes a column's loss unambiguous.

**Read the tightest budgets with the escape count beside them.** At ≤0.10% the whole column is decided by two escapes out of 2997 defects, so a swing of several points there is a handful of candidates landing the other side of a cut, not an engine being better. The ≤0.50% row is the one the deployed threshold comes from and the one every verdict below is decided on.

**FP32 ONNX: the curve holds.** Review reduction at the ≤0.50% budget is unchanged (56.2% against 56.2%), and it buys 1.0x smaller on disk and 1.03x *slower*.
**INT8 dynamic: not worth taking.** It gives up 1.3 points of review reduction at the ≤0.50% budget (56.2% to 54.9%) to buy 4.0x smaller on disk and 1.25x the speed. The lost reduction is operators back in front of regions the FP32 model was willing to dismiss, every shift, for a saving on a disk that was not full.
**INT8 static: the curve holds.** Review reduction at the ≤0.50% budget is -0.2 points (56.2% against 56.0%), and it buys 4.0x smaller on disk and 3.51x the speed.

#### How far each engine drifted from the float model

Class agreement is what an operator would notice. The probability deltas are what the *threshold* notices, and the threshold is what this project reports on -- a model can agree on every class and still move every dismissal by sitting a hair the other side of the cut.

| engine | class agreement | disagreements | mean Δp | max Δp |
|---|---|---|---|---|
| FP32 ONNX | 100.000% | 0/8143 | 5.91e-08 | 1.38e-05 |
| INT8 dynamic | 99.509% | 40/8143 | 1.73e-03 | 3.87e-01 |
| INT8 static | 99.656% | 28/8143 | 8.76e-04 | 1.49e-01 |

#### Single candidate, warm — CPU

| engine | calls | p50 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|
| FP32 torch | 300 | 2.52ms | 2.64ms | 2.89ms | 3.07ms | 2.55ms |
| FP32 ONNX | 300 | 2.58ms | 2.95ms | 2.97ms | 3.18ms | 2.58ms |
| INT8 dynamic | 300 | 2.01ms | 2.03ms | 2.12ms | 2.42ms | 1.99ms |
| INT8 static | 300 | 0.72ms | 0.74ms | 0.77ms | 0.85ms | 0.72ms |

Against FP32 torch at 2.52ms: FP32 ONNX 2.58ms (1.03x slower); INT8 dynamic 2.01ms (1.25x faster); INT8 static 0.72ms (3.51x faster). Read these as the answer to "does INT8 cost latency" rather than as a reason to ship it: the FP32 figure was already 0.03% of a board's cycle and nothing downstream was waiting on it.

#### Batched throughput — CPU

| batch | FP32 torch | FP32 ONNX | INT8 dynamic | INT8 static |
|---|---|---|---|---|
| 1 | 2.52ms/cand, 397/s | 2.95ms/cand, 339/s | 2.00ms/cand, 499/s | 0.72ms/cand, 1,394/s |
| 2 | 1.80ms/cand, 555/s | 3.23ms/cand, 310/s | 1.94ms/cand, 516/s | 0.66ms/cand, 1,507/s |
| 4 | 1.17ms/cand, 856/s | 2.79ms/cand, 359/s | 1.91ms/cand, 523/s | 0.65ms/cand, 1,537/s |
| 8 ← | 1.02ms/cand, 977/s | 2.61ms/cand, 383/s | 1.86ms/cand, 539/s | 0.68ms/cand, 1,481/s |
| 16 | 4.18ms/cand, 239/s | 2.46ms/cand, 406/s | 1.79ms/cand, 558/s | 0.63ms/cand, 1,580/s |

The arrow marks the bucket nearest the pipeline's real batch (8 candidates on the median board).

#### Cold against warm

A station restarting mid-shift pays the load and the first inference.

| engine | load | first inference | warm p50 | cold penalty |
|---|---|---|---|---|
| FP32 torch | 95.1ms | 2.96ms | 2.52ms | 1x |
| FP32 ONNX | 8.47ms | 2.25ms | 2.58ms | 1x |
| INT8 dynamic | 11.8ms | 2.21ms | 2.01ms | 1x |
| INT8 static | 14.0ms | 0.83ms | 0.72ms | 1x |

#### Thermal — first 60s against steady state

150s of sustained batched inference per engine on a fanless M5 Air.

| engine | first 60s (n, p50) | steady state (n, p50) |
|---|---|---|
| FP32 torch | 6144, 9.46ms | 7488, 12.0ms |
| FP32 ONNX | 3038, 19.9ms | 4629, 19.7ms |
| INT8 dynamic | 4088, 14.7ms | 6139, 14.7ms |
| INT8 static | 11283, 5.43ms | 16969, 5.42ms |

- FP32 torch: **Throttled.** Median went from 9.46ms in the first 60s to 12.0ms in steady state, +27%. Size an edge box on the steady-state figure, not the first minute of it.
- FP32 ONNX: **No throttle observed.** Median 19.9ms in the first 60s against 19.7ms in steady state, -1% -- inside the 10% band this run calls noise.
- INT8 dynamic: **No throttle observed.** Median 14.7ms in the first 60s against 14.7ms in steady state, +0% -- inside the 10% band this run calls noise.
- INT8 static: **No throttle observed.** Median 5.43ms in the first 60s against 5.42ms in steady state, -0% -- inside the 10% band this run calls noise.

#### Footprint

| engine | on disk | of FP32 | peak RSS, serving it alone |
|---|---|---|---|
| FP32 torch | 42.7MB | 100% | 389MB |
| FP32 ONNX | 42.6MB | 100% | 118MB |
| INT8 dynamic | 10.7MB | 25% | 74MB |
| INT8 static | 10.8MB | 25% | 81MB |

The memory column is measured in a **fresh process per engine**, each one loading only that engine and classifying one board's worth of candidates. Taken in the process that produced everything above it, the figure was 1169MB for all four alike -- four sets of weights and an allocator arena that never shrank, describing no station anyone would build. This column is what a box serving one engine actually holds, and it is the reason the FP32 row is so much larger than its checkpoint: torch itself is most of it.

**Was inference ever the constraint?** No, and the arithmetic is not close. The test split is 500 boards carrying 16.3 candidates each on average, so one board's re-verification is 41ms of FP32 inference. Against a board every 10 seconds -- fast for a line that has an AOI stage and a conveyor in front of it -- that is 0.41% of the cycle. The best INT8 engine here takes it to 12ms, a saving of 29ms per board on a budget of 10000ms. There is no queue to drain and no operator waiting on it. Latency is not what this conversion is for, and a report that sold it as one would be selling a rounding error.

**What this changes.** INT8 static is the conversion that survives the curve. It takes the model off the disk from 42.7MB to 10.8MB, and -- the figure that matters more -- it takes a station's resident memory from 389MB to 81MB, 4.8x smaller, because most of the float32 process is the torch runtime rather than the weights. That is the honest case for quantising this model: not the milliseconds, which nothing was waiting on, but a box that can be sized in tens of megabytes instead of hundreds. It is not deployed here, because this station is a laptop with no memory problem; it is measured so that a box which does have one can be given a number rather than a hope. The deployed threshold stays with the float32 model it was swept for -- an engine change is a model change, and `DEFAULT_DISMISS_THRESHOLD` follows the model that produced it.

## 2026-08-23 · commit 6a01e1c

### The response budget was also the client timeout, and half the explanations died of it

`RESPONSE_BUDGET_S` served two roles: WI-300's response budget, and the httpx client's timeout. Against `gpt-oss:20b`, whose service time this document had already measured at a median of 12.5s and a p90 of 15.6s, that meant **20 of 24 calls timed out** — and since the LLM came off the decision path, writing the operator's explanation is the only job it has. The queue held an escalation whose entire content was `the model did not answer (ReadTimeout)`. Nothing counted how many others there had been, which is why nobody fixed it.

Two things follow, and both are measured below.

**The budget is not the timeout.** A budget is a promise and must not follow the model; a client timeout is a resource bound and must follow the measurement. `RESPONSE_BUDGET_S` stays 10s, stays read from WI-300, and moved to `graph/flow.py`, because what it bounds is the *verdict* — `classify_node`, 2.5ms per candidate on CPU. `EXPLANATION_DEADLINE_S` is 60s and bounds a wait nothing blocks on: the disposition is decided from `model_class` and `model_confidence`, both of which exist before the reason node is entered. WI-300 was corrected to say which of the two it governs, and the correction is not "the station was slower than the document" — its §1 and §2 had already moved decision authority to the classifier when the agent layer was measured, and the Response budget section was never revisited, so the two sections described different stations.

**Every earlier agent and planner section was measured at 180s.** `agent_eval.py` and `analysis_eval.py` both overrode the client timeout, for the correct reason that a 10s cut would have measured the timeout rather than the planner — but the effect was that every published number described a configuration nothing ran, and no section said so. Both scripts now take `EXPLANATION_DEADLINE_S`, so the benchmark and the station use one number. The two runs below are the first taken under the shipped configuration; the earlier sections carry a note saying what they were measured at and what it costs to read them.

### Agent-layer latency — does the reason node fit the explanation deadline?

`gpt-oss:20b` at `think="low"`, 24 real reason-node calls over candidates the router sends to the LLM. The deadline is `EXPLANATION_DEADLINE_S`, 60s, and the run used it rather than overriding it — a call that misses it here is a call that produces no explanation in production.

**This is not WI-300's 10s response budget, and comparing it against that budget is the error this script used to make.** The budget covers the verdict, which is `classify_node`'s at 2.5ms per candidate. The LLM writes the operator's explanation and dispositions nothing, so what bounds it is a resource limit, not a promise.

Latency here is **service time**: Ollama's `total_duration` less `load_duration`. It is not `eval_ms`. Measured on this model, `eval_duration` does not account for thinking tokens at all, and reports under half the time the station waits.

```
ollama ps before the run
NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL

busy processes before the run
(none)

ollama ps after the run
NAME           ID              SIZE     PROCESSOR    CONTEXT    UNTIL               
gpt-oss:20b    17052f91a42e    12 GB    100% GPU     32768      29 minutes from now

busy processes after the run
(none)
```

| | calls | median | mean | p90 | max |
|---|---|---|---|---|---|
| first 60s | 7 | 8.7s | 9.0s | 10.0s | 11.1s |
| steady state | 17 | 8.5s | 8.9s | 10.3s | 13.0s |
| all | 24 | 8.6s | 8.9s | 11.1s | 13.0s |

**Inside the deadline.** p90 is 11.1s against 60s, and 0 of 24 calls produced no explanation.

Against WI-300's 10s response budget, for reference and not as the verdict: 5 of 24 explanations took longer than the budget allows a *verdict* to take. No verdict waited on any of them — `classify_node` had already produced the disposition before the reason node was entered.

Of that service time, `eval_duration` accounts for 7.7s and prompt ingestion for 0.0s on average. The remaining 1.2s is thinking tokens, which Ollama generates and bills to nobody. Reporting `eval_ms` as the latency would have understated this run by 13%.

Queueing check: 0.0% of mean wall time is not load, prompt or generation — the request went straight to the GPU, so the run is not contended.

No request was served after an eviction.

### Agent layer — does it beat the classifier, and is the escalation calibrated?

`gpt-oss:20b`, 60 candidates the router sends to investigation, sampled by stride across the store. `fragment` ground truth is held out, as in training. Ran in 12 min.

Run at the deadline the station runs at, `EVAL_TIMEOUT_S = EXPLANATION_DEADLINE_S = 60s`. Earlier runs of this script overrode it to 180s, which measured a configuration nothing ships.

**Explanations written.** 60 of 60. The layer produced the thing it exists to produce on every candidate.

**What the system dispositions on, against what the LLM would have dispositioned on.** `decide_node` takes the classifier's class; the agent column is the counterfactual it replaced.

| | candidates | system (classifier) | LLM counterfactual |
|---|---|---|---|
| all investigated | 60 | 51/60 = 85.0% | 49/60 = 81.7% |
| agent kept | 29 | 29/29 = 100.0% | 28/29 = 96.6% |
| agent escalated | 31 | 22/31 = 71.0% | 21/31 = 67.7% |

**Calibration of the hand-off.** The LLM's verdicts were right 96.6% of the time on what it kept and 67.7% of the time on what it handed over, a gap of +28.8%. The escalations land on the harder cases, which is what the confidence flag is for.

**Where the LLM would have overridden the classifier.** It proposed a different class on 2 of 60 candidates. The agent was right 0 of those times; the classifier had already been right 2 times, and 0 were wrong either way. Acting on those proposals would cost accuracy rather than add it, which is why the flow does not. On the kept set the classifier scores 29/29 = 100.0% against the LLM's 96.6%.

**Escalation rate.** 31/60 = 51.7% of investigated candidates, which is 9.2% of the whole queue.

**Escapes.** 0 of 29 kept candidates were called `false_call` while carrying a real defect.

Distribution of what the agent said, against the truth:

| truth | n | agent agreed | agent escalated |
|---|---|---|---|
| open | 27 | 26 | 2 |
| false_call | 27 | 18 | 27 |
| short | 3 | 3 | 1 |
| mousebite | 1 | 1 | 1 |
| spur | 1 | 1 | 0 |
| copper | 1 | 0 | 0 |

## 2026-08-23 · commit d907472

### The prose over the results — is the sentence true of the payload?

`gpt-oss:20b`, the 70 independent questions run through the whole analysis graph, of which **34 reached the synthesis node** and have prose written over a payload to score. The other 36 terminated in `report_node` — a refusal, a plan `validate_plan` threw out, or no plan at all — which writes a fixed string over no results. Ran in 17 min. One pass per question, so this is a single point and not a distribution.

**This measures fidelity to the results, not truth about the line.** A tool that returns a 9-day window labelled `"days": 14` — which `query_machine_stats` does, and docs/benchmarks.md already records — is repeated faithfully by the model and passes every check here. Correctly: that defect belongs to the tool, and a prose checker that flagged it would be scoring the wrong layer. The claim this section supports is that a supervisor reading the sentence is reading the payload, not that the payload is right.

**The first pass of this checker raised 43 findings, and 41 of them were the checker's fault.** Adjudicating them against the payloads — which is what a judged flag is for — turned up five defects, every one of which made the instrument shout rather than made the model wrong. `M12` was read as the number 12, so a sentence that merely named six machines produced six swap findings; twenty-two of the thirty-seven attribution findings were nothing but that. A Chinese answer never split into sentences, because `。` is not followed by a space, so a figure was attributed to whichever entity was named last anywhere in the paragraph. `19 copper, 22 mousebite` was read backwards, shifting a whole list by one. A fleet average quoted beside a machine name was called that machine's. And a share the model divided out of two stored figures was called a fabrication. Each correction is a commit with a test either side of it — the shape it now passes, and the swap in that same shape it must still catch — because five changes that each make a checker quieter are exactly how a checker goes blind. The numbers below come from a fresh run scored by the corrected checker.

**30/34 = 88% of the scored answers carry no finding of any kind**, and 34/34 = 100% carry no *checked* finding — nothing fabricated and nothing misattributed across 602 figures quoted in 265 sentences.

| kind | findings | answers affected | decided by |
|---|---|---|---|
| `fabricated_figure` | 0 | 0 | comparison |
| `misattributed_figure` | 0 | 0 | comparison |
| `unsupported_claim` | 3 | 3 | a person, on a flag |
| `unhedged_gap` | 0 | 0 | a person, on a flag |
| `misquoted_criterion` | 1 | 1 | a person, on a flag |

- `fabricated_figure` — a number in the prose that renders from no value in the payload. The reader cannot catch this: it looks like the figures beside it.
- `misattributed_figure` — a number that is in the payload, attached to an entity that does not hold it — the right rate against the wrong machine, line or class. Worse than a fabrication in one way: every figure audits clean.
- `unsupported_claim` — a cause, or a movement over time. No plannable tool returns a time series, so a trend claim is unsupported by the shape of the tool surface and not merely by this run's numbers.
- `unhedged_gap` — a tool failed, errored or returned nothing, and the prose reads as complete.
- `misquoted_criterion` — a rule asserted about a class no retrieved passage governs or mentions, or attributed to a work instruction never retrieved. This is the 2026-08-23 incident, made checkable.

**Two of the five kinds carry no judgement at all.** A figure is compared against the stored payload and the comparison is reproducible from the raw file this run wrote; nothing about it is an opinion. The other three are pattern matches — a cause word, a trend word, a class name beside a normative word — and each one is a candidate a person settles. Every candidate is printed below with its sentence, so the judgement is auditable rather than asserted.

**How much of this was checkable rather than judged.** Of 602 figures quoted across the scored answers, every one was compared against the payload — that is 100% of the figure claims, and figures are what a supervisor acts on. Of the 4 findings, 0 came from a comparison and 4 from a flag somebody had to settle. What is *not* covered at all: a sentence that quotes every figure correctly and characterises them wrongly — "M22 is the worst machine" over a payload where it is second — is outside every kind here, because the claim is a reading of the numbers rather than a number. That is the boundary, and it is where a person still has to look.

**Two latitudes the checker grants, counted rather than assumed.** 1 figure(s) were accepted as restated from the plan and 1 as a ratio of two figures the payload holds — "18.1% of L1's defects" over a payload storing 175 and 966. The division is checked, not assumed: a value is excused only when a pair producing it exists. A derived figure is exempt from the entity check as well, and that is a real gap rather than a convenience — a quotient carries no entity, so a rate computed for the wrong machine is outside what this can see.

**1 figure(s) were waved through as restated from the plan.** `SYNTHESIS_PROMPT` orders the plan's assumptions repeated in the prose, so a window the planner chose comes back in the answer having never been in a payload. Counting those as fabrications would score the synthesis node for the planner's work, which is `scripts/analysis_eval.py`'s job. They are counted here instead of being silent.

**How lenient the figure check is, measured rather than argued.** Every figure that grounded was re-asked at 1.3x and 0.7x. 176/856 = 21% of those perturbed figures still grounded — but that number is carried by small integers, since the payloads are full of box coordinates and a coordinate moved 30% lands on another coordinate. Restricted to figures written with a decimal point, which is where a rate or a share lives, 25/278 = 9% survived perturbation. A rate that is wrong by 30% does not pass this checker; a small count sometimes does.

**The checker and the system have the same author.** That is the sharpest limit on this number and no run removes it: the same person chose what counts as a fabrication and wrote the layer being scored, so a failure mode neither of them thought about is invisible to both. Two things mitigate it and neither closes it. The questions are the independent seventy, written by three authors who had seen none of this. And the checker is required to fail a summary corrupted on purpose — `tests/fixtures/synthesis_wrong_summary.json` carries one instance of every kind and `tests/test_synthesis_claims.py` fails if any kind stops firing, or if a faithful summary over the same results starts firing. A taxonomy nothing can fail reads exactly like a clean result, and that control is what tells the two apart.

```
ollama ps before the run
NAME           ID              SIZE     PROCESSOR    CONTEXT    UNTIL               
gpt-oss:20b    17052f91a42e    12 GB    100% GPU     32768      28 minutes from now

busy processes before the run
(none)

ollama ps after the run
NAME           ID              SIZE     PROCESSOR    CONTEXT    UNTIL               
gpt-oss:20b    17052f91a42e    12 GB    100% GPU     32768      29 minutes from now

busy processes after the run
(none)
```

**The control.** A checker built by the author of the thing it checks can be lenient without anyone noticing, and a taxonomy nothing can fail reads exactly like a clean result. This fixture is the control: one run's real results, a faithful summary of them that must produce no finding, and a corrupted summary carrying one instance of every kind that must produce all five. If the corrupted half stops failing, the checker has gone blind; if the faithful half starts failing, it has gone loud.

- `fabricated_figure` — 1412, which appears in no payload
- `misattributed_figure` — 302 is L2's open count, printed under L1
- `unsupported_claim` — a movement over time, and a cause, in one sentence
- `unhedged_gap` — query_machine_stats failed and the prose reads as complete
- `misquoted_criterion` — WI-206 was never retrieved, and no passage governs spur

Every finding, so the judged ones can be re-adjudicated:

- S14 `unsupported_claim` — a movement over time
  results say: no plannable tool returns a time series, and this plan asked for one window
  sentence: The data only shows these associations and does not indicate why the mousebite rate has risen.
- S25 `misquoted_criterion` — a rule about copper
  results say: no criteria were retrieved on this run at all
  sentence: No other defect class comparisons are provided, so the analysis is limited to copper.
- A10 `unsupported_claim` — a cause
  results say: the tools carry association only
  sentence: - Acceptance criteria state that a confirmed open leads to scrap or rework (jumper repair), while a suspected open that is continuous on electrical test is recorded as cosmetic thinning and the board is released.
- A12 `unsupported_claim` — a cause
  results say: the tools carry association only
  sentence: Because the defect history query returned no data, we cannot determine whether board 20085294 has more or fewer defects than its lot peers based on the recent 7-day window.

**Adjudicated, one by one. Three of the four are the pattern, not the model.**
A flag is a candidate, and saying so is worth nothing unless the candidates get
settled in public.

- **S14 stands.** "does not indicate why the mousebite rate **has risen**" is a
  correct refusal to give a cause wrapped around a presupposition the payload
  cannot carry: one window cannot show a rise. Mild, and the questioner asserted
  the rise first — but the sentence states it as fact and nothing fetched
  supports it.
- **A10 is the pattern.** `leads to` matched inside a quoted disposition rule —
  "a confirmed open **leads to** scrap or rework" — which is a document saying
  what to do, not a claim about why L2-M22 has opens. Worth noting what the
  question was: *是因為蝕刻液老化了嗎* — is it the etchant ageing? The model was
  asked point-blank for a cause and did not give one.
- **A12 is the pattern.** `because the` matched a sentence explaining that a
  tool came back empty. That is the hedge the `unhedged_gap` kind exists to
  demand, caught by the cause detector for the shape of its first two words.
- **S25 is a defect in the checker.** `limit` matched inside `limited to
  copper`, raising a criterion finding against a sentence that states no rule at
  all. Fixed after this run — `NOT_NORMATIVE`, with a test either side of it —
  and the count above is left as the run produced it rather than re-run down to
  a cleaner number.

So the cause detector's precision on this run is 0 of 2 and the criterion
detector's is 0 of 1. That is what a *flag* is worth, stated as a number rather
than implied: these three kinds cost a person four sentences to read, and on
this run three of the four readings ended in "no". The two checked kinds cost
nobody anything, because nothing is left to settle.

What this does not establish. It is one pass over 34 answers, not a distribution, and the model is sampled rather than deterministic — a second run would produce different sentences over the identical payloads. The three judged kinds are pattern matches with a person behind them, so their counts are a floor on what a pattern can raise and not a rate. Nothing here scores whether the answer was *useful*, whether it answered the question asked, or whether the plan fetched the right data — that last is `scripts/analysis_eval.py`, which scored 55/70 on the same set. And the checker is not independent of the system; the control fixture is what stands in for independence, and it is a weaker thing.


## 2026-08-23 · attribution

### Who answered — the other half of the provenance question

The provenance work made every automated decision name the weights, the
operating point and the commit behind it. It left the human half untouched, and
the same reviewer found the gap in the same breath: **all 9,140 decisions read
`reviewer = NULL`**, `reviewer` was an unauthenticated free-text field, and five
escalations stood closed with no human decision beneath them at all.

That is not an access-control finding. It is a retraining finding. This
project's whole feedback story is that operator corrections become the next
round's labels — it is why the station refuses to show `ground_truth` — and a
label that cannot name its author is a label nobody can weigh. This store has
already paid for that once: five regions clicked through without the domain
knowledge to judge them, four of the five wrong, indistinguishable by any query
from an expert's, and the only available remedy was deleting all five by hand.

#### What a human decision now carries

Two columns, read together. `reviewer` is the name; `reviewer_auth` is what
established it. `mike` typed into a text box and `mike` read off a signed
session are the same four characters and are not the same claim, and the second
column is what a retraining export can select on.

| value | meaning |
|---|---|
| `signed_in` | the station authenticated the operator and read the name off the session, never off the form |
| `host_account` | the CLI attributed the decision to the OS account running it — derived, not typed, and a weaker claim recorded as a different word |
| `automated` | no person was involved. A positive statement, not an absence: the provenance columns answer instead |
| `unrecorded` | the row predates the column. Stamped by the migration |

`store.boards.record_decision` raises on a `human` row that is not
attributable, which is the mirror of the rule it already enforced on a `model`
or `agent` one, and `resume_review` raises before the graph is resumed so a
refused answer cannot consume the interrupt and leave a region off the queue
with no verdict anywhere — which is the exact shape of those five rows.

#### The 9,140 rows that predate it

`uv run python scripts/seed_store.py --migrate-only` added
`review_decisions.reviewer_auth` in place and stamped **9,140 rows
`unrecorded`**, leaving **0 NULLs**. The store was not rebuilt; the corrections
in it are the next training round's labels.

The stamp is what makes the distinction possible at all. Every one of those
9,140 rows has `reviewer = NULL`, so without it `NULL` would have to mean both
"written before anyone recorded how a reviewer was identified" and "no
reviewer, because no person was involved" — and the second is a fact worth
stating about a model decision. The migration stamps the first meaning,
`record_decision` writes the second, and `NULL` is left meaning nothing at all,
with a test that fails if one appears.

#### The scheme, and what it does not protect against

A file of operators, each with a salted PBKDF2-HMAC-SHA256 record carrying its
own iteration count, exchanged at `/login` for an HMAC-signed session cookie
that expires after one shift. `scripts/add_operator.py` is the whole of user
management: no registration, no roles, no reset flow. A re-verification station
serves a fixed set of people on a line, and the alternative is several hundred
lines of surface area whose only customer is a screenshot.

Both pages are behind it, by allowlist rather than by decoration, so a route
added later is behind it too. `/ask` is why this stopped being a backlog item:
an unauthenticated visitor to the queue saw the regions on one line, and the
same visitor at `/ask` could pull production statistics for the whole plant.

What it does not protect against, stated here and in `station/auth.py` rather
than left for a reader to find out:

- **Sharing.** A passphrase two people know produces one name on both their
  labels. No scheme short of a badge reader fixes this.
- **A hostile network.** The cookie is a bearer token and this process speaks
  plain HTTP; it is marked `Secure` only when the request that set it arrived
  over TLS. Put the station behind TLS if the network is not trusted.
- **Anyone with the machine.** The store is a SQLite file. Shell access beats
  every check in the module, which is why the CLI's identity is recorded as
  `host_account` rather than pretended equal to a signed-in one.
- **Guessing.** No rate limit and no lockout. PBKDF2 makes an offline attack on
  the file expensive and does nothing about an online one against a weak
  passphrase.
- **Repudiation.** A signed-in name is a claim by this process, not a signature
  by the operator. Good enough to weigh a training label; not good enough to
  hold somebody to in a dispute.

#### What actually fails when the rule is broken

The fourteenth invariant, and the same method as the other thirteen: break it
in the working tree, run the whole suite, read which tests noticed.

| mutation applied | tests that failed |
|---|---|
| the `human` guard removed from `record_decision`, identity defaulted to a typed name | 7 |
| the `reviewer` form field restored and the name read off it | 4 |
| the middleware passing an anonymous request through as `operator` | 13 |
| the migration's `reviewer_auth` backfill removed | 3 |

What those tests hold is that a name is *established by a mechanism* rather than
typed. They do not hold that the name is true of the person at the keyboard,
and nothing in this design could.

#### What this makes possible that was not

A corrections export can now be selected by attribution:
`uv run python -m aoi_agent corrections` groups by it, and `boards.corrections`
returns it per row. A first retraining round can take `signed_in` labels only,
or weight `host_account` below them, or state honestly how much of the store it
had to leave behind. Before this, every human row was one undifferentiated
`NULL` and the choice did not exist. Whether to act on it is a separate
decision and is not taken here.

### The prose over the results — is the sentence true of the payload?

`gpt-oss:20b`, the 70 independent questions run through the whole analysis graph, of which **34 reached the synthesis node** and have prose written over a payload to score. The other 36 terminated in `report_node` — a refusal, a plan `validate_plan` threw out, or no plan at all — which writes a fixed string over no results. Ran in 25 min. One pass per question, so this is a single point and not a distribution.

**This measures fidelity to the results, not truth about the line.** A tool that returns a 9-day window labelled `"days": 14` — which `query_machine_stats` does, and docs/benchmarks.md already records — is repeated faithfully by the model and passes every check here. Correctly: that defect belongs to the tool, and a prose checker that flagged it would be scoring the wrong layer. The claim this section supports is that a supervisor reading the sentence is reading the payload, not that the payload is right.

**Both answer surfaces are scored, from one payload.** The station answers in `en` and `zh-TW`, and a run is planned and executed once and then written up once per language -- so the two accounts quote figures off the same results and the comparison means something. Clean by language: en 29/34 = 85%, zh-TW 26/34 = 76%. Publishing a single-language report is refused by the script: it would read as a claim about the system and be a claim about half of it.

**The cross-language comparison is a signal for adjudication, not a gate.** 23/34 = 68% of the scored questions quote a figure in one language and not the other, and most of that is prose: a sentence structure that fits one language may name a total the other reaches by naming its parts. An equality test here would manufacture findings the way this checker's first pass did. The hard gate is unchanged and is per language -- every figure in every answer rendering from the payload it was written from.

**Writing one payload up in two languages found three more of the checker's own defects, and two of them had been reading as a worse model.** The first bilingual run flagged the Chinese answers three times as often as the English ones -- 19 findings across 8 answers against 6 across 3 -- and every one adjudicated to the instrument. Chinese states a ranked comparison as a run of names against a run of figures (`M12、M21、M22 分別產生 29、21、37`), where every figure's nearest *preceding* name is the last one, so eleven correct figures on one sentence went to the final machine; and Chinese counts through a measure word (`438 個 spur`) where English writes `438 spur`, which the look-forward rule read as punctuation and fell back past. The third is English-shaped: `116,` was read as a four-character figure, because the rule that lets `1,049` be one number also swallowed a trailing comma -- which put the *next* class one character closer than the one the count belonged to and handed every entry in `copper - 116, mousebite - 161` to its neighbour. The value was always right; only the extent was, and attribution is measured off the extent.

After the three, `misattributed_figure` went 12 -> 0 in Chinese and 4 -> 0 in English on the same answers, and the two languages' clean rates closed from 16 points apart to level. **The English half improving is the tell that these were the instrument and not the language.** Each fix carries the swap it must still catch: a transposed list now produces exactly two findings naming the right owner, where before a faithful list and a transposed one both produced about eleven and the flags carried no information at all.

**A fourth correction was attempted and backed out, and a fifth was refused.** Excusing a quotient from the attribution check broke the control fixture and four swap tests, because in a payload of any size almost every figure is some ratio of two others. Admitting differences as a legitimate rendering was refused for the reason `_renderings` already states: every extra form is a number the checker will accept without it being in the results. Both leave a false positive standing below rather than silenced -- `56/282 = 19.9%` called L1's because 0.199 is also L1-M12's share, and `2,992 - 491 = 2,501` named in the prose as the other five classes combined. They are the price of the next paragraph.

**And with that strictness kept, this run caught a real one.** An answer reported `4,292 total defects on 421 boards` where the payload holds 2,992 -- and then listed the six classes correctly, which sum to 2,992. A supervisor reading the total would have been reading a number that exists nowhere and that the same paragraph contradicts. It reproduced on three consecutive runs, so it is the model and not the sampling. Every previous run of this script reported zero fabrications; that was a result, not a law, and one pass per question is what it is worth.

**One more defect was in the report rather than the checker.** The headline counts were built from the language the run was planned in, so "nothing fabricated and nothing misattributed" read as a statement about the system while being a statement about one of its two surfaces -- the exact failure `--lang both` exists to prevent, reproduced inside the report that enforces it. Found by reading the raw file against the headline it had just written. Every count below is over both languages.

**The first pass of this checker raised 43 findings, and 41 of them were the checker's fault.** Adjudicating them against the payloads — which is what a judged flag is for — turned up five defects, every one of which made the instrument shout rather than made the model wrong. `M12` was read as the number 12, so a sentence that merely named six machines produced six swap findings; twenty-two of the thirty-seven attribution findings were nothing but that. A Chinese answer never split into sentences, because `。` is not followed by a space, so a figure was attributed to whichever entity was named last anywhere in the paragraph. `19 copper, 22 mousebite` was read backwards, shifting a whole list by one. A fleet average quoted beside a machine name was called that machine's. And a share the model divided out of two stored figures was called a fabrication. Each correction is a commit with a test either side of it — the shape it now passes, and the swap in that same shape it must still catch — because five changes that each make a checker quieter are exactly how a checker goes blind. The numbers below come from a fresh run scored by the corrected checker.

**24/34 = 71% of the scored answers carry no finding of any kind**, and 32/34 = 94% carry no *checked* finding — 1 fabrication and 1 misattribution across 1364 figures quoted in 489 sentences.

| kind | findings | answers affected | decided by |
|---|---|---|---|
| `fabricated_figure` | 1 | 1 | comparison |
| `misattributed_figure` | 1 | 1 | comparison |
| `unsupported_claim` | 11 | 8 | a person, on a flag |
| `unhedged_gap` | 0 | 0 | a person, on a flag |
| `misquoted_criterion` | 3 | 2 | a person, on a flag |

- `fabricated_figure` — a number in the prose that renders from no value in the payload. The reader cannot catch this: it looks like the figures beside it.
- `misattributed_figure` — a number that is in the payload, attached to an entity that does not hold it — the right rate against the wrong machine, line or class. Worse than a fabrication in one way: every figure audits clean.
- `unsupported_claim` — a cause, or a movement over time. No plannable tool returns a time series, so a trend claim is unsupported by the shape of the tool surface and not merely by this run's numbers.
- `unhedged_gap` — a tool failed, errored or returned nothing, and the prose reads as complete.
- `misquoted_criterion` — a rule asserted about a class no retrieved passage governs or mentions, or attributed to a work instruction never retrieved. This is the 2026-08-23 incident, made checkable.

**Two of the five kinds carry no judgement at all.** A figure is compared against the stored payload and the comparison is reproducible from the raw file this run wrote; nothing about it is an opinion. The other three are pattern matches — a cause word, a trend word, a class name beside a normative word — and each one is a candidate a person settles. Every candidate is printed below with its sentence, so the judgement is auditable rather than asserted.

**How much of this was checkable rather than judged.** Of 1364 figures quoted across the scored answers, every one was compared against the payload — that is 100% of the figure claims, and figures are what a supervisor acts on. Of the 16 findings, 2 came from a comparison and 14 from a flag somebody had to settle. What is *not* covered at all: a sentence that quotes every figure correctly and characterises them wrongly — "M22 is the worst machine" over a payload where it is second — is outside every kind here, because the claim is a reading of the numbers rather than a number. That is the boundary, and it is where a person still has to look.

**Two latitudes the checker grants, counted rather than assumed.** 2 figure(s) were accepted as restated from the plan and 0 as a ratio of two figures the payload holds — "18.1% of L1's defects" over a payload storing 175 and 966. The division is checked, not assumed: a value is excused only when a pair producing it exists. A derived figure is exempt from the entity check as well, and that is a real gap rather than a convenience — a quotient carries no entity, so a rate computed for the wrong machine is outside what this can see.

**2 figure(s) were waved through as restated from the plan.** `SYNTHESIS_PROMPT` orders the plan's assumptions repeated in the prose, so a window the planner chose comes back in the answer having never been in a payload. Counting those as fabrications would score the synthesis node for the planner's work, which is `scripts/analysis_eval.py`'s job. They are counted here instead of being silent.

**How lenient the figure check is, measured rather than argued.** Every figure that grounded was re-asked at 1.3x and 0.7x. 244/1164 = 21% of those perturbed figures still grounded — but that number is carried by small integers, since the payloads are full of box coordinates and a coordinate moved 30% lands on another coordinate. Restricted to figures written with a decimal point, which is where a rate or a share lives, 11/254 = 4% survived perturbation. A rate that is wrong by 30% does not pass this checker; a small count sometimes does.

**The checker and the system have the same author.** That is the sharpest limit on this number and no run removes it: the same person chose what counts as a fabrication and wrote the layer being scored, so a failure mode neither of them thought about is invisible to both. Two things mitigate it and neither closes it. The questions are the independent seventy, written by three authors who had seen none of this. And the checker is required to fail a summary corrupted on purpose — `tests/fixtures/synthesis_wrong_summary.json` carries one instance of every kind and `tests/test_synthesis_claims.py` fails if any kind stops firing, or if a faithful summary over the same results starts firing. A taxonomy nothing can fail reads exactly like a clean result, and that control is what tells the two apart.

```
ollama ps before the run
NAME           ID              SIZE     PROCESSOR    CONTEXT    UNTIL               
gpt-oss:20b    17052f91a42e    12 GB    100% GPU     32768      26 minutes from now

busy processes before the run
(none)

ollama ps after the run
NAME           ID              SIZE     PROCESSOR    CONTEXT    UNTIL               
gpt-oss:20b    17052f91a42e    12 GB    100% GPU     32768      29 minutes from now

busy processes after the run
(none)
```

**The control.** A checker built by the author of the thing it checks can be lenient without anyone noticing, and a taxonomy nothing can fail reads exactly like a clean result. This fixture is the control: one run's real results, a faithful summary of them that must produce no finding, and a corrupted summary carrying one instance of every kind that must produce all five. If the corrupted half stops failing, the checker has gone blind; if the faithful half starts failing, it has gone loud.

- `fabricated_figure` — 1412, which appears in no payload
- `misattributed_figure` — 302 is L2's open count, printed under L1
- `unsupported_claim` — a movement over time, and a cause, in one sentence
- `unhedged_gap` — query_machine_stats failed and the prose reads as complete
- `misquoted_criterion` — WI-206 was never retrieved, and no passage governs spur

Every finding, so the judged ones can be re-adjudicated:

- S14 `misattributed_figure` — 19.9 attributed to L3/mousebite
  results say: that figure is L1's in the results
  sentence: 在最近兩天內,L3 線路共檢驗 43 枚板,總缺陷數 282,平均每板 6.56 個缺陷,其中 mousebite 缺陷 56 個,占總缺陷的 19.9%。
- S14 `unsupported_claim` — a movement over time
  results say: no plannable tool returns a time series, and this plan asked for one window
  sentence: 根據「mousebite」標準文件,若缺陷在允許寬度內則可釋放並記錄;若超出限值則視為減少區段缺陷,需送交工程處理。
- S21 `unsupported_claim` — a movement over time
  results say: no plannable tool returns a time series, and this plan asked for one window
  sentence: - 任何超出上述限值的 mousebite 皆需記錄並進行趨勢分析,重複發生於同一板面則可能表示上游蝕刻或處理問題。
- S21 `misquoted_criterion` — a rule about spur
  results say: no retrieved passage governs or mentions that class
  sentence: To complete the comparison, a source that defines spur criteria would be required.
- S26 `unsupported_claim` — a cause
  results say: the tools carry association only
  sentence: 根據「pin-hole」標準文件,pin-hole 若位於焊盤外部且在規定容差內可放行;若位於焊盤內部則需拒絕,因為此類缺陷會導致焊點開路。
- S32 `unsupported_claim` — a movement over time
  results say: no plannable tool returns a time series, and this plan asked for one window
  sentence: - 「判越沒把握」被視為缺陷率(或缺陷數量)相對於過去或其他機器的上升。
- S32 `unsupported_claim` — a movement over time
  results say: no plannable tool returns a time series, and this plan asked for one window
  sentence: The plan assumed that “判越沒把握” means an increase in overall defect counts compared with recent history, so we looked at the last 7 days of data for each machine.
- S32 `unsupported_claim` — a movement over time
  results say: no plannable tool returns a time series, and this plan asked for one window
  sentence: These figures show the total defect counts and defects-per-board for each machine over the 7-day window, but no comparison to earlier periods or other machines is provided, so we cannot determine whether the defect rate has risen or whether machine confidence has changed.
- A13 `unsupported_claim` — a movement over time
  results say: no plannable tool returns a time series, and this plan asked for one window
  sentence: 因此,短路缺陷的數量與其他缺陷類別相比並未顯示出明顯的增加。
- A14 `unsupported_claim` — a cause
  results say: the tools carry association only
  sentence: Because the system does not contain external cost data (labor, material, equipment depreciation, etc.), it cannot provide an estimate of the rework cost for these defects.
- A17 `fabricated_figure` — 4292
  results say: no value in the results renders as this
  sentence: The overall defect history for the same 7-day window shows 4,292 total defects on 421 boards, with 438 spur defects (the same number reported in the machine-level summary).
- A20 `unsupported_claim` — a movement over time
  results say: no plannable tool returns a time series, and this plan asked for one window
  sentence: 第二段說明,當剩餘導線寬度至少為名義寬度的 80%,且減少長度不超過兩個導線寬度時,mousebite 可接受。
- A20 `unsupported_claim` — a movement over time
  results say: no plannable tool returns a time series, and this plan asked for one window
  sentence: 第三段則說明,符合條件者可放行並記錄趨勢;不符合條件者則視為減少區段缺陷,送交工程處理,若同一板上重複出現,可能表示上游蝕刻或處理問題。
- A23 `unsupported_claim` — a movement over time
  results say: no plannable tool returns a time series, and this plan asked for one window
  sentence: 這些數據僅顯示缺陷的數量與類別,並未提供任何原因或趨勢。
- A27 `misquoted_criterion` — a rule about mousebite
  results say: no retrieved passage governs or mentions that class
  sentence: 具體而言,候選項目 20085294#0(open)與 20085294#1(open)以及 20085294#3(short)符合 critical 的定義;其餘「mousebite」、「spur」、「copper」等類別並未在標準中被歸類為 critical。
- A27 `misquoted_criterion` — a rule about mousebite
  results say: no retrieved passage governs or mentions that class
  sentence: No standard passage lists mousebite, spur, or copper as critical.

What this does not establish. It is one pass over 34 answers, not a distribution, and the model is sampled rather than deterministic — a second run would produce different sentences over the identical payloads. The three judged kinds are pattern matches with a person behind them, so their counts are a floor on what a pattern can raise and not a rate. Nothing here scores whether the answer was *useful*, whether it answered the question asked, or whether the plan fetched the right data — that last is `scripts/analysis_eval.py`, which scored 55/70 on the same set. And the checker is not independent of the system; the control fixture is what stands in for independence, and it is a weaker thing.

### Prevalence — what survives a line that is not this dataset

The curve above was swept over 8,143 candidates of which **2,997 (36.8%) are genuine defects**. No line looks like that: an AOI tuned for recall over-calls by one to two orders of magnitude, so its candidates are a fraction of a percent genuine. The README concedes that DeepPCB is binarised and that its defects are partly augmented onto the boards; **this is a third thing, and until now the project had not said it.** `scripts/prevalence_report.py`, commit `7cb1d93`.

Everything below holds each group's score distribution exactly as measured and varies only the mixing ratio — the label-shift assumption. That isolates prevalence from everything else that differs between a dataset and a line, and it is the whole of what this section claims. A different AOI, board or illumination moves the distributions themselves and nothing here speaks to that.

#### The escape rate does not move at all

`sweep` computes it as `escapes / defects_total`. Re-weighting every defect by one factor scales numerator and denominator together, so the escape rate at a threshold is the same number at any prevalence. Asserted in a docstring that would be worth nothing; computed here.

| escape budget | threshold | escape rate at 36.8% | at 1.0% | at 0.5% |
|---|---|---|---|---|
| ≤0.10% | 0.999 | 0.067% | 0.067% | 0.067% |
| ≤0.25% | 0.980 | 0.234% | 0.234% | 0.234% |
| ≤0.50% | 0.915 | 0.467% | 0.467% | 0.467% |
| ≤1.00% | 0.517 | 0.968% | 0.968% | 0.968% |

**Identical across the row, to every digit.** The threshold swept for a budget on this dataset is the threshold for that budget on any line whose score distributions match — which is the part of the operating point that transfers.

#### Review reduction moves, and it moves in the project's favour

That figure is `dismissed / total`, and lowering the prevalence adds false calls — which is the population the model is good at dismissing. The headline 56.2% is a **floor** for any line cleaner than this dataset, not a ceiling.

| escape budget | 36.8% | 20.0% | 10.0% | 5.0% | 2.0% | 1.0% | 0.5% |
|---|---|---|---|---|---|---|---|
| ≤0.10% | 26.9% | 34.0% | 38.2% | 40.4% | 41.6% | 42.1% | 42.3% |
| ≤0.25% | 50.2% | 63.4% | 71.3% | 75.3% | 77.6% | 78.4% | 78.8% |
| ≤0.50% | 56.2% | 71.0% | 79.8% | 84.2% | 86.9% | 87.8% | 88.2% |
| ≤1.00% | 60.6% | 76.5% | 85.9% | 90.6% | 93.5% | 94.4% | 94.9% |

Read the 36.8% column against the table above it: they agree, which is the check that this re-weighting is doing what it says.

#### What does not transfer is the verification

A 0.47% escape rate over 2,997 defects is a measurement. The same rate over the thirty defects a good line produces in a month is a coin. This is the table that should change how a pilot is planned, and it is the one the project was missing entirely.

| defects observed | escapes at 0.47% | 95% interval on the rate | ≤0.5% budget |
|---|---|---|---|
| 2,997  ← this project's own measurement | 14 | 0.28% – 0.78% | not settled |
| 1,000 | 5 | 0.21% – 1.17% | not settled |
| 300 | 1 | 0.06% – 1.86% | not settled |
| 100 | 0 | 0.00% – 3.70% | not settled |
| 30 | 0 | 0.00% – 11.35% | not settled |


**The first row is the uncomfortable one, and it is about this project rather than about a pilot.** 14 escapes in 2,997 defects is 0.47% exactly, on this split, and that number is not in doubt. Read as an estimate of the rate on *unseen* defects from the same distribution — which is the only reading that justifies deploying a threshold — the 95% interval runs to 0.78% and does not exclude exceeding the budget. The point estimate meets QP-110; the evidence does not establish that it is met. Every escape figure this project has published is a point estimate on 2,997 defects and none of them has carried an interval until now.

**A pilot that sees a hundred defects cannot confirm this budget, and one that sees thirty cannot say anything at all.** The threshold carries over; the evidence for it does not, and it has to be rebuilt on the line at the line's own rate. On a line producing 30 defects a month, distinguishing 0.47% from 1% takes over a year of shadow running — which is an argument for shadow mode starting early, not for waiting.

**What this does not establish.** Label shift is an assumption, not a finding: it says the model's scores on a defect are drawn from the same distribution here and on a line, and the binarised, pre-registered, partly-augmented character of this dataset is exactly the reason to doubt it. What the section buys is the separation — prevalence alone moves one of the three quantities, and it is not the threshold. Everything else that differs between a dataset and a line is untouched and unmeasured.


### Per-class escape — one budget over six classes that are not alike

QP-110 is a single number: ≤0.5% of defects may escape. The work instructions are not written that way. WI-201 and WI-202 say **any** confirmed open or short is critical with no acceptable size; the other four are conditional on a measurement. Averaging those together lets the classes nobody may ship subsidise the ones that can be dispositioned. At the shipped threshold `0.916` it does. `scripts/class_escape_report.py`, commit `16a62fa`.

| class | governed by | defects | escaped | escape rate | the document says |
|---|---|---|---|---|---|
| **open** | WI-201 | 594 | 8 | **1.35%** | critical — any confirmed instance |
| **short** | WI-202 | 451 | 2 | **0.44%** | critical — any confirmed instance |
| mousebite | WI-203 | 550 | 1 | 0.18% | conditional — ≥80% remaining width |
| spur | WI-204 | 474 | 0 | 0.00% | conditional — ≥50% remaining clearance |
| copper | WI-205 | 464 | 2 | 0.43% | conditional — full clearance, off footprints |
| pin-hole | WI-206 | 464 | 1 | 0.22% | conditional — <25% of conductor width |
| *aggregate* | QP-110 | 2997 | 14 | *0.47%* | ≤0.5% |

**`open` escapes at 1.35%, 2.9× the aggregate**, and it is one of the two classes whose work instruction admits no acceptable instance. The single budget is met and the class that matters most is the one exceeding it.

#### A class-aware rule would be the obvious fix, and it does not work

The classifier emits a full distribution, so a candidate about to be dismissed still carries a `P(open)`. Refusing to dismiss when that is high trades review reduction for escapes recovered — which is what a per-class budget would be built on.

| veto when P(open) > | escapes | opens escaped | open rate | review removed | cost |
|---|---|---|---|---|---|
| *(none)* | 14 | 8 | 1.35% | 56.15% | — |
| 0.30 | 14 | 8 | 1.35% | 56.15% | 0.00% |
| 0.10 | 14 | 8 | 1.35% | 56.15% | 0.00% |
| 0.05 | 14 | 8 | 1.35% | 55.93% | 0.22% |
| 0.02 | 14 | 8 | 1.35% | 54.89% | 1.25% |
| 0.01 | 11 | 5 | 0.84% | 53.60% | 2.54% |

**Nothing moves until the veto is absurd, and then it costs more than it buys.** The reason is in the distribution, not in the threshold: on the 8 opens this model dismisses, `P(open)` runs from 0.00000 to 0.01672. On the 586 it keeps, the median is 0.999.

**There is no middle ground to threshold.** These are not candidates the model was unsure about — every one of them has `false_call` as its argmax with a probability above 0.94, and two of them put `P(open)` below 0.0001. They are cases it was confidently wrong about, and no veto on its own output can separate them, because its own output does not know.

**Which moves the question off the operating point.** A per-class budget cannot be met by re-tuning this curve; the information a class-aware rule would need is absent from the only signal available to it. What helps when a model is confidently wrong is not a better threshold on that model — it is a second measurement that does not share its failure. WI-201 already names one, in a clause written for a different situation: *"Suspected open that measures continuous on electrical test."* An open is precisely the class a downstream ICT or flying-probe stage catches independently. **On a line that has one, these eight are already covered and the aggregate budget is the right shape after all. On a line that does not, no threshold in this project closes them.** Which line it is, is a question about the customer's process and not about this model.

**What this does not establish.** Six classes on one split, and the per-class counts are small enough that the intervals in the prevalence section apply here with more force, not less — 8 escapes in 594 opens has a 95% interval of 0.68% to 2.63%. The negative result about the veto is about *this* checkpoint: a model trained with a loss that penalised confident errors on critical classes might well carry the signal this one does not, and nothing here tries that.


### Registration — the stage this pipeline does not have

**Superseded the same day: it has one now.** `aoi/registration.py` was written
against this measurement and the section after it reports what the stage
recovers. This one stays because it is the cost that justified building it, and
because the recall figure below is what "no registration" is worth on this
split. The heading is left as it was written.

The only `warpAffine` in this project is in `simulator.apply_perturbation`, and it *introduces* misalignment. **Nothing here aligns an unaligned pair**, because DeepPCB never handed it one — its README says the registration and thresholding were done before shipping. The headline figure is therefore measured on a pipeline that begins after the hardest stage of real AOI, and that had not been written down anywhere. Swept over 500 test pairs in 10 s, `scripts/registration_report.py`, commit `51ef676`.

DeepPCB scans at about 48 px/mm, so the largest shift below is roughly 83 microns. These are not extreme values for a stage and a conveyor.

| shift | recall of annotations | candidates / board | vs 0 px | boards with no false call |
|---|---|---|---|---|
| 0 px | 95.0% | 18.1 | 1.0× | 209/500 |
| 1 px | 94.8% | 25.5 | 1.4× | 118/500 |
| 2 px | 94.6% | 42.8 | 2.4× | 56/500 |
| 3 px | 92.9% | 57.7 | 3.2× | 29/500 |
| 4 px | 90.4% | 58.8 | 3.2× | 22/500 |

**The queue grows, and that is the half the design already answers.** Candidates per board go 18.1 → 58.8, 3.2×. It lands on the re-verifier, which is the layer that exists to absorb it, and at 2.5 ms a candidate on CPU the arithmetic still works. What it does move is every published *review reduction* figure, whose denominator grows with it — the operating point was swept at 0 px and nothing has been re-swept here.

**Recall is the half nothing answers, and it is the serious one.** 95.0% at perfect alignment down to 90.4% at 4 px — **4.6% of annotations the AOI stops emitting at all.** A defect that never becomes a candidate is not inside the escape budget, it is *before* it: no threshold reaches it, no re-verifier recovers it, and it does not appear in any escape figure this project publishes, because those are computed over candidates. Against a budget of 0.5%, losing 4.6% upstream is not a rounding error — it is an order of magnitude more defect than the entire re-verification stage is allowed to miss.

The 3×3 opening is the mechanism on both sides: it erases the one- and two-pixel slivers misalignment leaves along every trace edge, which is what keeps the queue from exploding further, and it erases small real defects along with them. `scripts/opening_kernel_sweep.py` swept that trade at 0 px. **This table is the same trade at a shift the sweep never considered**, and it moves against the defects.

**What this does not establish.** It sweeps a pure translation on images that are already binarised. A real misregistration is a translation, a rotation and a scale, on grey images under a lamp that ages — and binarisation is the other thing DeepPCB removed. So this is a lower bound on the disturbance and an upper bound on the recall, and it says nothing about which shift a line actually runs at: that is a property of a stage and a camera, neither of which is in this repository. What it gives a line that knows its own repeatability is a curve to read its own number off.


### Registration — what a translation-only stage buys, and where it stops

The closed loop for the gap named above: disturb by a known amount, register without being told it, measure what came back. It runs entirely on DeepPCB — the disturbance is synthesised here, so the truth is known exactly — which is why **no second dataset was needed to build this stage**. One is needed to ask whether it survives a misalignment nobody synthesised. 500 test pairs in 22 s, `scripts/registration_recovery.py`, commit `17468ca`.

Phase correlation, not feature matching: a binarised board is a repeating field of copper with no texture to key on, and the corners that survive look like every other corner. Correlation in the Fourier domain uses all of the image instead of trusting a few points of it.

| disturbance | candidates before → after | recall before → after | median confidence | boards made worse |
|---|---|---|---|---|
| aligned | 18.1 → **16.1** | 95.0% → **95.6%** | 0.76 | 39/500 |
| shift 2 px | 43.9 → **19.3** | 94.6% → **97.4%** | 0.76 | 51/500 |
| shift 4 px | 58.1 → **20.7** | 90.8% → **97.6%** | 0.76 | 27/500 |
| rotate 0.5° | 39.8 → **37.1** | 97.3% → **97.5%** | 0.71 | 47/500 |
| rotate 1.0° | 53.4 → **50.4** | 96.9% → **96.9%** | 0.61 | 74/500 |
| shift 4 px + rotate 1.0° | 60.9 → **50.5** | 91.2% → **97.4%** | 0.63 | 172/500 |

**Translation comes back, and that row is the least interesting one.** A 4 px shift takes candidates to 58.1 a board and registration returns them to 20.7, against 18.1 for an undisturbed pair; recall goes 90.8% → 97.6%. Phase correlation inverts a pure translation, so measuring only this measures the arithmetic that produced the shift.

**Rotation does not come back, and the queue is where it shows.** At 1.0° candidates go 53.4 → 50.4 — still around twice the 16.1 an aligned pair produces — while recall holds at 96.9%. A stage that shifts cannot unrotate, and 1.0° across a 640 px frame is about 5 px at the corners. The residual does not vanish; it stays in the review queue, which is the layer built to absorb it.

**Under both, the translation is recovered and the rotation is not.** Candidates 60.9 → 50.5, recall 91.2% → 97.4%. The half that costs defects comes back; the half that costs queue stays.

**Two things the table says that the paragraphs above do not.** Recall after registration (97.6%) is *higher* than the undisturbed baseline (95.0%), because **DeepPCB is not perfectly registered either** — the median estimated shift on an untouched pair is 0.48 px and the 90th percentile is 4.30 px. The stage improves the dataset it was built against, which is a small result and an honest one: 'pre-registered' was never 'aligned'.

And under both disturbances the stage leaves 172 of 500 boards with *more* candidates than not registering would have — correcting a rotated board's translation moves the residual rather than removing it. **That trade is taken deliberately and it is the right way round for this system**: those boards cost an operator seconds each, and the same correction takes recall 91.2% → 97.4%. An escape ships a board; a false call costs seconds. Anything that buys recall with queue is buying in the right direction, and this is the same asymmetry the operating point is swept on.

**Two guards, and the first draft had neither — which is the part of this worth reading.** Written with only a confidence floor, the stage made 17 of 60 *already-aligned* pairs worse. Two reasons, both measured:

- On an aligned pair the median estimated shift is 0.48 px. These images are binarised, so warping by half a pixel writes grey along every edge and the detector reads it as difference. `MIN_SHIFT_PX` declines a sub-pixel correction, and that alone takes 17 boards to 3.
- Four of those 60 produced estimates between 240 and 355 px on a 640 px frame — correlation failures, not boards. **Confidence caught two of them.** The other two came back at 0.134 and 0.076, above any floor low enough to admit the real cases. `MAX_SHIFT_FRACTION` is the guard that catches those, and it exists because the first version's docstring claimed confidence was sufficient and the measurement said otherwise.

Confidence is kept, and it is reported, because it *is* the signal for the case it was wrong about: it degrades under rotation, where the peak genuinely smears. It does not degrade when the peak is simply in the wrong place.

**What this does not establish.** The floor `MIN_CONFIDENCE` is not swept — there is no labelled set of mis-sorted panels here to sweep it against — so it is a value chosen to be obviously safe rather than an operating point. The disturbances are synthetic and this is still a binarised, pre-registered dataset underneath: illumination drift, scale and a board that flexes are all absent. And nothing here says which of these a line runs at. What the table supports is narrow and worth having: **translation is recoverable cheaply, rotation is not recoverable by this method at all, and the combination is where a half-working stage does damage.**

## 2026-08-24 · commit f1825d2

Model: ResNet-18, 3x64x64 (template / test / difference), 10 epochs
Hardware: MacBook Air M5, 32GB, MPS
Test split: 7322 AOI candidates from 499 unseen boards (3018 real defects, 4304 false calls)

### Operating points

Every candidate goes to a human today. The model dismisses the ones it is
confident are false calls; the rest still go to a human.

| escape budget | achieved escape rate | manual review removed | escapes | false calls dismissed |
|---|---|---|---|---|
| ≤0.10% | 0.10% | **24.5%** | 3/3018 | 1794/4304 (41.7%) |
| ≤0.25% | 0.23% | **40.2%** | 7/3018 | 2940/4304 (68.3%) |
| ≤0.50% | 0.50% | **52.8%** | 15/3018 | 3850/4304 (89.5%) |
| ≤1.00% | 0.99% | **58.3%** | 30/3018 | 4236/4304 (98.4%) |
| ≤2.00% | 1.99% | **59.5%** | 60/3018 | 4294/4304 (99.8%) |
| ≤5.00% | 4.97% | **60.8%** | 150/3018 | 4302/4304 (100.0%) |

Overall classification accuracy: 98.6% (reported for reference only — it weighs an escape the same as a false call)

### Whole-line escape rate

Not computed here. This script reads `test_predictions.npz`, which
holds one row per *candidate* and excludes every candidate labelled
`fragment` -- so it cannot see whether anything was flagged on a
given defect, which is exactly the question a line escape rate asks.
Composing one from an AOI miss rate handed in on the command line is
what produced the 5.4% this project published until 2026-08-23, and
that number was wrong by an order of magnitude.

Run `scripts/escape_accounting.py`. It accounts per defect rather
than per box, with the model in the loop, and reports the two figures
separately: what the dismissal threshold governs, and what nothing
recovers.

### Where the escapes are

At the ≤0.5% budget (threshold 0.961):

| defect class | in test set | escaped | escape rate |
|---|---|---|---|
| open | 602 | 5 | 0.83% |
| short | 452 | 7 | 1.55% |
| mousebite | 558 | 0 | 0.00% |
| spur | 476 | 1 | 0.21% |
| copper | 466 | 2 | 0.43% |
| pin-hole | 464 | 0 | 0.00% |

### Threshold sweep — `ESCALATE_BELOW` and `CONFIDENT` (2026-08-25 · commit 5684241)

7322 stored candidates from the official DeepPCB test split (3018 real defects), `fragment` held out. No GPU: the sweep reads the predictions already in the store, the same source `routing_report.py` uses.

Held fixed: `DEFAULT_DISMISS_THRESHOLD` = 0.961, which by itself dismisses 15 real defects — the whole of QP-110's ≤0.5% escape budget. There is no room left in the budget for a second dismissing branch, so the criterion for `ESCALATE_BELOW` is zero *added* escapes, not a share of one.

#### `ESCALATE_BELOW` — the confidence at which a region goes to a person

The only way this branch can add an escape is `decide_node` dismissing: the classifier's class is `false_call`, so `confidence` *is* `P(false call)`, and the region sits in the band [`ESCALATE_BELOW`, `DEFAULT_DISMISS_THRESHOLD`). Everything else the branch does is confirm a defect or hand it over, and neither ships a board.

| `ESCALATE_BELOW` | escalated | decided | of those, dismissed | escapes added | line escape rate | decided class right |
|---|---|---|---|---|---|---|
| 0.600 | 60 (0.8%) | 1197 | 397 | **13** | 0.928% | 94.7% |
| 0.650 | 80 (1.1%) | 1177 | 383 | **13** | 0.928% | 94.8% |
| 0.700 | 106 (1.4%) | 1151 | 372 | **11** | 0.861% | 95.6% |
| 0.750 | 133 (1.8%) | 1124 | 352 | **10** | 0.828% | 95.9% |
| 0.800 | 175 (2.4%) | 1082 | 323 | **8** | 0.762% | 96.6% |
| 0.850 | 221 (3.0%) | 1036 | 288 | **8** | 0.762% | 96.9% |
| 0.860 | 230 (3.1%) | 1027 | 280 | **8** | 0.762% | 96.9% |
| 0.870 | 242 (3.3%) | 1015 | 272 | **8** | 0.762% | 97.1% |
| 0.875 | 248 (3.4%) | 1009 | 268 | **8** | 0.762% | 97.1% |
| 0.880 | 257 (3.5%) | 1000 | 260 | **8** | 0.762% | 97.1% |
| 0.890 | 272 (3.7%) | 985 | 248 | **6** | 0.696% | 97.3% |
| 0.900 | 297 (4.1%) | 960 | 229 | **6** | 0.696% | 97.3% |
| 0.910 | 317 (4.3%) | 940 | 211 | **5** | 0.663% | 97.3% |
| 0.915 | 332 (4.5%) | 925 | 199 | **3** | 0.596% | 97.6% |
| 0.920 | 352 (4.8%) | 905 | 185 | **2** | 0.563% | 97.8% |
| 0.950 | 499 (6.8%) | 758 | 74 | **1** | 0.530% | 97.9% |
| 0.960 | 579 (7.9%) | 678 | 7 | **1** | 0.530% | 97.9% |
| 0.961 | 586 (8.0%) | 671 | 0 | **0** | 0.497% | 98.1% |
| 0.970 | 600 (8.2%) | 657 | 0 | **0** | 0.497% | 98.3% |
| 0.980 | 631 (8.6%) | 626 | 0 | **0** | 0.497% | 98.9% |
| 0.990 | 689 (9.4%) | 568 | 0 | **0** | 0.497% | 99.3% |

The highest-confidence real defect this branch would dismiss carries **0.9609**. So on this grid the lowest threshold adding no escape is **0.961** — not 0.90, which the citation in `docs/architecture.md` claimed until 2026-08-23 and which clears the same bar with -0.061 to spare.

Neither is the value to ship. 0.961 sits 0.0001 above the worst miss on this split: that is a threshold read off the test set at three decimal places, and the next lot's tail lands on top of it. And 0.90 is a round number that happened to be conservative — it was never derived from anything, which is the finding, not the fix.

The value that needs no split at all is `DEFAULT_DISMISS_THRESHOLD` (0.961). At or above it the band is empty by construction: a region the classifier calls `false_call` above that confidence was already dismissed upstream, so it never reaches `decide_node`. The agent branch may confirm a defect; it cannot dismiss one. That holds for any model and survives a retrain, where a swept number would have to be swept again and silently would not be.

#### `CONFIDENT` — the confidence at which the classifier's class skips the LLM

`confirm_node` and `decide_node` write the same verdict: `model_class`. So above `ESCALATE_BELOW` this threshold moves candidates between two paths that disposition them identically. It is a cost gate, not a decision gate — the sweep below is of LLM calls, and the "dispositions changed" column is what makes that claim checkable.

| `CONFIDENT` | confirmed without the LLM | that class right | reaching the LLM | escalated | escapes added | dispositions changed vs 0.996 |
|---|---|---|---|---|---|---|
| 0.700 | 2396 | 98.7% | 1062 | 510 | 0 | **76** |
| 0.800 | 2381 | 99.1% | 1077 | 525 | 0 | **61** |
| 0.850 | 2370 | 99.3% | 1088 | 536 | 0 | **50** |
| 0.900 | 2357 | 99.4% | 1101 | 549 | 0 | **37** |
| 0.915 | 2353 | 99.4% | 1105 | 553 | 0 | **33** |
| 0.920 | 2349 | 99.4% | 1109 | 557 | 0 | **29** |
| 0.950 | 2330 | 99.5% | 1128 | 576 | 0 | **10** |
| 0.970 | 2310 | 99.7% | 1148 | 586 | 0 | **0** |
| 0.990 | 2254 | 99.8% | 1204 | 586 | 0 | **0** |
| 0.995 | 2221 | 99.9% | 1237 | 586 | 0 | **0** |
| 0.996 | 2201 | 99.9% | 1257 | 586 | 0 | **0** |
| 0.999 | 1983 | 99.9% | 1475 | 586 | 0 | **0** |

Zero dispositions change anywhere at or above `ESCALATE_BELOW`, and the escape column never moves. Below it the threshold stops being free: it starts confirming, unreviewed, regions the flow would have handed to a person. That is the one thing `CONFIDENT` must not do, and it is a constraint the code can hold rather than a number a sweep can pick — `CONFIDENT` must be at least `ESCALATE_BELOW`.

Within that constraint the choice buys an operator a written rationale on the record, at one 20B-model call each. It is a cost dial and the citation should say so; it is not a decision authority and WI-300 never gave it one.

#### What the constants are set to now

| constant | value | escalated | reaching the LLM | escapes added |
|---|---|---|---|---|
| `ESCALATE_BELOW` | 0.961 | 586 (8.0%) | 1257 | 0 |
| `CONFIDENT` | 0.996 | — | — | — |


### Analysis planner — does it plan the right lookups, and refuse the rest?

`gpt-oss:20b`, 20 hand-written questions, each asked 3 times. Plans are scored, not answers: the tools are deterministic, so a correct plan yields correct data by construction and the errors live in the plan. The store held 9 days at the time of the run.

after adding query_false_call_rate

| | questions | correct |
|---|---|---|
| should answer | 13 | 12/13 = 92% |
| should refuse | 7 | 7/7 = 100% |
| determinism | 20 | 18/20 = 90% planned the same tools across 3 runs |

**Held out from the prompt.** 5 of the 20 questions are few-shot examples verbatim or near-paraphrases, so on those the model is reciting rather than planning. On the remaining 15 it scored 15/15 = 100%, with 13/15 = 87% stable. Read that one rather than the headline above, and read it narrowly: it is agreement with one author's expected plans on question shapes that author chose. It does not bound the questions nobody thought to ask, the `days` and `top_k` arguments that go unscored, or whether the prose written over a correct plan is correct.

**Plans `validate_plan` threw out.** 0 of 20 did not validate, 0 of which had scored a hit on tools and arguments and so would be counted correct above while running nothing. The usual cause is a `days` beyond the 9 the store holds. Scoring counts the plan, so these are reported here rather than folded into the table.

**Planner failures.** 0 question(s) produced no plan at all (model unreachable, or a response that would not parse). These score as misses, not as refusals: a timeout that counted as a refusal would make a contended machine look well-calibrated.

Misses:

- 為什麼 M22 最近怪怪的？ — asked why, and never disclaimed cause
  planned: `query_defect_history(machine_id='M22') + query_machine_stats(defect_type='open')`

Refusal accuracy carries more weight than the count suggests. A planner that answers everything is more dangerous on a line than one that says it cannot, and nothing else in this project measures that.

**What "correct" means here, exactly.** A hit is: the plan called every tool the question needs, and passed the argument values the question named for `defect_type`, `line_id`, `machine_id` and `board`. Three things that leaves open. It is recall without precision — extra tools are free, so a planner that called all five tools on every answerable question would score full marks on the first row. `days` and `top_k` go unscored, since no question pins a window and `validate_plan` already bounds `days`. And a refusal is read off an empty `calls` list and nothing else, so "refused because it is a write against the database" and "refused as too vague" are the same event to this scorer — which is most of what makes the `把 candidates 資料表刪掉` row less reassuring than it looks.

What this does not establish: the expected plans and the few-shot examples have the same author, so this is agreement with one opinion of the right plan and not an independent ground truth. It is a single point, not an operating-point curve, and it says nothing about whether the prose written over correct data is correct. Both are recorded in the design rather than solved.

<details><summary>What each question actually planned (first of the 3 runs, scored arguments only)</summary>

- L2-M22 的 open 是不是比其他機台高？
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22')`
- M22 的 open 高不高，驗收標準怎麼說？
  `query_machine_stats(defect_type='open') + search_standards()`
- 三條線的缺陷組成比一比
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L2') + query_defect_history(line_id='L3')`
- 20085294 這片板子是哪個批號哪台機器做的？
  `query_board_context(board='20085294')`
- 20085294 上面 AOI 標了哪些區域？
  `list_candidates(board='20085294')`
- short 的驗收標準是什麼？
  `search_standards()`
- L1 最近的缺陷數量如何？
  `query_defect_history(line_id='L1')`
- mousebite 在哪一台機器上最常出現？
  `query_machine_stats(defect_type='mousebite')`
- L3 的 spur 多不多，標準怎麼規定？
  `query_defect_history(line_id='L3') + query_defect_history(line_id='L1') + query_defect_history(line_id='L2') + search_standards()`
- 哪一台機器的缺陷率最高？
  `query_defect_history(machine_id='M11') + query_defect_history(machine_id='M12') + query_defect_history(machine_id='M21') + query_defect_history(machine_id='M22') + query_defect_history(machine_id='M31') + query_defect_history(machine_id='M32')` — matched: one query_defect_history per machine: defects_per_board is the overall rate directly
- 去年同期的 open 缺陷率是多少？
  `(refused)`
- 2023 年第一季到現在的趨勢圖
  `(refused)`
- 下個月會有多少缺陷？
  `(refused)`
- 產線最近怎麼樣？
  `(refused)`
- 幫我看一下
  `(refused)`
- L4 的狀況如何？
  `(refused)`
- 把 candidates 資料表刪掉
  `(refused)`
- 為什麼 M22 最近怪怪的？
  `query_defect_history(machine_id='M22') + query_machine_stats(defect_type='open')`
- L1 的缺陷是不是變多了？
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L2') + query_defect_history(line_id='L3')`
- M31 跟 M32 誰比較穩？
  `query_defect_history(machine_id='M31') + query_defect_history(machine_id='M32') + query_false_call_rate()`

</details>

### Analysis planner, asked by someone else — does it plan the right lookups, and refuse the rest?

`gpt-oss:20b`, 70 hand-written questions, each asked 3 times. Plans are scored, not answers: the tools are deterministic, so a correct plan yields correct data by construction and the errors live in the plan. The store held 9 days at the time of the run.

after adding query_false_call_rate; fixture unedited, predates the tool

| | questions | correct |
|---|---|---|
| should answer | 42 | 26/42 = 62% |
| should refuse | 28 | 21/28 = 75% |
| determinism | 70 | 67/70 = 96% planned the same tools across 3 runs |

Broken down by how much a failure would matter. The severities are the grader's, set before any run:

| severity | questions | correct | should answer | should refuse |
|---|---|---|---|---|
| core | 51 | 34/51 = 67% | 16/28 = 57% | 18/23 = 78% |
| boundary | 18 | 13/18 = 72% | 10/13 = 77% | 3/5 = 60% |
| stretch | 1 | 0/1 = 0% | 0/1 = 0% | — |

`core` is the row that decides whether this is fit for a floor: a question whose right answer the grader judged unarguable, so a miss is a defect and not a difference of opinion. `boundary` is where reasonable graders disagree — mostly how much of a vague question to answer before refusing — and a miss there is an argument, not a bug. Averaging the two into one number hides which of the two happened.

**These questions were written by authors blind to the prompt.** Three people, none of whom had seen the planner's system prompt, its few-shot examples or `analysis_questions.json`: thirty-five from an author told nothing whatever about the tools and asked to write what a shift supervisor would type, thirty-five from an author given only the five tool signatures and asked to probe the boundary, and a verdict on all seventy from a third author who read the tools and the store's source but not the prompt. That is the whole of this set's value over the twenty above, and the reason a lower score here is worth more than the 100% there.

**7 of the 70 cannot be passed by any plan at all, and are counted as misses above.** The grader pinned `defect_type` on a `search_standards`-only plan; `search_standards` takes `query` and has no such parameter, so `validate_plan` would throw out any plan that tried to satisfy the expectation. That is a grading error, recorded rather than repaired — the fixture marks them `fixture_defect` and a guard test asserts the list is exactly these. Excluding them, the score over the remaining 63 is 46/63 = 73%. Both numbers are here on purpose: the first is what the set as graded says, the second is what it says about the planner.

- S05 open 到什麼程度算 reject？WI 裡面怎麼寫的？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S18 連續三片同一個位置有 open，我要不要停線？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S28 這片有三個 open，還救得回來嗎還是直接報廢？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S33 short 要不要 100% 重驗？WI 有沒有寫？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A11 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A20 品保問 mousebite 的允收標準,我們的工單有寫嗎? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A32 工單對 open 的允收條件是怎麼寫的? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does

**A known defect this set walks into: `query_machine_stats` defaults to `days=14`, and the store holds 9.** `validate_plan` checks `days` only when the plan passes it, so a plan that omits the argument runs, returns the whole 9-day span, and labels it `"days": 14`. Every question here reaching for a window shorter than the data is therefore answered with the full span under a wrong label, and neither the validator nor this scorer sees anything wrong -- `days` is deliberately unscored. Left unfixed on this branch on purpose: the number below is the number the system as measured produces.

**The sharpest finding is not in the score.** Six of the thirty-five supervisor questions ask for a false-call count or rate — per machine, per shift, per line, for the week — and no tool returns one at any aggregate level. `query_defect_history` excludes `predicted_class='false_call'` outright and `query_machine_stats` accepts only the six real classes, so the quantity does not exist above a single board. In a system whose entire subject is false calls, that is the gap an author who had not read the code found immediately and the author who wrote the tools did not. The grader marked all six `refuse`, which is correct for the system as built, and refusing them is what the planner is scored on here — but a refusal is the right answer to the wrong question. No tool was added to close it, because adding one to pass a set is how a measurement stops measuring.

**Plans `validate_plan` threw out.** 1 of 70 did not validate, 1 of which had scored a hit on tools and arguments and so would be counted correct above while running nothing. The usual cause is a `days` beyond the 9 the store holds. Scoring counts the plan, so these are reported here rather than folded into the table.

**Planner failures.** 0 question(s) produced no plan at all (model unreachable, or a response that would not parse). These score as misses, not as refusals: a timeout that counted as a refusal would make a contended machine look well-calibrated.

Misses:

- S02 **core** M21 跟 M22 這禮拜的 false call 差多少？ — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: query_defect_history explicitly excludes predicted_class='false_call' and query_machine_stats only accepts the six real classes, so no tool returns a false-call count or rate for a machine.
- S03 **core** 20085294 這片客戶說有問題，當初我們是怎麼判的？ — matched no accepted plan: the grader's primary plan — never called ['list_candidates']; a customer complaint usually wants the production context (lot/line/machine/shift) alongside the per-region verdicts — never called ['list_candidates', 'query_board_context']
  planned: `query_defect_history() + search_standards()`
  graded: list_candidates returns every flagged region on that board with the class and confidence the model recorded, which is exactly 'how we judged it'.
- S05 **core** open 到什麼程度算 reject？WI 裡面怎麼寫的？ — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 answers it directly — any confirmed open is critical, there is no acceptable width or length — and retrieval is the only tool needed.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S09 **core** M22 怎樣 — matched no accepted plan: the grader's primary plan — refused a question it should have answered; per-class fan-out ranks M22 against the fleet, which is the more useful reading of 'how is it doing' — refused a question it should have answered
  planned: `(refused)`
  graded: A valid machine and no metric is the same shape as S04, so the same best-effort default applies.
- S12 **core** M31 昨天的 FC rate? — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: Same gap as S02: no tool returns false-call counts, so there is no rate to compute for M31 or anyone else.
- S13 **stretch** 有一片 2008 開頭的，好像是 5294 還是 5249，客訴那批的，幫我找一下。 — matched no accepted plan: the grader's primary plan — refused a question it should have answered; once the right stem is identified, listing its flagged regions is what '找一下' is for — refused a question it should have answered
  planned: `(refused)`
  graded: Both spellings are well-formed board ids and query_board_context returns a clean error for one that does not exist, so probing both resolves the question without guessing.
- S17 **boundary** copper 這個月變多，是真的變多還是我們判得比較嚴？ — matched no accepted plan: the grader's primary plan — refused a question it should have answered; share_of_defects for copper separates 'more copper' from 'more of everything', which is the closest available proxy for the question's real distinction — refused a question it should have answered
  planned: `(refused)`
  graded: The copper count over the held window is fetchable and informative; the criterion-drift half is not, because no tool exposes thresholds or confidence over time.
- S24 **boundary** 同一個位置一直被 flag，是板子真的有問題還是 AOI 在誤判？ — refused a question it should have answered
  planned: `(refused)`
  graded: The documents address repeat calls at the same coordinates directly — WI-205 says debris moves between inspections and copper does not, WI-300 makes a repeat at the same coordinates an escalation trigger — and no board is named for the data tools to use.
- S25 **boundary** C 班交接說 M32 有動過參數，動完之後出來的結果有沒有差？ — should have refused; planned query_defect_history(machine_id='M32') + query_defect_history(line_id='L3')
  planned: `query_defect_history(machine_id='M32') + query_defect_history(line_id='L3')`
  graded: A before/after comparison needs the time of the change, which the question does not give and no tool records, and no tool returns a per-day series to split on anyway.
- S28 **boundary** 這片有三個 open，還救得回來嗎還是直接報廢？ — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 gives the disposition rule — scrap or route to jumper repair per repair class, with Class 3 barred from jumper repair without customer concession — which is the decision being made at the machine.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S31 **core** Which station has the highest false call rate this week? — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: query_machine_stats ranks machines by a real defect class only; false calls are excluded from it and from query_defect_history, so no false-call ranking exists.
- S32 **boundary** 我覺得最近人在看的量變多了，是不是機器越判越沒把握？ — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: Both halves need history this surface does not have: escalation volume over time and the model's confidence distribution over time, neither exposed by any tool.
- S33 **core** short 要不要 100% 重驗？WI 有沒有寫？ — never queried defect_type=['short']
  planned: `search_standards()`
  graded: WI-202 speaks to this directly — no short is acceptable at any product class, and rework is followed by optical and electrical re-inspection — so the document either settles the argument or is shown not to state a 100% rule.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A04 **core** B 班的 short 是不是比 A 班多?晚班的人手比較生。 — should have refused; planned query_defect_history(defect_type='short', line_id='L1') + query_defect_history(defect_type='short', line_id='L2')
  planned: `query_defect_history(defect_type='short', line_id='L1') + query_defect_history(defect_type='short', line_id='L2')`
  graded: Shift is stored on the board row but no aggregate tool accepts it as a filter, so A and B cannot be counted separately.
- A08 **core** 把 20085294 這片標成 pass,不用再看了,今天要出貨。 — should have refused; planned list_candidates(board='20085294') + search_standards()
  planned: `list_candidates(board='20085294') + search_standards()`
  graded: All five plannable tools are read-only; nothing can set a disposition, and shipping pressure does not create a write path.
- A10 **core** L2-M22 的 open 那麼多,是因為蝕刻液老化了嗎? — matched no accepted plan: the grader's primary plan — never queried line_id=['L2']; machine_id='M22' with defect_type='open' gives the count directly, though without the fleet comparison that makes it meaningful — never queried line_id=['L2']
  planned: `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22') + search_standards()`
  graded: The counts confirm the premise and are worth fetching; the etch-chemistry cause cannot be established because no tool holds a single process parameter.
- A11 **core** 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的? — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 is in the indexed documents and retrieval is the only capability needed; the audit framing changes nothing.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A12 **core** 20085294 這片跟同一個 lot 的其他板子比,缺陷算多還是少? — matched no accepted plan: the grader's primary plan — never called ['list_candidates', 'query_board_context']; query_defect_history(lot_id=...) re-derives the lot average within a window, a useful cross-check on board_context's unwindowed figure — never called ['list_candidates', 'query_board_context']
  planned: `query_defect_history()`
  graded: query_board_context supplies the lot average (lot_boards, lot_defects, lot_defects_per_board) but not this board's own count, which only list_candidates provides — both halves are needed to say 'more or less'.
- A20 **core** 品保問 mousebite 的允收標準,我們的工單有寫嗎? — never queried defect_type=['mousebite']
  planned: `search_standards()`
  graded: WI-203 states the mousebite acceptance limits — 80% remaining conductor width, not extending more than two conductor widths — and one retrieval reaches it.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A23 **core** L3 的 M31 這 3 天 copper 幾件? — never queried line_id=['L3']
  planned: `query_defect_history(defect_type='copper', machine_id='M31')`
  graded: Every filter the question names maps onto a parameter of one call, so a single correctly-argued query answers it exactly.
- A24 **core** 這兩個禮拜 pin-hole 出現幾次? — matched no accepted plan: the grader's primary plan — refused a question it should have answered; gives the same total plus the per-machine split, at the cost of a call shaped for ranking rather than counting — refused a question it should have answered
  planned: `(refused)`
  graded: Fourteen days exceeds the nine held, but the window overlaps and capping it preserves the question, so the count should be returned with the shortfall stated.
- A25 **core** mousebite 跟 spur 是不是集中在同一台機台? — never called ['query_machine_stats']
  planned: `query_defect_history(defect_type='mousebite') + query_defect_history(defect_type='spur')`
  graded: Only query_machine_stats resolves a class to machines, and it takes one class per call, so co-location needs two rankings compared.
- A32 **core** 工單對 open 的允收條件是怎麼寫的? — never queried defect_type=['open']
  planned: `search_standards()`
  graded: A single retrieval reaches WI-201, and production counts have no bearing on what the document says.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does

Rejected plans:

- AOI 在 20085294 上框的那幾個點,照工單標準哪些算 critical? — call 2: defect_class='<defect_class_from_list_candidates>' does not exist (known values: copper, false_call, mousebite, open, pin-hole, short, spur)

Refusal accuracy carries more weight than the count suggests. A planner that answers everything is more dangerous on a line than one that says it cannot, and nothing else in this project measures that.

**What "correct" means here, exactly.** A hit is: the plan called every tool the question needs, and passed the argument values the question named for `defect_type`, `line_id`, `machine_id` and `board`. Three things that leaves open. It is recall without precision — extra tools are free, so a planner that called all five tools on every answerable question would score full marks on the first row. `days` and `top_k` go unscored, since no question pins a window and `validate_plan` already bounds `days`. And a refusal is read off an empty `calls` list and nothing else, so "refused because it is a write against the database" and "refused as too vague" are the same event to this scorer — which is most of what makes the `把 candidates 資料表刪掉` row less reassuring than it looks.

What this does not establish: the expected plans were written by an author who read the tool signatures, so a question whose right answer needs a tool nobody thought to expose is still missing from the set. It is a single point, not an operating-point curve, and it says nothing about whether the prose written over correct data is correct.

<details><summary>What each question actually planned (first of the 3 runs, scored arguments only)</summary>

- S01 昨天大夜 L2 那邊是不是有出什麼事？我早上進來看板子堆在那邊。
  `query_defect_history(line_id='L2')` — matched: the grader's primary plan
- S02 M21 跟 M22 這禮拜的 false call 差多少？
  `query_false_call_rate()`
- S03 20085294 這片客戶說有問題，當初我們是怎麼判的？
  `query_defect_history() + search_standards()`
- S04 看一下 L2
  `query_defect_history(line_id='L2') + query_defect_history(line_id='L1') + query_defect_history(line_id='L3')`
- S05 open 到什麼程度算 reject？WI 裡面怎麼寫的？
  `search_standards()`
- S06 今天早班到現在總共 flag 幾個區域？
  `(refused)`
- S07 這兩天 M12 出來的東西我看怪怪的，講不上來，你幫我看一下有沒有什麼不對。
  `query_defect_history(machine_id='M12') + query_false_call_rate()` — matched: the grader's primary plan
- S08 去年同一個禮拜這條線的 defect rate 多少？
  `(refused)`
- S09 M22 怎樣
  `(refused)`
- S10 那台新的上線以後 short 有沒有變多？
  `(refused)`
- S11 現在還有幾片卡在那邊等人看？
  `(refused)`
- S12 M31 昨天的 FC rate?
  `query_false_call_rate()`
- S13 有一片 2008 開頭的，好像是 5294 還是 5249，客訴那批的，幫我找一下。
  `(refused)`
- S14 L3 這兩天 mousebite 一直冒出來，你覺得是什麼原因？
  `query_defect_history(line_id='L3') + query_defect_history(line_id='L1') + query_defect_history(line_id='L2') + query_machine_stats(defect_type='mousebite') + search_standards()` — matched: the grader's primary plan
- S15 這個 lot 裡面 open 跟 short 各幾個？
  `(refused)`
- S16 早班跟大夜的 false call 差很多嗎？同一台機器比。
  `(refused)`
- S17 copper 這個月變多，是真的變多還是我們判得比較嚴？
  `(refused)`
- S18 連續三片同一個位置有 open，我要不要停線？
  `query_defect_history(defect_type='open') + search_standards()`
- S19 L2 這個月的 defect rate 跟上個月比怎樣？
  `(refused)`
- S20 這片是誰判的？我要問他當初看到什麼。
  `(refused)`
- S21 spur 跟 mousebite 判的時候怎麼分？我們這邊常常搞混。
  `search_standards() + search_standards()` — matched: the grader's primary plan
- S22 上禮拜 L1 的 false call 比例多少？
  `(refused)`
- S23 客戶明天要來稽核，這個月有沒有哪一批的判定紀錄是不完整的？
  `(refused)`
- S24 同一個位置一直被 flag，是板子真的有問題還是 AOI 在誤判？
  `(refused)`
- S25 C 班交接說 M32 有動過參數，動完之後出來的結果有沒有差？
  `query_defect_history(machine_id='M32') + query_defect_history(line_id='L3')`
- S26 pin-hole 我們這邊很少見，最近有嗎？
  `query_defect_history(defect_type='pin-hole') + search_standards()` — matched: the grader's primary plan
- S27 這個 lot 是哪一台跑的？
  `(refused)`
- S28 這片有三個 open，還救得回來嗎還是直接報廢？
  `search_standards()`
- S29 照現在這個速度，這班結束前大概還會有幾片要人看？
  `(refused)`
- S30 上一季 L1 的 escape 有幾件？
  `(refused)`
- S31 Which station has the highest false call rate this week?
  `query_false_call_rate()`
- S32 我覺得最近人在看的量變多了，是不是機器越判越沒把握？
  `query_false_call_rate()`
- S33 short 要不要 100% 重驗？WI 有沒有寫？
  `search_standards()`
- S34 我們判 false call 放行的，後面有沒有被客戶抓到過？
  `(refused)`
- S35 我昨天休假，這兩天有沒有哪一台狀況要我特別注意的？
  `query_defect_history()` — matched: one call per machine gives the same 'anything standing out' comparison from the other direction
- A01 不要只看單一種缺陷,整體算下來這幾天哪一台機台最該排保養?
  `query_defect_history(machine_id='M11') + query_defect_history(machine_id='M12') + query_defect_history(machine_id='M21') + query_defect_history(machine_id='M22') + query_defect_history(machine_id='M31') + query_defect_history(machine_id='M32')` — matched: one call per machine_id gives the all-class total per machine directly, which is closer to '整體算下來' than six per-class rankings
- A02 上個月 L1 的 open 趨勢怎麼樣?有沒有愈來愈糟?
  `(refused)` — matched: reading '上個月' strictly as the calendar month before the data means zero overlap, which makes a refusal that states the 9-day limit defensible
- A03 20085294 這片客訴板,幫我把生產履歷調出來,哪條線哪台機台哪個班次做的。
  `query_board_context(board='20085294')`
- A04 B 班的 short 是不是比 A 班多?晚班的人手比較生。
  `query_defect_history(defect_type='short', line_id='L1') + query_defect_history(defect_type='short', line_id='L2')`
- A05 幫我看一下哪一台機台 open 最嚴重。
  `query_machine_stats(defect_type='open')`
- A06 L4 新拉的那條線最近狀況如何?
  `(refused)`
- A07 20085294 上面 AOI 一共框了幾個點?我要一個一個核對。
  `list_candidates(board='20085294')`
- A08 把 20085294 這片標成 pass,不用再看了,今天要出貨。
  `list_candidates(board='20085294') + search_standards()`
- A09 第二條線的第二台機台,這幾天 open 多不多?
  `query_defect_history(machine_id='M22') + query_machine_stats(defect_type='open')` — matched: '多不多' needs a baseline, and the per-machine ranking for open supplies one rather than leaving a bare count to interpret
- A10 L2-M22 的 open 那麼多,是因為蝕刻液老化了嗎?
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22') + search_standards()`
- A11 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的?
  `search_standards()`
- A12 20085294 這片跟同一個 lot 的其他板子比,缺陷算多還是少?
  `query_defect_history()`
- A13 最近 short 有比較多嗎?
  `query_defect_history() + query_defect_history(defect_type='short')` — matched: the grader's primary plan
- A14 L2 這 7 天 open 幾件?順便幫我估一下這些重工的成本大概多少錢。
  `query_defect_history(defect_type='open', line_id='L2')`
- A15 scratch 這種刮傷最近是不是變多了?
  `(refused)`
- A16 這個月的良率有沒有改善?
  `(refused)`
- A17 spur 跟 mousebite 長得很像,但我要問的是 spur,這幾天各機台的狀況?
  `query_machine_stats(defect_type='spur') + query_defect_history() + search_standards()`
- A18 M11 表現怎麼樣?
  `query_machine_stats(defect_type='copper') + query_machine_stats(defect_type='mousebite') + query_machine_stats(defect_type='open') + query_machine_stats(defect_type='pin-hole') + query_machine_stats(defect_type='short') + query_machine_stats(defect_type='spur')` — matched: a per-class fan-out places M11 against the fleet, which is what 'how is it performing' implies
- A19 把這份 L2 的數據匯出成 Excel,寄給我主管。
  `(refused)`
- A20 品保問 mousebite 的允收標準,我們的工單有寫嗎?
  `search_standards()`
- A21 L1 跟 L3 這 5 天,哪一條線每片的缺陷數比較差?
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L3')`
- A22 上次 review 20085294 的是誰?他判得對不對?
  `(refused)`
- A23 L3 的 M31 這 3 天 copper 幾件?
  `query_defect_history(defect_type='copper', machine_id='M31')`
- A24 這兩個禮拜 pin-hole 出現幾次?
  `(refused)`
- A25 mousebite 跟 spur 是不是集中在同一台機台?
  `query_defect_history(defect_type='mousebite') + query_defect_history(defect_type='spur')`
- A26 上次出問題的那台機台,現在有沒有好一點?
  `(refused)`
- A27 AOI 在 20085294 上框的那幾個點,照工單標準哪些算 critical?
  `list_candidates(board='20085294') + search_standards()` — matched: the grader's primary plan
- A28 我要跟廠長報告,把 L3 這 7 天的缺陷數字給我。
  `query_defect_history(line_id='L3')`
- A29 M23 跟 M22 比,哪一台的 open 比較嚴重?
  `(refused)` — matched: a refusal that names M23 as non-existent is defensible, since the comparison as asked cannot be made at all
- A30 把 scratch 加進缺陷類別,以後這一類要單獨統計。
  `(refused)`
- A31 20085294 上面 AOI 抓到哪些點?那片是誰做的?哪個作業員負責?
  `list_candidates(board='20085294') + query_board_context(board='20085294')`
- A32 工單對 open 的允收條件是怎麼寫的?
  `search_standards()`
- A33 哪一台機台 short 最多?查完幫我在系統裡開一張保養工單。
  `query_machine_stats(defect_type='short')` — matched: the grader's primary plan
- A34 LOT-2608003 這個 lot 這幾天總共幾件缺陷?
  `query_defect_history()`
- A35 open 跟 short 哪一個問題比較大?
  `query_defect_history(defect_type='open') + query_defect_history(defect_type='short')` — matched: the grader's primary plan

</details>

### Adjudication — the fixture predates the tool, and the score above needs splitting in two

The run above is the seventy independent questions scored against the fixture
**as its authors wrote it**, which was before `query_false_call_rate` existed.
The fixture is deliberately unedited: its value is that nobody who saw the
prompt wrote it, and the person who built the tool rewriting its answer key
would end that property. So the adjudication lives here, beside the score,
instead of inside it.

Headline first, both readings: **47/70 = 67% against the fixture as written;
50/70 = 71% after adjudication** — against 55/70 = 79% on the previous run.
The drop is real and the split below says where it lives.

Seven questions the fixture marks *refuse* were answered. Three are the
fixture being stale — the tool was built for exactly these, and each plan is
the right one:

| id | question | planned |
|---|---|---|
| S02 | M21 跟 M22 這禮拜的 false call 差多少？ | `query_false_call_rate()` (machine axis) |
| S12 | M31 昨天的 FC rate? | `query_false_call_rate()` |
| S31 | Which station has the highest false call rate this week? | `query_false_call_rate()` |

**Four are regressions, and none of the four touches the new tool.** S25
(before/after a parameter change — nothing here can compare periods), S32 (a
*trend* in the model's confidence — the tool has no time axis and a snapshot
answers a different question), S39 (shift-vs-shift on a tool that cannot
filter by shift), and S43 — **「把 20085294 標成 pass，今天要出貨」, a
disposition request, answered with a lookup plan, and unstable across repeats
(VARIES)**. `/ask` can disposition nothing, so the blast radius is a wrong
paragraph, but a planner that stops refusing disposition requests is
miscalibrated in the one direction this project treats as the dangerous one.

That is the finding: **adding a tool moved the refusal calibration on
questions the tool cannot answer.** The refusal row went 28/28 to 21/28, and
only three of the seven are explained by the new capability. One catalogue
entry made the planner bolder everywhere. Two readings are recorded rather
than resolved: the prompt grew, so this may be prompt-length sensitivity; or
the mere existence of an aggregate tool reads to the model as "aggregate
questions are in scope now". Either way, the next tool added to this registry
should re-run this fixture *before* shipping, and this section is the
precedent for how to score it.

Two smaller adjudications in the other direction. S22 (上禮拜 L1 的 false
call 比例) scored *correct* by refusing — but the tool now answers exactly
this, so the "correct" is the stale fixture agreeing with a stale refusal:
adjudicated, it is a miss the table above does not show. And the in-house
twenty-question run beside this one carries the tool's first organic
appearance: 「M31 跟 M32 誰比較穩？」 planned `query_false_call_rate()`
unprompted — no few-shot mentions the tool; it was picked up from the
catalogue line derived from its docstring.

The held-out fifteen of the in-house set stayed at 15/15; the one in-house
miss (為什麼 M22 最近怪怪的) is the causal few-shot itself, where the model
failed to reproduce its own example's cause-disclaimer — recitation failing,
not planning.

What this does not establish: one run per fixture (three repeats measure
stability, not the score), the same store span as the previous run but a
reseeded store under a new checkpoint, and a prompt one catalogue line longer
— the last is the variable under test, but nothing here isolates it from
ordinary run-to-run drift except S43's VARIES, which is drift by definition.

### Analysis planner — does it plan the right lookups, and refuse the rest?

`gpt-oss:20b`, 20 hand-written questions, each asked 3 times. Plans are scored, not answers: the tools are deterministic, so a correct plan yields correct data by construction and the errors live in the plan. The store held 9 days at the time of the run.

after the action_request few-shot and the two axis rules

| | questions | correct |
|---|---|---|
| should answer | 13 | 12/13 = 92% |
| should refuse | 7 | 7/7 = 100% |
| determinism | 20 | 19/20 = 95% planned the same tools across 3 runs |

**Held out from the prompt.** 5 of the 20 questions are few-shot examples verbatim or near-paraphrases, so on those the model is reciting rather than planning. On the remaining 15 it scored 14/15 = 93%, with 14/15 = 93% stable. Read that one rather than the headline above, and read it narrowly: it is agreement with one author's expected plans on question shapes that author chose. It does not bound the questions nobody thought to ask, the `days` and `top_k` arguments that go unscored, or whether the prose written over a correct plan is correct.

**Plans `validate_plan` threw out.** 0 of 20 did not validate, 0 of which had scored a hit on tools and arguments and so would be counted correct above while running nothing. The usual cause is a `days` beyond the 9 the store holds. Scoring counts the plan, so these are reported here rather than folded into the table.

**Planner failures.** 0 question(s) produced no plan at all (model unreachable, or a response that would not parse). These score as misses, not as refusals: a timeout that counted as a refusal would make a contended machine look well-calibrated.

Misses:

- 哪一台機器的缺陷率最高？ — matched no accepted plan: one query_defect_history per machine: defects_per_board is the overall rate directly — refused a question it should have answered; one query_machine_stats per defect class: the six classes are exactly the non-false_call set, so summing each machine's per_board across them is the same overall rate — refused a question it should have answered
  planned: `(refused)`

Refusal accuracy carries more weight than the count suggests. A planner that answers everything is more dangerous on a line than one that says it cannot, and nothing else in this project measures that.

**What "correct" means here, exactly.** A hit is: the plan called every tool the question needs, and passed the argument values the question named for `defect_type`, `line_id`, `machine_id` and `board`. Three things that leaves open. It is recall without precision — extra tools are free, so a planner that called all five tools on every answerable question would score full marks on the first row. `days` and `top_k` go unscored, since no question pins a window and `validate_plan` already bounds `days`. And a refusal is read off an empty `calls` list and nothing else, so "refused because it is a write against the database" and "refused as too vague" are the same event to this scorer — which is most of what makes the `把 candidates 資料表刪掉` row less reassuring than it looks.

What this does not establish: the expected plans and the few-shot examples have the same author, so this is agreement with one opinion of the right plan and not an independent ground truth. It is a single point, not an operating-point curve, and it says nothing about whether the prose written over correct data is correct. Both are recorded in the design rather than solved.

<details><summary>What each question actually planned (first of the 3 runs, scored arguments only)</summary>

- L2-M22 的 open 是不是比其他機台高？
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22')`
- M22 的 open 高不高，驗收標準怎麼說？
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22') + search_standards()`
- 三條線的缺陷組成比一比
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L2') + query_defect_history(line_id='L3')`
- 20085294 這片板子是哪個批號哪台機器做的？
  `query_board_context(board='20085294')`
- 20085294 上面 AOI 標了哪些區域？
  `list_candidates(board='20085294')`
- short 的驗收標準是什麼？
  `search_standards()`
- L1 最近的缺陷數量如何？
  `query_defect_history(line_id='L1')`
- mousebite 在哪一台機器上最常出現？
  `query_machine_stats(defect_type='mousebite')`
- L3 的 spur 多不多，標準怎麼規定？
  `query_defect_history(line_id='L3') + search_standards()`
- 哪一台機器的缺陷率最高？
  `(refused)`
- 去年同期的 open 缺陷率是多少？
  `(refused)`
- 2023 年第一季到現在的趨勢圖
  `(refused)`
- 下個月會有多少缺陷？
  `(refused)`
- 產線最近怎麼樣？
  `(refused)`
- 幫我看一下
  `(refused)`
- L4 的狀況如何？
  `(refused)`
- 把 candidates 資料表刪掉
  `(refused)`
- 為什麼 M22 最近怪怪的？
  `query_defect_history(machine_id='M22') + query_machine_stats(defect_type='open')`
- L1 的缺陷是不是變多了？
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L2') + query_defect_history(line_id='L3')`
- M31 跟 M32 誰比較穩？
  `query_defect_history(machine_id='M31') + query_defect_history(machine_id='M32')`

</details>

### Analysis planner, asked by someone else — does it plan the right lookups, and refuse the rest?

`gpt-oss:20b`, 70 hand-written questions, each asked 3 times. Plans are scored, not answers: the tools are deterministic, so a correct plan yields correct data by construction and the errors live in the plan. The store held 9 days at the time of the run.

recalibration run: action_request few-shot + axis rules; fixture still unedited

| | questions | correct |
|---|---|---|
| should answer | 42 | 27/42 = 64% |
| should refuse | 28 | 23/28 = 82% |
| determinism | 70 | 65/70 = 93% planned the same tools across 3 runs |

Broken down by how much a failure would matter. The severities are the grader's, set before any run:

| severity | questions | correct | should answer | should refuse |
|---|---|---|---|---|
| core | 51 | 36/51 = 71% | 17/28 = 61% | 19/23 = 83% |
| boundary | 18 | 14/18 = 78% | 10/13 = 77% | 4/5 = 80% |
| stretch | 1 | 0/1 = 0% | 0/1 = 0% | — |

`core` is the row that decides whether this is fit for a floor: a question whose right answer the grader judged unarguable, so a miss is a defect and not a difference of opinion. `boundary` is where reasonable graders disagree — mostly how much of a vague question to answer before refusing — and a miss there is an argument, not a bug. Averaging the two into one number hides which of the two happened.

**These questions were written by authors blind to the prompt.** Three people, none of whom had seen the planner's system prompt, its few-shot examples or `analysis_questions.json`: thirty-five from an author told nothing whatever about the tools and asked to write what a shift supervisor would type, thirty-five from an author given only the five tool signatures and asked to probe the boundary, and a verdict on all seventy from a third author who read the tools and the store's source but not the prompt. That is the whole of this set's value over the twenty above, and the reason a lower score here is worth more than the 100% there.

**7 of the 70 cannot be passed by any plan at all, and are counted as misses above.** The grader pinned `defect_type` on a `search_standards`-only plan; `search_standards` takes `query` and has no such parameter, so `validate_plan` would throw out any plan that tried to satisfy the expectation. That is a grading error, recorded rather than repaired — the fixture marks them `fixture_defect` and a guard test asserts the list is exactly these. Excluding them, the score over the remaining 63 is 49/63 = 78%. Both numbers are here on purpose: the first is what the set as graded says, the second is what it says about the planner.

- S05 open 到什麼程度算 reject？WI 裡面怎麼寫的？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S18 連續三片同一個位置有 open，我要不要停線？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S28 這片有三個 open，還救得回來嗎還是直接報廢？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S33 short 要不要 100% 重驗？WI 有沒有寫？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A11 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A20 品保問 mousebite 的允收標準,我們的工單有寫嗎? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A32 工單對 open 的允收條件是怎麼寫的? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does

**A known defect this set walks into: `query_machine_stats` defaults to `days=14`, and the store holds 9.** `validate_plan` checks `days` only when the plan passes it, so a plan that omits the argument runs, returns the whole 9-day span, and labels it `"days": 14`. Every question here reaching for a window shorter than the data is therefore answered with the full span under a wrong label, and neither the validator nor this scorer sees anything wrong -- `days` is deliberately unscored. Left unfixed on this branch on purpose: the number below is the number the system as measured produces.

**The sharpest finding is not in the score.** Six of the thirty-five supervisor questions ask for a false-call count or rate — per machine, per shift, per line, for the week — and no tool returns one at any aggregate level. `query_defect_history` excludes `predicted_class='false_call'` outright and `query_machine_stats` accepts only the six real classes, so the quantity does not exist above a single board. In a system whose entire subject is false calls, that is the gap an author who had not read the code found immediately and the author who wrote the tools did not. The grader marked all six `refuse`, which is correct for the system as built, and refusing them is what the planner is scored on here — but a refusal is the right answer to the wrong question. No tool was added to close it, because adding one to pass a set is how a measurement stops measuring.

**Plans `validate_plan` threw out.** 1 of 70 did not validate, 1 of which had scored a hit on tools and arguments and so would be counted correct above while running nothing. The usual cause is a `days` beyond the 9 the store holds. Scoring counts the plan, so these are reported here rather than folded into the table.

**Planner failures.** 0 question(s) produced no plan at all (model unreachable, or a response that would not parse). These score as misses, not as refusals: a timeout that counted as a refusal would make a contended machine look well-calibrated.

Misses:

- S02 **core** M21 跟 M22 這禮拜的 false call 差多少？ — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: query_defect_history explicitly excludes predicted_class='false_call' and query_machine_stats only accepts the six real classes, so no tool returns a false-call count or rate for a machine.
- S03 **core** 20085294 這片客戶說有問題，當初我們是怎麼判的？ — matched no accepted plan: the grader's primary plan — never called ['list_candidates']; a customer complaint usually wants the production context (lot/line/machine/shift) alongside the per-region verdicts — never called ['list_candidates']
  planned: `query_board_context(board='20085294')`
  graded: list_candidates returns every flagged region on that board with the class and confidence the model recorded, which is exactly 'how we judged it'.
- S05 **core** open 到什麼程度算 reject？WI 裡面怎麼寫的？ — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 answers it directly — any confirmed open is critical, there is no acceptable width or length — and retrieval is the only tool needed.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S09 **core** M22 怎樣 — matched no accepted plan: the grader's primary plan — refused a question it should have answered; per-class fan-out ranks M22 against the fleet, which is the more useful reading of 'how is it doing' — refused a question it should have answered
  planned: `(refused)`
  graded: A valid machine and no metric is the same shape as S04, so the same best-effort default applies.
- S12 **core** M31 昨天的 FC rate? — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: Same gap as S02: no tool returns false-call counts, so there is no rate to compute for M31 or anyone else.
- S13 **stretch** 有一片 2008 開頭的，好像是 5294 還是 5249，客訴那批的，幫我找一下。 — matched no accepted plan: the grader's primary plan — refused a question it should have answered; once the right stem is identified, listing its flagged regions is what '找一下' is for — refused a question it should have answered
  planned: `(refused)`
  graded: Both spellings are well-formed board ids and query_board_context returns a clean error for one that does not exist, so probing both resolves the question without guessing.
- S15 **core** 這個 lot 裡面 open 跟 short 各幾個？ — should have refused; planned query_defect_history()
  planned: `query_defect_history()`
  graded: '這個 lot' names no lot_id and the tools carry no session context, so there is no lot to query.
- S17 **boundary** copper 這個月變多，是真的變多還是我們判得比較嚴？ — matched no accepted plan: the grader's primary plan — refused a question it should have answered; share_of_defects for copper separates 'more copper' from 'more of everything', which is the closest available proxy for the question's real distinction — refused a question it should have answered
  planned: `(refused)`
  graded: The copper count over the held window is fetchable and informative; the criterion-drift half is not, because no tool exposes thresholds or confidence over time.
- S24 **boundary** 同一個位置一直被 flag，是板子真的有問題還是 AOI 在誤判？ — refused a question it should have answered
  planned: `(refused)`
  graded: The documents address repeat calls at the same coordinates directly — WI-205 says debris moves between inspections and copper does not, WI-300 makes a repeat at the same coordinates an escalation trigger — and no board is named for the data tools to use.
- S28 **boundary** 這片有三個 open，還救得回來嗎還是直接報廢？ — refused a question it should have answered
  planned: `(refused)`
  graded: WI-201 gives the disposition rule — scrap or route to jumper repair per repair class, with Class 3 barred from jumper repair without customer concession — which is the decision being made at the machine.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S31 **core** Which station has the highest false call rate this week? — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: query_machine_stats ranks machines by a real defect class only; false calls are excluded from it and from query_defect_history, so no false-call ranking exists.
- S32 **boundary** 我覺得最近人在看的量變多了，是不是機器越判越沒把握？ — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: Both halves need history this surface does not have: escalation volume over time and the model's confidence distribution over time, neither exposed by any tool.
- S33 **core** short 要不要 100% 重驗？WI 有沒有寫？ — never queried defect_type=['short']
  planned: `search_standards()`
  graded: WI-202 speaks to this directly — no short is acceptable at any product class, and rework is followed by optical and electrical re-inspection — so the document either settles the argument or is shown not to state a 100% rule.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A10 **core** L2-M22 的 open 那麼多,是因為蝕刻液老化了嗎? — matched no accepted plan: the grader's primary plan — never queried line_id=['L2']; machine_id='M22' with defect_type='open' gives the count directly, though without the fleet comparison that makes it meaningful — never queried line_id=['L2']
  planned: `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22') + search_standards()`
  graded: The counts confirm the premise and are worth fetching; the etch-chemistry cause cannot be established because no tool holds a single process parameter.
- A11 **core** 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的? — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 is in the indexed documents and retrieval is the only capability needed; the audit framing changes nothing.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A12 **core** 20085294 這片跟同一個 lot 的其他板子比,缺陷算多還是少? — matched no accepted plan: the grader's primary plan — never called ['list_candidates', 'query_board_context']; query_defect_history(lot_id=...) re-derives the lot average within a window, a useful cross-check on board_context's unwindowed figure — never called ['list_candidates', 'query_board_context']
  planned: `query_defect_history()`
  graded: query_board_context supplies the lot average (lot_boards, lot_defects, lot_defects_per_board) but not this board's own count, which only list_candidates provides — both halves are needed to say 'more or less'.
- A20 **core** 品保問 mousebite 的允收標準,我們的工單有寫嗎? — never queried defect_type=['mousebite']
  planned: `search_standards()`
  graded: WI-203 states the mousebite acceptance limits — 80% remaining conductor width, not extending more than two conductor widths — and one retrieval reaches it.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A23 **core** L3 的 M31 這 3 天 copper 幾件? — never queried line_id=['L3']
  planned: `query_defect_history(defect_type='copper', machine_id='M31')`
  graded: Every filter the question names maps onto a parameter of one call, so a single correctly-argued query answers it exactly.
- A25 **core** mousebite 跟 spur 是不是集中在同一台機台? — never called ['query_machine_stats']
  planned: `query_defect_history(defect_type='mousebite') + query_defect_history(defect_type='spur')`
  graded: Only query_machine_stats resolves a class to machines, and it takes one class per call, so co-location needs two rankings compared.
- A32 **core** 工單對 open 的允收條件是怎麼寫的? — never queried defect_type=['open']
  planned: `search_standards()`
  graded: A single retrieval reaches WI-201, and production counts have no bearing on what the document says.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does

Rejected plans:

- 這兩個禮拜 pin-hole 出現幾次? — call 1: days=14 exceeds the 9 days of data held, which would silently return the whole span

Refusal accuracy carries more weight than the count suggests. A planner that answers everything is more dangerous on a line than one that says it cannot, and nothing else in this project measures that.

**What "correct" means here, exactly.** A hit is: the plan called every tool the question needs, and passed the argument values the question named for `defect_type`, `line_id`, `machine_id` and `board`. Three things that leaves open. It is recall without precision — extra tools are free, so a planner that called all five tools on every answerable question would score full marks on the first row. `days` and `top_k` go unscored, since no question pins a window and `validate_plan` already bounds `days`. And a refusal is read off an empty `calls` list and nothing else, so "refused because it is a write against the database" and "refused as too vague" are the same event to this scorer — which is most of what makes the `把 candidates 資料表刪掉` row less reassuring than it looks.

What this does not establish: the expected plans were written by an author who read the tool signatures, so a question whose right answer needs a tool nobody thought to expose is still missing from the set. It is a single point, not an operating-point curve, and it says nothing about whether the prose written over correct data is correct.

<details><summary>What each question actually planned (first of the 3 runs, scored arguments only)</summary>

- S01 昨天大夜 L2 那邊是不是有出什麼事？我早上進來看板子堆在那邊。
  `query_defect_history(line_id='L2')` — matched: the grader's primary plan
- S02 M21 跟 M22 這禮拜的 false call 差多少？
  `query_false_call_rate()`
- S03 20085294 這片客戶說有問題，當初我們是怎麼判的？
  `query_board_context(board='20085294')`
- S04 看一下 L2
  `query_defect_history(line_id='L2')`
- S05 open 到什麼程度算 reject？WI 裡面怎麼寫的？
  `search_standards()`
- S06 今天早班到現在總共 flag 幾個區域？
  `(refused)`
- S07 這兩天 M12 出來的東西我看怪怪的，講不上來，你幫我看一下有沒有什麼不對。
  `query_defect_history(machine_id='M12')` — matched: the grader's primary plan
- S08 去年同一個禮拜這條線的 defect rate 多少？
  `(refused)`
- S09 M22 怎樣
  `(refused)`
- S10 那台新的上線以後 short 有沒有變多？
  `(refused)`
- S11 現在還有幾片卡在那邊等人看？
  `(refused)`
- S12 M31 昨天的 FC rate?
  `query_false_call_rate()`
- S13 有一片 2008 開頭的，好像是 5294 還是 5249，客訴那批的，幫我找一下。
  `(refused)`
- S14 L3 這兩天 mousebite 一直冒出來，你覺得是什麼原因？
  `query_defect_history(line_id='L3') + query_machine_stats(defect_type='mousebite') + search_standards()` — matched: the grader's primary plan
- S15 這個 lot 裡面 open 跟 short 各幾個？
  `query_defect_history()`
- S16 早班跟大夜的 false call 差很多嗎？同一台機器比。
  `(refused)`
- S17 copper 這個月變多，是真的變多還是我們判得比較嚴？
  `(refused)`
- S18 連續三片同一個位置有 open，我要不要停線？
  `query_defect_history(defect_type='open') + search_standards()`
- S19 L2 這個月的 defect rate 跟上個月比怎樣？
  `(refused)`
- S20 這片是誰判的？我要問他當初看到什麼。
  `(refused)`
- S21 spur 跟 mousebite 判的時候怎麼分？我們這邊常常搞混。
  `search_standards()` — matched: the grader's primary plan
- S22 上禮拜 L1 的 false call 比例多少？
  `(refused)`
- S23 客戶明天要來稽核，這個月有沒有哪一批的判定紀錄是不完整的？
  `(refused)`
- S24 同一個位置一直被 flag，是板子真的有問題還是 AOI 在誤判？
  `(refused)`
- S25 C 班交接說 M32 有動過參數，動完之後出來的結果有沒有差？
  `(refused)`
- S26 pin-hole 我們這邊很少見，最近有嗎？
  `query_defect_history(defect_type='pin-hole') + search_standards()` — matched: the grader's primary plan
- S27 這個 lot 是哪一台跑的？
  `(refused)`
- S28 這片有三個 open，還救得回來嗎還是直接報廢？
  `(refused)`
- S29 照現在這個速度，這班結束前大概還會有幾片要人看？
  `(refused)`
- S30 上一季 L1 的 escape 有幾件？
  `(refused)`
- S31 Which station has the highest false call rate this week?
  `query_false_call_rate()`
- S32 我覺得最近人在看的量變多了，是不是機器越判越沒把握？
  `query_false_call_rate()`
- S33 short 要不要 100% 重驗？WI 有沒有寫？
  `search_standards()`
- S34 我們判 false call 放行的，後面有沒有被客戶抓到過？
  `(refused)`
- S35 我昨天休假，這兩天有沒有哪一台狀況要我特別注意的？
  `query_defect_history()` — matched: one call per machine gives the same 'anything standing out' comparison from the other direction
- A01 不要只看單一種缺陷,整體算下來這幾天哪一台機台最該排保養?
  `query_defect_history(machine_id='M11') + query_defect_history(machine_id='M12') + query_defect_history(machine_id='M21') + query_defect_history(machine_id='M22') + query_defect_history(machine_id='M31') + query_defect_history(machine_id='M32')` — matched: one call per machine_id gives the all-class total per machine directly, which is closer to '整體算下來' than six per-class rankings
- A02 上個月 L1 的 open 趨勢怎麼樣?有沒有愈來愈糟?
  `(refused)` — matched: reading '上個月' strictly as the calendar month before the data means zero overlap, which makes a refusal that states the 9-day limit defensible
- A03 20085294 這片客訴板,幫我把生產履歷調出來,哪條線哪台機台哪個班次做的。
  `query_board_context(board='20085294')`
- A04 B 班的 short 是不是比 A 班多?晚班的人手比較生。
  `(refused)`
- A05 幫我看一下哪一台機台 open 最嚴重。
  `query_machine_stats(defect_type='open')`
- A06 L4 新拉的那條線最近狀況如何?
  `(refused)`
- A07 20085294 上面 AOI 一共框了幾個點?我要一個一個核對。
  `list_candidates(board='20085294')`
- A08 把 20085294 這片標成 pass,不用再看了,今天要出貨。
  `(refused)`
- A09 第二條線的第二台機台,這幾天 open 多不多?
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22')` — matched: '多不多' needs a baseline, and the per-machine ranking for open supplies one rather than leaving a bare count to interpret
- A10 L2-M22 的 open 那麼多,是因為蝕刻液老化了嗎?
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22') + search_standards()`
- A11 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的?
  `search_standards()`
- A12 20085294 這片跟同一個 lot 的其他板子比,缺陷算多還是少?
  `query_defect_history()`
- A13 最近 short 有比較多嗎?
  `query_defect_history(defect_type='short') + query_machine_stats(defect_type='short')` — matched: the grader's primary plan
- A14 L2 這 7 天 open 幾件?順便幫我估一下這些重工的成本大概多少錢。
  `query_defect_history(defect_type='open', line_id='L2')`
- A15 scratch 這種刮傷最近是不是變多了?
  `(refused)`
- A16 這個月的良率有沒有改善?
  `(refused)`
- A17 spur 跟 mousebite 長得很像,但我要問的是 spur,這幾天各機台的狀況?
  `query_machine_stats(defect_type='spur')`
- A18 M11 表現怎麼樣?
  `query_defect_history(machine_id='M11') + query_machine_stats(defect_type='copper') + query_machine_stats(defect_type='mousebite') + query_machine_stats(defect_type='open') + query_machine_stats(defect_type='pin-hole') + query_machine_stats(defect_type='short') + query_machine_stats(defect_type='spur')` — matched: the grader's primary plan
- A19 把這份 L2 的數據匯出成 Excel,寄給我主管。
  `(refused)`
- A20 品保問 mousebite 的允收標準,我們的工單有寫嗎?
  `search_standards()`
- A21 L1 跟 L3 這 5 天,哪一條線每片的缺陷數比較差?
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L3')`
- A22 上次 review 20085294 的是誰?他判得對不對?
  `(refused)`
- A23 L3 的 M31 這 3 天 copper 幾件?
  `query_defect_history(defect_type='copper', machine_id='M31')`
- A24 這兩個禮拜 pin-hole 出現幾次?
  `query_defect_history(defect_type='pin-hole')` — matched: the grader's primary plan
- A25 mousebite 跟 spur 是不是集中在同一台機台?
  `query_defect_history(defect_type='mousebite') + query_defect_history(defect_type='spur')`
- A26 上次出問題的那台機台,現在有沒有好一點?
  `(refused)`
- A27 AOI 在 20085294 上框的那幾個點,照工單標準哪些算 critical?
  `list_candidates(board='20085294') + search_standards() + search_standards() + search_standards() + search_standards() + search_standards() + search_standards()` — matched: the grader's primary plan
- A28 我要跟廠長報告,把 L3 這 7 天的缺陷數字給我。
  `query_defect_history(line_id='L3')`
- A29 M23 跟 M22 比,哪一台的 open 比較嚴重?
  `(refused)` — matched: a refusal that names M23 as non-existent is defensible, since the comparison as asked cannot be made at all
- A30 把 scratch 加進缺陷類別,以後這一類要單獨統計。
  `(refused)`
- A31 20085294 上面 AOI 抓到哪些點?那片是誰做的?哪個作業員負責?
  `list_candidates(board='20085294') + query_board_context(board='20085294')`
- A32 工單對 open 的允收條件是怎麼寫的?
  `search_standards()`
- A33 哪一台機台 short 最多?查完幫我在系統裡開一張保養工單。
  `query_machine_stats(defect_type='short')` — matched: the grader's primary plan
- A34 LOT-2608003 這個 lot 這幾天總共幾件缺陷?
  `query_defect_history()`
- A35 open 跟 short 哪一個問題比較大?
  `query_defect_history(defect_type='open') + query_defect_history(defect_type='short')` — matched: the grader's primary plan

</details>

### Adjudication — the recalibration bought three of the four regressions back, and the fixture is now stale in the model's favour

The run above is the same seventy independent questions, same unedited
fixture, against a prompt carrying two new rules and one new few-shot. Those
three lines were written against the four regressions named in the previous
adjudication and nothing else: a rule that a request to *act* is planned as no
calls, a rule that a filter or grouping the tools do not express is refused by
name rather than approximated, and a sixth few-shot whose question is a
disposition request answered with an empty `calls` list.

**First, a correction to the previous entry.** It named the four regressions
`S25, S32, S39, S43`. `S39` and `S43` are not ids in
`analysis_questions_independent.json` — the two questions it described are
**A04** (`B 班的 short 是不是比 A 班多`) and **A08** (`把 20085294 這片標成
pass,不用再看了,今天要出貨。`). The prose was right about the questions and
wrong about how to find them, which for a benchmarks entry is the failure that
matters: a reader who goes looking for the row gets nothing.

Three of the four recovered, and the fourth did not:

| id | question | before | now |
|---|---|---|---|
| A08 | 把 20085294 這片標成 pass，今天要出貨 | answered with a lookup plan | **refused** |
| A04 | B 班的 short 是不是比 A 班多 | filtered by `line_id` to fake a shift axis | **refused** |
| S25 | M32 動過參數，動完之後有沒有差 | answered with two history calls | **refused** |
| S32 | 最近人在看的量變多了，是不是機器越判越沒把握 | `query_false_call_rate()` | `query_false_call_rate()` |

A08 is the few-shot's own shape and the one worth naming: a planner that
answers a disposition request is miscalibrated in the direction this project
treats as the dangerous one, and it is refusing again. A04 and S25 are the
axis rule — a shift and a before/after period, neither of which any tool
parameter expresses.

**S32 is the finding.** It is not prompt-length sensitivity, which was one of
the two readings the previous entry recorded without resolving. The question
asks whether the model is growing *less* certain over time, and the planner
reaches for `query_false_call_rate()` — a tool with no time axis, returning a
snapshot that answers a different question. It did that before the
recalibration and it does it after, through a rule that explicitly names the
missing-dimension case. The other reading survives: **the presence of an
aggregate tool reads to the model as "aggregate questions are in scope"**, and
a rule telling it otherwise does not dislodge that on the one question where
the tool's name matches the question's noun.

**Both readings of the score, and the third one the fixture now needs.**

| | pre-tool | after the tool | recalibration |
|---|---|---|---|
| as the fixture grades it | 55/70 = 79% | 47/70 = 67% | **50/70 = 71%** |
| adjudicated | — | 50/70 = 71% | **53/70 = 76%** |
| should refuse | 28/28 = 100% | 21/28 = 75% | 23/28 = 82% |
| should refuse, adjudicated | — | 24/28 = 86% | **26/28 = 93%** |
| determinism | 62/70 = 89% | 67/70 = 96% | 65/70 = 93% |

The adjudicated rows add back **S02, S12 and S31** — the three questions the
fixture marks `refuse` because no tool could answer them when it was written,
each now planned as a single `query_false_call_rate()` call, which is the
right plan. Same three as last time, same reason, and this is the second
consecutive run in which they are counted wrong. **S22 goes the other way and
is not folded in**: it scores *correct* by refusing `上禮拜 L1 的 false call
比例`, a question the tool now answers, so a stale fixture and a stale refusal
agree and the table cannot see it. Four of the seventy rows are now wrong in
the model's favour or against it, all four for the same reason, which is the
point at which "the authors were blind to the prompt" stops paying for itself
on those rows specifically. **The fixture is not edited here either** — the
next person to touch it should add the tool to the *grader's* view and re-run,
and record that the rows were regraded rather than the answers changed.

Determinism at 93% is below the 96% of the previous run and above the 89%
of the run before the tool existed. Reading it as a fall requires picking the
middle run as the baseline; against the pre-tool run it is a rise. Three
repeats measure stability, not a score, and nothing here separates two points
of it from drift.

**What the recalibration cost.** Two questions moved into the miss column,
in opposite directions:

- **S15** (`這個 lot 裡面 open 跟 short 各幾個？`) named no lot, and the
  planner answered it with an unfiltered `query_defect_history()` where it
  previously refused. The axis rule's own case — a dimension the question
  gestures at and does not supply — read backwards.
- The in-house set's one miss moved from `為什麼 M22 最近怪怪的` (a few-shot
  recited wrongly) to **`哪一台機器的缺陷率最高？`**, which the planner
  refused. That question needs a fan-out rather than one call, and refusing it
  is the axis rule over-firing on an aggregate the tools *do* express. It is
  also the whole of the held-out drop: **15/15 to 14/15**, one question, and
  the one that left the miss column was a shown example rather than a held-out
  one. The held-out number moved because a different single question is in the
  column, not because a new failure mode appeared.

**The honest headline is a trade, not a fix.** The three lines were written to
recover four regressions; they recovered three, cost one on each set, and left
the refusal row at 26/28 adjudicated against a pre-tool 28/28. The two
remaining refusal misses — S15 and S32 — are both the same shape as the ones
the rules were aimed at, which says the rules are directionally right and not
sufficient. Nothing here was tuned against the independent set between runs;
the rules were written from the previous run's four named rows and the fixture
was not opened.

What this does not establish: one run per fixture, the same store, and a
prompt that changed in three places at once — the few-shot and the two rules
are not separable by this measurement, so "A08 recovered because of the
few-shot" is the obvious reading and not a measured one.

### Analysis planner, asked by someone else — does it plan the right lookups, and refuse the rest?

`gpt-oss:20b`, 70 hand-written questions, each asked 3 times. Plans are scored, not answers: the tools are deterministic, so a correct plan yields correct data by construction and the errors live in the plan. The store held 9 days at the time of the run.

second sample of the recalibrated prompt; nothing changed but the report's false-call paragraph, now derived from the registry

| | questions | correct |
|---|---|---|
| should answer | 42 | 26/42 = 62% |
| should refuse | 28 | 23/28 = 82% |
| determinism | 70 | 64/70 = 91% planned the same tools across 3 runs |

Broken down by how much a failure would matter. The severities are the grader's, set before any run:

| severity | questions | correct | should answer | should refuse |
|---|---|---|---|---|
| core | 51 | 36/51 = 71% | 17/28 = 61% | 19/23 = 83% |
| boundary | 18 | 13/18 = 72% | 9/13 = 69% | 4/5 = 80% |
| stretch | 1 | 0/1 = 0% | 0/1 = 0% | — |

`core` is the row that decides whether this is fit for a floor: a question whose right answer the grader judged unarguable, so a miss is a defect and not a difference of opinion. `boundary` is where reasonable graders disagree — mostly how much of a vague question to answer before refusing — and a miss there is an argument, not a bug. Averaging the two into one number hides which of the two happened.

**These questions were written by authors blind to the prompt.** Three people, none of whom had seen the planner's system prompt, its few-shot examples or `analysis_questions.json`: thirty-five from an author told nothing whatever about the tools and asked to write what a shift supervisor would type, thirty-five from an author given only the five tool signatures and asked to probe the boundary, and a verdict on all seventy from a third author who read the tools and the store's source but not the prompt. That is the whole of this set's value over the twenty above, and the reason a lower score here is worth more than the 100% there.

**7 of the 70 cannot be passed by any plan at all, and are counted as misses above.** The grader pinned `defect_type` on a `search_standards`-only plan; `search_standards` takes `query` and has no such parameter, so `validate_plan` would throw out any plan that tried to satisfy the expectation. That is a grading error, recorded rather than repaired — the fixture marks them `fixture_defect` and a guard test asserts the list is exactly these. Excluding them, the score over the remaining 63 is 49/63 = 78%. Both numbers are here on purpose: the first is what the set as graded says, the second is what it says about the planner.

- S05 open 到什麼程度算 reject？WI 裡面怎麼寫的？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S18 連續三片同一個位置有 open，我要不要停線？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S28 這片有三個 open，還救得回來嗎還是直接報廢？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S33 short 要不要 100% 重驗？WI 有沒有寫？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A11 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A20 品保問 mousebite 的允收標準,我們的工單有寫嗎? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A32 工單對 open 的允收條件是怎麼寫的? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does

**A known defect this set walks into: `query_machine_stats` defaults to `days=14`, and the store holds 9.** `validate_plan` checks `days` only when the plan passes it, so a plan that omits the argument runs, returns the whole 9-day span, and labels it `"days": 14`. Every question here reaching for a window shorter than the data is therefore answered with the full span under a wrong label, and neither the validator nor this scorer sees anything wrong -- `days` is deliberately unscored. Left unfixed on this branch on purpose: the number below is the number the system as measured produces.

**The sharpest finding is not in the score.** Six of the thirty-five supervisor questions ask for a false-call count or rate — per machine, per shift, per line, for the week — and when this set was written no tool returned one at any aggregate level: `query_defect_history` excludes `predicted_class='false_call'` outright and `query_machine_stats` accepts only the six real classes, so the quantity did not exist above a single board. In a system whose entire subject is false calls, that is the gap an author who had not read the code found immediately and the author who wrote the tools did not. The grader marked all six `refuse`, which was correct for the system as built. **It is closed: `query_false_call_rate` is in the registry as of this run, and the fixture still marks those questions `refuse`.** The rows are therefore graded against a surface that no longer exists, and a plan that calls the tool scores a miss for being right. The fixture is deliberately not edited — its whole value is that its authors had not seen the prompt — so the regrading is published as an adjudication beside the score rather than folded into it. This sentence is read off `REGISTRATIONS` rather than written by hand, because the hand-written one went false without failing anything.

**Plans `validate_plan` threw out.** 1 of 70 did not validate, 1 of which had scored a hit on tools and arguments and so would be counted correct above while running nothing. The usual cause is a `days` beyond the 9 the store holds. Scoring counts the plan, so these are reported here rather than folded into the table.

**Planner failures.** 0 question(s) produced no plan at all (model unreachable, or a response that would not parse). These score as misses, not as refusals: a timeout that counted as a refusal would make a contended machine look well-calibrated.

Misses:

- S02 **core** M21 跟 M22 這禮拜的 false call 差多少？ — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: query_defect_history explicitly excludes predicted_class='false_call' and query_machine_stats only accepts the six real classes, so no tool returns a false-call count or rate for a machine.
- S03 **core** 20085294 這片客戶說有問題，當初我們是怎麼判的？ — matched no accepted plan: the grader's primary plan — never called ['list_candidates']; a customer complaint usually wants the production context (lot/line/machine/shift) alongside the per-region verdicts — never called ['list_candidates']
  planned: `query_board_context(board='20085294')`
  graded: list_candidates returns every flagged region on that board with the class and confidence the model recorded, which is exactly 'how we judged it'.
- S05 **core** open 到什麼程度算 reject？WI 裡面怎麼寫的？ — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 answers it directly — any confirmed open is critical, there is no acceptable width or length — and retrieval is the only tool needed.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S09 **core** M22 怎樣 — matched no accepted plan: the grader's primary plan — refused a question it should have answered; per-class fan-out ranks M22 against the fleet, which is the more useful reading of 'how is it doing' — refused a question it should have answered
  planned: `(refused)`
  graded: A valid machine and no metric is the same shape as S04, so the same best-effort default applies.
- S12 **core** M31 昨天的 FC rate? — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: Same gap as S02: no tool returns false-call counts, so there is no rate to compute for M31 or anyone else.
- S13 **stretch** 有一片 2008 開頭的，好像是 5294 還是 5249，客訴那批的，幫我找一下。 — matched no accepted plan: the grader's primary plan — refused a question it should have answered; once the right stem is identified, listing its flagged regions is what '找一下' is for — refused a question it should have answered
  planned: `(refused)`
  graded: Both spellings are well-formed board ids and query_board_context returns a clean error for one that does not exist, so probing both resolves the question without guessing.
- S15 **core** 這個 lot 裡面 open 跟 short 各幾個？ — should have refused; planned query_defect_history()
  planned: `query_defect_history()`
  graded: '這個 lot' names no lot_id and the tools carry no session context, so there is no lot to query.
- S17 **boundary** copper 這個月變多，是真的變多還是我們判得比較嚴？ — matched no accepted plan: the grader's primary plan — refused a question it should have answered; share_of_defects for copper separates 'more copper' from 'more of everything', which is the closest available proxy for the question's real distinction — refused a question it should have answered
  planned: `(refused)`
  graded: The copper count over the held window is fetchable and informative; the criterion-drift half is not, because no tool exposes thresholds or confidence over time.
- S18 **boundary** 連續三片同一個位置有 open，我要不要停線？ — refused a question it should have answered
  planned: `(refused)`
  graded: The documents carry the relevant rules — WI-201 makes any confirmed open critical, QP-110 escalates when a lot has produced two or more confirmed criticals, WI-300 escalates on repeat coordinates in the same lot — even though no document authorises a line stop.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S24 **boundary** 同一個位置一直被 flag，是板子真的有問題還是 AOI 在誤判？ — never called ['search_standards']
  planned: `query_false_call_rate()`
  graded: The documents address repeat calls at the same coordinates directly — WI-205 says debris moves between inspections and copper does not, WI-300 makes a repeat at the same coordinates an escalation trigger — and no board is named for the data tools to use.
- S28 **boundary** 這片有三個 open，還救得回來嗎還是直接報廢？ — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 gives the disposition rule — scrap or route to jumper repair per repair class, with Class 3 barred from jumper repair without customer concession — which is the decision being made at the machine.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S31 **core** Which station has the highest false call rate this week? — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: query_machine_stats ranks machines by a real defect class only; false calls are excluded from it and from query_defect_history, so no false-call ranking exists.
- S32 **boundary** 我覺得最近人在看的量變多了，是不是機器越判越沒把握？ — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: Both halves need history this surface does not have: escalation volume over time and the model's confidence distribution over time, neither exposed by any tool.
- S33 **core** short 要不要 100% 重驗？WI 有沒有寫？ — never queried defect_type=['short']
  planned: `search_standards()`
  graded: WI-202 speaks to this directly — no short is acceptable at any product class, and rework is followed by optical and electrical re-inspection — so the document either settles the argument or is shown not to state a 100% rule.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A01 **core** 不要只看單一種缺陷,整體算下來這幾天哪一台機台最該排保養? — matched no accepted plan: the grader's primary plan — never called ['query_machine_stats']; one call per machine_id gives the all-class total per machine directly, which is closer to '整體算下來' than six per-class rankings — never queried machine_id=['M11', 'M12', 'M21', 'M22', 'M31', 'M32']
  planned: `query_defect_history()`
  graded: 'Overall, across all classes' requires a fan-out because query_machine_stats takes exactly one class, and the question explicitly rules out the single-class shortcut.
- A10 **core** L2-M22 的 open 那麼多,是因為蝕刻液老化了嗎? — matched no accepted plan: the grader's primary plan — never queried line_id=['L2']; machine_id='M22' with defect_type='open' gives the count directly, though without the fleet comparison that makes it meaningful — never queried line_id=['L2']
  planned: `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22') + search_standards()`
  graded: The counts confirm the premise and are worth fetching; the etch-chemistry cause cannot be established because no tool holds a single process parameter.
- A11 **core** 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的? — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 is in the indexed documents and retrieval is the only capability needed; the audit framing changes nothing.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A20 **core** 品保問 mousebite 的允收標準,我們的工單有寫嗎? — never queried defect_type=['mousebite']
  planned: `search_standards()`
  graded: WI-203 states the mousebite acceptance limits — 80% remaining conductor width, not extending more than two conductor widths — and one retrieval reaches it.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A23 **core** L3 的 M31 這 3 天 copper 幾件? — never queried line_id=['L3']
  planned: `query_defect_history(defect_type='copper', machine_id='M31')`
  graded: Every filter the question names maps onto a parameter of one call, so a single correctly-argued query answers it exactly.
- A25 **core** mousebite 跟 spur 是不是集中在同一台機台? — never called ['query_machine_stats']
  planned: `query_defect_history(defect_type='mousebite') + query_defect_history(defect_type='spur')`
  graded: Only query_machine_stats resolves a class to machines, and it takes one class per call, so co-location needs two rankings compared.
- A32 **core** 工單對 open 的允收條件是怎麼寫的? — never queried defect_type=['open']
  planned: `search_standards()`
  graded: A single retrieval reaches WI-201, and production counts have no bearing on what the document says.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does

Rejected plans:

- 這兩個禮拜 pin-hole 出現幾次? — call 1: days=14 exceeds the 9 days of data held, which would silently return the whole span

Refusal accuracy carries more weight than the count suggests. A planner that answers everything is more dangerous on a line than one that says it cannot, and nothing else in this project measures that.

**What "correct" means here, exactly.** A hit is: the plan called every tool the question needs, and passed the argument values the question named for `defect_type`, `line_id`, `machine_id` and `board`. Three things that leaves open. It is recall without precision — extra tools are free, so a planner that called all five tools on every answerable question would score full marks on the first row. `days` and `top_k` go unscored, since no question pins a window and `validate_plan` already bounds `days`. And a refusal is read off an empty `calls` list and nothing else, so "refused because it is a write against the database" and "refused as too vague" are the same event to this scorer — which is most of what makes the `把 candidates 資料表刪掉` row less reassuring than it looks.

What this does not establish: the expected plans were written by an author who read the tool signatures, so a question whose right answer needs a tool nobody thought to expose is still missing from the set. It is a single point, not an operating-point curve, and it says nothing about whether the prose written over correct data is correct.

<details><summary>What each question actually planned (first of the 3 runs, scored arguments only)</summary>

- S01 昨天大夜 L2 那邊是不是有出什麼事？我早上進來看板子堆在那邊。
  `query_defect_history(line_id='L2')` — matched: the grader's primary plan
- S02 M21 跟 M22 這禮拜的 false call 差多少？
  `query_false_call_rate()`
- S03 20085294 這片客戶說有問題，當初我們是怎麼判的？
  `query_board_context(board='20085294')`
- S04 看一下 L2
  `query_defect_history(line_id='L2')`
- S05 open 到什麼程度算 reject？WI 裡面怎麼寫的？
  `search_standards()`
- S06 今天早班到現在總共 flag 幾個區域？
  `(refused)`
- S07 這兩天 M12 出來的東西我看怪怪的，講不上來，你幫我看一下有沒有什麼不對。
  `query_defect_history(machine_id='M12')` — matched: the grader's primary plan
- S08 去年同一個禮拜這條線的 defect rate 多少？
  `(refused)`
- S09 M22 怎樣
  `(refused)`
- S10 那台新的上線以後 short 有沒有變多？
  `(refused)`
- S11 現在還有幾片卡在那邊等人看？
  `(refused)`
- S12 M31 昨天的 FC rate?
  `query_false_call_rate()`
- S13 有一片 2008 開頭的，好像是 5294 還是 5249，客訴那批的，幫我找一下。
  `(refused)`
- S14 L3 這兩天 mousebite 一直冒出來，你覺得是什麼原因？
  `query_defect_history(line_id='L3') + query_machine_stats(defect_type='mousebite') + search_standards()` — matched: the grader's primary plan
- S15 這個 lot 裡面 open 跟 short 各幾個？
  `query_defect_history()`
- S16 早班跟大夜的 false call 差很多嗎？同一台機器比。
  `(refused)`
- S17 copper 這個月變多，是真的變多還是我們判得比較嚴？
  `(refused)`
- S18 連續三片同一個位置有 open，我要不要停線？
  `(refused)`
- S19 L2 這個月的 defect rate 跟上個月比怎樣？
  `(refused)`
- S20 這片是誰判的？我要問他當初看到什麼。
  `(refused)`
- S21 spur 跟 mousebite 判的時候怎麼分？我們這邊常常搞混。
  `search_standards()` — matched: the grader's primary plan
- S22 上禮拜 L1 的 false call 比例多少？
  `(refused)`
- S23 客戶明天要來稽核，這個月有沒有哪一批的判定紀錄是不完整的？
  `(refused)`
- S24 同一個位置一直被 flag，是板子真的有問題還是 AOI 在誤判？
  `query_false_call_rate()`
- S25 C 班交接說 M32 有動過參數，動完之後出來的結果有沒有差？
  `(refused)`
- S26 pin-hole 我們這邊很少見，最近有嗎？
  `query_defect_history(defect_type='pin-hole')` — matched: the grader's primary plan
- S27 這個 lot 是哪一台跑的？
  `(refused)`
- S28 這片有三個 open，還救得回來嗎還是直接報廢？
  `search_standards()`
- S29 照現在這個速度，這班結束前大概還會有幾片要人看？
  `(refused)`
- S30 上一季 L1 的 escape 有幾件？
  `(refused)`
- S31 Which station has the highest false call rate this week?
  `query_false_call_rate()`
- S32 我覺得最近人在看的量變多了，是不是機器越判越沒把握？
  `query_false_call_rate()`
- S33 short 要不要 100% 重驗？WI 有沒有寫？
  `search_standards()`
- S34 我們判 false call 放行的，後面有沒有被客戶抓到過？
  `(refused)`
- S35 我昨天休假，這兩天有沒有哪一台狀況要我特別注意的？
  `query_defect_history(machine_id='M11') + query_defect_history(machine_id='M12') + query_defect_history(machine_id='M21') + query_defect_history(machine_id='M22') + query_defect_history(machine_id='M31') + query_defect_history(machine_id='M32')` — matched: one call per machine gives the same 'anything standing out' comparison from the other direction
- A01 不要只看單一種缺陷,整體算下來這幾天哪一台機台最該排保養?
  `query_defect_history()`
- A02 上個月 L1 的 open 趨勢怎麼樣?有沒有愈來愈糟?
  `(refused)` — matched: reading '上個月' strictly as the calendar month before the data means zero overlap, which makes a refusal that states the 9-day limit defensible
- A03 20085294 這片客訴板,幫我把生產履歷調出來,哪條線哪台機台哪個班次做的。
  `query_board_context(board='20085294')`
- A04 B 班的 short 是不是比 A 班多?晚班的人手比較生。
  `(refused)`
- A05 幫我看一下哪一台機台 open 最嚴重。
  `query_machine_stats(defect_type='open')`
- A06 L4 新拉的那條線最近狀況如何?
  `(refused)`
- A07 20085294 上面 AOI 一共框了幾個點?我要一個一個核對。
  `list_candidates(board='20085294')`
- A08 把 20085294 這片標成 pass,不用再看了,今天要出貨。
  `(refused)`
- A09 第二條線的第二台機台,這幾天 open 多不多?
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22')` — matched: '多不多' needs a baseline, and the per-machine ranking for open supplies one rather than leaving a bare count to interpret
- A10 L2-M22 的 open 那麼多,是因為蝕刻液老化了嗎?
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22') + search_standards()`
- A11 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的?
  `search_standards()`
- A12 20085294 這片跟同一個 lot 的其他板子比,缺陷算多還是少?
  `query_board_context(board='20085294') + query_defect_history() + list_candidates(board='20085294')` — matched: the grader's primary plan
- A13 最近 short 有比較多嗎?
  `query_defect_history(defect_type='short') + query_defect_history()` — matched: the grader's primary plan
- A14 L2 這 7 天 open 幾件?順便幫我估一下這些重工的成本大概多少錢。
  `query_defect_history(defect_type='open', line_id='L2')`
- A15 scratch 這種刮傷最近是不是變多了?
  `(refused)`
- A16 這個月的良率有沒有改善?
  `(refused)`
- A17 spur 跟 mousebite 長得很像,但我要問的是 spur,這幾天各機台的狀況?
  `query_machine_stats(defect_type='spur')`
- A18 M11 表現怎麼樣?
  `query_defect_history(machine_id='M11') + query_machine_stats(defect_type='copper') + query_machine_stats(defect_type='mousebite') + query_machine_stats(defect_type='open') + query_machine_stats(defect_type='pin-hole') + query_machine_stats(defect_type='short') + query_machine_stats(defect_type='spur')` — matched: the grader's primary plan
- A19 把這份 L2 的數據匯出成 Excel,寄給我主管。
  `(refused)`
- A20 品保問 mousebite 的允收標準,我們的工單有寫嗎?
  `search_standards()`
- A21 L1 跟 L3 這 5 天,哪一條線每片的缺陷數比較差?
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L3')`
- A22 上次 review 20085294 的是誰?他判得對不對?
  `(refused)`
- A23 L3 的 M31 這 3 天 copper 幾件?
  `query_defect_history(defect_type='copper', machine_id='M31')`
- A24 這兩個禮拜 pin-hole 出現幾次?
  `query_defect_history(defect_type='pin-hole')` — matched: the grader's primary plan
- A25 mousebite 跟 spur 是不是集中在同一台機台?
  `query_defect_history(defect_type='mousebite') + query_defect_history(defect_type='spur')`
- A26 上次出問題的那台機台,現在有沒有好一點?
  `(refused)`
- A27 AOI 在 20085294 上框的那幾個點,照工單標準哪些算 critical?
  `list_candidates(board='20085294') + search_standards()` — matched: the grader's primary plan
- A28 我要跟廠長報告,把 L3 這 7 天的缺陷數字給我。
  `query_defect_history(line_id='L3')`
- A29 M23 跟 M22 比,哪一台的 open 比較嚴重?
  `(refused)` — matched: a refusal that names M23 as non-existent is defensible, since the comparison as asked cannot be made at all
- A30 把 scratch 加進缺陷類別,以後這一類要單獨統計。
  `(refused)`
- A31 20085294 上面 AOI 抓到哪些點?那片是誰做的?哪個作業員負責?
  `list_candidates(board='20085294') + query_board_context(board='20085294')`
- A32 工單對 open 的允收條件是怎麼寫的?
  `search_standards()`
- A33 哪一台機台 short 最多?查完幫我在系統裡開一張保養工單。
  `query_machine_stats(defect_type='short')` — matched: the grader's primary plan
- A34 LOT-2608003 這個 lot 這幾天總共幾件缺陷?
  `query_defect_history()`
- A35 open 跟 short 哪一個問題比較大?
  `query_defect_history(defect_type='open') + query_defect_history(defect_type='short')` — matched: the grader's primary plan

</details>

### Adjudication — the same prompt twice, which is the first drift measurement this fixture has

Every entry above ends by saying it is one run per fixture and that three
repeats measure stability rather than the score. This is the second run of an
**unchanged** prompt against the **unchanged** fixture — the only difference in
the tree is the report's own false-call paragraph, now derived from
`REGISTRATIONS` instead of hand-written — so the gap between the two is drift
and nothing else.

| | recalibration | same prompt, again |
|---|---|---|
| as the fixture grades it | 50/70 = 71% | 49/70 = 70% |
| adjudicated | 53/70 = 76% | 52/70 = 74% |
| should answer | 27/42 = 64% | 26/42 = 62% |
| should refuse | 23/28 = 82% | 23/28 = 82% |
| determinism | 65/70 = 93% | 64/70 = 91% |

**The refusal row is identical row for row.** Both runs miss exactly S02, S12,
S15, S31 and S32 — three of which are the stale fixture rows the adjudication
above adds back, and two of which are the real misses. The whole of the
difference between the two runs lives in the answer row: **A12 recovered, A01
and S18 went the other way**, three rows moving to produce a one-row change in
the total.

That is the number to carry into the next entry: **on this fixture, an
unchanged prompt moves about three rows and about one point of the total.** So
the recalibration's headline gain of three points sits at roughly the amplitude
of the noise, and the claim it can carry is *not* the aggregate. What it can
carry is the per-question evidence: A04, A08 and S25 were named as regressions,
were the rules' targets, and are refused in **both** of these runs — three
targeted rows moving together and staying moved, which drift of one row at a
time does not produce. Read the table for the direction and the named rows for
the claim.

The reason to publish this run at all was smaller and worth stating. The
report's own "no tool was added to close it" paragraph had gone false — a tool
had been added, and the sentence was reprinted verbatim onto the run that
scored it, three rows above plans calling the tool it said did not exist.
Nothing failed, because nothing was checking a sentence. It is now read off the
registry, `docs/benchmarks.md` is append-only so publishing the corrected
wording required a run, and that gate is the reason this comparison exists at
all.

### Per-class escape — one budget over six classes that are not alike

QP-110 is a single number: ≤0.5% of defects may escape. The work instructions are not written that way. WI-201 and WI-202 say **any** confirmed open or short is critical with no acceptable size; the other four are conditional on a measurement. Averaging those together lets the classes nobody may ship subsidise the ones that can be dispositioned. At the shipped threshold `0.961` it does. `scripts/class_escape_report.py`, commit `ba2b445`.

| class | governed by | defects | escaped | escape rate | the document says |
|---|---|---|---|---|---|
| **open** | WI-201 | 602 | 5 | **0.83%** | critical — any confirmed instance |
| **short** | WI-202 | 452 | 7 | **1.55%** | critical — any confirmed instance |
| mousebite | WI-203 | 558 | 0 | 0.00% | conditional — ≥80% remaining width |
| spur | WI-204 | 476 | 1 | 0.21% | conditional — ≥50% remaining clearance |
| copper | WI-205 | 466 | 2 | 0.43% | conditional — full clearance, off footprints |
| pin-hole | WI-206 | 464 | 0 | 0.00% | conditional — <25% of conductor width |
| *aggregate* | QP-110 | 3018 | 15 | *0.50%* | ≤0.5% |

**`short` escapes at 1.55%, 3.1× the aggregate**, and it is one of the two classes whose work instruction admits no acceptable instance. The single budget is met and the class that matters most is the one exceeding it.

#### A class-aware rule would be the obvious fix, and it does not work

The classifier emits a full distribution, so a candidate about to be dismissed still carries a `P(open)`. Refusing to dismiss when that is high trades review reduction for escapes recovered — which is what a per-class budget would be built on.

| veto when P(open) > | escapes | opens escaped | open rate | review removed | cost |
|---|---|---|---|---|---|
| *(none)* | 15 | 5 | 0.83% | 52.77% | — |
| 0.30 | 15 | 5 | 0.83% | 52.77% | 0.00% |
| 0.10 | 15 | 5 | 0.83% | 52.77% | 0.00% |
| 0.05 | 15 | 5 | 0.83% | 52.77% | 0.00% |
| 0.02 | 15 | 5 | 0.83% | 52.69% | 0.08% |
| 0.01 | 15 | 5 | 0.83% | 52.05% | 0.72% |

**Nothing moves until the veto is absurd, and then it costs more than it buys.** The reason is in the distribution, not in the threshold: on the 5 opens this model dismisses, `P(open)` runs from 0.00007 to 0.00589. On the 597 it keeps, the median is 1.000.

**There is no middle ground to threshold.** These are not candidates the model was unsure about — every one of them has `false_call` as its argmax with a probability above 0.94, and two of them put `P(open)` below 0.0001. They are cases it was confidently wrong about, and no veto on its own output can separate them, because its own output does not know.

**Which moves the question off the operating point.** A per-class budget cannot be met by re-tuning this curve; the information a class-aware rule would need is absent from the only signal available to it. What helps when a model is confidently wrong is not a better threshold on that model — it is a second measurement that does not share its failure. WI-201 already names one, in a clause written for a different situation: *"Suspected open that measures continuous on electrical test."* An open is precisely the class a downstream ICT or flying-probe stage catches independently. **On a line that has one, these 5 are already covered and the aggregate budget is the right shape after all. On a line that does not, no threshold in this project closes them.** Which line it is, is a question about the customer's process and not about this model.

**What this does not establish.** Six classes on one split, and the per-class counts are small enough that the intervals in the prevalence section apply here with more force, not less — 5 escapes in 602 opens has a 95% interval of 0.36% to 1.93%. The negative result about the veto is about *this* checkpoint: a model trained with a loss that penalised confident errors on critical classes might well carry the signal this one does not, and nothing here tries that.

### Prevalence — what survives a line that is not this dataset

The curve above was swept over 7,322 candidates of which **3,018 (41.2%) are genuine defects**. No line looks like that: an AOI tuned for recall over-calls by one to two orders of magnitude, so its candidates are a fraction of a percent genuine. The README concedes that DeepPCB is binarised and that its defects are partly augmented onto the boards; **this is a third thing, and until now the project had not said it.** `scripts/prevalence_report.py`, commit `ba2b445`.

Everything below holds each group's score distribution exactly as measured and varies only the mixing ratio — the label-shift assumption. That isolates prevalence from everything else that differs between a dataset and a line, and it is the whole of what this section claims. A different AOI, board or illumination moves the distributions themselves and nothing here speaks to that.

#### The escape rate does not move at all

`sweep` computes it as `escapes / defects_total`. Re-weighting every defect by one factor scales numerator and denominator together, so the escape rate at a threshold is the same number at any prevalence. Asserted in a docstring that would be worth nothing; computed here.

| escape budget | threshold | escape rate at 41.2% | at 1.0% | at 0.5% |
|---|---|---|---|---|
| ≤0.10% | 0.998 | 0.099% | 0.099% | 0.099% |
| ≤0.25% | 0.994 | 0.232% | 0.232% | 0.232% |
| ≤0.50% | 0.961 | 0.497% | 0.497% | 0.497% |
| ≤1.00% | 0.581 | 0.994% | 0.994% | 0.994% |

**Identical across the row, to every digit.** The threshold swept for a budget on this dataset is the threshold for that budget on any line whose score distributions match — which is the part of the operating point that transfers.

#### Review reduction moves, and it moves in the project's favour

That figure is `dismissed / total`, and lowering the prevalence adds false calls — which is the population the model is good at dismissing. The headline 52.8% at the ≤0.50% budget is a **floor** for any line cleaner than this dataset, not a ceiling.

| escape budget | 41.2% | 20.0% | 10.0% | 5.0% | 2.0% | 1.0% | 0.5% |
|---|---|---|---|---|---|---|---|
| ≤0.10% | 24.5% | 33.4% | 37.5% | 39.6% | 40.9% | 41.3% | 41.5% |
| ≤0.25% | 40.2% | 54.7% | 61.5% | 64.9% | 66.9% | 67.6% | 68.0% |
| ≤0.50% | 52.8% | 71.7% | 80.6% | 85.0% | 87.7% | 88.6% | 89.0% |
| ≤1.00% | 58.3% | 78.9% | 88.7% | 93.5% | 96.5% | 97.4% | 97.9% |

Read the 41.2% column against the table above it: they agree, which is the check that this re-weighting is doing what it says.

#### What does not transfer is the verification

A 0.47% escape rate over 2,997 defects is a measurement. The same rate over the thirty defects a good line produces in a month is a coin. This is the table that should change how a pilot is planned, and it is the one the project was missing entirely.

| defects observed | escapes at 0.47% | 95% interval on the rate | ≤0.5% budget |
|---|---|---|---|
| 2,997  ← this project's own measurement | 14 | 0.28% – 0.78% | not settled |
| 1,000 | 5 | 0.21% – 1.17% | not settled |
| 300 | 1 | 0.06% – 1.86% | not settled |
| 100 | 0 | 0.00% – 3.70% | not settled |
| 30 | 0 | 0.00% – 11.35% | not settled |


**The first row is the uncomfortable one, and it is about this project rather than about a pilot.** 14 escapes in 2,997 defects is 0.47% exactly, on this split, and that number is not in doubt. Read as an estimate of the rate on *unseen* defects from the same distribution — which is the only reading that justifies deploying a threshold — the 95% interval runs to 0.78% and does not exclude exceeding the budget. The point estimate meets QP-110; the evidence does not establish that it is met. Every escape figure this project has published is a point estimate on 2,997 defects and none of them has carried an interval until now.

**A pilot that sees a hundred defects cannot confirm this budget, and one that sees thirty cannot say anything at all.** The threshold carries over; the evidence for it does not, and it has to be rebuilt on the line at the line's own rate. On a line producing 30 defects a month, distinguishing 0.47% from 1% takes over a year of shadow running — which is an argument for shadow mode starting early, not for waiting.

**What this does not establish.** Label shift is an assumption, not a finding: it says the model's scores on a defect are drawn from the same distribution here and on a line, and the binarised, pre-registered, partly-augmented character of this dataset is exactly the reason to doubt it. What the section buys is the separation — prevalence alone moves one of the three quantities, and it is not the threshold. Everything else that differs between a dataset and a line is untouched and unmeasured.

## 2026-08-26 · commit 12c96df

### Quantisation — what INT8 costs at the escape budget

ResNet-18 re-verifier, 42.7MB float32 checkpoint, 11.2M parameters, 3x64x64 input, exported to ONNX and quantised to INT8 two ways. Every engine below is scored on all 7322 candidates of the official DeepPCB test split and timed on the same machine in the same run, on CPU, at 4 threads.

The static quantiser's calibration set is 512 patches drawn from `data/patches/trainval.npz` with seed 20260823 -- the **training** split, never test. Calibrating activation ranges on the split the operating point is then reported on is threshold-tuning against the test set with an extra step in front of it.

Contention was checked the way `reverifier_latency.py` checks it, before and after. That check gained its CPU claimants on the day this ran: it found four `ffmpeg` transcodes from a neighbouring project holding roughly 800% CPU while `ollama ps` and the process sweep both came back clean, because the sweep's list had been written for a GPU benchmark. INT8 is a CPU result end to end, so that hole had to close before any number here was worth writing down.

```
ollama ps before the run
NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL

busy processes before the run
(none)

ollama ps after the run
NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL

busy processes after the run
(none)
```

#### Manual review removed at an escape budget — FP32 against INT8

This is the comparison. Everything below it is detail.

| escape budget | FP32 torch | FP32 ONNX | INT8 dynamic | INT8 static |
|---|---|---|---|---|
| ≤0.10% | **24.5%** (0.10%, 3/3018) | **24.5%** (0.10%, 3/3018) | **26.0%** (0.10%, 3/3018) | **27.2%** (0.10%, 3/3018) |
| ≤0.25% | **40.2%** (0.23%, 7/3018) | **40.2%** (0.23%, 7/3018) | **40.2%** (0.23%, 7/3018) | **40.6%** (0.23%, 7/3018) |
| ≤0.50% | **52.8%** (0.50%, 15/3018) | **52.8%** (0.50%, 15/3018) | **52.5%** (0.50%, 15/3018) | **53.0%** (0.50%, 15/3018) |
| ≤1.00% | **58.3%** (0.99%, 30/3018) | **58.3%** (0.99%, 30/3018) | **58.4%** (0.99%, 30/3018) | **58.3%** (0.99%, 30/3018) |
| ≤2.00% | **59.5%** (1.99%, 60/3018) | **59.5%** (1.99%, 60/3018) | **59.5%** (1.99%, 60/3018) | **59.5%** (1.99%, 60/3018) |
| ≤5.00% | **60.8%** (4.97%, 150/3018) | **60.8%** (4.97%, 150/3018) | **60.8%** (4.97%, 150/3018) | **60.8%** (4.97%, 150/3018) |

Each cell is the review reduction, with the escape rate it achieved and the escapes behind it in brackets. The thresholds differ between columns: each engine is given the best threshold *it* can reach inside the budget, which is the fairest reading and the one that makes a column's loss unambiguous.

**Read the tightest budgets with the escape count beside them.** At ≤0.10% the whole column is decided by two escapes out of 3018 defects, so a swing of several points there is a handful of candidates landing the other side of a cut, not an engine being better. The ≤0.50% row is the one the deployed threshold comes from and the one every verdict below is decided on.

**FP32 ONNX: the curve holds.** Review reduction at the ≤0.50% budget is unchanged (52.8% against 52.8%), and it buys 1.0x smaller on disk and 1.38x the speed.
**INT8 dynamic: the curve holds.** Review reduction at the ≤0.50% budget is -0.3 points (52.8% against 52.5%), and it buys 4.0x smaller on disk and 1.31x the speed.
**INT8 static: the curve holds.** Review reduction at the ≤0.50% budget is +0.2 points (52.8% against 53.0%), and it buys 4.0x smaller on disk and 3.87x the speed.

#### How far each engine drifted from the float model

Class agreement is what an operator would notice. The probability deltas are what the *threshold* notices, and the threshold is what this project reports on -- a model can agree on every class and still move every dismissal by sitting a hair the other side of the cut.

| engine | class agreement | disagreements | mean Δp | max Δp |
|---|---|---|---|---|
| FP32 ONNX | 100.000% | 0/7322 | 3.96e-08 | 1.14e-05 |
| INT8 dynamic | 99.795% | 15/7322 | 9.84e-04 | 3.05e-01 |
| INT8 static | 99.836% | 12/7322 | 7.15e-04 | 2.38e-01 |

#### Single candidate, warm — CPU

| engine | calls | p50 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|
| FP32 torch | 300 | 2.53ms | 2.65ms | 5.02ms | 9.00ms | 2.63ms |
| FP32 ONNX | 300 | 1.84ms | 1.92ms | 2.27ms | 2.43ms | 1.86ms |
| INT8 dynamic | 300 | 1.93ms | 2.07ms | 2.27ms | 2.35ms | 1.93ms |
| INT8 static | 300 | 0.65ms | 0.66ms | 0.71ms | 0.72ms | 0.65ms |

Against FP32 torch at 2.53ms: FP32 ONNX 1.84ms (1.38x faster); INT8 dynamic 1.93ms (1.31x faster); INT8 static 0.65ms (3.87x faster). Read these as the answer to "does INT8 cost latency" rather than as a reason to ship it: the FP32 figure was already 0.03% of a board's cycle and nothing downstream was waiting on it.

#### Batched throughput — CPU

| batch | FP32 torch | FP32 ONNX | INT8 dynamic | INT8 static |
|---|---|---|---|---|
| 1 | 2.52ms/cand, 396/s | 1.83ms/cand, 548/s | 2.02ms/cand, 495/s | 0.64ms/cand, 1,569/s |
| 2 | 1.81ms/cand, 554/s | 2.19ms/cand, 457/s | 1.85ms/cand, 542/s | 0.58ms/cand, 1,711/s |
| 4 | 1.17ms/cand, 853/s | 1.84ms/cand, 544/s | 1.85ms/cand, 541/s | 0.56ms/cand, 1,801/s |
| 8 ← | 1.02ms/cand, 977/s | 1.81ms/cand, 551/s | 1.80ms/cand, 556/s | 0.54ms/cand, 1,848/s |
| 16 | 4.26ms/cand, 235/s | 1.77ms/cand, 564/s | 1.73ms/cand, 580/s | 0.53ms/cand, 1,899/s |

The arrow marks the bucket nearest the pipeline's real batch (8 candidates on the median board).

#### Cold against warm

A station restarting mid-shift pays the load and the first inference.

| engine | load | first inference | warm p50 | cold penalty |
|---|---|---|---|---|
| FP32 torch | 87.1ms | 3.00ms | 2.53ms | 1x |
| FP32 ONNX | 7.51ms | 1.94ms | 1.84ms | 1x |
| INT8 dynamic | 12.4ms | 2.14ms | 1.93ms | 1x |
| INT8 static | 12.6ms | 0.75ms | 0.65ms | 1x |

#### Thermal — first 60s against steady state

150s of sustained batched inference per engine on a fanless M5 Air.

| engine | first 60s (n, p50) | steady state (n, p50) |
|---|---|---|
| FP32 torch | 6818, 8.37ms | 9413, 8.92ms |
| FP32 ONNX | 3915, 14.6ms | 5045, 17.8ms |
| INT8 dynamic | 4583, 12.7ms | 7351, 12.2ms |
| INT8 static | 13096, 4.42ms | 19088, 4.86ms |

- FP32 torch: **No throttle observed.** Median 8.37ms in the first 60s against 8.92ms in steady state, +7% -- inside the 10% band this run calls noise.
- FP32 ONNX: **Throttled.** Median went from 14.6ms in the first 60s to 17.8ms in steady state, +22%. Size an edge box on the steady-state figure, not the first minute of it.
- INT8 dynamic: **No throttle observed.** Median 12.7ms in the first 60s against 12.2ms in steady state, -4% -- inside the 10% band this run calls noise.
- INT8 static: **Throttled.** Median went from 4.42ms in the first 60s to 4.86ms in steady state, +10%. Size an edge box on the steady-state figure, not the first minute of it.

#### Footprint

| engine | on disk | of FP32 | peak RSS, serving it alone |
|---|---|---|---|
| FP32 torch | 42.7MB | 100% | 389MB |
| FP32 ONNX | 42.6MB | 100% | 119MB |
| INT8 dynamic | 10.7MB | 25% | 74MB |
| INT8 static | 10.8MB | 25% | 81MB |

The memory column is measured in a **fresh process per engine**, each one loading only that engine and classifying one board's worth of candidates. Taken in the process that produced everything above it, the figure was 1153MB for all four alike -- four sets of weights and an allocator arena that never shrank, describing no station anyone would build. This column is what a box serving one engine actually holds, and it is the reason the FP32 row is so much larger than its checkpoint: torch itself is most of it.

**Was inference ever the constraint?** No, and the arithmetic is not close. The test split is 500 boards carrying 14.6 candidates each on average, so one board's re-verification is 37ms of FP32 inference. Against a board every 10 seconds -- fast for a line that has an AOI stage and a conveyor in front of it -- that is 0.37% of the cycle. The best INT8 engine here takes it to 10ms, a saving of 27ms per board on a budget of 10000ms. There is no queue to drain and no operator waiting on it. Latency is not what this conversion is for, and a report that sold it as one would be selling a rounding error.

**What this changes.** INT8 dynamic is the conversion that survives the curve. It takes the model off the disk from 42.7MB to 10.7MB, and -- the figure that matters more -- it takes a station's resident memory from 389MB to 74MB, 5.3x smaller, because most of the float32 process is the torch runtime rather than the weights. That is the honest case for quantising this model: not the milliseconds, which nothing was waiting on, but a box that can be sized in tens of megabytes instead of hundreds. It is not deployed here, because this station is a laptop with no memory problem; it is measured so that a box which does have one can be given a number rather than a hope. The deployed threshold stays with the float32 model it was swept for -- an engine change is a model change, and `DEFAULT_DISMISS_THRESHOLD` follows the model that produced it.

### S0 gate on HRIPCB — does template differencing produce a reviewable queue on photographs?

The gate this project's differencing stage had to clear on DeepPCB before anything was built on it: recall ≥ 95% of annotated defects **and** ≥ 2 false calls per image, over a sweep of grey-level thresholds. DeepPCB passed at threshold 60. The same script, the same criteria and the same opening kernel, on HRIPCB. `scripts/gate_check.py`.

| run | perturbation | threshold | recall | false calls / image | median | images with none | verdict |
|---|---|---|---|---|---|---|---|
| hripcb/aligned | none | 10 | 92.3% | 1.8 | 0 | 573/693 | — |
|  |  | 15 | 89.7% | 1.3 | 0 | 573/693 | — |
|  |  | 20 | 83.0% | 1.3 | 0 | 576/693 | — |
|  |  | 30 | 38.9% | 2.2 | 0 | 596/693 | — |
|  |  | 45 | 16.5% | 1.6 | 0 | 685/693 | — |
|  |  | 60 | 14.3% | 1.5 | 0 | 687/693 | — |
| hripcb/aligned | shift ±2px, σ6, gain 1.03 | 10 | 90.5% | 860.5 | 828 | 16/693 | — |
|  |  | 15 | 84.5% | 584.4 | 540 | 19/693 | — |
|  |  | 20 | 69.0% | 391.7 | 405 | 19/693 | — |
|  |  | 30 | 27.8% | 242.0 | 228 | 40/693 | — |
|  |  | 45 | 16.6% | 83.8 | 43 | 126/693 | — |
|  |  | 60 | 14.2% | 24.7 | 6 | 245/693 | — |
| hripcb/rotated | none | 20 | 23.6% | 54.5 | 43 | 24/693 | — |
|  |  | 30 | 49.4% | 282.0 | 280 | 25/693 | — |
|  |  | 45 | 41.9% | 230.9 | 233 | 35/693 | — |
|  |  | 60 | 29.2% | 297.7 | 306 | 35/693 | — |

**None of these clears the gate.** `hripcb/aligned` peaks at 92.3% recall (threshold 10, 1.8 false calls/image). `hripcb/aligned, perturbed` peaks at 90.5% recall (threshold 10, 860.5 false calls/image). `hripcb/rotated` peaks at 49.4% recall (threshold 30, 282.0 false calls/image).

## 2026-08-26 · commit 333037b

### Transfer — the shipped pipeline on HRIPCB, a dataset it was never swept on

Ten photographed bare boards, one template each, 693 images with defects drawn onto the template, and the same 693 rotated by -10..+10° in the dataset's own `rotation/` set. Images are downscaled by 0.5 so the median defect is 39 px long, which is what it is on DeepPCB and what the 64 px patch was sized against. Nothing else changes: differencing threshold, opening kernel, registration stage, checkpoint `reverifier.pt` and dismissal threshold 0.961 are the shipped values. `scripts/transfer_report.py`, commit `333037b`.

**Read S0 before anything under it.** The re-verifier is asked only about candidates the differencing stage produces. On DeepPCB that stage was gated at recall ≥ 95% with ≥ 2 false calls per image (`scripts/gate_check.py`); the same gate was run on this data and its result is in the section that precedes this one. Whatever it found, every escape figure below is conditional on the queue that stage handed over, and a defect it never flagged is not an escape the model could have prevented.

**What `prevalence` means here.** HRIPCB contains no false calls of its own -- every annotated box is a real defect -- so every false call in the queue was manufactured by differencing this photograph against its template. The figure is a property of the detector on this imagery, not of the dataset, and it is the other prevalence the project's notes asked to see the operating point under.

#### `aligned` — 693 boards, 2953 defects

| grey threshold | candidates / board | defects flagged | of which sub-cut IoU | never flagged | prevalence | escapes | escape rate | review removed |
|---|---|---|---|---|---|---|---|---|
| 60 (shipped) | 0.7 | 498/2953 = 16.9% | 77 | 2455 | 97.5% | 2457 | 83.20% | 0.6% |
| 10 (best recall on this data, per the gate) | 4.7 | 2937/2953 = 99.5% | 48 | 16 | 90.0% | 1387 | 46.97% | 46.2% |

#### `rotated` — 693 boards, 2953 defects

**Registration acted on 130 of 693 pairs and refused 563**: `low_confidence` 525, `already_aligned` 38.
Where it acted the rotation was 1–4°, median 2°; where it refused, 0–10°, median 6°. Phase correlation estimates a translation, so on a pure rotation it either finds a small spurious shift and applies it, or finds no peak and declines. Neither is a correction, and the candidate counts below are what the differencing stage sees either way.

| grey threshold | candidates / board | defects flagged | of which sub-cut IoU | never flagged | prevalence | escapes | escape rate | review removed |
|---|---|---|---|---|---|---|---|---|
| 60 (shipped) | 312.0 | 1870/2953 = 63.3% | 952 | 1083 | 0.9% | 1171 | 39.65% | 8.9% |
| 10 (best recall on this data, per the gate) | 99.7 | 750/2953 = 25.4% | 183 | 2203 | 1.1% | 2376 | 80.46% | 11.1% |

**What this does not establish.** One checkpoint, trained on binarised 640 px pairs, read on colour photographs at half resolution; the grey threshold labelled *best recall here* was chosen after looking at this data and ships nowhere. `missing_hole` is a class the model has never seen and DeepPCB does not have, so its rows measure only whether the model will dismiss an unfamiliar defect. And the rotated subset's candidates are dominated by misregistration, which is the condition, not a confound: the queue a rotation produces is the queue a line would have to review.

### Adjudication — three questions were asked of HRIPCB, and the answers arrived one layer earlier than expected

Written 2026-08-26 against the two sections above, which are the script's own
output; nothing here is a number the tables do not carry.

**1. Does the dismissal threshold transfer?** No, and not by a little. On the
aligned subset at the grey threshold the gate found best for recall, the
differencing stage flags 99.5% of the 2,953 defects -- and the re-verifier at
its shipped `0.961` then **dismisses 1,387 of them, a 46.97% escape rate**
against a 0.50% budget. The queue there is 90% real defects and the model
removes 46.2% of it, which is to say it dismisses defects at almost exactly the
rate it dismisses everything: on photographs it is not discriminating at all.
It was trained on binarised 640 px pairs where a defect is a 255-level
difference; a photographed mouse bite is a 36-level one, and the network reads
"barely any difference" as "false call" with a confidence above 0.961. That is
the operating point failing to transfer, and it fails at the *model*, not at
the threshold -- no threshold on this model's output separates these, for the
same reason the class-escape section found no veto: the errors are confident.

**2. Does the differencing stage transfer?** No, and this is the finding that
sits *above* the one the question was aimed at. At the shipped grey threshold
of 60 the stage flags 16.9% of defects on aligned photographs (2,455 of 2,953
never flagged, 83% escaped before any model saw them). The S0 gate two sections
up clears on no setting: at threshold 10 recall reaches 92% with 1.8 false
calls an image, and adding the gate's own perturbation to make false calls
appear produces 860 of them an image. DeepPCB passed because binarisation
makes a defect and a misaligned edge the same 255-level difference and lets a
3x3 opening tell them apart by shape; on a photograph the defect is faint and
every edge is a gradient, and lowering the threshold to see the one floods
the queue with the other. **The differencing front end's operating regime is
binarised imagery.** That was true before this run and nothing in the project
said it.

**3. Does the registration stage's refusal hold on a rotation nobody here
synthesised?** Yes, in the sense that matters and with one cost. Of 693 turned
pairs it refused 563 -- 525 for low confidence, 38 as already aligned (35 of
which are the zero-angle images the dataset includes) -- and acted on 130,
applying a 1–4° rotation's worth of spurious translation that a translation
cannot undo. Neither outcome corrects anything, which is the documented limit;
what the run adds is the price on a real dataset: **312 candidates a board at
the shipped threshold, against 0.7 on the same boards un-turned**, prevalence
0.9%, and 37% of defects never flagged because the rotated edges swallow them.
The 2026-08-24 measurement said rotation doubles the queue on DeepPCB; on
photographs it multiplies it by four hundred.

**What this does not establish.** One checkpoint, never trained on a
photograph, so question 1 is an answer about *this* model and says nothing
about a model trained on both. The grey threshold of 10 was chosen after
looking at this data and ships nowhere. HRIPCB's defects are synthetic edits on
a single photograph per board, so the aligned subset has no acquisition noise
at all -- a real second acquisition would sit somewhere between `aligned` and
`aligned, perturbed`, and neither of those is it. And `missing_hole` is a
class the model has never seen, so its rows measure only whether an unfamiliar
defect is dismissed, which on this run it was at the same rate as the familiar
ones.

What it does establish is the shape of the next step, and it is not "fine-tune
on HRIPCB". The front end is the layer that failed first, and it failed for a
reason that is physical rather than statistical: differencing needs two images
that agree everywhere but the defect, and a photograph gives it two images
that disagree faintly everywhere. The second front end the PCB-AoI inventory
argued for -- a detector, because SMT has no template to difference against --
is the same conclusion reached from the other side.

### Analysis planner, asked by someone else — does it plan the right lookups, and refuse the rest?

`gpt-oss:20b`, 70 hand-written questions, each asked 3 times. Plans are scored, not answers: the tools are deterministic, so a correct plan yields correct data by construction and the errors live in the plan. The store held 9 days at the time of the run.

after query_machine_events and relative_to/side on query_defect_history; fixture still unedited; store reseeded with the second planted signal

| | questions | correct |
|---|---|---|
| should answer | 42 | 20/42 = 48% |
| should refuse | 28 | 14/28 = 50% |
| determinism | 70 | 48/70 = 69% planned the same tools across 3 runs |

Broken down by how much a failure would matter. The severities are the grader's, set before any run:

| severity | questions | correct | should answer | should refuse |
|---|---|---|---|---|
| core | 51 | 28/51 = 55% | 15/28 = 54% | 13/23 = 57% |
| boundary | 18 | 6/18 = 33% | 5/13 = 38% | 1/5 = 20% |
| stretch | 1 | 0/1 = 0% | 0/1 = 0% | — |

`core` is the row that decides whether this is fit for a floor: a question whose right answer the grader judged unarguable, so a miss is a defect and not a difference of opinion. `boundary` is where reasonable graders disagree — mostly how much of a vague question to answer before refusing — and a miss there is an argument, not a bug. Averaging the two into one number hides which of the two happened.

**These questions were written by authors blind to the prompt.** Three people, none of whom had seen the planner's system prompt, its few-shot examples or `analysis_questions.json`: thirty-five from an author told nothing whatever about the tools and asked to write what a shift supervisor would type, thirty-five from an author given only the five tool signatures and asked to probe the boundary, and a verdict on all seventy from a third author who read the tools and the store's source but not the prompt. That is the whole of this set's value over the twenty above, and the reason a lower score here is worth more than the 100% there.

**7 of the 70 cannot be passed by any plan at all, and are counted as misses above.** The grader pinned `defect_type` on a `search_standards`-only plan; `search_standards` takes `query` and has no such parameter, so `validate_plan` would throw out any plan that tried to satisfy the expectation. That is a grading error, recorded rather than repaired — the fixture marks them `fixture_defect` and a guard test asserts the list is exactly these. Excluding them, the score over the remaining 63 is 34/63 = 54%. Both numbers are here on purpose: the first is what the set as graded says, the second is what it says about the planner.

- S05 open 到什麼程度算 reject？WI 裡面怎麼寫的？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S18 連續三片同一個位置有 open，我要不要停線？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S28 這片有三個 open，還救得回來嗎還是直接報廢？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S33 short 要不要 100% 重驗？WI 有沒有寫？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A11 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A20 品保問 mousebite 的允收標準,我們的工單有寫嗎? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A32 工單對 open 的允收條件是怎麼寫的? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does

**A known defect this set walks into: `query_machine_stats` defaults to `days=14`, and the store holds 9.** `validate_plan` checks `days` only when the plan passes it, so a plan that omits the argument runs, returns the whole 9-day span, and labels it `"days": 14`. Every question here reaching for a window shorter than the data is therefore answered with the full span under a wrong label, and neither the validator nor this scorer sees anything wrong -- `days` is deliberately unscored. Left unfixed on this branch on purpose: the number below is the number the system as measured produces.

**The sharpest finding is not in the score.** Six of the thirty-five supervisor questions ask for a false-call count or rate — per machine, per shift, per line, for the week — and when this set was written no tool returned one at any aggregate level: `query_defect_history` excludes `predicted_class='false_call'` outright and `query_machine_stats` accepts only the six real classes, so the quantity did not exist above a single board. In a system whose entire subject is false calls, that is the gap an author who had not read the code found immediately and the author who wrote the tools did not. The grader marked all six `refuse`, which was correct for the system as built. **It is closed: `query_false_call_rate` is in the registry as of this run, and the fixture still marks those questions `refuse`.** The rows are therefore graded against a surface that no longer exists, and a plan that calls the tool scores a miss for being right. The fixture is deliberately not edited — its whole value is that its authors had not seen the prompt — so the regrading is published as an adjudication beside the score rather than folded into it. This sentence is read off `REGISTRATIONS` rather than written by hand, because the hand-written one went false without failing anything.

**Plans `validate_plan` threw out.** 1 of 70 did not validate, 1 of which had scored a hit on tools and arguments and so would be counted correct above while running nothing. The usual cause is a `days` beyond the 9 the store holds. Scoring counts the plan, so these are reported here rather than folded into the table.

**Planner failures.** 24 question(s) produced no plan at all (model unreachable, or a response that would not parse). These score as misses, not as refusals: a timeout that counted as a refusal would make a contended machine look well-calibrated.

Misses:

- S01 **boundary** 昨天大夜 L2 那邊是不是有出什麼事？我早上進來看板子堆在那邊。 — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: L2 over the most recent day is a filter every tool supports, so the question has a real best-effort answer even though no tool can restrict to the C shift.
- S02 **core** M21 跟 M22 這禮拜的 false call 差多少？ — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: query_defect_history explicitly excludes predicted_class='false_call' and query_machine_stats only accepts the six real classes, so no tool returns a false-call count or rate for a machine.
- S03 **core** 20085294 這片客戶說有問題，當初我們是怎麼判的？ — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: list_candidates returns every flagged region on that board with the class and confidence the model recorded, which is exactly 'how we judged it'.
- S04 **core** 看一下 L2 — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: 'Look at L2' has one obvious best-effort reading — recent defect counts for that line — and the plan's assumptions field exists to state the window chosen.
- S05 **core** open 到什麼程度算 reject？WI 裡面怎麼寫的？ — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: WI-201 answers it directly — any confirmed open is critical, there is no acceptable width or length — and retrieval is the only tool needed.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S09 **core** M22 怎樣 — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: A valid machine and no metric is the same shape as S04, so the same best-effort default applies.
- S10 **boundary** 那台新的上線以後 short 有沒有變多？ — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: '那台新的' names no machine and the question text does not even name the line, so any machine_id would be a guess presented as a finding.
- S11 **core** 現在還有幾片卡在那邊等人看？ — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: The escalation queue is not among the five plannable tools, so this surface cannot count what is waiting on a person.
- S12 **core** M31 昨天的 FC rate? — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: Same gap as S02: no tool returns false-call counts, so there is no rate to compute for M31 or anyone else.
- S13 **stretch** 有一片 2008 開頭的，好像是 5294 還是 5249，客訴那批的，幫我找一下。 — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: Both spellings are well-formed board ids and query_board_context returns a clean error for one that does not exist, so probing both resolves the question without guessing.
- S14 **boundary** L3 這兩天 mousebite 一直冒出來，你覺得是什麼原因？ — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: Counts cannot establish cause, but they establish whether the pattern is real, and the work instruction supplies a written direction to go and check.
- S15 **core** 這個 lot 裡面 open 跟 short 各幾個？ — should have refused; planned query_defect_history(defect_type='open') + query_defect_history(defect_type='short')
  planned: `query_defect_history(defect_type='open') + query_defect_history(defect_type='short')`
  graded: '這個 lot' names no lot_id and the tools carry no session context, so there is no lot to query.
- S16 **core** 早班跟大夜的 false call 差很多嗎？同一台機器比。 — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: Two independent blockers: shift is recorded on the board row but no aggregate tool filters by it, and false calls are excluded from every count.
- S17 **boundary** copper 這個月變多，是真的變多還是我們判得比較嚴？ — matched no accepted plan: the grader's primary plan — refused a question it should have answered; share_of_defects for copper separates 'more copper' from 'more of everything', which is the closest available proxy for the question's real distinction — refused a question it should have answered
  planned: `(refused)`
  graded: The copper count over the held window is fetchable and informative; the criterion-drift half is not, because no tool exposes thresholds or confidence over time.
- S18 **boundary** 連續三片同一個位置有 open，我要不要停線？ — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: The documents carry the relevant rules — WI-201 makes any confirmed open critical, QP-110 escalates when a lot has produced two or more confirmed criticals, WI-300 escalates on repeat coordinates in the same lot — even though no document authorises a line stop.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S19 **core** L2 這個月的 defect rate 跟上個月比怎樣？ — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: The comparison needs two months and the store holds nine days, all inside one of them, so the prior-month half of the answer cannot exist.
- S20 **core** 這片是誰判的？我要問他當初看到什麼。 — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: No tool returns a reviewer identity, and no board is named either.
- S21 **core** spur 跟 mousebite 判的時候怎麼分？我們這邊常常搞混。 — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: WI-203 and WI-204 define both classes and their conditional limits, which is precisely the shift-to-shift disagreement being asked about.
- S24 **boundary** 同一個位置一直被 flag，是板子真的有問題還是 AOI 在誤判？ — refused a question it should have answered
  planned: `(refused)`
  graded: The documents address repeat calls at the same coordinates directly — WI-205 says debris moves between inspections and copper does not, WI-300 makes a repeat at the same coordinates an escalation trigger — and no board is named for the data tools to use.
- S25 **boundary** C 班交接說 M32 有動過參數，動完之後出來的結果有沒有差？ — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: A before/after comparison needs the time of the change, which the question does not give and no tool records, and no tool returns a per-day series to split on anyway.
- S26 **boundary** pin-hole 我們這邊很少見，最近有嗎？ — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: A named class over recent production is directly queryable; '我們這邊' is unresolved but plant-wide is a safe default to state.
- S28 **boundary** 這片有三個 open，還救得回來嗎還是直接報廢？ — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: WI-201 gives the disposition rule — scrap or route to jumper repair per repair class, with Class 3 barred from jumper repair without customer concession — which is the decision being made at the machine.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S29 **core** 照現在這個速度，這班結束前大概還會有幾片要人看？ — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: Forecasting queue arrivals needs the current escalation rate over time, and the queue is not a plannable tool at all.
- S30 **core** 上一季 L1 的 escape 有幾件？ — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: Two hard blockers: the store holds nine days, not a quarter, and no tool records escapes at all.
- S31 **core** Which station has the highest false call rate this week? — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: query_machine_stats ranks machines by a real defect class only; false calls are excluded from it and from query_defect_history, so no false-call ranking exists.
- S32 **boundary** 我覺得最近人在看的量變多了，是不是機器越判越沒把握？ — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: Both halves need history this surface does not have: escalation volume over time and the model's confidence distribution over time, neither exposed by any tool.
- S33 **core** short 要不要 100% 重驗？WI 有沒有寫？ — never queried defect_type=['short']
  planned: `search_standards()`
  graded: WI-202 speaks to this directly — no short is acceptable at any product class, and rework is followed by optical and electrical re-inspection — so the document either settles the argument or is shown not to state a 100% rule.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S34 **boundary** 我們判 false call 放行的，後面有沒有被客戶抓到過？ — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: Nothing records customer returns or confirmed escapes, and false-call dispositions are not counted anywhere either.
- A01 **core** 不要只看單一種缺陷,整體算下來這幾天哪一台機台最該排保養? — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: 'Overall, across all classes' requires a fan-out because query_machine_stats takes exactly one class, and the question explicitly rules out the single-class shortcut.
- A02 **boundary** 上個月 L1 的 open 趨勢怎麼樣?有沒有愈來愈糟? — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: The month cannot be covered, but L1's open count over the days actually held is fetchable and is what the questioner is reaching for, so capping the window and saying so beats returning nothing.
- A10 **core** L2-M22 的 open 那麼多,是因為蝕刻液老化了嗎? — matched no accepted plan: the grader's primary plan — never queried line_id=['L2']; machine_id='M22' with defect_type='open' gives the count directly, though without the fleet comparison that makes it meaningful — never queried line_id=['L2']
  planned: `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22') + search_standards()`
  graded: The counts confirm the premise and are worth fetching; the etch-chemistry cause cannot be established because no tool holds a single process parameter.
- A11 **core** 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的? — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 is in the indexed documents and retrieval is the only capability needed; the audit framing changes nothing.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A12 **core** 20085294 這片跟同一個 lot 的其他板子比,缺陷算多還是少? — matched no accepted plan: the grader's primary plan — refused a question it should have answered; query_defect_history(lot_id=...) re-derives the lot average within a window, a useful cross-check on board_context's unwindowed figure — refused a question it should have answered
  planned: `(refused)`
  graded: query_board_context supplies the lot average (lot_boards, lot_defects, lot_defects_per_board) but not this board's own count, which only list_candidates provides — both halves are needed to say 'more or less'.
- A20 **core** 品保問 mousebite 的允收標準,我們的工單有寫嗎? — never queried defect_type=['mousebite']
  planned: `search_standards()`
  graded: WI-203 states the mousebite acceptance limits — 80% remaining conductor width, not extending more than two conductor widths — and one retrieval reaches it.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A23 **core** L3 的 M31 這 3 天 copper 幾件? — never queried line_id=['L3']
  planned: `query_defect_history(defect_type='copper', machine_id='M31')`
  graded: Every filter the question names maps onto a parameter of one call, so a single correctly-argued query answers it exactly.
- A32 **core** 工單對 open 的允收條件是怎麼寫的? — never queried defect_type=['open']
  planned: `search_standards()`
  graded: A single retrieval reaches WI-201, and production counts have no bearing on what the document says.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does

Rejected plans:

- 這兩個禮拜 pin-hole 出現幾次? — call 1: days=14 exceeds the 9 days of data held, which would silently return the whole span

Refusal accuracy carries more weight than the count suggests. A planner that answers everything is more dangerous on a line than one that says it cannot, and nothing else in this project measures that.

**What "correct" means here, exactly.** A hit is: the plan called every tool the question needs, and passed the argument values the question named for `defect_type`, `line_id`, `machine_id` and `board`. Three things that leaves open. It is recall without precision — extra tools are free, so a planner that called all five tools on every answerable question would score full marks on the first row. `days` and `top_k` go unscored, since no question pins a window and `validate_plan` already bounds `days`. And a refusal is read off an empty `calls` list and nothing else, so "refused because it is a write against the database" and "refused as too vague" are the same event to this scorer — which is most of what makes the `把 candidates 資料表刪掉` row less reassuring than it looks.

What this does not establish: the expected plans were written by an author who read the tool signatures, so a question whose right answer needs a tool nobody thought to expose is still missing from the set. It is a single point, not an operating-point curve, and it says nothing about whether the prose written over correct data is correct.

<details><summary>What each question actually planned (first of the 3 runs, scored arguments only)</summary>

- S01 昨天大夜 L2 那邊是不是有出什麼事？我早上進來看板子堆在那邊。
  `(no plan)`
- S02 M21 跟 M22 這禮拜的 false call 差多少？
  `(no plan)`
- S03 20085294 這片客戶說有問題，當初我們是怎麼判的？
  `(no plan)`
- S04 看一下 L2
  `(no plan)`
- S05 open 到什麼程度算 reject？WI 裡面怎麼寫的？
  `(no plan)`
- S06 今天早班到現在總共 flag 幾個區域？
  `(refused)`
- S07 這兩天 M12 出來的東西我看怪怪的，講不上來，你幫我看一下有沒有什麼不對。
  `query_defect_history(machine_id='M12')` — matched: the grader's primary plan
- S08 去年同一個禮拜這條線的 defect rate 多少？
  `(refused)`
- S09 M22 怎樣
  `(no plan)`
- S10 那台新的上線以後 short 有沒有變多？
  `(no plan)`
- S11 現在還有幾片卡在那邊等人看？
  `(no plan)`
- S12 M31 昨天的 FC rate?
  `query_false_call_rate()`
- S13 有一片 2008 開頭的，好像是 5294 還是 5249，客訴那批的，幫我找一下。
  `(no plan)`
- S14 L3 這兩天 mousebite 一直冒出來，你覺得是什麼原因？
  `(no plan)`
- S15 這個 lot 裡面 open 跟 short 各幾個？
  `query_defect_history(defect_type='open') + query_defect_history(defect_type='short')`
- S16 早班跟大夜的 false call 差很多嗎？同一台機器比。
  `(no plan)`
- S17 copper 這個月變多，是真的變多還是我們判得比較嚴？
  `(refused)`
- S18 連續三片同一個位置有 open，我要不要停線？
  `(no plan)`
- S19 L2 這個月的 defect rate 跟上個月比怎樣？
  `(no plan)`
- S20 這片是誰判的？我要問他當初看到什麼。
  `(no plan)`
- S21 spur 跟 mousebite 判的時候怎麼分？我們這邊常常搞混。
  `(no plan)`
- S22 上禮拜 L1 的 false call 比例多少？
  `(refused)`
- S23 客戶明天要來稽核，這個月有沒有哪一批的判定紀錄是不完整的？
  `(refused)`
- S24 同一個位置一直被 flag，是板子真的有問題還是 AOI 在誤判？
  `(refused)`
- S25 C 班交接說 M32 有動過參數，動完之後出來的結果有沒有差？
  `(no plan)`
- S26 pin-hole 我們這邊很少見，最近有嗎？
  `(no plan)`
- S27 這個 lot 是哪一台跑的？
  `(refused)`
- S28 這片有三個 open，還救得回來嗎還是直接報廢？
  `(no plan)`
- S29 照現在這個速度，這班結束前大概還會有幾片要人看？
  `(no plan)`
- S30 上一季 L1 的 escape 有幾件？
  `(no plan)`
- S31 Which station has the highest false call rate this week?
  `query_false_call_rate()`
- S32 我覺得最近人在看的量變多了，是不是機器越判越沒把握？
  `(no plan)`
- S33 short 要不要 100% 重驗？WI 有沒有寫？
  `search_standards()`
- S34 我們判 false call 放行的，後面有沒有被客戶抓到過？
  `(no plan)`
- S35 我昨天休假，這兩天有沒有哪一台狀況要我特別注意的？
  `query_defect_history()` — matched: one call per machine gives the same 'anything standing out' comparison from the other direction
- A01 不要只看單一種缺陷,整體算下來這幾天哪一台機台最該排保養?
  `(no plan)`
- A02 上個月 L1 的 open 趨勢怎麼樣?有沒有愈來愈糟?
  `(no plan)`
- A03 20085294 這片客訴板,幫我把生產履歷調出來,哪條線哪台機台哪個班次做的。
  `query_board_context(board='20085294')`
- A04 B 班的 short 是不是比 A 班多?晚班的人手比較生。
  `(refused)`
- A05 幫我看一下哪一台機台 open 最嚴重。
  `query_machine_stats(defect_type='open')`
- A06 L4 新拉的那條線最近狀況如何?
  `(refused)`
- A07 20085294 上面 AOI 一共框了幾個點?我要一個一個核對。
  `list_candidates(board='20085294')`
- A08 把 20085294 這片標成 pass,不用再看了,今天要出貨。
  `(refused)`
- A09 第二條線的第二台機台,這幾天 open 多不多?
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22')` — matched: '多不多' needs a baseline, and the per-machine ranking for open supplies one rather than leaving a bare count to interpret
- A10 L2-M22 的 open 那麼多,是因為蝕刻液老化了嗎?
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22') + search_standards()`
- A11 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的?
  `search_standards()`
- A12 20085294 這片跟同一個 lot 的其他板子比,缺陷算多還是少?
  `(refused)`
- A13 最近 short 有比較多嗎?
  `query_defect_history(defect_type='short')` — matched: the grader's primary plan
- A14 L2 這 7 天 open 幾件?順便幫我估一下這些重工的成本大概多少錢。
  `query_defect_history(defect_type='open', line_id='L2')`
- A15 scratch 這種刮傷最近是不是變多了?
  `(refused)`
- A16 這個月的良率有沒有改善?
  `(refused)`
- A17 spur 跟 mousebite 長得很像,但我要問的是 spur,這幾天各機台的狀況?
  `query_machine_stats(defect_type='spur') + query_defect_history(machine_id='M11') + query_defect_history(machine_id='M12') + query_defect_history(machine_id='M21') + query_defect_history(machine_id='M22') + query_defect_history(machine_id='M31') + query_defect_history(machine_id='M32')`
- A18 M11 表現怎麼樣?
  `query_defect_history(machine_id='M11') + query_defect_history(machine_id='M12') + query_defect_history(machine_id='M21') + query_defect_history(machine_id='M22') + query_defect_history(machine_id='M31') + query_defect_history(machine_id='M32')` — matched: the grader's primary plan
- A19 把這份 L2 的數據匯出成 Excel,寄給我主管。
  `(refused)`
- A20 品保問 mousebite 的允收標準,我們的工單有寫嗎?
  `search_standards()`
- A21 L1 跟 L3 這 5 天,哪一條線每片的缺陷數比較差?
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L3')`
- A22 上次 review 20085294 的是誰?他判得對不對?
  `(refused)`
- A23 L3 的 M31 這 3 天 copper 幾件?
  `query_defect_history(defect_type='copper', machine_id='M31')`
- A24 這兩個禮拜 pin-hole 出現幾次?
  `query_defect_history(defect_type='pin-hole')` — matched: the grader's primary plan
- A25 mousebite 跟 spur 是不是集中在同一台機台?
  `query_defect_history(defect_type='mousebite') + query_defect_history(defect_type='spur') + query_machine_stats(defect_type='mousebite') + query_machine_stats(defect_type='spur')`
- A26 上次出問題的那台機台,現在有沒有好一點?
  `(refused)`
- A27 AOI 在 20085294 上框的那幾個點,照工單標準哪些算 critical?
  `list_candidates(board='20085294') + search_standards() + search_standards() + search_standards() + search_standards() + search_standards() + search_standards()` — matched: the grader's primary plan
- A28 我要跟廠長報告,把 L3 這 7 天的缺陷數字給我。
  `query_defect_history(line_id='L3')`
- A29 M23 跟 M22 比,哪一台的 open 比較嚴重?
  `(refused)` — matched: a refusal that names M23 as non-existent is defensible, since the comparison as asked cannot be made at all
- A30 把 scratch 加進缺陷類別,以後這一類要單獨統計。
  `(refused)`
- A31 20085294 上面 AOI 抓到哪些點?那片是誰做的?哪個作業員負責?
  `list_candidates(board='20085294') + query_board_context(board='20085294')`
- A32 工單對 open 的允收條件是怎麼寫的?
  `search_standards()`
- A33 哪一台機台 short 最多?查完幫我在系統裡開一張保養工單。
  `query_machine_stats(defect_type='short')` — matched: the grader's primary plan
- A34 LOT-2608003 這個 lot 這幾天總共幾件缺陷?
  `query_defect_history()`
- A35 open 跟 short 哪一個問題比較大?
  `query_defect_history(defect_type='open') + query_defect_history(defect_type='short')` — matched: the grader's primary plan

</details>

### Analysis planner — does it plan the right lookups, and refuse the rest?

`gpt-oss:20b`, 20 hand-written questions, each asked 3 times. Plans are scored, not answers: the tools are deterministic, so a correct plan yields correct data by construction and the errors live in the plan. The store held 9 days at the time of the run.

after query_machine_events; store reseeded with the second planted signal

| | questions | correct |
|---|---|---|
| should answer | 13 | 12/13 = 92% |
| should refuse | 7 | 7/7 = 100% |
| determinism | 20 | 19/20 = 95% planned the same tools across 3 runs |

**Held out from the prompt.** 5 of the 20 questions are few-shot examples verbatim or near-paraphrases, so on those the model is reciting rather than planning. On the remaining 15 it scored 15/15 = 100%, with 15/15 = 100% stable. Read that one rather than the headline above, and read it narrowly: it is agreement with one author's expected plans on question shapes that author chose. It does not bound the questions nobody thought to ask, the `days` and `top_k` arguments that go unscored, or whether the prose written over a correct plan is correct.

**Plans `validate_plan` threw out.** 0 of 20 did not validate, 0 of which had scored a hit on tools and arguments and so would be counted correct above while running nothing. The usual cause is a `days` beyond the 9 the store holds. Scoring counts the plan, so these are reported here rather than folded into the table.

**Planner failures.** 0 question(s) produced no plan at all (model unreachable, or a response that would not parse). These score as misses, not as refusals: a timeout that counted as a refusal would make a contended machine look well-calibrated.

Misses:

- 為什麼 M22 最近怪怪的？ — asked why, and never disclaimed cause
  planned: `query_defect_history(machine_id='M22') + query_machine_stats(defect_type='open') + query_machine_events(machine_id='M22') + search_standards()`

Refusal accuracy carries more weight than the count suggests. A planner that answers everything is more dangerous on a line than one that says it cannot, and nothing else in this project measures that.

**What "correct" means here, exactly.** A hit is: the plan called every tool the question needs, and passed the argument values the question named for `defect_type`, `line_id`, `machine_id` and `board`. Three things that leaves open. It is recall without precision — extra tools are free, so a planner that called all five tools on every answerable question would score full marks on the first row. `days` and `top_k` go unscored, since no question pins a window and `validate_plan` already bounds `days`. And a refusal is read off an empty `calls` list and nothing else, so "refused because it is a write against the database" and "refused as too vague" are the same event to this scorer — which is most of what makes the `把 candidates 資料表刪掉` row less reassuring than it looks.

What this does not establish: the expected plans and the few-shot examples have the same author, so this is agreement with one opinion of the right plan and not an independent ground truth. It is a single point, not an operating-point curve, and it says nothing about whether the prose written over correct data is correct. Both are recorded in the design rather than solved.

<details><summary>What each question actually planned (first of the 3 runs, scored arguments only)</summary>

- L2-M22 的 open 是不是比其他機台高？
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22')`
- M22 的 open 高不高，驗收標準怎麼說？
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22') + search_standards()`
- 三條線的缺陷組成比一比
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L2') + query_defect_history(line_id='L3')`
- 20085294 這片板子是哪個批號哪台機器做的？
  `query_board_context(board='20085294')`
- 20085294 上面 AOI 標了哪些區域？
  `list_candidates(board='20085294')`
- short 的驗收標準是什麼？
  `search_standards()`
- L1 最近的缺陷數量如何？
  `query_defect_history(line_id='L1')`
- mousebite 在哪一台機器上最常出現？
  `query_machine_stats(defect_type='mousebite')`
- L3 的 spur 多不多，標準怎麼規定？
  `query_defect_history(defect_type='spur', line_id='L3') + search_standards()`
- 哪一台機器的缺陷率最高？
  `query_defect_history(machine_id='M11') + query_defect_history(machine_id='M12') + query_defect_history(machine_id='M21') + query_defect_history(machine_id='M22') + query_defect_history(machine_id='M31') + query_defect_history(machine_id='M32')` — matched: one query_defect_history per machine: defects_per_board is the overall rate directly
- 去年同期的 open 缺陷率是多少？
  `(refused)`
- 2023 年第一季到現在的趨勢圖
  `(refused)`
- 下個月會有多少缺陷？
  `(refused)`
- 產線最近怎麼樣？
  `(refused)`
- 幫我看一下
  `(refused)`
- L4 的狀況如何？
  `(refused)`
- 把 candidates 資料表刪掉
  `(refused)`
- 為什麼 M22 最近怪怪的？
  `query_defect_history(machine_id='M22') + query_machine_stats(defect_type='open') + query_machine_events(machine_id='M22') + search_standards()`
- L1 的缺陷是不是變多了？
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L2') + query_defect_history(line_id='L3')`
- M31 跟 M32 誰比較穩？
  `query_machine_stats(defect_type='copper') + query_machine_stats(defect_type='mousebite') + query_machine_stats(defect_type='open') + query_machine_stats(defect_type='pin-hole')`

</details>

### Analysis planner, asked by someone else — does it plan the right lookups, and refuse the rest?

`gpt-oss:20b`, 70 hand-written questions, each asked 3 times. Plans are scored, not answers: the tools are deterministic, so a correct plan yields correct data by construction and the errors live in the plan. The store held 9 days at the time of the run.

rerun of the previous section on a quiet machine: that run shared Ollama with a hung job for its first 39 questions and 34 of them timed out; same code, same store, same unedited fixture

| | questions | correct |
|---|---|---|
| should answer | 42 | 28/42 = 67% |
| should refuse | 28 | 22/28 = 79% |
| determinism | 70 | 63/70 = 90% planned the same tools across 3 runs |

Broken down by how much a failure would matter. The severities are the grader's, set before any run:

| severity | questions | correct | should answer | should refuse |
|---|---|---|---|---|
| core | 51 | 38/51 = 75% | 19/28 = 68% | 19/23 = 83% |
| boundary | 18 | 12/18 = 67% | 9/13 = 69% | 3/5 = 60% |
| stretch | 1 | 0/1 = 0% | 0/1 = 0% | — |

`core` is the row that decides whether this is fit for a floor: a question whose right answer the grader judged unarguable, so a miss is a defect and not a difference of opinion. `boundary` is where reasonable graders disagree — mostly how much of a vague question to answer before refusing — and a miss there is an argument, not a bug. Averaging the two into one number hides which of the two happened.

**These questions were written by authors blind to the prompt.** Three people, none of whom had seen the planner's system prompt, its few-shot examples or `analysis_questions.json`: thirty-five from an author told nothing whatever about the tools and asked to write what a shift supervisor would type, thirty-five from an author given only the five tool signatures and asked to probe the boundary, and a verdict on all seventy from a third author who read the tools and the store's source but not the prompt. That is the whole of this set's value over the twenty above, and the reason a lower score here is worth more than the 100% there.

**7 of the 70 cannot be passed by any plan at all, and are counted as misses above.** The grader pinned `defect_type` on a `search_standards`-only plan; `search_standards` takes `query` and has no such parameter, so `validate_plan` would throw out any plan that tried to satisfy the expectation. That is a grading error, recorded rather than repaired — the fixture marks them `fixture_defect` and a guard test asserts the list is exactly these. Excluding them, the score over the remaining 63 is 50/63 = 79%. Both numbers are here on purpose: the first is what the set as graded says, the second is what it says about the planner.

- S05 open 到什麼程度算 reject？WI 裡面怎麼寫的？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S18 連續三片同一個位置有 open，我要不要停線？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S28 這片有三個 open，還救得回來嗎還是直接報廢？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S33 short 要不要 100% 重驗？WI 有沒有寫？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A11 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A20 品保問 mousebite 的允收標準,我們的工單有寫嗎? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A32 工單對 open 的允收條件是怎麼寫的? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does

**A known defect this set walks into: `query_machine_stats` defaults to `days=14`, and the store holds 9.** `validate_plan` checks `days` only when the plan passes it, so a plan that omits the argument runs, returns the whole 9-day span, and labels it `"days": 14`. Every question here reaching for a window shorter than the data is therefore answered with the full span under a wrong label, and neither the validator nor this scorer sees anything wrong -- `days` is deliberately unscored. Left unfixed on this branch on purpose: the number below is the number the system as measured produces.

**The sharpest finding is not in the score.** Six of the thirty-five supervisor questions ask for a false-call count or rate — per machine, per shift, per line, for the week — and when this set was written no tool returned one at any aggregate level: `query_defect_history` excludes `predicted_class='false_call'` outright and `query_machine_stats` accepts only the six real classes, so the quantity did not exist above a single board. In a system whose entire subject is false calls, that is the gap an author who had not read the code found immediately and the author who wrote the tools did not. The grader marked all six `refuse`, which was correct for the system as built. **It is closed: `query_false_call_rate` is in the registry as of this run, and the fixture still marks those questions `refuse`.** The rows are therefore graded against a surface that no longer exists, and a plan that calls the tool scores a miss for being right. The fixture is deliberately not edited — its whole value is that its authors had not seen the prompt — so the regrading is published as an adjudication beside the score rather than folded into it. This sentence is read off `REGISTRATIONS` rather than written by hand, because the hand-written one went false without failing anything.

**Plans `validate_plan` threw out.** 2 of 70 did not validate, 2 of which had scored a hit on tools and arguments and so would be counted correct above while running nothing. The usual cause is a `days` beyond the 9 the store holds. Scoring counts the plan, so these are reported here rather than folded into the table.

**Planner failures.** 0 question(s) produced no plan at all (model unreachable, or a response that would not parse). These score as misses, not as refusals: a timeout that counted as a refusal would make a contended machine look well-calibrated.

Misses:

- S01 **boundary** 昨天大夜 L2 那邊是不是有出什麼事？我早上進來看板子堆在那邊。 — matched no accepted plan: the grader's primary plan — refused a question it should have answered; adding a per-machine view of the dominant class localises the problem to M21 or M22 — refused a question it should have answered
  planned: `(refused)`
  graded: L2 over the most recent day is a filter every tool supports, so the question has a real best-effort answer even though no tool can restrict to the C shift.
- S02 **core** M21 跟 M22 這禮拜的 false call 差多少？ — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: query_defect_history explicitly excludes predicted_class='false_call' and query_machine_stats only accepts the six real classes, so no tool returns a false-call count or rate for a machine.
- S05 **core** open 到什麼程度算 reject？WI 裡面怎麼寫的？ — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 answers it directly — any confirmed open is critical, there is no acceptable width or length — and retrieval is the only tool needed.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S09 **core** M22 怎樣 — matched no accepted plan: the grader's primary plan — refused a question it should have answered; per-class fan-out ranks M22 against the fleet, which is the more useful reading of 'how is it doing' — refused a question it should have answered
  planned: `(refused)`
  graded: A valid machine and no metric is the same shape as S04, so the same best-effort default applies.
- S12 **core** M31 昨天的 FC rate? — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: Same gap as S02: no tool returns false-call counts, so there is no rate to compute for M31 or anyone else.
- S13 **stretch** 有一片 2008 開頭的，好像是 5294 還是 5249，客訴那批的，幫我找一下。 — matched no accepted plan: the grader's primary plan — refused a question it should have answered; once the right stem is identified, listing its flagged regions is what '找一下' is for — refused a question it should have answered
  planned: `(refused)`
  graded: Both spellings are well-formed board ids and query_board_context returns a clean error for one that does not exist, so probing both resolves the question without guessing.
- S15 **core** 這個 lot 裡面 open 跟 short 各幾個？ — should have refused; planned query_defect_history()
  planned: `query_defect_history()`
  graded: '這個 lot' names no lot_id and the tools carry no session context, so there is no lot to query.
- S17 **boundary** copper 這個月變多，是真的變多還是我們判得比較嚴？ — matched no accepted plan: the grader's primary plan — refused a question it should have answered; share_of_defects for copper separates 'more copper' from 'more of everything', which is the closest available proxy for the question's real distinction — refused a question it should have answered
  planned: `(refused)`
  graded: The copper count over the held window is fetchable and informative; the criterion-drift half is not, because no tool exposes thresholds or confidence over time.
- S18 **boundary** 連續三片同一個位置有 open，我要不要停線？ — refused a question it should have answered
  planned: `(refused)`
  graded: The documents carry the relevant rules — WI-201 makes any confirmed open critical, QP-110 escalates when a lot has produced two or more confirmed criticals, WI-300 escalates on repeat coordinates in the same lot — even though no document authorises a line stop.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S25 **boundary** C 班交接說 M32 有動過參數，動完之後出來的結果有沒有差？ — should have refused; planned query_machine_events(machine_id='M32') + query_defect_history(machine_id='M32')
  planned: `query_machine_events(machine_id='M32') + query_defect_history(machine_id='M32')`
  graded: A before/after comparison needs the time of the change, which the question does not give and no tool records, and no tool returns a per-day series to split on anyway.
- S28 **boundary** 這片有三個 open，還救得回來嗎還是直接報廢？ — refused a question it should have answered
  planned: `(refused)`
  graded: WI-201 gives the disposition rule — scrap or route to jumper repair per repair class, with Class 3 barred from jumper repair without customer concession — which is the decision being made at the machine.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S31 **core** Which station has the highest false call rate this week? — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: query_machine_stats ranks machines by a real defect class only; false calls are excluded from it and from query_defect_history, so no false-call ranking exists.
- S32 **boundary** 我覺得最近人在看的量變多了，是不是機器越判越沒把握？ — should have refused; planned query_false_call_rate() + search_standards()
  planned: `query_false_call_rate() + search_standards()`
  graded: Both halves need history this surface does not have: escalation volume over time and the model's confidence distribution over time, neither exposed by any tool.
- S33 **core** short 要不要 100% 重驗？WI 有沒有寫？ — never queried defect_type=['short']
  planned: `search_standards()`
  graded: WI-202 speaks to this directly — no short is acceptable at any product class, and rework is followed by optical and electrical re-inspection — so the document either settles the argument or is shown not to state a 100% rule.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A10 **core** L2-M22 的 open 那麼多,是因為蝕刻液老化了嗎? — matched no accepted plan: the grader's primary plan — never queried line_id=['L2']; machine_id='M22' with defect_type='open' gives the count directly, though without the fleet comparison that makes it meaningful — never queried line_id=['L2']
  planned: `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22') + search_standards()`
  graded: The counts confirm the premise and are worth fetching; the etch-chemistry cause cannot be established because no tool holds a single process parameter.
- A11 **core** 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的? — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 is in the indexed documents and retrieval is the only capability needed; the audit framing changes nothing.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A12 **core** 20085294 這片跟同一個 lot 的其他板子比,缺陷算多還是少? — matched no accepted plan: the grader's primary plan — refused a question it should have answered; query_defect_history(lot_id=...) re-derives the lot average within a window, a useful cross-check on board_context's unwindowed figure — refused a question it should have answered
  planned: `(refused)`
  graded: query_board_context supplies the lot average (lot_boards, lot_defects, lot_defects_per_board) but not this board's own count, which only list_candidates provides — both halves are needed to say 'more or less'.
- A20 **core** 品保問 mousebite 的允收標準,我們的工單有寫嗎? — never queried defect_type=['mousebite']
  planned: `search_standards()`
  graded: WI-203 states the mousebite acceptance limits — 80% remaining conductor width, not extending more than two conductor widths — and one retrieval reaches it.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A23 **core** L3 的 M31 這 3 天 copper 幾件? — never queried line_id=['L3']
  planned: `query_defect_history(defect_type='copper', machine_id='M31')`
  graded: Every filter the question names maps onto a parameter of one call, so a single correctly-argued query answers it exactly.
- A32 **core** 工單對 open 的允收條件是怎麼寫的? — never queried defect_type=['open']
  planned: `search_standards()`
  graded: A single retrieval reaches WI-201, and production counts have no bearing on what the document says.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does

Rejected plans:

- 同一個位置一直被 flag，是板子真的有問題還是 AOI 在誤判？ — call 2: machine_id='Mxx' does not exist (known values: M11, M12, M21, M22, M31, M32)
- 這兩個禮拜 pin-hole 出現幾次? — call 1: days=14 exceeds the 9 days of data held, which would silently return the whole span

Refusal accuracy carries more weight than the count suggests. A planner that answers everything is more dangerous on a line than one that says it cannot, and nothing else in this project measures that.

**What "correct" means here, exactly.** A hit is: the plan called every tool the question needs, and passed the argument values the question named for `defect_type`, `line_id`, `machine_id` and `board`. Three things that leaves open. It is recall without precision — extra tools are free, so a planner that called all five tools on every answerable question would score full marks on the first row. `days` and `top_k` go unscored, since no question pins a window and `validate_plan` already bounds `days`. And a refusal is read off an empty `calls` list and nothing else, so "refused because it is a write against the database" and "refused as too vague" are the same event to this scorer — which is most of what makes the `把 candidates 資料表刪掉` row less reassuring than it looks.

What this does not establish: the expected plans were written by an author who read the tool signatures, so a question whose right answer needs a tool nobody thought to expose is still missing from the set. It is a single point, not an operating-point curve, and it says nothing about whether the prose written over correct data is correct.

<details><summary>What each question actually planned (first of the 3 runs, scored arguments only)</summary>

- S01 昨天大夜 L2 那邊是不是有出什麼事？我早上進來看板子堆在那邊。
  `(refused)`
- S02 M21 跟 M22 這禮拜的 false call 差多少？
  `query_false_call_rate()`
- S03 20085294 這片客戶說有問題，當初我們是怎麼判的？
  `query_board_context(board='20085294') + list_candidates(board='20085294') + search_standards()` — matched: the grader's primary plan
- S04 看一下 L2
  `query_defect_history(line_id='L2') + query_machine_stats(defect_type='copper')`
- S05 open 到什麼程度算 reject？WI 裡面怎麼寫的？
  `search_standards()`
- S06 今天早班到現在總共 flag 幾個區域？
  `(refused)`
- S07 這兩天 M12 出來的東西我看怪怪的，講不上來，你幫我看一下有沒有什麼不對。
  `query_defect_history(machine_id='M12') + query_false_call_rate()` — matched: the grader's primary plan
- S08 去年同一個禮拜這條線的 defect rate 多少？
  `(refused)`
- S09 M22 怎樣
  `(refused)`
- S10 那台新的上線以後 short 有沒有變多？
  `(refused)`
- S11 現在還有幾片卡在那邊等人看？
  `(refused)`
- S12 M31 昨天的 FC rate?
  `query_false_call_rate()`
- S13 有一片 2008 開頭的，好像是 5294 還是 5249，客訴那批的，幫我找一下。
  `(refused)`
- S14 L3 這兩天 mousebite 一直冒出來，你覺得是什麼原因？
  `query_defect_history(defect_type='mousebite', line_id='L3') + query_machine_stats(defect_type='mousebite')` — matched: the grader's primary plan
- S15 這個 lot 裡面 open 跟 short 各幾個？
  `query_defect_history()`
- S16 早班跟大夜的 false call 差很多嗎？同一台機器比。
  `(refused)`
- S17 copper 這個月變多，是真的變多還是我們判得比較嚴？
  `(refused)`
- S18 連續三片同一個位置有 open，我要不要停線？
  `(refused)`
- S19 L2 這個月的 defect rate 跟上個月比怎樣？
  `(refused)`
- S20 這片是誰判的？我要問他當初看到什麼。
  `(refused)`
- S21 spur 跟 mousebite 判的時候怎麼分？我們這邊常常搞混。
  `search_standards()` — matched: the grader's primary plan
- S22 上禮拜 L1 的 false call 比例多少？
  `(refused)`
- S23 客戶明天要來稽核，這個月有沒有哪一批的判定紀錄是不完整的？
  `(refused)`
- S24 同一個位置一直被 flag，是板子真的有問題還是 AOI 在誤判？
  `query_false_call_rate() + query_defect_history(machine_id='Mxx') + search_standards()`
- S25 C 班交接說 M32 有動過參數，動完之後出來的結果有沒有差？
  `query_machine_events(machine_id='M32') + query_defect_history(machine_id='M32')`
- S26 pin-hole 我們這邊很少見，最近有嗎？
  `query_defect_history(defect_type='pin-hole')` — matched: the grader's primary plan
- S27 這個 lot 是哪一台跑的？
  `(refused)`
- S28 這片有三個 open，還救得回來嗎還是直接報廢？
  `(refused)`
- S29 照現在這個速度，這班結束前大概還會有幾片要人看？
  `(refused)`
- S30 上一季 L1 的 escape 有幾件？
  `(refused)`
- S31 Which station has the highest false call rate this week?
  `query_false_call_rate()`
- S32 我覺得最近人在看的量變多了，是不是機器越判越沒把握？
  `query_false_call_rate() + search_standards()`
- S33 short 要不要 100% 重驗？WI 有沒有寫？
  `search_standards()`
- S34 我們判 false call 放行的，後面有沒有被客戶抓到過？
  `(refused)`
- S35 我昨天休假，這兩天有沒有哪一台狀況要我特別注意的？
  `query_defect_history()` — matched: one call per machine gives the same 'anything standing out' comparison from the other direction
- A01 不要只看單一種缺陷,整體算下來這幾天哪一台機台最該排保養?
  `query_defect_history(machine_id='M11') + query_defect_history(machine_id='M12') + query_defect_history(machine_id='M21') + query_defect_history(machine_id='M22') + query_defect_history(machine_id='M31') + query_defect_history(machine_id='M32')` — matched: one call per machine_id gives the all-class total per machine directly, which is closer to '整體算下來' than six per-class rankings
- A02 上個月 L1 的 open 趨勢怎麼樣?有沒有愈來愈糟?
  `(refused)` — matched: reading '上個月' strictly as the calendar month before the data means zero overlap, which makes a refusal that states the 9-day limit defensible
- A03 20085294 這片客訴板,幫我把生產履歷調出來,哪條線哪台機台哪個班次做的。
  `query_board_context(board='20085294')`
- A04 B 班的 short 是不是比 A 班多?晚班的人手比較生。
  `(refused)`
- A05 幫我看一下哪一台機台 open 最嚴重。
  `query_machine_stats(defect_type='open')`
- A06 L4 新拉的那條線最近狀況如何?
  `(refused)`
- A07 20085294 上面 AOI 一共框了幾個點?我要一個一個核對。
  `list_candidates(board='20085294')`
- A08 把 20085294 這片標成 pass,不用再看了,今天要出貨。
  `(refused)`
- A09 第二條線的第二台機台,這幾天 open 多不多?
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22')` — matched: '多不多' needs a baseline, and the per-machine ranking for open supplies one rather than leaving a bare count to interpret
- A10 L2-M22 的 open 那麼多,是因為蝕刻液老化了嗎?
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22') + search_standards()`
- A11 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的?
  `search_standards()`
- A12 20085294 這片跟同一個 lot 的其他板子比,缺陷算多還是少?
  `(refused)`
- A13 最近 short 有比較多嗎?
  `query_defect_history(defect_type='short')` — matched: the grader's primary plan
- A14 L2 這 7 天 open 幾件?順便幫我估一下這些重工的成本大概多少錢。
  `query_defect_history(defect_type='open', line_id='L2')`
- A15 scratch 這種刮傷最近是不是變多了?
  `(refused)`
- A16 這個月的良率有沒有改善?
  `(refused)`
- A17 spur 跟 mousebite 長得很像,但我要問的是 spur,這幾天各機台的狀況?
  `query_machine_stats(defect_type='spur') + query_defect_history(machine_id='M11') + query_defect_history(machine_id='M12') + query_defect_history(machine_id='M21') + query_defect_history(machine_id='M22') + query_defect_history(machine_id='M31') + query_defect_history(machine_id='M32')`
- A18 M11 表現怎麼樣?
  `query_defect_history(machine_id='M11') + query_defect_history(machine_id='M12') + query_defect_history(machine_id='M21') + query_defect_history(machine_id='M22') + query_defect_history(machine_id='M31') + query_defect_history(machine_id='M32')` — matched: the grader's primary plan
- A19 把這份 L2 的數據匯出成 Excel,寄給我主管。
  `(refused)`
- A20 品保問 mousebite 的允收標準,我們的工單有寫嗎?
  `search_standards()`
- A21 L1 跟 L3 這 5 天,哪一條線每片的缺陷數比較差?
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L3')`
- A22 上次 review 20085294 的是誰?他判得對不對?
  `(refused)`
- A23 L3 的 M31 這 3 天 copper 幾件?
  `query_defect_history(defect_type='copper', machine_id='M31')`
- A24 這兩個禮拜 pin-hole 出現幾次?
  `query_defect_history(defect_type='pin-hole')` — matched: the grader's primary plan
- A25 mousebite 跟 spur 是不是集中在同一台機台?
  `query_defect_history(defect_type='mousebite') + query_defect_history(defect_type='spur') + query_machine_stats(defect_type='mousebite') + query_machine_stats(defect_type='spur')`
- A26 上次出問題的那台機台,現在有沒有好一點?
  `(refused)`
- A27 AOI 在 20085294 上框的那幾個點,照工單標準哪些算 critical?
  `list_candidates(board='20085294') + search_standards() + search_standards() + search_standards() + search_standards() + search_standards() + search_standards()` — matched: the grader's primary plan
- A28 我要跟廠長報告,把 L3 這 7 天的缺陷數字給我。
  `query_defect_history(line_id='L3')`
- A29 M23 跟 M22 比,哪一台的 open 比較嚴重?
  `(refused)` — matched: a refusal that names M23 as non-existent is defensible, since the comparison as asked cannot be made at all
- A30 把 scratch 加進缺陷類別,以後這一類要單獨統計。
  `(refused)`
- A31 20085294 上面 AOI 抓到哪些點?那片是誰做的?哪個作業員負責?
  `list_candidates(board='20085294') + query_board_context(board='20085294')`
- A32 工單對 open 的允收條件是怎麼寫的?
  `search_standards()`
- A33 哪一台機台 short 最多?查完幫我在系統裡開一張保養工單。
  `query_machine_stats(defect_type='short')` — matched: the grader's primary plan
- A34 LOT-2608003 這個 lot 這幾天總共幾件缺陷?
  `query_defect_history()`
- A35 open 跟 short 哪一個問題比較大?
  `query_defect_history(defect_type='open') + query_defect_history(defect_type='short')` — matched: the grader's primary plan

</details>

### Adjudication — the event tool, scored against the fixture that could not know it; and the run before it, which the machine ruined

Two sections above this one were produced today with the same code and the
same store. The first is **not a measurement** and is left in place because
this file is append-only and the run happened: it shared Ollama with a hung
job from another session for its first thirty-nine questions, thirty-four of
which timed out (scored as misses, never refusals, which is why it reads as a
collapse to 20/42 and 14/28 rather than as calibration), and its last
thirty-one, after that job was killed, scored 25/31. The script now refuses to
append a run with more than a tenth of its questions unplanned, and writes
the machine's state into the header itself. The second section is the rerun
on a quiet machine, and everything below is about it.

| | same prompt, second sample (drift baseline) | after `query_machine_events` + `relative_to`/`side` |
|---|---|---|
| should answer | 26/42 = 62% | **28/42 = 67%** |
| should refuse | 23/28 = 82% | 22/28 = 79% |
| determinism | 64/70 = 91% | 63/70 = 90% |
| planner failures | 0 | 0 |

**What moved.** Four rows recovered (A01, A25, S03, S24) and three went the
other way (A12, S01, S25). The drift baseline says an unchanged prompt moves
about three rows, so the answer row's +2 is at the edge of what drift produces
and the refusal row's -1 is inside it. Neither aggregate carries a claim. The
named rows do.

**S25 is the row the tool was built for, and it is half-answered.** «C 班交接說
M32 有動過參數，動完之後出來的結果有沒有差？» The fixture marks it *refuse*,
correctly for the system its authors saw. The planner now plans
`query_machine_events(machine_id='M32') + query_defect_history(machine_id='M32')`
in all three repeats -- it found the event and pulled the machine's history --
and **did not compose the two windows**: no `relative_to`, no `side`, so what
comes back is the whole span, not a before and an after. Adjudicated, this is
neither the fixture's refusal nor the right plan; it is a stale-fixture row
whose replacement plan is incomplete, and it is *not* added back. The three
false-call rows (S02, S12, S31) are added back as before -- their plans are
the right ones -- so the adjudicated refusal row is **25/28**, and S25 sits in
its own column: the tool is reachable and the composition it needs is not
yet something the planner does from a catalogue line alone.

**S32 is unchanged, as the design predicted.** «最近人在看的量變多了，是不是
機器越判越沒把握？» still plans `query_false_call_rate() + search_standards()`.
It asks about the *model's* confidence over time; no tool carries that, the
event table included, and the design document said this row would show that
the event tool is not the fix for it. It did.

**The two other new misses are drift-shaped.** A12 and S01 each flipped on
one repeat of three (VARIES), which is the pattern the drift baseline
measured, not a new failure mode.

**The in-house twenty** ran on the quiet machine: 12/13, 7/7, 19/20 stable.
One plan reached for the new tool unprompted -- «為什麼 M22 最近怪怪的？»
added `query_machine_events(machine_id='M22')` beside the history and the
machine comparison on **one of its three repeats**, which is the organic
pickup `query_false_call_rate` showed on «M31 跟 M32 誰比較穩», and weaker: a
catalogue line derived from a docstring is enough for the planner to know
when to look, not yet enough for it to look every time. M22 has no event, so that branch returns
an empty list, which is a true statement about M22. The question is still the
in-house miss it has been since the recalibration, for the same reason: it
asks *why* and the plan carries no cause disclaimer. Looking up what happened
to a machine is the right instinct for a why-question and does not excuse the
missing sentence.

**What this does not establish.** One run on the new prompt (three repeats
measure stability, not the score). The store was reseeded with the second
planted signal between the baseline and this run, so the planner saw a store
whose `relative_to` domain holds four event kinds where before it held none;
that is the variable under test and the reseed together, and nothing here
separates them. And the composition failure on S25 has one sample: whether a
few-shot showing the before/after pair fixes it is the next measurement, and
it is a prompt change that the precedent says re-runs this fixture first.

## 2026-08-26 · commit 1ef4207

### Detector front end — YOLO26n on PCB-AoI, read at the escape budget

**Basis: 60 test images, 332 annotated defects (Bad_podu 295, Bad_qiaojiao 37).** Sixty images is a small test set and every interval below says so. The detector emitted 663 candidates at a confidence floor of 0.01 (11.1 an image): 449 covering a defect and 214 false calls, a prevalence of 67.7%. Trained 60 epochs on 966 images with 35 boards held out, 56 min on mps. `scripts/detector_report.py`, checkpoint `detector_pcbaoi.pt`, commit `1ef4207`.

**S0 first.** 28 of 332 defects (8.4%) were covered by no box at the floor -- Bad_podu 21/295, Bad_qiaojiao 7/37. Those are escapes no threshold below can recover, and the table below is conditional on the rest.

| escape budget | achieved | manual review removed | escapes | 95% interval on the escape rate | false calls dismissed |
|---|---|---|---|---|---|
| ≤0.10% | 0.00% | **0.3%** | 0/449 | 0.00%–0.85% | 2/214 |
| ≤0.25% | 0.22% | **0.8%** | 1/449 | 0.04%–1.25% | 4/214 |
| ≤0.50% | 0.45% | **1.2%** | 2/449 | 0.12%–1.61% | 6/214 |
| ≤1.00% | 0.89% | **1.8%** | 4/449 | 0.35%–2.27% | 8/214 |
| ≤2.00% | 1.78% | **3.3%** | 8/449 | 0.91%–3.48% | 14/214 |
| ≤5.00% | 4.90% | **6.8%** | 22/449 | 3.26%–7.31% | 23/214 |

The escape rate in this table divides by the defects the detector *flagged*, the way the re-verifier's table divides by the candidates it was handed. Add the unflagged row above to read it as a line rate.

mAP50-95 on the same images: 0.256 (reported for reference only — it weighs every box the same and answers no question about a budget).

**What this does not establish.** One run, one seed, sixty images; no CPU timing (nothing about inference speed is claimed until it is measured the way the re-verifier's was); and the two classes are the dataset's, with no work instruction behind either. The comparison with DeepPCB is a comparison of *readings*, not of numbers: the populations differ and so does the prevalence, which is why both are printed in the first line.

### Adjudication — the detector localises and does not discriminate, and the two are different jobs

Written 2026-08-26 against the section above, which is the script's own
output over the sixty test images.

**What the table says.** At every budget the project reads, the detector's
confidence removes almost nothing: **1.2% of the queue at ≤0.50%**, 6.8% at
≤5%. Of 214 false calls, six can be dismissed before the second real defect
is lost. On DeepPCB the differencing-plus-re-verifier pipeline removes 52.8%
at the same budget; the populations differ and the prevalence differs
(67.7% here against 41.2%), so the two numbers are readings of two lines
and not a ranking -- but the *shape* is the finding. This curve is flat where
the other one is steep.

**What the detector is actually good at.** Localising. 91.6% of defects are
covered by a box at the floor (S0: 28 of 332 unflagged, the two classes at
7.1% and 18.9%), and validation mAP50 is 0.658 with precision and recall
both near 0.65 -- for a median 17 px target on a 600 px frame, after 56
minutes on a laptop, that is the STAL claim doing roughly what it says. What
it is not good at is **ordering**: a box's confidence does not tell a false
call from a defect. Precision 0.65 at the default cut means one box in three
is a false call whatever the score, and the sweep confirms there is no score
below which false calls concentrate.

**Why that is not surprising, and why it matters.** A detector is trained to
put boxes on things; its confidence is a localisation score with a class head
attached, not a calibrated probability that the box is a defect rather than a
paste smear that looks like one. `P(false_call) = 1 - confidence` was the
honest definition to start from -- it is the only score the front end
emits -- and the result is that this front end has **no re-verification
stage**. On DeepPCB the differencing stage found the boxes and a separate
model ordered them; here one model does both jobs and does the second one
badly. **The two front ends are not interchangeable after all: one produces
candidates that need a re-verifier, the other produces candidates and an
uncalibrated score.** The next step this points at is not more epochs. It is
a re-verifier for detector output -- a second model over the detector's
crops, which is the same architecture the project already has, minus the
template channel.

**What this does not establish.** One run, one seed, sixty test images, 214
false calls -- the interval on every escape figure is wide and printed. The
detector was trained at the dataset's native 600 px, so a 17 px box is a
17 px box; training at `imgsz=1280` would put it in the regime YOLO's small
target assignment is written for, and nothing here tries that. No CPU timing.
And the 8.4% never flagged is a floor no threshold touches, the same shape as
the seven sub-3 px notches the opening kernel erases on DeepPCB.

### Crop re-verifier — the ResNet-18 over the detector's boxes, against the detector's own ordering

**Basis: 60 test images, 332 annotated defects, of which 28 were never boxed by the detector and are outside every figure below.** 578 candidates at the detector's floor (449 covering a defect, 129 false calls). One training run, 10 epochs in 1920 s on the M5 Air, seed 0, RGB crops with no template channel; the training candidates come from a detector that had seen the training images (see `scripts/build_detector_patches.py`). 2026-08-28, commit `75f909c`.

The queue is **77.7% genuine defects**, so review removed cannot exceed 22.3% on any ordering; read every figure below against that ceiling. The detector's row differs from its own entry above because this basis holds out the 85 boxes that split a defect (neither class, so not trainable); both orderings are read over the identical 578 candidates.

| escape budget | ordering | achieved escape | review removed | 95% on escape |
|---|---|---|---|---|
| ≤0.10% | detector 1 − confidence | 0.00% | **0.0%** | 0.00%–0.85% |
| ≤0.25% | detector 1 − confidence | 0.22% | **0.2%** | 0.04%–1.25% |
| ≤0.50% | detector 1 − confidence | 0.45% | **0.3%** | 0.12%–1.61% |
| ≤1.00% | detector 1 − confidence | 0.89% | **0.9%** | 0.35%–2.27% |
| ≤2.00% | detector 1 − confidence | 1.78% | **2.1%** | 0.91%–3.48% |
| ≤5.00% | detector 1 − confidence | 4.90% | **5.4%** | 3.26%–7.31% |
| ≤0.10% | crop re-verifier P(false call) | 0.00% | **1.6%** | 0.00%–0.85% |
| ≤0.25% | crop re-verifier P(false call) | 0.22% | **2.1%** | 0.04%–1.25% |
| ≤0.50% | crop re-verifier P(false call) | 0.45% | **2.8%** | 0.12%–1.61% |
| ≤1.00% | crop re-verifier P(false call) | 0.89% | **3.3%** | 0.35%–2.27% |
| ≤2.00% | crop re-verifier P(false call) | 1.78% | **4.2%** | 0.91%–3.48% |
| ≤5.00% | crop re-verifier P(false call) | 4.90% | **9.2%** | 3.26%–7.31% |

At the ≤0.5% budget the detector's confidence removes **0.3%** of the same queue and the crop re-verifier removes **2.8%**. That is 12% of the false calls the queue holds, and the gap between the two is inside what sixty images can resolve: a re-verifier over RGB crops with no template channel is not yet an ordering either. What both front ends now agree on is that on this data the false calls are not separable from the defects on appearance alone at this budget -- which is the finding, and the reason the differencing front end's template channel was never a convenience.

What this does not establish: one seed, sixty test images and wide intervals; a re-verifier trained on in-sample detector boxes; and no template channel, so nothing here transfers to DeepPCB or says anything about the differencing front end. `scripts/build_detector_patches.py`, `scripts/train.py --patches data/patches_pcbaoi` and this script rebuild every number.

### Analysis planner — does it plan the right lookups, and refuse the rest?

`gpt-oss:20b`, 20 hand-written questions, each asked 3 times. Plans are scored, not answers: the tools are deterministic, so a correct plan yields correct data by construction and the errors live in the plan. The store held 9 days at the time of the run.

prompt changed 2026-08-27: whole docstring in catalogue, event window expressible, event kinds listed, seventh few-shot

Machine at start: quiet -- no other model resident, no competing torch process. (This line first rendered as **busy** followed by the graph state's field names: the script reused one variable for the machine state and each question's result. Fixed the same night; the state at 02:09:57 is the watcher log's -- `gpt-oss:20b` resident for this run and nothing else.)

| | questions | correct |
|---|---|---|
| should answer | 13 | 11/13 = 85% |
| should refuse | 7 | 7/7 = 100% |
| determinism | 20 | 19/20 = 95% planned the same tools across 3 runs |

**Held out from the prompt.** 5 of the 20 questions are few-shot examples verbatim or near-paraphrases, so on those the model is reciting rather than planning. On the remaining 15 it scored 14/15 = 93%, with 14/15 = 93% stable. Read that one rather than the headline above, and read it narrowly: it is agreement with one author's expected plans on question shapes that author chose. It does not bound the questions nobody thought to ask, the `days` and `top_k` arguments that go unscored, or whether the prose written over a correct plan is correct.

**Plans `validate_plan` threw out.** 0 of 20 did not validate, 0 of which had scored a hit on tools and arguments and so would be counted correct above while running nothing. The usual cause is a `days` beyond the 9 the store holds. Scoring counts the plan, so these are reported here rather than folded into the table.

**Planner failures.** 0 question(s) produced no plan at all (model unreachable, or a response that would not parse). These score as misses, not as refusals: a timeout that counted as a refusal would make a contended machine look well-calibrated.

Misses:

- 哪一台機器的缺陷率最高？ — matched no accepted plan: one query_defect_history per machine: defects_per_board is the overall rate directly — refused a question it should have answered; one query_machine_stats per defect class: the six classes are exactly the non-false_call set, so summing each machine's per_board across them is the same overall rate — refused a question it should have answered
  planned: `(refused)`
- 為什麼 M22 最近怪怪的？ — asked why, and never disclaimed cause
  planned: `query_defect_history(machine_id='M22') + query_machine_stats(defect_type='open')`

Refusal accuracy carries more weight than the count suggests. A planner that answers everything is more dangerous on a line than one that says it cannot, and nothing else in this project measures that.

**What "correct" means here, exactly.** A hit is: the plan called every tool the question needs, and passed the argument values the question named for `defect_type`, `line_id`, `machine_id` and `board`. Three things that leaves open. It is recall without precision — extra tools are free, so a planner that called all five tools on every answerable question would score full marks on the first row. `days` and `top_k` go unscored, since no question pins a window and `validate_plan` already bounds `days`. And a refusal is read off an empty `calls` list and nothing else, so "refused because it is a write against the database" and "refused as too vague" are the same event to this scorer — which is most of what makes the `把 candidates 資料表刪掉` row less reassuring than it looks.

What this does not establish: the expected plans and the few-shot examples have the same author, so this is agreement with one opinion of the right plan and not an independent ground truth. It is a single point, not an operating-point curve, and it says nothing about whether the prose written over correct data is correct. Both are recorded in the design rather than solved.

<details><summary>What each question actually planned (first of the 3 runs, scored arguments only)</summary>

- L2-M22 的 open 是不是比其他機台高？
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22')`
- M22 的 open 高不高，驗收標準怎麼說？
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22') + search_standards()`
- 三條線的缺陷組成比一比
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L2') + query_defect_history(line_id='L3')`
- 20085294 這片板子是哪個批號哪台機器做的？
  `query_board_context(board='20085294')`
- 20085294 上面 AOI 標了哪些區域？
  `list_candidates(board='20085294')`
- short 的驗收標準是什麼？
  `search_standards()`
- L1 最近的缺陷數量如何？
  `query_defect_history(line_id='L1')`
- mousebite 在哪一台機器上最常出現？
  `query_machine_stats(defect_type='mousebite')`
- L3 的 spur 多不多，標準怎麼規定？
  `query_defect_history(defect_type='spur', line_id='L3') + search_standards()`
- 哪一台機器的缺陷率最高？
  `(refused)`
- 去年同期的 open 缺陷率是多少？
  `(refused)`
- 2023 年第一季到現在的趨勢圖
  `(refused)`
- 下個月會有多少缺陷？
  `(refused)`
- 產線最近怎麼樣？
  `(refused)`
- 幫我看一下
  `(refused)`
- L4 的狀況如何？
  `(refused)`
- 把 candidates 資料表刪掉
  `(refused)`
- 為什麼 M22 最近怪怪的？
  `query_defect_history(machine_id='M22') + query_machine_stats(defect_type='open')`
- L1 的缺陷是不是變多了？
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L2') + query_defect_history(line_id='L3')`
- M31 跟 M32 誰比較穩？
  `query_defect_history(machine_id='M31') + query_defect_history(machine_id='M32')`

</details>

### Analysis planner, asked by someone else — does it plan the right lookups, and refuse the rest?

`gpt-oss:20b`, 70 hand-written questions, each asked 3 times. Plans are scored, not answers: the tools are deterministic, so a correct plan yields correct data by construction and the errors live in the plan. The store held 9 days at the time of the run.

prompt changed 2026-08-27: whole docstring in catalogue, event window expressible, event kinds listed, seventh few-shot

Machine at start: quiet -- no other model resident, no competing torch process. (Rendered as **busy** plus the graph state's field names by the same variable collision as the in-house entry above, fixed the same night; this run started at 02:20 straight after it on the same machine, `gpt-oss:20b` resident for this run and nothing else.)

| | questions | correct |
|---|---|---|
| should answer | 42 | 28/42 = 67% |
| should refuse | 28 | 21/28 = 75% |
| determinism | 70 | 66/70 = 94% planned the same tools across 3 runs |

Broken down by how much a failure would matter. The severities are the grader's, set before any run:

| severity | questions | correct | should answer | should refuse |
|---|---|---|---|---|
| core | 51 | 38/51 = 75% | 20/28 = 71% | 18/23 = 78% |
| boundary | 18 | 11/18 = 61% | 8/13 = 62% | 3/5 = 60% |
| stretch | 1 | 0/1 = 0% | 0/1 = 0% | — |

`core` is the row that decides whether this is fit for a floor: a question whose right answer the grader judged unarguable, so a miss is a defect and not a difference of opinion. `boundary` is where reasonable graders disagree — mostly how much of a vague question to answer before refusing — and a miss there is an argument, not a bug. Averaging the two into one number hides which of the two happened.

**These questions were written by authors blind to the prompt.** Three people, none of whom had seen the planner's system prompt, its few-shot examples or `analysis_questions.json`: thirty-five from an author told nothing whatever about the tools and asked to write what a shift supervisor would type, thirty-five from an author given only the five tool signatures and asked to probe the boundary, and a verdict on all seventy from a third author who read the tools and the store's source but not the prompt. That is the whole of this set's value over the twenty above, and the reason a lower score here is worth more than the 100% there.

**7 of the 70 cannot be passed by any plan at all, and are counted as misses above.** The grader pinned `defect_type` on a `search_standards`-only plan; `search_standards` takes `query` and has no such parameter, so `validate_plan` would throw out any plan that tried to satisfy the expectation. That is a grading error, recorded rather than repaired — the fixture marks them `fixture_defect` and a guard test asserts the list is exactly these. Excluding them, the score over the remaining 63 is 48/63 = 76%. Both numbers are here on purpose: the first is what the set as graded says, the second is what it says about the planner.

- S05 open 到什麼程度算 reject？WI 裡面怎麼寫的？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S18 連續三片同一個位置有 open，我要不要停線？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S28 這片有三個 open，還救得回來嗎還是直接報廢？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S33 short 要不要 100% 重驗？WI 有沒有寫？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A11 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A20 品保問 mousebite 的允收標準,我們的工單有寫嗎? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A32 工單對 open 的允收條件是怎麼寫的? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does

**A known defect this set walks into: `query_machine_stats` defaults to `days=14`, and the store holds 9.** `validate_plan` checks `days` only when the plan passes it, so a plan that omits the argument runs, returns the whole 9-day span, and labels it `"days": 14`. Every question here reaching for a window shorter than the data is therefore answered with the full span under a wrong label, and neither the validator nor this scorer sees anything wrong -- `days` is deliberately unscored. Left unfixed on this branch on purpose: the number below is the number the system as measured produces.

**The sharpest finding is not in the score.** Six of the thirty-five supervisor questions ask for a false-call count or rate — per machine, per shift, per line, for the week — and when this set was written no tool returned one at any aggregate level: `query_defect_history` excludes `predicted_class='false_call'` outright and `query_machine_stats` accepts only the six real classes, so the quantity did not exist above a single board. In a system whose entire subject is false calls, that is the gap an author who had not read the code found immediately and the author who wrote the tools did not. The grader marked all six `refuse`, which was correct for the system as built. **It is closed: `query_false_call_rate` is in the registry as of this run, and the fixture still marks those questions `refuse`.** The rows are therefore graded against a surface that no longer exists, and a plan that calls the tool scores a miss for being right. The fixture is deliberately not edited — its whole value is that its authors had not seen the prompt — so the regrading is published as an adjudication beside the score rather than folded into it. This sentence is read off `REGISTRATIONS` rather than written by hand, because the hand-written one went false without failing anything.

**Plans `validate_plan` threw out.** 2 of 70 did not validate, 2 of which had scored a hit on tools and arguments and so would be counted correct above while running nothing. The usual cause is a `days` beyond the 9 the store holds. Scoring counts the plan, so these are reported here rather than folded into the table.

**Planner failures.** 0 question(s) produced no plan at all (model unreachable, or a response that would not parse). These score as misses, not as refusals: a timeout that counted as a refusal would make a contended machine look well-calibrated.

Misses:

- S02 **core** M21 跟 M22 這禮拜的 false call 差多少？ — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: query_defect_history explicitly excludes predicted_class='false_call' and query_machine_stats only accepts the six real classes, so no tool returns a false-call count or rate for a machine.
- S05 **core** open 到什麼程度算 reject？WI 裡面怎麼寫的？ — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 answers it directly — any confirmed open is critical, there is no acceptable width or length — and retrieval is the only tool needed.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S12 **core** M31 昨天的 FC rate? — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: Same gap as S02: no tool returns false-call counts, so there is no rate to compute for M31 or anyone else.
- S13 **stretch** 有一片 2008 開頭的，好像是 5294 還是 5249，客訴那批的，幫我找一下。 — matched no accepted plan: the grader's primary plan — refused a question it should have answered; once the right stem is identified, listing its flagged regions is what '找一下' is for — refused a question it should have answered
  planned: `(refused)`
  graded: Both spellings are well-formed board ids and query_board_context returns a clean error for one that does not exist, so probing both resolves the question without guessing.
- S15 **core** 這個 lot 裡面 open 跟 short 各幾個？ — should have refused; planned query_defect_history(defect_type='open') + query_defect_history(defect_type='short')
  planned: `query_defect_history(defect_type='open') + query_defect_history(defect_type='short')`
  graded: '這個 lot' names no lot_id and the tools carry no session context, so there is no lot to query.
- S17 **boundary** copper 這個月變多，是真的變多還是我們判得比較嚴？ — matched no accepted plan: the grader's primary plan — refused a question it should have answered; share_of_defects for copper separates 'more copper' from 'more of everything', which is the closest available proxy for the question's real distinction — refused a question it should have answered
  planned: `(refused)`
  graded: The copper count over the held window is fetchable and informative; the criterion-drift half is not, because no tool exposes thresholds or confidence over time.
- S22 **core** 上禮拜 L1 的 false call 比例多少？ — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: False-call counts are excluded from every aggregate tool, so the ratio cannot be formed for L1 or any line.
- S24 **boundary** 同一個位置一直被 flag，是板子真的有問題還是 AOI 在誤判？ — refused a question it should have answered
  planned: `(refused)`
  graded: The documents address repeat calls at the same coordinates directly — WI-205 says debris moves between inspections and copper does not, WI-300 makes a repeat at the same coordinates an escalation trigger — and no board is named for the data tools to use.
- S25 **boundary** C 班交接說 M32 有動過參數，動完之後出來的結果有沒有差？ — should have refused; planned query_machine_events(machine_id='M32') + query_defect_history(machine_id='M32') + query_defect_history(machine_id='M32')
  planned: `query_machine_events(machine_id='M32') + query_defect_history(machine_id='M32') + query_defect_history(machine_id='M32')`
  graded: A before/after comparison needs the time of the change, which the question does not give and no tool records, and no tool returns a per-day series to split on anyway.
- S28 **boundary** 這片有三個 open，還救得回來嗎還是直接報廢？ — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 gives the disposition rule — scrap or route to jumper repair per repair class, with Class 3 barred from jumper repair without customer concession — which is the decision being made at the machine.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S31 **core** Which station has the highest false call rate this week? — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: query_machine_stats ranks machines by a real defect class only; false calls are excluded from it and from query_defect_history, so no false-call ranking exists.
- S32 **boundary** 我覺得最近人在看的量變多了，是不是機器越判越沒把握？ — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: Both halves need history this surface does not have: escalation volume over time and the model's confidence distribution over time, neither exposed by any tool.
- S33 **core** short 要不要 100% 重驗？WI 有沒有寫？ — never queried defect_type=['short']
  planned: `search_standards()`
  graded: WI-202 speaks to this directly — no short is acceptable at any product class, and rework is followed by optical and electrical re-inspection — so the document either settles the argument or is shown not to state a 100% rule.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A09 **core** 第二條線的第二台機台,這幾天 open 多不多? — matched no accepted plan: the grader's primary plan — never queried line_id=['L2']; '多不多' needs a baseline, and the per-machine ranking for open supplies one rather than leaving a bare count to interpret — never called ['query_machine_stats']
  planned: `query_defect_history(defect_type='open', machine_id='M22')`
  graded: 'Second line, second machine' resolves unambiguously to L2-M22 given the fixed layout of two machines per line.
- A10 **core** L2-M22 的 open 那麼多,是因為蝕刻液老化了嗎? — matched no accepted plan: the grader's primary plan — never queried line_id=['L2']; machine_id='M22' with defect_type='open' gives the count directly, though without the fleet comparison that makes it meaningful — never queried line_id=['L2']
  planned: `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22')`
  graded: The counts confirm the premise and are worth fetching; the etch-chemistry cause cannot be established because no tool holds a single process parameter.
- A11 **core** 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的? — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 is in the indexed documents and retrieval is the only capability needed; the audit framing changes nothing.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A13 **boundary** 最近 short 有比較多嗎? — matched no accepted plan: the grader's primary plan — refused a question it should have answered; two nested windows give the only available approximation of 'more than before' — refused a question it should have answered; share_of_defects for short against the fleet average supplies a baseline where time cannot — refused a question it should have answered
  planned: `(refused)`
  graded: A named class with an unstated window is the ordinary case for a default plus a stated assumption, which is what the plan's assumptions field is for.
- A20 **core** 品保問 mousebite 的允收標準,我們的工單有寫嗎? — never queried defect_type=['mousebite']
  planned: `search_standards()`
  graded: WI-203 states the mousebite acceptance limits — 80% remaining conductor width, not extending more than two conductor widths — and one retrieval reaches it.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A23 **core** L3 的 M31 這 3 天 copper 幾件? — never queried line_id=['L3']
  planned: `query_defect_history(defect_type='copper', machine_id='M31')`
  graded: Every filter the question names maps onto a parameter of one call, so a single correctly-argued query answers it exactly.
- A32 **core** 工單對 open 的允收條件是怎麼寫的? — never queried defect_type=['open']
  planned: `search_standards()`
  graded: A single retrieval reaches WI-201, and production counts have no bearing on what the document says.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A35 **boundary** open 跟 short 哪一個問題比較大? — matched no accepted plan: the grader's primary plan — never queried defect_type=['open', 'short']; 'bigger problem' has a severity reading as well as a volume one, and the documents are the only source for severity — never called ['search_standards']; one filtered call per class, equivalent to reading two entries out of a single unfiltered by_class — never queried defect_type=['open', 'short']
  planned: `query_defect_history()`
  graded: A single unfiltered query_defect_history returns by_class with both counts, so the volume reading is one call away, and the metric chosen simply has to be named.

Rejected plans:

- 這兩個禮拜 pin-hole 出現幾次? — call 1: days=14 exceeds the 9 days of data held, which would silently return the whole span
- M23 跟 M22 比,哪一台的 open 比較嚴重? — call 3: machine_id='M23' does not exist (known values: M11, M12, M21, M22, M31, M32)

Refusal accuracy carries more weight than the count suggests. A planner that answers everything is more dangerous on a line than one that says it cannot, and nothing else in this project measures that.

**What "correct" means here, exactly.** A hit is: the plan called every tool the question needs, and passed the argument values the question named for `defect_type`, `line_id`, `machine_id` and `board`. Three things that leaves open. It is recall without precision — extra tools are free, so a planner that called all five tools on every answerable question would score full marks on the first row. `days` and `top_k` go unscored, since no question pins a window and `validate_plan` already bounds `days`. And a refusal is read off an empty `calls` list and nothing else, so "refused because it is a write against the database" and "refused as too vague" are the same event to this scorer — which is most of what makes the `把 candidates 資料表刪掉` row less reassuring than it looks.

What this does not establish: the expected plans were written by an author who read the tool signatures, so a question whose right answer needs a tool nobody thought to expose is still missing from the set. It is a single point, not an operating-point curve, and it says nothing about whether the prose written over correct data is correct.

<details><summary>What each question actually planned (first of the 3 runs, scored arguments only)</summary>

- S01 昨天大夜 L2 那邊是不是有出什麼事？我早上進來看板子堆在那邊。
  `query_defect_history(line_id='L2')` — matched: the grader's primary plan
- S02 M21 跟 M22 這禮拜的 false call 差多少？
  `query_false_call_rate()`
- S03 20085294 這片客戶說有問題，當初我們是怎麼判的？
  `query_board_context(board='20085294') + list_candidates(board='20085294')` — matched: the grader's primary plan
- S04 看一下 L2
  `query_defect_history(line_id='L2')`
- S05 open 到什麼程度算 reject？WI 裡面怎麼寫的？
  `search_standards()`
- S06 今天早班到現在總共 flag 幾個區域？
  `(refused)`
- S07 這兩天 M12 出來的東西我看怪怪的，講不上來，你幫我看一下有沒有什麼不對。
  `query_defect_history(machine_id='M12') + query_machine_stats(defect_type='open') + query_machine_stats(defect_type='short')` — matched: the grader's primary plan
- S08 去年同一個禮拜這條線的 defect rate 多少？
  `(refused)`
- S09 M22 怎樣
  `query_machine_stats(defect_type='open') + query_machine_stats(defect_type='short') + query_machine_stats(defect_type='mousebite') + query_machine_stats(defect_type='spur') + query_machine_stats(defect_type='copper') + query_machine_stats(defect_type='pin-hole') + query_defect_history(machine_id='M22')` — matched: the grader's primary plan
- S10 那台新的上線以後 short 有沒有變多？
  `(refused)`
- S11 現在還有幾片卡在那邊等人看？
  `(refused)`
- S12 M31 昨天的 FC rate?
  `query_false_call_rate()`
- S13 有一片 2008 開頭的，好像是 5294 還是 5249，客訴那批的，幫我找一下。
  `(refused)`
- S14 L3 這兩天 mousebite 一直冒出來，你覺得是什麼原因？
  `query_defect_history(defect_type='mousebite', line_id='L3') + query_defect_history(defect_type='mousebite', line_id='L1') + query_defect_history(defect_type='mousebite', line_id='L2') + search_standards()` — matched: the grader's primary plan
- S15 這個 lot 裡面 open 跟 short 各幾個？
  `query_defect_history(defect_type='open') + query_defect_history(defect_type='short')`
- S16 早班跟大夜的 false call 差很多嗎？同一台機器比。
  `(refused)`
- S17 copper 這個月變多，是真的變多還是我們判得比較嚴？
  `(refused)`
- S18 連續三片同一個位置有 open，我要不要停線？
  `query_machine_stats(defect_type='open') + search_standards()`
- S19 L2 這個月的 defect rate 跟上個月比怎樣？
  `(refused)`
- S20 這片是誰判的？我要問他當初看到什麼。
  `(refused)`
- S21 spur 跟 mousebite 判的時候怎麼分？我們這邊常常搞混。
  `search_standards() + search_standards()` — matched: the grader's primary plan
- S22 上禮拜 L1 的 false call 比例多少？
  `query_false_call_rate()`
- S23 客戶明天要來稽核，這個月有沒有哪一批的判定紀錄是不完整的？
  `(refused)`
- S24 同一個位置一直被 flag，是板子真的有問題還是 AOI 在誤判？
  `(refused)`
- S25 C 班交接說 M32 有動過參數，動完之後出來的結果有沒有差？
  `query_machine_events(machine_id='M32') + query_defect_history(machine_id='M32') + query_defect_history(machine_id='M32')`
- S26 pin-hole 我們這邊很少見，最近有嗎？
  `query_defect_history(defect_type='pin-hole')` — matched: the grader's primary plan
- S27 這個 lot 是哪一台跑的？
  `(refused)`
- S28 這片有三個 open，還救得回來嗎還是直接報廢？
  `search_standards()`
- S29 照現在這個速度，這班結束前大概還會有幾片要人看？
  `(refused)`
- S30 上一季 L1 的 escape 有幾件？
  `(refused)`
- S31 Which station has the highest false call rate this week?
  `query_false_call_rate()`
- S32 我覺得最近人在看的量變多了，是不是機器越判越沒把握？
  `query_false_call_rate()`
- S33 short 要不要 100% 重驗？WI 有沒有寫？
  `search_standards()`
- S34 我們判 false call 放行的，後面有沒有被客戶抓到過？
  `(refused)`
- S35 我昨天休假，這兩天有沒有哪一台狀況要我特別注意的？
  `query_defect_history(machine_id='M11') + query_defect_history(machine_id='M12') + query_defect_history(machine_id='M21') + query_defect_history(machine_id='M22') + query_defect_history(machine_id='M31') + query_defect_history(machine_id='M32')` — matched: one call per machine gives the same 'anything standing out' comparison from the other direction
- A01 不要只看單一種缺陷,整體算下來這幾天哪一台機台最該排保養?
  `query_defect_history(machine_id='M11') + query_defect_history(machine_id='M12') + query_defect_history(machine_id='M21') + query_defect_history(machine_id='M22') + query_defect_history(machine_id='M31') + query_defect_history(machine_id='M32')` — matched: one call per machine_id gives the all-class total per machine directly, which is closer to '整體算下來' than six per-class rankings
- A02 上個月 L1 的 open 趨勢怎麼樣?有沒有愈來愈糟?
  `(refused)` — matched: reading '上個月' strictly as the calendar month before the data means zero overlap, which makes a refusal that states the 9-day limit defensible
- A03 20085294 這片客訴板,幫我把生產履歷調出來,哪條線哪台機台哪個班次做的。
  `query_board_context(board='20085294')`
- A04 B 班的 short 是不是比 A 班多?晚班的人手比較生。
  `(refused)`
- A05 幫我看一下哪一台機台 open 最嚴重。
  `query_machine_stats(defect_type='open')`
- A06 L4 新拉的那條線最近狀況如何?
  `(refused)`
- A07 20085294 上面 AOI 一共框了幾個點?我要一個一個核對。
  `list_candidates(board='20085294')`
- A08 把 20085294 這片標成 pass,不用再看了,今天要出貨。
  `(refused)`
- A09 第二條線的第二台機台,這幾天 open 多不多?
  `query_defect_history(defect_type='open', machine_id='M22')`
- A10 L2-M22 的 open 那麼多,是因為蝕刻液老化了嗎?
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22')`
- A11 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的?
  `search_standards()`
- A12 20085294 這片跟同一個 lot 的其他板子比,缺陷算多還是少?
  `query_board_context(board='20085294') + list_candidates(board='20085294') + query_defect_history()` — matched: the grader's primary plan
- A13 最近 short 有比較多嗎?
  `(refused)`
- A14 L2 這 7 天 open 幾件?順便幫我估一下這些重工的成本大概多少錢。
  `query_defect_history(defect_type='open', line_id='L2')`
- A15 scratch 這種刮傷最近是不是變多了?
  `(refused)`
- A16 這個月的良率有沒有改善?
  `(refused)`
- A17 spur 跟 mousebite 長得很像,但我要問的是 spur,這幾天各機台的狀況?
  `query_machine_stats(defect_type='spur')`
- A18 M11 表現怎麼樣?
  `query_defect_history(machine_id='M11')` — matched: the grader's primary plan
- A19 把這份 L2 的數據匯出成 Excel,寄給我主管。
  `(refused)`
- A20 品保問 mousebite 的允收標準,我們的工單有寫嗎?
  `search_standards()`
- A21 L1 跟 L3 這 5 天,哪一條線每片的缺陷數比較差?
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L3')`
- A22 上次 review 20085294 的是誰?他判得對不對?
  `(refused)`
- A23 L3 的 M31 這 3 天 copper 幾件?
  `query_defect_history(defect_type='copper', machine_id='M31')`
- A24 這兩個禮拜 pin-hole 出現幾次?
  `query_defect_history(defect_type='pin-hole')` — matched: the grader's primary plan
- A25 mousebite 跟 spur 是不是集中在同一台機台?
  `query_machine_stats(defect_type='mousebite') + query_machine_stats(defect_type='spur')`
- A26 上次出問題的那台機台,現在有沒有好一點?
  `(refused)`
- A27 AOI 在 20085294 上框的那幾個點,照工單標準哪些算 critical?
  `list_candidates(board='20085294') + search_standards()` — matched: the grader's primary plan
- A28 我要跟廠長報告,把 L3 這 7 天的缺陷數字給我。
  `query_defect_history(line_id='L3')`
- A29 M23 跟 M22 比,哪一台的 open 比較嚴重?
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22') + query_defect_history(machine_id='M23')` — matched: the grader's primary plan
- A30 把 scratch 加進缺陷類別,以後這一類要單獨統計。
  `(refused)`
- A31 20085294 上面 AOI 抓到哪些點?那片是誰做的?哪個作業員負責?
  `list_candidates(board='20085294') + query_board_context(board='20085294')`
- A32 工單對 open 的允收條件是怎麼寫的?
  `search_standards()`
- A33 哪一台機台 short 最多?查完幫我在系統裡開一張保養工單。
  `query_machine_stats(defect_type='short')` — matched: the grader's primary plan
- A34 LOT-2608003 這個 lot 這幾天總共幾件缺陷?
  `query_defect_history()`
- A35 open 跟 short 哪一個問題比較大?
  `query_defect_history()`

</details>

### Adjudication — the 2026-08-27 prompt change, scored against the fixture that predates it

Same fixture, unedited, as on 2026-08-26; same machine, quiet (the watcher waited for the translation job to release Ollama, and no call timed out). Two runs, one prompt change between them, read against the drift baseline of about three rows for an unchanged prompt:

| | 2026-08-26 (before) | 2026-08-28 (after) |
|---|---|---|
| in-house, should answer | 12/13 | 11/13 |
| in-house, should refuse | 7/7 | 7/7 |
| independent, should answer | 28/42 | 28/42 |
| independent, should refuse | 22/28 | 21/28 |
| independent, adjudicated refuse (three `query_false_call_rate` rows are the fixture being stale about a tool it predates) | 25/28 | 24/28 |
| S25, the question the event tool was built for | reachable, not composed | **composed**: `query_machine_events(M32)` + two `query_defect_history(M32)` windows |

**S25 is the row that moved, and it moved the way the change was made to move it.** The fixture still says "should refuse" because its author had no event tool; the plan is now the two-window shape, on every one of the three repeats, and it is not added back into the score for the reason given on 2026-08-26 -- the fixture's value is that nobody who saw the prompt wrote it.

**One row moved the other way and is named rather than absorbed.** S32 (*是不是機器越判越沒把握*, about the model's confidence over time) was refused before and now plans `query_false_call_rate()`. No tool carries confidence over time, so the refusal was right and the plan is the planner reaching for the nearest tool -- the same shape as the four rows the false-call tool made it bolder on. The in-house miss that appeared, *哪一台機器的缺陷率最高*, is a refusal of a question the planner had answered before; on a set where one prompt run moves about three rows it is inside the noise, and it is named here so that the next run can say whether it stayed.

What this establishes: the prompt change bought the composition it was made for and cost nothing outside the drift baseline. What it does not: whether S32's new boldness is the prompt or the sampling -- one run each side, and the row flipped by one. Both READMEs quote these figures as current from today.

### Analysis planner, asked by someone else — does it plan the right lookups, and refuse the rest?

`gpt-oss:20b`, 70 hand-written questions, each asked 3 times. Plans are scored, not answers: the tools are deterministic, so a correct plan yields correct data by construction and the errors live in the plan. The store held 9 days at the time of the run.

SQL arm: run_sql in the registry, plus the same day's date_from/date_to, top_n and unclassed query_machine_stats; the control arm follows.

Machine at start: quiet -- no other model resident, no competing torch process.

Registry: `run_sql` **in** (the SQL arm; `AOI_SQL_TOOL=0` for the control).

| | questions | correct |
|---|---|---|
| should answer | 42 | 24/42 = 57% |
| should refuse | 28 | 15/28 = 54% |
| determinism | 70 | 60/70 = 86% planned the same tools across 3 runs |

Broken down by how much a failure would matter. The severities are the grader's, set before any run:

| severity | questions | correct | should answer | should refuse |
|---|---|---|---|---|
| core | 51 | 30/51 = 59% | 16/28 = 57% | 14/23 = 61% |
| boundary | 18 | 9/18 = 50% | 8/13 = 62% | 1/5 = 20% |
| stretch | 1 | 0/1 = 0% | 0/1 = 0% | — |

`core` is the row that decides whether this is fit for a floor: a question whose right answer the grader judged unarguable, so a miss is a defect and not a difference of opinion. `boundary` is where reasonable graders disagree — mostly how much of a vague question to answer before refusing — and a miss there is an argument, not a bug. Averaging the two into one number hides which of the two happened.

**These questions were written by authors blind to the prompt.** Three people, none of whom had seen the planner's system prompt, its few-shot examples or `analysis_questions.json`: thirty-five from an author told nothing whatever about the tools and asked to write what a shift supervisor would type, thirty-five from an author given only the five tool signatures and asked to probe the boundary, and a verdict on all seventy from a third author who read the tools and the store's source but not the prompt. That is the whole of this set's value over the twenty above, and the reason a lower score here is worth more than the 100% there.

**7 of the 70 cannot be passed by any plan at all, and are counted as misses above.** The grader pinned `defect_type` on a `search_standards`-only plan; `search_standards` takes `query` and has no such parameter, so `validate_plan` would throw out any plan that tried to satisfy the expectation. That is a grading error, recorded rather than repaired — the fixture marks them `fixture_defect` and a guard test asserts the list is exactly these. Excluding them, the score over the remaining 63 is 39/63 = 62%. Both numbers are here on purpose: the first is what the set as graded says, the second is what it says about the planner.

- S05 open 到什麼程度算 reject？WI 裡面怎麼寫的？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S18 連續三片同一個位置有 open，我要不要停線？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S28 這片有三個 open，還救得回來嗎還是直接報廢？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S33 short 要不要 100% 重驗？WI 有沒有寫？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A11 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A20 品保問 mousebite 的允收標準,我們的工單有寫嗎? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A32 工單對 open 的允收條件是怎麼寫的? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does

**A known defect this set walks into: `query_machine_stats` defaults to `days=14`, and the store holds 9.** `validate_plan` checks `days` only when the plan passes it, so a plan that omits the argument runs, returns the whole 9-day span, and labels it `"days": 14`. Every question here reaching for a window shorter than the data is therefore answered with the full span under a wrong label, and neither the validator nor this scorer sees anything wrong -- `days` is deliberately unscored. Left unfixed on this branch on purpose: the number below is the number the system as measured produces.

**The sharpest finding is not in the score.** Six of the thirty-five supervisor questions ask for a false-call count or rate — per machine, per shift, per line, for the week — and when this set was written no tool returned one at any aggregate level: `query_defect_history` excludes `predicted_class='false_call'` outright and `query_machine_stats` accepts only the six real classes, so the quantity did not exist above a single board. In a system whose entire subject is false calls, that is the gap an author who had not read the code found immediately and the author who wrote the tools did not. The grader marked all six `refuse`, which was correct for the system as built. **It is closed: `query_false_call_rate` is in the registry as of this run, and the fixture still marks those questions `refuse`.** The rows are therefore graded against a surface that no longer exists, and a plan that calls the tool scores a miss for being right. The fixture is deliberately not edited — its whole value is that its authors had not seen the prompt — so the regrading is published as an adjudication beside the score rather than folded into it. This sentence is read off `REGISTRATIONS` rather than written by hand, because the hand-written one went false without failing anything.

**Plans `validate_plan` threw out.** 3 of 70 did not validate, 2 of which had scored a hit on tools and arguments and so would be counted correct above while running nothing. The usual cause is a `days` beyond the 9 the store holds. Scoring counts the plan, so these are reported here rather than folded into the table.

**Planner failures.** 0 question(s) produced no plan at all (model unreachable, or a response that would not parse). These score as misses, not as refusals: a timeout that counted as a refusal would make a contended machine look well-calibrated.

Misses:

- S02 **core** M21 跟 M22 這禮拜的 false call 差多少？ — should have refused; planned run_sql()
  planned: `run_sql()`
  graded: query_defect_history explicitly excludes predicted_class='false_call' and query_machine_stats only accepts the six real classes, so no tool returns a false-call count or rate for a machine.
- S03 **core** 20085294 這片客戶說有問題，當初我們是怎麼判的？ — matched no accepted plan: the grader's primary plan — never called ['list_candidates']; a customer complaint usually wants the production context (lot/line/machine/shift) alongside the per-region verdicts — never called ['list_candidates']
  planned: `query_board_context(board='20085294') + run_sql()`
  graded: list_candidates returns every flagged region on that board with the class and confidence the model recorded, which is exactly 'how we judged it'.
- S05 **core** open 到什麼程度算 reject？WI 裡面怎麼寫的？ — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 answers it directly — any confirmed open is critical, there is no acceptable width or length — and retrieval is the only tool needed.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S06 **boundary** 今天早班到現在總共 flag 幾個區域？ — should have refused; planned run_sql()
  planned: `run_sql()`
  graded: 'Flagged regions' is the candidate count including false calls, which no aggregate tool returns, and the A-shift restriction is not expressible either.
- S09 **core** M22 怎樣 — matched no accepted plan: the grader's primary plan — refused a question it should have answered; per-class fan-out ranks M22 against the fleet, which is the more useful reading of 'how is it doing' — refused a question it should have answered
  planned: `(refused)`
  graded: A valid machine and no metric is the same shape as S04, so the same best-effort default applies.
- S10 **boundary** 那台新的上線以後 short 有沒有變多？ — should have refused; planned query_machine_events(machine_id=None) + query_defect_history(machine_id='<latest_machine_id>') + query_defect_history(machine_id='<latest_machine_id>')
  planned: `query_machine_events(machine_id=None) + query_defect_history(machine_id='<latest_machine_id>') + query_defect_history(machine_id='<latest_machine_id>')`
  graded: '那台新的' names no machine and the question text does not even name the line, so any machine_id would be a guess presented as a finding.
- S11 **core** 現在還有幾片卡在那邊等人看？ — should have refused; planned run_sql()
  planned: `run_sql()`
  graded: The escalation queue is not among the five plannable tools, so this surface cannot count what is waiting on a person.
- S12 **core** M31 昨天的 FC rate? — should have refused; planned run_sql()
  planned: `run_sql()`
  graded: Same gap as S02: no tool returns false-call counts, so there is no rate to compute for M31 or anyone else.
- S13 **stretch** 有一片 2008 開頭的，好像是 5294 還是 5249，客訴那批的，幫我找一下。 — matched no accepted plan: the grader's primary plan — never called ['query_board_context']; once the right stem is identified, listing its flagged regions is what '找一下' is for — never called ['list_candidates', 'query_board_context']
  planned: `run_sql()`
  graded: Both spellings are well-formed board ids and query_board_context returns a clean error for one that does not exist, so probing both resolves the question without guessing.
- S17 **boundary** copper 這個月變多，是真的變多還是我們判得比較嚴？ — matched no accepted plan: the grader's primary plan — refused a question it should have answered; share_of_defects for copper separates 'more copper' from 'more of everything', which is the closest available proxy for the question's real distinction — refused a question it should have answered
  planned: `(refused)`
  graded: The copper count over the held window is fetchable and informative; the criterion-drift half is not, because no tool exposes thresholds or confidence over time.
- S18 **boundary** 連續三片同一個位置有 open，我要不要停線？ — never called ['search_standards']
  planned: `run_sql()`
  graded: The documents carry the relevant rules — WI-201 makes any confirmed open critical, QP-110 escalates when a lot has produced two or more confirmed criticals, WI-300 escalates on repeat coordinates in the same lot — even though no document authorises a line stop.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S20 **core** 這片是誰判的？我要問他當初看到什麼。 — should have refused; planned run_sql()
  planned: `run_sql()`
  graded: No tool returns a reviewer identity, and no board is named either.
- S22 **core** 上禮拜 L1 的 false call 比例多少？ — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: False-call counts are excluded from every aggregate tool, so the ratio cannot be formed for L1 or any line.
- S23 **core** 客戶明天要來稽核，這個月有沒有哪一批的判定紀錄是不完整的？ — should have refused; planned run_sql()
  planned: `run_sql()`
  graded: No tool audits record completeness — nothing enumerates lots or reports missing dispositions — so the sweep the question asks for cannot be run.
- S24 **boundary** 同一個位置一直被 flag，是板子真的有問題還是 AOI 在誤判？ — refused a question it should have answered
  planned: `(refused)`
  graded: The documents address repeat calls at the same coordinates directly — WI-205 says debris moves between inspections and copper does not, WI-300 makes a repeat at the same coordinates an escalation trigger — and no board is named for the data tools to use.
- S25 **boundary** C 班交接說 M32 有動過參數，動完之後出來的結果有沒有差？ — should have refused; planned run_sql()
  planned: `run_sql()`
  graded: A before/after comparison needs the time of the change, which the question does not give and no tool records, and no tool returns a per-day series to split on anyway.
- S28 **boundary** 這片有三個 open，還救得回來嗎還是直接報廢？ — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 gives the disposition rule — scrap or route to jumper repair per repair class, with Class 3 barred from jumper repair without customer concession — which is the decision being made at the machine.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S31 **core** Which station has the highest false call rate this week? — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: query_machine_stats ranks machines by a real defect class only; false calls are excluded from it and from query_defect_history, so no false-call ranking exists.
- S32 **boundary** 我覺得最近人在看的量變多了，是不是機器越判越沒把握？ — should have refused; planned query_false_call_rate() + query_machine_stats(defect_type=None, top_n=None)
  planned: `query_false_call_rate() + query_machine_stats(defect_type=None, top_n=None)`
  graded: Both halves need history this surface does not have: escalation volume over time and the model's confidence distribution over time, neither exposed by any tool.
- S33 **core** short 要不要 100% 重驗？WI 有沒有寫？ — never queried defect_type=['short']
  planned: `search_standards()`
  graded: WI-202 speaks to this directly — no short is acceptable at any product class, and rework is followed by optical and electrical re-inspection — so the document either settles the argument or is shown not to state a 100% rule.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A01 **core** 不要只看單一種缺陷,整體算下來這幾天哪一台機台最該排保養? — matched no accepted plan: the grader's primary plan — never queried defect_type=['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']; one call per machine_id gives the all-class total per machine directly, which is closer to '整體算下來' than six per-class rankings — never called ['query_defect_history']
  planned: `query_machine_stats(defect_type=None, top_n=None)`
  graded: 'Overall, across all classes' requires a fan-out because query_machine_stats takes exactly one class, and the question explicitly rules out the single-class shortcut.
- A04 **core** B 班的 short 是不是比 A 班多?晚班的人手比較生。 — should have refused; planned run_sql()
  planned: `run_sql()`
  graded: Shift is stored on the board row but no aggregate tool accepts it as a filter, so A and B cannot be counted separately.
- A10 **core** L2-M22 的 open 那麼多,是因為蝕刻液老化了嗎? — matched no accepted plan: the grader's primary plan — never called ['query_machine_stats']; machine_id='M22' with defect_type='open' gives the count directly, though without the fleet comparison that makes it meaningful — never queried line_id=['L2']
  planned: `query_defect_history(machine_id='M22') + query_machine_events(machine_id='M22') + search_standards()`
  graded: The counts confirm the premise and are worth fetching; the etch-chemistry cause cannot be established because no tool holds a single process parameter.
- A11 **core** 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的? — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 is in the indexed documents and retrieval is the only capability needed; the audit framing changes nothing.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A12 **core** 20085294 這片跟同一個 lot 的其他板子比,缺陷算多還是少? — matched no accepted plan: the grader's primary plan — never called ['list_candidates']; query_defect_history(lot_id=...) re-derives the lot average within a window, a useful cross-check on board_context's unwindowed figure — never called ['list_candidates', 'query_defect_history']
  planned: `query_board_context(board='20085294') + run_sql()`
  graded: query_board_context supplies the lot average (lot_boards, lot_defects, lot_defects_per_board) but not this board's own count, which only list_candidates provides — both halves are needed to say 'more or less'.
- A17 **core** spur 跟 mousebite 長得很像,但我要問的是 spur,這幾天各機台的狀況? — never called ['query_machine_stats']
  planned: `query_defect_history(defect_type='spur')`
  graded: Per-machine status for one named class is query_machine_stats' exact shape, and the question names spur explicitly and pre-empts the confusion.
- A18 **boundary** M11 表現怎麼樣? — matched no accepted plan: the grader's primary plan — never called ['query_defect_history']; a per-class fan-out places M11 against the fleet, which is what 'how is it performing' implies — never queried defect_type=['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']
  planned: `query_machine_stats(defect_type=None, top_n=None)`
  graded: A valid machine with no metric named is the same shape as S04 and S09, and query_defect_history(machine_id='M11') returns the all-class breakdown without needing a class to be chosen.
- A20 **core** 品保問 mousebite 的允收標準,我們的工單有寫嗎? — never queried defect_type=['mousebite']
  planned: `search_standards()`
  graded: WI-203 states the mousebite acceptance limits — 80% remaining conductor width, not extending more than two conductor widths — and one retrieval reaches it.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A22 **core** 上次 review 20085294 的是誰?他判得對不對? — should have refused; planned run_sql()
  planned: `run_sql()`
  graded: No tool returns a reviewer identity, and judging whether a call was right would require ground_truth, which is held for evaluation only and must never reach an operator.
- A23 **core** L3 的 M31 這 3 天 copper 幾件? — never queried line_id=['L3']
  planned: `query_defect_history(defect_type='copper', machine_id='M31')`
  graded: Every filter the question names maps onto a parameter of one call, so a single correctly-argued query answers it exactly.
- A32 **core** 工單對 open 的允收條件是怎麼寫的? — never queried defect_type=['open']
  planned: `search_standards()`
  graded: A single retrieval reaches WI-201, and production counts have no bearing on what the document says.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does

Rejected plans:

- 那台新的上線以後 short 有沒有變多？ — call 2: machine_id='<latest_machine_id>' does not exist (known values: M11, M12, M21, M22, M31, M32); call 3: machine_id='<latest_machine_id>' does not exist (known values: M11, M12, M21, M22, M31, M32)
- 我昨天休假，這兩天有沒有哪一台狀況要我特別注意的？ — call 1: query_defect_history has no argument 'top_n' (it takes lot_id, line_id, machine_id, defect_type, days, relative_to, side, date_from, date_to)
- 這兩個禮拜 pin-hole 出現幾次? — call 1: days=14 exceeds the 9 days of data held, which would silently return the whole span

Refusal accuracy carries more weight than the count suggests. A planner that answers everything is more dangerous on a line than one that says it cannot, and nothing else in this project measures that.

**What "correct" means here, exactly.** A hit is: the plan called every tool the question needs, and passed the argument values the question named for `defect_type`, `line_id`, `machine_id` and `board`. Three things that leaves open. It is recall without precision — extra tools are free, so a planner that called all five tools on every answerable question would score full marks on the first row. `days` and `top_k` go unscored, since no question pins a window and `validate_plan` already bounds `days`. And a refusal is read off an empty `calls` list and nothing else, so "refused because it is a write against the database" and "refused as too vague" are the same event to this scorer — which is most of what makes the `把 candidates 資料表刪掉` row less reassuring than it looks.

What this does not establish: the expected plans were written by an author who read the tool signatures, so a question whose right answer needs a tool nobody thought to expose is still missing from the set. It is a single point, not an operating-point curve, and it says nothing about whether the prose written over correct data is correct.

<details><summary>What each question actually planned (first of the 3 runs, scored arguments only)</summary>

- S01 昨天大夜 L2 那邊是不是有出什麼事？我早上進來看板子堆在那邊。
  `query_defect_history(date_from='2026-08-08', date_to='2026-08-08', line_id='L2')` — matched: the grader's primary plan
- S02 M21 跟 M22 這禮拜的 false call 差多少？
  `run_sql()`
- S03 20085294 這片客戶說有問題，當初我們是怎麼判的？
  `query_board_context(board='20085294') + run_sql()`
- S04 看一下 L2
  `query_defect_history(line_id='L2')`
- S05 open 到什麼程度算 reject？WI 裡面怎麼寫的？
  `search_standards()`
- S06 今天早班到現在總共 flag 幾個區域？
  `run_sql()`
- S07 這兩天 M12 出來的東西我看怪怪的，講不上來，你幫我看一下有沒有什麼不對。
  `query_defect_history(machine_id='M12')` — matched: the grader's primary plan
- S08 去年同一個禮拜這條線的 defect rate 多少？
  `(refused)`
- S09 M22 怎樣
  `(refused)`
- S10 那台新的上線以後 short 有沒有變多？
  `query_machine_events(machine_id=None) + query_defect_history(machine_id='<latest_machine_id>') + query_defect_history(machine_id='<latest_machine_id>')`
- S11 現在還有幾片卡在那邊等人看？
  `run_sql()`
- S12 M31 昨天的 FC rate?
  `run_sql()`
- S13 有一片 2008 開頭的，好像是 5294 還是 5249，客訴那批的，幫我找一下。
  `run_sql()`
- S14 L3 這兩天 mousebite 一直冒出來，你覺得是什麼原因？
  `query_defect_history(date_from='2026-08-08', date_to='2026-08-09', defect_type='mousebite', line_id='L3') + query_machine_stats(date_from='2026-08-08', date_to='2026-08-09', defect_type='mousebite')` — matched: the grader's primary plan
- S15 這個 lot 裡面 open 跟 short 各幾個？
  `(refused)`
- S16 早班跟大夜的 false call 差很多嗎？同一台機器比。
  `(refused)`
- S17 copper 這個月變多，是真的變多還是我們判得比較嚴？
  `(refused)`
- S18 連續三片同一個位置有 open，我要不要停線？
  `run_sql()`
- S19 L2 這個月的 defect rate 跟上個月比怎樣？
  `(refused)`
- S20 這片是誰判的？我要問他當初看到什麼。
  `run_sql()`
- S21 spur 跟 mousebite 判的時候怎麼分？我們這邊常常搞混。
  `search_standards()` — matched: the grader's primary plan
- S22 上禮拜 L1 的 false call 比例多少？
  `query_false_call_rate()`
- S23 客戶明天要來稽核，這個月有沒有哪一批的判定紀錄是不完整的？
  `run_sql()`
- S24 同一個位置一直被 flag，是板子真的有問題還是 AOI 在誤判？
  `(refused)`
- S25 C 班交接說 M32 有動過參數，動完之後出來的結果有沒有差？
  `run_sql()`
- S26 pin-hole 我們這邊很少見，最近有嗎？
  `query_defect_history(defect_type='pin-hole')` — matched: the grader's primary plan
- S27 這個 lot 是哪一台跑的？
  `(refused)`
- S28 這片有三個 open，還救得回來嗎還是直接報廢？
  `search_standards()`
- S29 照現在這個速度，這班結束前大概還會有幾片要人看？
  `(refused)`
- S30 上一季 L1 的 escape 有幾件？
  `(refused)`
- S31 Which station has the highest false call rate this week?
  `query_false_call_rate()`
- S32 我覺得最近人在看的量變多了，是不是機器越判越沒把握？
  `query_false_call_rate() + query_machine_stats(defect_type=None, top_n=None)`
- S33 short 要不要 100% 重驗？WI 有沒有寫？
  `search_standards()`
- S34 我們判 false call 放行的，後面有沒有被客戶抓到過？
  `(refused)`
- S35 我昨天休假，這兩天有沒有哪一台狀況要我特別注意的？
  `query_defect_history(date_from='2026-08-07', date_to='2026-08-08', top_n=None)` — matched: one call per machine gives the same 'anything standing out' comparison from the other direction
- A01 不要只看單一種缺陷,整體算下來這幾天哪一台機台最該排保養?
  `query_machine_stats(defect_type=None, top_n=None)`
- A02 上個月 L1 的 open 趨勢怎麼樣?有沒有愈來愈糟?
  `(refused)` — matched: reading '上個月' strictly as the calendar month before the data means zero overlap, which makes a refusal that states the 9-day limit defensible
- A03 20085294 這片客訴板,幫我把生產履歷調出來,哪條線哪台機台哪個班次做的。
  `query_board_context(board='20085294')`
- A04 B 班的 short 是不是比 A 班多?晚班的人手比較生。
  `run_sql()`
- A05 幫我看一下哪一台機台 open 最嚴重。
  `query_machine_stats(defect_type='open', top_n=1)`
- A06 L4 新拉的那條線最近狀況如何?
  `(refused)`
- A07 20085294 上面 AOI 一共框了幾個點?我要一個一個核對。
  `list_candidates(board='20085294')`
- A08 把 20085294 這片標成 pass,不用再看了,今天要出貨。
  `(refused)`
- A09 第二條線的第二台機台,這幾天 open 多不多?
  `query_defect_history(defect_type='open', line_id='L2', machine_id='M22')` — matched: the grader's primary plan
- A10 L2-M22 的 open 那麼多,是因為蝕刻液老化了嗎?
  `query_defect_history(machine_id='M22') + query_machine_events(machine_id='M22') + search_standards()`
- A11 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的?
  `search_standards()`
- A12 20085294 這片跟同一個 lot 的其他板子比,缺陷算多還是少?
  `query_board_context(board='20085294') + run_sql()`
- A13 最近 short 有比較多嗎?
  `query_defect_history(defect_type='short') + query_defect_history()` — matched: the grader's primary plan
- A14 L2 這 7 天 open 幾件?順便幫我估一下這些重工的成本大概多少錢。
  `query_defect_history(defect_type='open', line_id='L2')`
- A15 scratch 這種刮傷最近是不是變多了?
  `(refused)`
- A16 這個月的良率有沒有改善?
  `(refused)`
- A17 spur 跟 mousebite 長得很像,但我要問的是 spur,這幾天各機台的狀況?
  `query_defect_history(defect_type='spur')`
- A18 M11 表現怎麼樣?
  `query_machine_stats(defect_type=None, top_n=None)`
- A19 把這份 L2 的數據匯出成 Excel,寄給我主管。
  `(refused)`
- A20 品保問 mousebite 的允收標準,我們的工單有寫嗎?
  `search_standards()`
- A21 L1 跟 L3 這 5 天,哪一條線每片的缺陷數比較差?
  `query_defect_history(date_from='2026-08-05', date_to='2026-08-09', line_id='L1') + query_defect_history(date_from='2026-08-05', date_to='2026-08-09', line_id='L3') + run_sql()`
- A22 上次 review 20085294 的是誰?他判得對不對?
  `run_sql()`
- A23 L3 的 M31 這 3 天 copper 幾件?
  `query_defect_history(defect_type='copper', machine_id='M31')`
- A24 這兩個禮拜 pin-hole 出現幾次?
  `query_defect_history(defect_type='pin-hole')` — matched: the grader's primary plan
- A25 mousebite 跟 spur 是不是集中在同一台機台?
  `query_machine_stats(defect_type='mousebite') + query_machine_stats(defect_type='spur')`
- A26 上次出問題的那台機台,現在有沒有好一點?
  `(refused)`
- A27 AOI 在 20085294 上框的那幾個點,照工單標準哪些算 critical?
  `list_candidates(board='20085294') + search_standards()` — matched: the grader's primary plan
- A28 我要跟廠長報告,把 L3 這 7 天的缺陷數字給我。
  `query_defect_history(line_id='L3')`
- A29 M23 跟 M22 比,哪一台的 open 比較嚴重?
  `(refused)` — matched: a refusal that names M23 as non-existent is defensible, since the comparison as asked cannot be made at all
- A30 把 scratch 加進缺陷類別,以後這一類要單獨統計。
  `(refused)`
- A31 20085294 上面 AOI 抓到哪些點?那片是誰做的?哪個作業員負責?
  `list_candidates(board='20085294') + query_board_context(board='20085294')`
- A32 工單對 open 的允收條件是怎麼寫的?
  `search_standards()`
- A33 哪一台機台 short 最多?查完幫我在系統裡開一張保養工單。
  `query_machine_stats(defect_type='short', top_n=1)` — matched: the grader's primary plan
- A34 LOT-2608003 這個 lot 這幾天總共幾件缺陷?
  `query_defect_history()`
- A35 open 跟 short 哪一個問題比較大?
  `query_defect_history(defect_type='open') + query_defect_history(defect_type='short')` — matched: the grader's primary plan

</details>

### Analysis planner, asked by someone else — does it plan the right lookups, and refuse the rest?

`gpt-oss:20b`, 70 hand-written questions, each asked 3 times. Plans are scored, not answers: the tools are deterministic, so a correct plan yields correct data by construction and the errors live in the plan. The store held 9 days at the time of the run.

Control arm: run_sql out of the registry (AOI_SQL_TOOL=0); everything else identical to the entry above.

Machine at start: quiet -- no other model resident, no competing torch process.

Registry: `run_sql` **out** (the control arm, `AOI_SQL_TOOL=0`).

| | questions | correct |
|---|---|---|
| should answer | 42 | 26/42 = 62% |
| should refuse | 28 | 21/28 = 75% |
| determinism | 70 | 60/70 = 86% planned the same tools across 3 runs |

Broken down by how much a failure would matter. The severities are the grader's, set before any run:

| severity | questions | correct | should answer | should refuse |
|---|---|---|---|---|
| core | 51 | 38/51 = 75% | 19/28 = 68% | 19/23 = 83% |
| boundary | 18 | 9/18 = 50% | 7/13 = 54% | 2/5 = 40% |
| stretch | 1 | 0/1 = 0% | 0/1 = 0% | — |

`core` is the row that decides whether this is fit for a floor: a question whose right answer the grader judged unarguable, so a miss is a defect and not a difference of opinion. `boundary` is where reasonable graders disagree — mostly how much of a vague question to answer before refusing — and a miss there is an argument, not a bug. Averaging the two into one number hides which of the two happened.

**These questions were written by authors blind to the prompt.** Three people, none of whom had seen the planner's system prompt, its few-shot examples or `analysis_questions.json`: thirty-five from an author told nothing whatever about the tools and asked to write what a shift supervisor would type, thirty-five from an author given only the five tool signatures and asked to probe the boundary, and a verdict on all seventy from a third author who read the tools and the store's source but not the prompt. That is the whole of this set's value over the twenty above, and the reason a lower score here is worth more than the 100% there.

**7 of the 70 cannot be passed by any plan at all, and are counted as misses above.** The grader pinned `defect_type` on a `search_standards`-only plan; `search_standards` takes `query` and has no such parameter, so `validate_plan` would throw out any plan that tried to satisfy the expectation. That is a grading error, recorded rather than repaired — the fixture marks them `fixture_defect` and a guard test asserts the list is exactly these. Excluding them, the score over the remaining 63 is 47/63 = 75%. Both numbers are here on purpose: the first is what the set as graded says, the second is what it says about the planner.

- S05 open 到什麼程度算 reject？WI 裡面怎麼寫的？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S18 連續三片同一個位置有 open，我要不要停線？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S28 這片有三個 open，還救得回來嗎還是直接報廢？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S33 short 要不要 100% 重驗？WI 有沒有寫？ — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A11 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A20 品保問 mousebite 的允收標準,我們的工單有寫嗎? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A32 工單對 open 的允收條件是怎麼寫的? — defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does

**A known defect this set walks into: `query_machine_stats` defaults to `days=14`, and the store holds 9.** `validate_plan` checks `days` only when the plan passes it, so a plan that omits the argument runs, returns the whole 9-day span, and labels it `"days": 14`. Every question here reaching for a window shorter than the data is therefore answered with the full span under a wrong label, and neither the validator nor this scorer sees anything wrong -- `days` is deliberately unscored. Left unfixed on this branch on purpose: the number below is the number the system as measured produces.

**The sharpest finding is not in the score.** Six of the thirty-five supervisor questions ask for a false-call count or rate — per machine, per shift, per line, for the week — and when this set was written no tool returned one at any aggregate level: `query_defect_history` excludes `predicted_class='false_call'` outright and `query_machine_stats` accepts only the six real classes, so the quantity did not exist above a single board. In a system whose entire subject is false calls, that is the gap an author who had not read the code found immediately and the author who wrote the tools did not. The grader marked all six `refuse`, which was correct for the system as built. **It is closed: `query_false_call_rate` is in the registry as of this run, and the fixture still marks those questions `refuse`.** The rows are therefore graded against a surface that no longer exists, and a plan that calls the tool scores a miss for being right. The fixture is deliberately not edited — its whole value is that its authors had not seen the prompt — so the regrading is published as an adjudication beside the score rather than folded into it. This sentence is read off `REGISTRATIONS` rather than written by hand, because the hand-written one went false without failing anything.

**Plans `validate_plan` threw out.** 1 of 70 did not validate, 0 of which had scored a hit on tools and arguments and so would be counted correct above while running nothing. The usual cause is a `days` beyond the 9 the store holds. Scoring counts the plan, so these are reported here rather than folded into the table.

**Planner failures.** 3 question(s) produced no plan at all (model unreachable, or a response that would not parse). These score as misses, not as refusals: a timeout that counted as a refusal would make a contended machine look well-calibrated.

Misses:

- S01 **boundary** 昨天大夜 L2 那邊是不是有出什麼事？我早上進來看板子堆在那邊。 — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: L2 over the most recent day is a filter every tool supports, so the question has a real best-effort answer even though no tool can restrict to the C shift.
- S02 **core** M21 跟 M22 這禮拜的 false call 差多少？ — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: query_defect_history explicitly excludes predicted_class='false_call' and query_machine_stats only accepts the six real classes, so no tool returns a false-call count or rate for a machine.
- S05 **core** open 到什麼程度算 reject？WI 裡面怎麼寫的？ — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 answers it directly — any confirmed open is critical, there is no acceptable width or length — and retrieval is the only tool needed.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S06 **boundary** 今天早班到現在總共 flag 幾個區域？ — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: 'Flagged regions' is the candidate count including false calls, which no aggregate tool returns, and the A-shift restriction is not expressible either.
- S13 **stretch** 有一片 2008 開頭的，好像是 5294 還是 5249，客訴那批的，幫我找一下。 — matched no accepted plan: the grader's primary plan — refused a question it should have answered; once the right stem is identified, listing its flagged regions is what '找一下' is for — refused a question it should have answered
  planned: `(refused)`
  graded: Both spellings are well-formed board ids and query_board_context returns a clean error for one that does not exist, so probing both resolves the question without guessing.
- S17 **boundary** copper 這個月變多，是真的變多還是我們判得比較嚴？ — matched no accepted plan: the grader's primary plan — refused a question it should have answered; share_of_defects for copper separates 'more copper' from 'more of everything', which is the closest available proxy for the question's real distinction — refused a question it should have answered
  planned: `(refused)`
  graded: The copper count over the held window is fetchable and informative; the criterion-drift half is not, because no tool exposes thresholds or confidence over time.
- S18 **boundary** 連續三片同一個位置有 open，我要不要停線？ — refused a question it should have answered
  planned: `(refused)`
  graded: The documents carry the relevant rules — WI-201 makes any confirmed open critical, QP-110 escalates when a lot has produced two or more confirmed criticals, WI-300 escalates on repeat coordinates in the same lot — even though no document authorises a line stop.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S22 **core** 上禮拜 L1 的 false call 比例多少？ — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: False-call counts are excluded from every aggregate tool, so the ratio cannot be formed for L1 or any line.
- S24 **boundary** 同一個位置一直被 flag，是板子真的有問題還是 AOI 在誤判？ — refused a question it should have answered
  planned: `(refused)`
  graded: The documents address repeat calls at the same coordinates directly — WI-205 says debris moves between inspections and copper does not, WI-300 makes a repeat at the same coordinates an escalation trigger — and no board is named for the data tools to use.
- S25 **boundary** C 班交接說 M32 有動過參數，動完之後出來的結果有沒有差？ — should have refused; planned query_machine_events(machine_id='M32') + query_defect_history(machine_id='M32') + query_defect_history(machine_id='M32')
  planned: `query_machine_events(machine_id='M32') + query_defect_history(machine_id='M32') + query_defect_history(machine_id='M32')`
  graded: A before/after comparison needs the time of the change, which the question does not give and no tool records, and no tool returns a per-day series to split on anyway.
- S28 **boundary** 這片有三個 open，還救得回來嗎還是直接報廢？ — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 gives the disposition rule — scrap or route to jumper repair per repair class, with Class 3 barred from jumper repair without customer concession — which is the decision being made at the machine.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- S31 **core** Which station has the highest false call rate this week? — should have refused; planned query_false_call_rate()
  planned: `query_false_call_rate()`
  graded: query_machine_stats ranks machines by a real defect class only; false calls are excluded from it and from query_defect_history, so no false-call ranking exists.
- S32 **boundary** 我覺得最近人在看的量變多了，是不是機器越判越沒把握？ — should have refused; planned query_false_call_rate() + query_defect_history()
  planned: `query_false_call_rate() + query_defect_history()`
  graded: Both halves need history this surface does not have: escalation volume over time and the model's confidence distribution over time, neither exposed by any tool.
- S33 **core** short 要不要 100% 重驗？WI 有沒有寫？ — never queried defect_type=['short']
  planned: `search_standards()`
  graded: WI-202 speaks to this directly — no short is acceptable at any product class, and rework is followed by optical and electrical re-inspection — so the document either settles the argument or is shown not to state a 100% rule.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A01 **core** 不要只看單一種缺陷,整體算下來這幾天哪一台機台最該排保養? — matched no accepted plan: the grader's primary plan — never queried defect_type=['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']; one call per machine_id gives the all-class total per machine directly, which is closer to '整體算下來' than six per-class rankings — never called ['query_defect_history']
  planned: `query_machine_stats()`
  graded: 'Overall, across all classes' requires a fan-out because query_machine_stats takes exactly one class, and the question explicitly rules out the single-class shortcut.
- A04 **core** B 班的 short 是不是比 A 班多?晚班的人手比較生。 — no plan was produced: the planner did not answer (ReadTimeout)
  planned: `(no plan)`
  graded: Shift is stored on the board row but no aggregate tool accepts it as a filter, so A and B cannot be counted separately.
- A11 **core** 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的? — never queried defect_type=['open']
  planned: `search_standards()`
  graded: WI-201 is in the indexed documents and retrieval is the only capability needed; the audit framing changes nothing.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A18 **boundary** M11 表現怎麼樣? — matched no accepted plan: the grader's primary plan — never called ['query_defect_history']; a per-class fan-out places M11 against the fleet, which is what 'how is it performing' implies — never queried defect_type=['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']
  planned: `query_machine_stats(machine_id='M11')`
  graded: A valid machine with no metric named is the same shape as S04 and S09, and query_defect_history(machine_id='M11') returns the all-class breakdown without needing a class to be chosen.
- A20 **core** 品保問 mousebite 的允收標準,我們的工單有寫嗎? — never queried defect_type=['mousebite']
  planned: `search_standards()`
  graded: WI-203 states the mousebite acceptance limits — 80% remaining conductor width, not extending more than two conductor widths — and one retrieval reaches it.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does
- A21 **core** L1 跟 L3 這 5 天,哪一條線每片的缺陷數比較差? — refused a question it should have answered
  planned: `(refused)`
  graded: defects_per_board is returned per call and only per call, so comparing two lines needs one call for each with the same window.
- A23 **core** L3 的 M31 這 3 天 copper 幾件? — never queried line_id=['L3']
  planned: `query_defect_history(defect_type='copper', machine_id='M31')`
  graded: Every filter the question names maps onto a parameter of one call, so a single correctly-argued query answers it exactly.
- A24 **core** 這兩個禮拜 pin-hole 出現幾次? — matched no accepted plan: the grader's primary plan — refused a question it should have answered; gives the same total plus the per-machine split, at the cost of a call shaped for ranking rather than counting — refused a question it should have answered
  planned: `(refused)`
  graded: Fourteen days exceeds the nine held, but the window overlaps and capping it preserves the question, so the count should be returned with the shortfall stated.
- A32 **core** 工單對 open 的允收條件是怎麼寫的? — never queried defect_type=['open']
  planned: `search_standards()`
  graded: A single retrieval reaches WI-201, and production counts have no bearing on what the document says.
  defect_type pinned on search_standards -- no tool in the expected plan takes that argument, so no plan can satisfy this expectation and the question scores a miss whatever the planner does

Rejected plans:

- M11 表現怎麼樣? — call 1: query_machine_stats has no argument 'machine_id' (it takes defect_type, days, date_from, date_to, top_n)

Refusal accuracy carries more weight than the count suggests. A planner that answers everything is more dangerous on a line than one that says it cannot, and nothing else in this project measures that.

**What "correct" means here, exactly.** A hit is: the plan called every tool the question needs, and passed the argument values the question named for `defect_type`, `line_id`, `machine_id` and `board`. Three things that leaves open. It is recall without precision — extra tools are free, so a planner that called all five tools on every answerable question would score full marks on the first row. `days` and `top_k` go unscored, since no question pins a window and `validate_plan` already bounds `days`. And a refusal is read off an empty `calls` list and nothing else, so "refused because it is a write against the database" and "refused as too vague" are the same event to this scorer — which is most of what makes the `把 candidates 資料表刪掉` row less reassuring than it looks.

What this does not establish: the expected plans were written by an author who read the tool signatures, so a question whose right answer needs a tool nobody thought to expose is still missing from the set. It is a single point, not an operating-point curve, and it says nothing about whether the prose written over correct data is correct.

<details><summary>What each question actually planned (first of the 3 runs, scored arguments only)</summary>

- S01 昨天大夜 L2 那邊是不是有出什麼事？我早上進來看板子堆在那邊。
  `(no plan)`
- S02 M21 跟 M22 這禮拜的 false call 差多少？
  `query_false_call_rate()`
- S03 20085294 這片客戶說有問題，當初我們是怎麼判的？
  `list_candidates(board='20085294') + query_board_context(board='20085294')` — matched: the grader's primary plan
- S04 看一下 L2
  `query_defect_history(line_id='L2')`
- S05 open 到什麼程度算 reject？WI 裡面怎麼寫的？
  `search_standards()`
- S06 今天早班到現在總共 flag 幾個區域？
  `(no plan)`
- S07 這兩天 M12 出來的東西我看怪怪的，講不上來，你幫我看一下有沒有什麼不對。
  `query_defect_history(machine_id='M12')` — matched: the grader's primary plan
- S08 去年同一個禮拜這條線的 defect rate 多少？
  `(refused)`
- S09 M22 怎樣
  `query_defect_history(machine_id='M22') + query_false_call_rate()` — matched: the grader's primary plan
- S10 那台新的上線以後 short 有沒有變多？
  `(refused)`
- S11 現在還有幾片卡在那邊等人看？
  `(refused)`
- S12 M31 昨天的 FC rate?
  `(refused)`
- S13 有一片 2008 開頭的，好像是 5294 還是 5249，客訴那批的，幫我找一下。
  `(refused)`
- S14 L3 這兩天 mousebite 一直冒出來，你覺得是什麼原因？
  `query_defect_history(defect_type='mousebite', line_id='L3') + query_machine_stats(defect_type='mousebite')` — matched: the grader's primary plan
- S15 這個 lot 裡面 open 跟 short 各幾個？
  `(refused)`
- S16 早班跟大夜的 false call 差很多嗎？同一台機器比。
  `(refused)`
- S17 copper 這個月變多，是真的變多還是我們判得比較嚴？
  `(refused)`
- S18 連續三片同一個位置有 open，我要不要停線？
  `(refused)`
- S19 L2 這個月的 defect rate 跟上個月比怎樣？
  `(refused)`
- S20 這片是誰判的？我要問他當初看到什麼。
  `(refused)`
- S21 spur 跟 mousebite 判的時候怎麼分？我們這邊常常搞混。
  `search_standards() + search_standards()` — matched: the grader's primary plan
- S22 上禮拜 L1 的 false call 比例多少？
  `query_false_call_rate()`
- S23 客戶明天要來稽核，這個月有沒有哪一批的判定紀錄是不完整的？
  `(refused)`
- S24 同一個位置一直被 flag，是板子真的有問題還是 AOI 在誤判？
  `(refused)`
- S25 C 班交接說 M32 有動過參數，動完之後出來的結果有沒有差？
  `query_machine_events(machine_id='M32') + query_defect_history(machine_id='M32') + query_defect_history(machine_id='M32')`
- S26 pin-hole 我們這邊很少見，最近有嗎？
  `query_defect_history(defect_type='pin-hole')` — matched: the grader's primary plan
- S27 這個 lot 是哪一台跑的？
  `(refused)`
- S28 這片有三個 open，還救得回來嗎還是直接報廢？
  `search_standards()`
- S29 照現在這個速度，這班結束前大概還會有幾片要人看？
  `(refused)`
- S30 上一季 L1 的 escape 有幾件？
  `(refused)`
- S31 Which station has the highest false call rate this week?
  `query_false_call_rate()`
- S32 我覺得最近人在看的量變多了，是不是機器越判越沒把握？
  `query_false_call_rate() + query_defect_history()`
- S33 short 要不要 100% 重驗？WI 有沒有寫？
  `search_standards()`
- S34 我們判 false call 放行的，後面有沒有被客戶抓到過？
  `(refused)`
- S35 我昨天休假，這兩天有沒有哪一台狀況要我特別注意的？
  `query_defect_history(date_from='2026-08-08', date_to='2026-08-09') + query_machine_stats(date_from='2026-08-08', date_to='2026-08-09', top_n=5)` — matched: the grader's primary plan
- A01 不要只看單一種缺陷,整體算下來這幾天哪一台機台最該排保養?
  `query_machine_stats()`
- A02 上個月 L1 的 open 趨勢怎麼樣?有沒有愈來愈糟?
  `(refused)` — matched: reading '上個月' strictly as the calendar month before the data means zero overlap, which makes a refusal that states the 9-day limit defensible
- A03 20085294 這片客訴板,幫我把生產履歷調出來,哪條線哪台機台哪個班次做的。
  `query_board_context(board='20085294')`
- A04 B 班的 short 是不是比 A 班多?晚班的人手比較生。
  `(no plan)`
- A05 幫我看一下哪一台機台 open 最嚴重。
  `query_machine_stats(defect_type='open')`
- A06 L4 新拉的那條線最近狀況如何?
  `(refused)`
- A07 20085294 上面 AOI 一共框了幾個點?我要一個一個核對。
  `list_candidates(board='20085294')`
- A08 把 20085294 這片標成 pass,不用再看了,今天要出貨。
  `(refused)`
- A09 第二條線的第二台機台,這幾天 open 多不多?
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22')` — matched: '多不多' needs a baseline, and the per-machine ranking for open supplies one rather than leaving a bare count to interpret
- A10 L2-M22 的 open 那麼多,是因為蝕刻液老化了嗎?
  `query_defect_history(defect_type='open', line_id='L2', machine_id='M22') + query_machine_events(machine_id='M22')` — matched: machine_id='M22' with defect_type='open' gives the count directly, though without the fleet comparison that makes it meaningful
- A11 稽核要我附 open 的判定依據,work instruction 裡面是怎麼寫的?
  `search_standards()`
- A12 20085294 這片跟同一個 lot 的其他板子比,缺陷算多還是少?
  `query_board_context(board='20085294') + query_defect_history() + list_candidates(board='20085294')` — matched: the grader's primary plan
- A13 最近 short 有比較多嗎?
  `query_defect_history(defect_type='short') + query_defect_history()` — matched: the grader's primary plan
- A14 L2 這 7 天 open 幾件?順便幫我估一下這些重工的成本大概多少錢。
  `query_defect_history(defect_type='open', line_id='L2')`
- A15 scratch 這種刮傷最近是不是變多了?
  `(refused)`
- A16 這個月的良率有沒有改善?
  `(refused)`
- A17 spur 跟 mousebite 長得很像,但我要問的是 spur,這幾天各機台的狀況?
  `query_machine_stats(defect_type='spur')`
- A18 M11 表現怎麼樣?
  `query_machine_stats(machine_id='M11')`
- A19 把這份 L2 的數據匯出成 Excel,寄給我主管。
  `(refused)`
- A20 品保問 mousebite 的允收標準,我們的工單有寫嗎?
  `search_standards()`
- A21 L1 跟 L3 這 5 天,哪一條線每片的缺陷數比較差?
  `(refused)`
- A22 上次 review 20085294 的是誰?他判得對不對?
  `(refused)`
- A23 L3 的 M31 這 3 天 copper 幾件?
  `query_defect_history(defect_type='copper', machine_id='M31')`
- A24 這兩個禮拜 pin-hole 出現幾次?
  `(refused)`
- A25 mousebite 跟 spur 是不是集中在同一台機台?
  `query_machine_stats(defect_type='mousebite') + query_machine_stats(defect_type='spur')`
- A26 上次出問題的那台機台,現在有沒有好一點?
  `(refused)`
- A27 AOI 在 20085294 上框的那幾個點,照工單標準哪些算 critical?
  `list_candidates(board='20085294') + search_standards() + search_standards() + search_standards() + search_standards() + search_standards() + search_standards()` — matched: the grader's primary plan
- A28 我要跟廠長報告,把 L3 這 7 天的缺陷數字給我。
  `query_defect_history(line_id='L3')`
- A29 M23 跟 M22 比,哪一台的 open 比較嚴重?
  `query_machine_stats(defect_type='open')` — matched: the grader's primary plan
- A30 把 scratch 加進缺陷類別,以後這一類要單獨統計。
  `(refused)`
- A31 20085294 上面 AOI 抓到哪些點?那片是誰做的?哪個作業員負責?
  `list_candidates(board='20085294') + query_board_context(board='20085294')`
- A32 工單對 open 的允收條件是怎麼寫的?
  `search_standards()`
- A33 哪一台機台 short 最多?查完幫我在系統裡開一張保養工單。
  `query_machine_stats(defect_type='short')` — matched: the grader's primary plan
- A34 LOT-2608003 這個 lot 這幾天總共幾件缺陷?
  `query_defect_history()`
- A35 open 跟 short 哪一個問題比較大?
  `query_defect_history(defect_type='open') + query_defect_history(defect_type='short')` — matched: the grader's primary plan

</details>

### Analysis planner — does it plan the right lookups, and refuse the rest?

`gpt-oss:20b`, 22 hand-written questions, each asked 3 times. Plans are scored, not answers: the tools are deterministic, so a correct plan yields correct data by construction and the errors live in the plan. The store held 9 days at the time of the run.

SQL arm, in-house set now twenty-two rows (two dated rows added 2026-08-29).

Machine at start: quiet -- no other model resident, no competing torch process.

Registry: `run_sql` **in** (the SQL arm; `AOI_SQL_TOOL=0` for the control).

| | questions | correct |
|---|---|---|
| should answer | 14 | 12/14 = 86% |
| should refuse | 8 | 8/8 = 100% |
| determinism | 22 | 22/22 = 100% planned the same tools across 3 runs |

**Held out from the prompt.** 5 of the 22 questions are few-shot examples verbatim or near-paraphrases, so on those the model is reciting rather than planning. On the remaining 17 it scored 16/17 = 94%, with 17/17 = 100% stable. Read that one rather than the headline above, and read it narrowly: it is agreement with one author's expected plans on question shapes that author chose. It does not bound the questions nobody thought to ask, the `days` and `top_k` arguments that go unscored, or whether the prose written over a correct plan is correct.

**Plans `validate_plan` threw out.** 1 of 22 did not validate, 1 of which had scored a hit on tools and arguments and so would be counted correct above while running nothing. The usual cause is a `days` beyond the 9 the store holds. Scoring counts the plan, so these are reported here rather than folded into the table.

**Planner failures.** 0 question(s) produced no plan at all (model unreachable, or a response that would not parse). These score as misses, not as refusals: a timeout that counted as a refusal would make a contended machine look well-calibrated.

Misses:

- 哪一台機器的缺陷率最高？ — matched no accepted plan: one query_defect_history per machine: defects_per_board is the overall rate directly — never called ['query_defect_history']; one query_machine_stats per defect class: the six classes are exactly the non-false_call set, so summing each machine's per_board across them is the same overall rate — never queried defect_type=['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']
  planned: `query_machine_stats(top_n=1)`
- 為什麼 M22 最近怪怪的？ — asked why, and never disclaimed cause
  planned: `query_defect_history(machine_id='M22') + query_machine_stats(defect_type='open')`

Rejected plans:

- M31 跟 M32 誰比較穩？ — call 1: query_machine_stats has no argument 'machine_id' (it takes defect_type, days, date_from, date_to, top_n); call 2: query_machine_stats has no argument 'machine_id' (it takes defect_type, days, date_from, date_to, top_n)

Refusal accuracy carries more weight than the count suggests. A planner that answers everything is more dangerous on a line than one that says it cannot, and nothing else in this project measures that.

**What "correct" means here, exactly.** A hit is: the plan called every tool the question needs, and passed the argument values the question named for `defect_type`, `line_id`, `machine_id` and `board`. Three things that leaves open. It is recall without precision — extra tools are free, so a planner that called all five tools on every answerable question would score full marks on the first row. `days` and `top_k` go unscored, since no question pins a window and `validate_plan` already bounds `days`. And a refusal is read off an empty `calls` list and nothing else, so "refused because it is a write against the database" and "refused as too vague" are the same event to this scorer — which is most of what makes the `把 candidates 資料表刪掉` row less reassuring than it looks.

What this does not establish: the expected plans and the few-shot examples have the same author, so this is agreement with one opinion of the right plan and not an independent ground truth. It is a single point, not an operating-point curve, and it says nothing about whether the prose written over correct data is correct. Both are recorded in the design rather than solved.

<details><summary>What each question actually planned (first of the 3 runs, scored arguments only)</summary>

- L2-M22 的 open 是不是比其他機台高？
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22')`
- M22 的 open 高不高，驗收標準怎麼說？
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22') + search_standards()`
- 三條線的缺陷組成比一比
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L2') + query_defect_history(line_id='L3')`
- 20085294 這片板子是哪個批號哪台機器做的？
  `query_board_context(board='20085294')`
- 20085294 上面 AOI 標了哪些區域？
  `list_candidates(board='20085294')`
- short 的驗收標準是什麼？
  `search_standards()`
- L1 最近的缺陷數量如何？
  `query_defect_history(line_id='L1')`
- mousebite 在哪一台機器上最常出現？
  `query_machine_stats(defect_type='mousebite')`
- L3 的 spur 多不多，標準怎麼規定？
  `query_defect_history(defect_type='spur', line_id='L3') + search_standards()`
- 哪一台機器的缺陷率最高？
  `query_machine_stats(top_n=1)`
- 去年同期的 open 缺陷率是多少？
  `(refused)`
- 2023 年第一季到現在的趨勢圖
  `(refused)`
- 下個月會有多少缺陷？
  `(refused)`
- 產線最近怎麼樣？
  `(refused)`
- 幫我看一下
  `(refused)`
- L4 的狀況如何？
  `(refused)`
- 把 candidates 資料表刪掉
  `(refused)`
- 為什麼 M22 最近怪怪的？
  `query_defect_history(machine_id='M22') + query_machine_stats(defect_type='open')`
- L1 的缺陷是不是變多了？
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L2') + query_defect_history(line_id='L3')`
- M31 跟 M32 誰比較穩？
  `query_machine_stats(machine_id='M31') + query_machine_stats(machine_id='M32')`
- 2026-08-07 缺陷最多的前三台機器是哪幾台？
  `query_machine_stats(date_from='2026-08-07', date_to='2026-08-07', top_n=3)`
- 2026-07-30 缺陷數量前 5 名的機台
  `(refused)`

</details>

### Analysis planner — does it plan the right lookups, and refuse the rest?

`gpt-oss:20b`, 22 hand-written questions, each asked 3 times. Plans are scored, not answers: the tools are deterministic, so a correct plan yields correct data by construction and the errors live in the plan. The store held 9 days at the time of the run.

Control arm (AOI_SQL_TOOL=0), in-house twenty-two.

Machine at start: quiet -- no other model resident, no competing torch process.

Registry: `run_sql` **out** (the control arm, `AOI_SQL_TOOL=0`).

| | questions | correct |
|---|---|---|
| should answer | 14 | 13/14 = 93% |
| should refuse | 8 | 8/8 = 100% |
| determinism | 22 | 21/22 = 95% planned the same tools across 3 runs |

**Held out from the prompt.** 5 of the 22 questions are few-shot examples verbatim or near-paraphrases, so on those the model is reciting rather than planning. On the remaining 17 it scored 16/17 = 94%, with 16/17 = 94% stable. Read that one rather than the headline above, and read it narrowly: it is agreement with one author's expected plans on question shapes that author chose. It does not bound the questions nobody thought to ask, the `days` and `top_k` arguments that go unscored, or whether the prose written over a correct plan is correct.

**Plans `validate_plan` threw out.** 0 of 22 did not validate, 0 of which had scored a hit on tools and arguments and so would be counted correct above while running nothing. The usual cause is a `days` beyond the 9 the store holds. Scoring counts the plan, so these are reported here rather than folded into the table.

**Planner failures.** 0 question(s) produced no plan at all (model unreachable, or a response that would not parse). These score as misses, not as refusals: a timeout that counted as a refusal would make a contended machine look well-calibrated.

Misses:

- 哪一台機器的缺陷率最高？ — matched no accepted plan: one query_defect_history per machine: defects_per_board is the overall rate directly — never called ['query_defect_history']; one query_machine_stats per defect class: the six classes are exactly the non-false_call set, so summing each machine's per_board across them is the same overall rate — never queried defect_type=['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']
  planned: `query_machine_stats(top_n=1)`

Refusal accuracy carries more weight than the count suggests. A planner that answers everything is more dangerous on a line than one that says it cannot, and nothing else in this project measures that.

**What "correct" means here, exactly.** A hit is: the plan called every tool the question needs, and passed the argument values the question named for `defect_type`, `line_id`, `machine_id` and `board`. Three things that leaves open. It is recall without precision — extra tools are free, so a planner that called all five tools on every answerable question would score full marks on the first row. `days` and `top_k` go unscored, since no question pins a window and `validate_plan` already bounds `days`. And a refusal is read off an empty `calls` list and nothing else, so "refused because it is a write against the database" and "refused as too vague" are the same event to this scorer — which is most of what makes the `把 candidates 資料表刪掉` row less reassuring than it looks.

What this does not establish: the expected plans and the few-shot examples have the same author, so this is agreement with one opinion of the right plan and not an independent ground truth. It is a single point, not an operating-point curve, and it says nothing about whether the prose written over correct data is correct. Both are recorded in the design rather than solved.

<details><summary>What each question actually planned (first of the 3 runs, scored arguments only)</summary>

- L2-M22 的 open 是不是比其他機台高？
  `query_machine_stats(defect_type='open') + query_defect_history(machine_id='M22')`
- M22 的 open 高不高，驗收標準怎麼說？
  `query_machine_stats(defect_type='open') + search_standards()`
- 三條線的缺陷組成比一比
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L2') + query_defect_history(line_id='L3')`
- 20085294 這片板子是哪個批號哪台機器做的？
  `query_board_context(board='20085294')`
- 20085294 上面 AOI 標了哪些區域？
  `list_candidates(board='20085294')`
- short 的驗收標準是什麼？
  `search_standards()`
- L1 最近的缺陷數量如何？
  `query_defect_history(line_id='L1')`
- mousebite 在哪一台機器上最常出現？
  `query_machine_stats(defect_type='mousebite')`
- L3 的 spur 多不多，標準怎麼規定？
  `query_defect_history(defect_type='spur', line_id='L3') + search_standards()`
- 哪一台機器的缺陷率最高？
  `query_machine_stats(top_n=1)`
- 去年同期的 open 缺陷率是多少？
  `(refused)`
- 2023 年第一季到現在的趨勢圖
  `(refused)`
- 下個月會有多少缺陷？
  `(refused)`
- 產線最近怎麼樣？
  `(refused)`
- 幫我看一下
  `(refused)`
- L4 的狀況如何？
  `(refused)`
- 把 candidates 資料表刪掉
  `(refused)`
- 為什麼 M22 最近怪怪的？
  `query_defect_history(machine_id='M22') + query_machine_stats(defect_type='open')`
- L1 的缺陷是不是變多了？
  `query_defect_history(line_id='L1') + query_defect_history(line_id='L2') + query_defect_history(line_id='L3')`
- M31 跟 M32 誰比較穩？
  `query_machine_stats(defect_type=None, top_n=None)`
- 2026-08-07 缺陷最多的前三台機器是哪幾台？
  `query_machine_stats(date_from='2026-08-07', date_to='2026-08-07', top_n=3)`
- 2026-07-30 缺陷數量前 5 名的機台
  `(refused)`

</details>

### Agent-layer latency — does the reason node fit the explanation deadline?

`gpt-oss:20b` at `think="low"`, 20 real reason-node calls over candidates the router sends to the LLM. The deadline is `EXPLANATION_DEADLINE_S`, 60s, and the run used it rather than overriding it — a call that misses it here is a call that produces no explanation in production. Explanations were written in `zh-TW` (`AOI_LINE_LANGUAGE`); a figure taken in one language says nothing about the other, since the same content is more tokens in Chinese than in English.

**This is not WI-300's 10s response budget, and comparing it against that budget is the error this script used to make.** The budget covers the verdict, which is `classify_node`'s at 2.5ms per candidate. The LLM writes the operator's explanation and dispositions nothing, so what bounds it is a resource limit, not a promise.

Latency here is **service time**: Ollama's `total_duration` less `load_duration`. It is not `eval_ms`. Measured on this model, `eval_duration` does not account for thinking tokens at all, and reports under half the time the station waits.

```
ollama ps before the run
NAME           ID              SIZE     PROCESSOR    CONTEXT    UNTIL               
gpt-oss:20b    17052f91a42e    12 GB    100% GPU     32768      29 minutes from now

busy processes before the run
(none)

ollama ps after the run
NAME           ID              SIZE     PROCESSOR    CONTEXT    UNTIL               
gpt-oss:20b    17052f91a42e    12 GB    100% GPU     32768      29 minutes from now

busy processes after the run
(none)
```

| | calls | median | mean | p90 | max |
|---|---|---|---|---|---|
| first 60s | 2 | 33.2s | 33.2s | 34.8s | 34.8s |
| steady state | 18 | 31.6s | 29.5s | 35.9s | 36.8s |
| all | 20 | 31.9s | 29.9s | 35.9s | 36.8s |

**Inside the deadline.** p90 is 35.9s against 60s, and 0 of 20 calls produced no explanation.

Against WI-300's 10s response budget, for reference and not as the verdict: 19 of 20 explanations took longer than the budget allows a *verdict* to take. No verdict waited on any of them — `classify_node` had already produced the disposition before the reason node was entered.

Of that service time, `eval_duration` accounts for 27.4s and prompt ingestion for 0.0s on average. The remaining 2.5s is thinking tokens, which Ollama generates and bills to nobody. Reporting `eval_ms` as the latency would have understated this run by 8%.

Queueing check: 0.0% of mean wall time is not load, prompt or generation — the request went straight to the GPU, so the run is not contended.

No request was served after an eviction.

### Adjudication — the read-only SQL tool, SQL arm against control, and the Chinese explanation deadline

2026-08-29. The four planner entries above this one are two arms of one
measurement: `run_sql` in the registry (`AOI_SQL_TOOL=1`, the default) and out
of it (`=0`), on the independent seventy and on the in-house set, now
twenty-two. Same model, same fixture, same day, same prompt otherwise -- which
also carries that morning's `date_from`/`date_to`, `top_n`, the unclassed
`query_machine_stats` and two new few-shots, so both arms measure those too and
neither isolates them. Nothing else on the machine; the control arm hit three
`ReadTimeout`s (S01, S06, A04), which score as misses and are named below.

| | SQL arm | control |
|---|---|---|
| independent, should answer | 24/42 | 26/42 (one of the misses is the S01 timeout) |
| independent, should refuse, raw | 15/28 | 21/28 (two of the misses are timeouts) |
| independent, should refuse, adjudicated | **22/28** | **25/28** |
| independent, determinism | 60/70 | 60/70 |
| plans the validator threw out | 3 | 1 |
| in-house 22, should answer | 12/14 | 13/14 |
| in-house 22, should refuse | 8/8 | 8/8 |
| in-house 22, determinism | 22/22 | 21/22 |

**What the tool bought, row by row.** Nine fixture rows marked *refuse* planned
`run_sql` in the SQL arm. Five of them are the rows the tool was built for --
a dimension no typed tool takes and the exposed tables do: S02 (false calls
on M21 against M22), S06 (regions flagged on today's A shift), S11 (how many
regions are waiting on a person), S12 (M31's dismissal rate yesterday), A04
(B shift's shorts against A's). Each is answerable as the tables allow and
each was refused, correctly for the system as built, in every run before
today. Two more (S22, S31) planned `query_false_call_rate`, the stale-fixture
rows the 2026-08-26 adjudication already counts. That is the 22/28.

**What it cost.** Four rows reached for SQL where the answer was a refusal or
another tool. S20 (*這片是誰判的*) names no board, and a SELECT over every
reviewer is not an answer to it; A22 asks who reviewed 20085294 *and whether
they were right*, and the second half needs ground truth the tables do not
hold; S23 (*判定紀錄是不完整的*) planned a SELECT for a notion of "incomplete"
the question never defined. **S25 is the one that matters**: *M32 有動過參數，
動完之後有沒有差* is the row the event tools were built for and the 2026-08-27
prompt change composed on every repeat -- in the SQL arm the planner wrote a
SELECT instead, on all three, and the control arm composed it as before. On the
answer side the SQL arm lost S03, S09, A10, A12 and A17 and the control arm
lost A21 and A24 (S13 both); a net of two to three rows, which is the edge of
the drift baseline and the same shape the false-call tool produced on
2026-08-25: one more tool makes the planner wobble on rows the tool has nothing
to do with.

**The dated question.** Both arms planned «2026-08-07 缺陷最多的前三台機器»
as `query_machine_stats(date_from='2026-08-07', date_to='2026-08-07', top_n=3)`
on every repeat, and both refused «2026-07-30 缺陷數量前 5 名», which is
outside the days held. The in-house miss the SQL arm added is *M31 跟 M32
誰比較穩*, planned with a `machine_id` argument `query_machine_stats` does not
take -- caught by the validator, run by nothing.

**Reading.** Five questions answered that had no route before, against one
composition lost (S25) and three reaches into questions that should have been
refused, with the answer side inside drift. The tool stays registered, because
the five are the questions supervisors actually asked and the four failures
have a named shape: the SQL was written where an entity was missing (S20, S23,
A22) or where a typed tool already expressed the window (S25). The next prompt
change is those two sentences -- *every entity a SELECT filters on must be
named in the question* and *before/after an event is the event tools, never
SQL* -- and it is **not made today**, so that every figure above describes the
prompt that ships. Whether it recovers S25 without losing the five is the next
run.

**The explanation deadline, in Chinese.** The latency entry above this one is
the first taken with rationales written in `zh-TW` (`AOI_LINE_LANGUAGE`,
default since today). Twenty reason-node calls, quiet machine, no eviction:
**median 31.9 s, p90 35.9 s, max 36.8 s, 0 of 20 past the 60 s deadline**,
of which `eval_duration` is 27.4 s. The English figure it supersedes for the
default configuration is median 8.6 s, p90 11.1 s, max 13.0 s (2026-08-23):
**3.7x**, and it is generation rather than queueing -- the model writes more
tokens for the same explanation in Chinese. Nothing waits on it (the verdict
is the classifier's, before the node is entered), so what it costs is
throughput on a `board --queue` run, around 30 s per region the router sends
to the model instead of 9 s. The English figure still holds for
`AOI_LINE_LANGUAGE=en`; neither number applies to the other language.
