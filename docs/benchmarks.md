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

CLAUDE.md states twelve invariants. They are the part of this repository a
reader is asked to trust and an agent is asked not to undo, and until yesterday
nothing checked that breaking one broke anything. An outside audit found that
**five of the twelve** would have failed a test. The worst of them was the
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
| Report an operating-point curve | `BUDGETS = []` in `scripts/report.py` | **0** |
| No free-form text-to-SQL | `run_query(sql: str)` added to `PLANNABLE_TOOLS` | **0** |
| The fan-out is not a latency optimisation | the graph's docstring rewritten to call it a speed-up | **0** |
| Use the official DeepPCB split | `load_split` reads `trainval.txt` for both splits | 3, all dataset-marked |

The bold rows are the ones with nothing behind them. `0 → 5` is the one this
branch closed.

#### The count

**7 enforced, 4 partly enforced, 1 unenforceable.**

- **enforced** — breaking it fails a named test that runs in CI: the LLM off the
  decision path, the escalation direction, checkpoint durability, the
  `ground_truth` boundary, class-scoped retrieval, threshold citations, and the
  train/val split as of this branch.
- **partly enforced** — a real part of the rule is held and a named part is not:
  - *Report an operating-point curve.* The sweep, the escape budget and the
    two-stage compounding all fail when the arithmetic moves. Nothing imports
    `scripts/report.py`, so emptying its budget list -- publishing one accuracy
    and no curve -- leaves the suite green.
  - *No free-form text-to-SQL.* The validator is held hard against the registry
    that exists: unknown tool, unknown argument, out-of-domain value all refused
    before anything runs. Nothing pins the registry's own parameter surface, so
    `run_query(sql: str)` passes all of it -- `sql` is a known argument of a
    known tool, and the value has no domain to be outside of.
  - *The fan-out is not a latency optimisation.* The comparison the claim rests
    on is measured, stored per run and rendered, and `tools_wall` is held to
    cover the scheduling rather than only the slowest branch. The prohibition
    itself is on prose, and no test reads prose.
  - *Use the official DeepPCB split.* The only test that would catch a re-split
    is dataset-marked, so CI never runs it -- and it checks a count, so a
    size-preserving reshuffle of the same 1,500 boards passes.
- **unenforceable** — *Say what is simulated.* Prose discipline. No assertion
  distinguishes an honest sentence from a missing one. It is declared
  unenforceable with a reason rather than counted as passing, which is the whole
  point of having the category.

#### The gap this branch closed

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
