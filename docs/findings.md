<!-- Moved out of README.md on 2026-08-28: the README keeps the one-line summary
     of each finding and links here for the full account. -->

# What measurement changed here

The interesting part of this repository is not the pipeline. It is the five
things that were measured and turned out to be wrong — three of them in the
project's own favour — and what the measurement did to the code afterwards.

### The LLM was on the decision path. It was measured, and taken off it.

The original design had a local LLM read the evidence and produce the verdict.
Measured over 60 candidates the router sends to investigation, it scored
**43/60 = 71.7%** against the classifier's **51/60 = 85.0%** on the same
regions. It changed the class on 12 of them and was right **once**; the
classifier had already been right on 9, and 2 were wrong either way.

Its `confident` flag — the thing that chose who needed a person — also lost, to
a plain threshold on the number the classifier had already produced. So both
jobs went back to the model that is better at them: `route_after_reason` routes
on `ESCALATE_BELOW`, `decide_node` takes `model_class`, and the LLM writes the
rationale the operator reads. It explains; it no longer decides who reads it.
The re-run after the change is in the same file: **27/30 = 90.0%** for the
system as it now stands, against **22/30 = 73.3%** for the counterfactual it
replaced.

The measurement is pinned in the code that acts on it — `decide_node`'s
docstring carries the 12-and-1, and
`test_the_classifier_class_stands_when_the_llm_disagrees` fails if the verdict
comes off the LLM again.

What it does not establish: 60 candidates, one model (`gpt-oss:20b` at
`think="low"`), one prompt. It is a result about this classifier and this
prompt, not about language models.
→ [the run](benchmarks.md#agent-layer--does-it-beat-the-classifier-and-is-the-escalation-calibrated)

### The planner was graded on questions its author never saw.

A second entrance, `/ask`, turns a supervisor's question into a typed plan of
tool calls or refuses it. Graded on 20 questions written by the person who wrote
the prompt, it scored 100% — which says more about the question set than about
the planner, and the section says so.

So seventy more were written by three authors blind to the prompt, the few-shot
examples and the fixture: thirty-five from an author told nothing about the
tools and asked to write what a shift supervisor would type, thirty-five from an
author given only the tool signatures and asked to probe the boundary, and the
expected plans from a third who read the tools and the store but not the prompt.

**55/70 = 79%**, and the shape of the failure is the point:

| | questions | correct |
|---|---|---|
| should answer | 42 | **27/42 = 64%** |
| should refuse | 28 | **28/28 = 100%** |

Seven of the fifteen misses are refusals of questions it should have answered.
It is **timid, not reckless**, which is the survivable direction on a line: a
planner that answers everything is more dangerous than one that says it cannot.
Graded by severity, the questions whose right answer the grader called
unarguable score 42/51 = 82%; the ones where graders could reasonably disagree
score 13/18 = 72%. Seven of the seventy cannot be passed by any plan at all —
the grader pinned an argument onto a tool that does not take it — and they are
counted as misses anyway; over the remaining 63 the score is 55/63 = 87%.

The set also found two real defects that the score does not contain:

- `query_machine_stats` defaults to `days=14` and the store holds 9. A plan that
  omits the argument runs, returns the whole 9-day span, and labels it `14`.
- **No tool returns a false-call rate at any aggregate level**, in a system
  whose entire subject is false calls. Six of the supervisor's thirty-five
  questions ask for one, per machine or per shift or per line. Refusing them is
  correct for the system as built and is the right answer to the wrong question.
  Neither has been papered over, because adding a tool to pass a set is how a
  measurement stops measuring.

  **2026-08-25: the tool was then built, and the fixture deliberately was not
  touched** -- its value is that nobody who saw the prompt wrote it, and the
  person adding the tool rewriting its answer key would end that. Re-run
  against the fixture as written: 47/70, adjudicated 50/70, against 55/70
  before. Three of the seven new "wrongly answered" rows are the fixture being
  stale about `query_false_call_rate`; **four are a real finding -- adding one
  tool made the planner bolder on questions the tool cannot answer**, including
  a disposition request it had previously refused. The adjudication table and
  both readings are in benchmarks; the rate the tool reports is labelled in
  its own payload as the re-verifier's judgement, never ground truth.

- **A second tool was added the same way, and half-answered the question it
  was built for.** The independent set asked twice about change over time on
  one machine; the store had no time axis. `machine_events` and a
  `relative_to`/`side` pair on `query_defect_history` give it one, seeded as
  one effect and three controls so the tool can be wrong. Re-run against the
  unedited fixture on a quiet machine: 28/42 answered (from 26), 22/28
  refused (from 23), and the target question planned the event lookup and the
  machine's history **without composing the two windows** -- reachable, not
  yet composed. The design predicted the other time-axis question would stay
  unanswered, and it did. The run before it, taken beside a hung job from
  another session, timed out on 34 of 70 and is kept in `docs/benchmarks.md`
  as what it was; the script now refuses to publish such a run.

- **2026-08-27: the first real before/after question asked at the station was
  refused**, with the reason "neither tool allows filtering by a time boundary
  relative to an event" — true of what the planner was shown, false of the
  tool. The catalogue gave it one docstring line per tool, so `relative_to`
  and `side` reached it as bare names; the rule listing dimensions no tool
  expresses still used "a before/after boundary" as its example, written
  before the event tool existed; and nothing listed the event kinds, so
  "換燈" was anchored on `parameter_change`. All three are fixed and held by
  tests, and two real planning calls now compose the two-window shape on the
  effect machine and on a control. **Re-run on 2026-08-28 on a quiet
  machine, fixture unedited, no timeouts:** independent set 28/42 answered
  (unchanged), 21/28 refused (from 22), 66/70 stable; adjudicated refusals
  24/28 against 25/28 before, with the three false-call-rate rows still the
  fixture's staleness. S25 -- the question the event tool exists for -- now
  plans the two-window shape on every repeat, and is kept out of the score
  for the same reason as before. One row went the other way and is named:
  S32 was refused and now plans a false-call rate for a question about
  confidence over time. Everything else is inside the three-row drift
  baseline. The in-house twenty read 18/20 (11/13 answered, 7/7 refused).
- **2026-08-29, the read-only SQL tool, two arms.** With `run_sql` registered:
  independent 24/42 answered, adjudicated refusals 22/28; without it (the
  control, `AOI_SQL_TOOL=0`): 26/42 and 25/28. Five refused questions gained a
  route (a shift on one machine, today's flagged regions, the waiting count);
  S25's event composition was lost to a SELECT and three SELECTs were written
  for questions naming no entity. In-house, now twenty-two rows, 20/22 against
  21/22, with the new dated question planned correctly in both arms. The tool
  stays; the two prompt sentences that address the failures are named in the
  adjudication and not yet made.
- **2026-08-29, afternoon, re-run under the two sentences.** SQL arm 28/42
  answered, 23/28 refused after adjudication; control 25/42 (four timeouts)
  and 25/28. S25 composes on the event tools again, S20 refuses again, the
  five gains hold. Two SELECTs were valid and wrong -- a state value that does
  not exist, a column read as a position -- and each returned a number; the
  guard now refuses an equality on a value no row holds, unmeasured as yet.

What it does not establish: the questions were written by LLM authors working
from different briefs, not by a real shift supervisor, so this bounds the shapes
those briefs produce and nothing else. Plans are scored, not prose — the
sentence written over correct data is the section below.
→ [the independent run](benchmarks.md#analysis-planner-asked-by-someone-else--does-it-plan-the-right-lookups-and-refuse-the-rest)

### And then the prose over those plans was checked against the payload.

*"How do I know when it is making a figure up?"* The tools are deterministic, so for a
figure that question is arithmetic: it either renders from a value in the
payload or it does not, and the entity it is attached to either holds it or does
not. Five failure kinds, reported apart, because a fabricated number and a
missing hedge are not one accuracy:

| kind | findings | decided by |
|---|---|---|
| a figure that is in no payload | **0** | comparison |
| a figure attached to the wrong machine, line or class | **0** | comparison |
| a cause, or a movement over time | 3 | a person, on a flag |
| a failed tool the prose does not admit to | 0 | a person, on a flag |
| a rule about a class nothing retrieved | 1 | a person, on a flag |

**Across 602 figures in 265 sentences over 34 answers, nothing was fabricated
and nothing was misattributed.** The four flags were adjudicated one by one and
three of them are the pattern rather than the model — `leads to` inside a quoted
disposition rule, `because the` inside a correct hedge about an empty tool, and
`limit` inside `limited to copper`, which was a bug and is fixed. On the one
question that asked point-blank for a cause — *是因為蝕刻液老化了嗎* — the model
declined to give one.

**The first pass raised 43 findings and 41 of them were the checker's fault**,
which is the more useful half of this. `M12` was read as the number 12; a
Chinese answer never split into sentences because `。` carries no following
space; `19 copper, 22 mousebite` was read backwards. Every correction makes a
checker quieter, which is how one goes blind, so each has a test on both sides —
the shape it now passes, and the same shape with one value swapped that it must
still catch.

What it does not establish: it is one sampled pass, the checker shares an author
with the system, and a sentence that quotes every figure correctly and reads
them wrongly — *"M22 is the worst machine"* where it is second — is outside every
kind here. A deliberately corrupted summary is the control that stops the
taxonomy being one nothing can fail.
→ [the run](benchmarks.md#the-prose-over-the-results--is-the-sentence-true-of-the-payload)

### The criteria retrieval was answering about one class out of another's rules.

Found by reading the queue, not by a failing test. All five escalations in the
store told the operator that an open must be judged by whether it sits inside a
pad. No document says that: it is WI-206's rule and it governs pin holes. WI-201
says any confirmed open is critical, because continuity is binary.

Measured over six classes × six real phrasings at `top_k=2`, **27.8% of
retrieved passages came from the wrong class's work instruction** — worst for
`short` at 67%, and on the disposition path's own `open` query the pin-hole
disposition section ranked *first*, ahead of WI-201's own. The fix is at the
retrieval boundary rather than in the prompt: every document declares the class
it governs, the declaration rides on every passage, and a caller with a class in
hand gets that class's document plus the two that govern every class. Now
**0.0%**, held by tests over the real documents.

The project's standing defence — the LLM only explains, the classifier decides —
does not cover this. The fabricated rule went to the people who *do* decide, and
it pointed at releasing a critical defect. The eight explanations already written
under the old retrieval are marked in place rather than deleted or regenerated.

What it does not establish: this counts which document a passage came from, not
whether the passage answers the question the operator has. For `open` it still
returns how to *disposition* one, when the person looking at the images needs to
know how to *confirm* one. That is a documents problem now, and it is open.
→ [the contamination table](benchmarks.md#cross-class-contamination-in-the-criteria-retrieval)

### Two thresholds cited sources that did not say what they claimed.

`ESCALATE_BELOW` was cited as "the lowest threshold adding no escape to the
budget" and no sweep of it had ever been run. `CONFIDENT` was cited to a clause
of WI-300 that names no number. The sweep was then written, and it found the
first claim false in both directions: the lowest zero-escape threshold on this
split is **0.875**, not the 0.90 the code carried, and 0.90 had not come from
any sweep — it was a round number that happened to be conservative.

Neither number shipped. 0.875 clears the highest-confidence real defect that
branch would dismiss by **0.003**, which is a test split read to three decimal
places. The value that needs no split is the dismissal threshold itself:
`ESCALATE_BELOW` is now *equal* to it, which empties by construction the only
band in which the agent branch could dismiss a real defect. **The agent branch
may confirm a defect; it may never dismiss one** — and that survives a retrain,
where a swept number would have to be swept again and silently would not be. It
cost 289 more escalations out of 7,322 candidates, 3.9% of the queue --
and 229 of those 289 were agent *dismissals*, of which there are now none.

`CONFIDENT` turned out not to be a quality gate at all. `confirm_node` and
`decide_node` write the same verdict, so anywhere at or above `ESCALATE_BELOW`
it changes **zero** dispositions and adds zero escapes; swept to 0.999 it still
changes zero. It decides who gets an LLM call and a written rationale, not what
happens to a board. What it must not do is fall *below* `ESCALATE_BELOW`, where
it starts confirming, unreviewed, regions the flow would have handed to a
person. The constraint is the citation; the value inside it is a dial.

Every threshold now cites a script a reader can run or a line in a document that
states the number, and **29 tests fail** if a value drifts from its source, if a
source stops resolving, or if a threshold reaches the code with no row in the
table.
→ [the sweep](benchmarks.md#threshold-sweep--escalate_below-and-confident-2026-08-23--commit-68e90b6)
· [the table](architecture.md#thresholds-and-where-they-come-from)

### The whole-line escape rate was overstated by nearly an order of magnitude.

It read **5.4%**. It is **0.61%**. The old figure added a 5.0% "AOI stage miss
rate" to the re-verifier's own, under the sentence "already gone and no
threshold recovers them" — which was true of 7 defects and was being applied to
157. The 5.0% never counted defects the detector failed to find; it counted
defects whose best candidate did not clear DeepPCB's IoU 0.33 cut, and 150 of
those 157 have a candidate sitting on them. A statistic about how tightly this
detector draws a box was being published as a detection failure.

Recounted on defects rather than boxes, it is two numbers and not one:
**0.22%** never flagged at all (7 of 3,140 — unrecoverable) and **0.38%**
flagged and then dismissed by the re-verifier (the number the dismissal
threshold governs). Adding them into one headline is what produced the 5.4%,
and it told a reader to go tune the thing that cannot move.

This one was wrong in the project's favour: it charged 150 defects to a stage
that had not failed, while removing them from the only measurement that could
contradict it.
→ [the accounting](benchmarks.md#whole-line-escape-rate-recounted-on-defects-instead-of-boxes)

