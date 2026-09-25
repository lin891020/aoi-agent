> Moved out of CLAUDE.md on 2026-09-25; before that date this section lived there.

## Still open

**The planner's prompt changed on 2026-08-27 and every `analysis_eval.py`
figure in docs/benchmarks.md predates it.** The first before/after question
asked at the station was refused -- "neither tool allows filtering by a time
boundary relative to an event" -- which was true of what the planner was shown
and false of the tool: the catalogue gave it one docstring line per tool, so
`relative_to` and `side` reached it as bare names, and the rule listing
dimensions no tool expresses still used "a before/after boundary" as its
example, written before the event tool existed. Three changes, each held by
`tests/test_analysis_prompts.py`: the whole docstring is in the catalogue,
the rule names the event window as expressible, and the domain note lists the
event kinds the store holds (without it, "換燈" was anchored on
`parameter_change`). A seventh few-shot shows the two-call shape. Two real
planning calls compose it now, on both the effect machine and a control.
**Re-run 2026-08-28** on a quiet machine, fixture unedited, no timeouts:
independent 28/42 answered (unchanged), 21/28 refused (from 22), adjudicated
24/28; S25 composes the two windows on every repeat and stays out of the score
as before; S32 flipped from a refusal to planning a false-call rate and is
named in the adjudication; in-house 18/20. Inside the drift baseline except
for the row the change was made to move. The eval's header had rendered the
machine state as the graph state's field names under **busy** -- a variable
reused inside `main` -- and `machine_line` now refuses a dict.


**The read-only SQL tool is registered, measured twice, and its guard has one
rule that is not yet in a measurement.** 2026-08-29. Morning: both arms on the
independent seventy, `run_sql` in and out, in docs/benchmarks.md under
"Adjudication — the read-only SQL tool"; five refused questions gained a
route, S25's event composition was lost to a SELECT, three SELECTs were
written for questions naming no entity. Afternoon: the two sentences that
address those went into the prompt and both arms were re-run -- SQL arm 28/42
answered and 23/28 refused after adjudication, control 25/42 (four timeouts)
and 25/28; S25 composes again, S20 refuses again, the five gains hold. What
the second reading found is the failure the old invariant was written for:
two SELECTs that were valid, read-only, bounded and wrong -- a state value
that does not exist, a column read as a position -- each returning a number.
The guard now refuses a bare `column = literal` that matches no row and names
the values held; read against every SELECT of a third run (prompt unchanged,
plans near-identical) it refuses S11's wrong state and A22's `boards.id =
'20085294'`, and cannot see S24's column-read-as-position, which is left to
the prompt rule and the SQL on the page. The
per-question report with every SELECT is linked from the benchmarks entry.

**The planner fans out over a listed set instead of guessing a member or
refusing, since 2026-08-30, and the question it still cannot plan is the one
that names the next mechanism.** "The worst machine", "the line with the most
opens": the machines, lines, shifts and event kinds are enumerated in the
prompt, so the rule is the ranking call plus the dependent call for every
member; two limits (a tool that ranks the set in one call is the plan; boards,
lots and reviewers are not listed sets). Measured three wordings deep, final
against the 08-29 baseline: independent 28/42 unchanged, refusals 17/28 from
15, stable 64/70 from 58, in-house 14/16 and 9/9. What it does not buy is
"最差的那台機台，最近有沒有發生什麼事？" -- the planner wants the ranking's top
row passed into the events call, writes a placeholder or refuses, and the
architecture has no such channel. That is the dependent-argument mechanism:
a typed reference resolved by code after the first branch returns, validated
like any other value. Not built; not a loop. `validate_plan` refuses
placeholders in typed arguments and inside a SELECT
(`<machine_id_from_previous_call>`, `'YOUR_BOARD_ID'`), and the guard refuses
`b.id = '這片'` at run time, so the page shows a refusal rather than an
empty answer. docs/benchmarks.md, "Adjudication — the listed-set rule".

**The operator's rationale is scored now, and what the score cannot see is
named.** 2026-09-01/02, `scripts/rationale_eval.py`, in docs/benchmarks.md
under "Is the operator's rationale true of what the model was shown?". Three
exact checks and three pattern flags over the same 60 candidates in both
languages; the flags are published with the sentence for a person to
resolve, and their counts are a floor on what a pattern can raise, not a
rate. What is open: the third flag keeps raising a sentence that is true and
reads as permission -- on `open` and `short`, whose work instructions admit
nothing, the model writes 「仍屬於可接受範圍」 about the *machine's* rate
next to criteria saying any confirmed one is critical. That is a wording
question for whoever owns the prompt, not a count, and narrowing the flag to
require the class name near the phrase is named in the entry and not made.
The 09-02 re-read found the same shape with teeth: **5 of 60 Chinese
rationales invert WI-201's disposition clause** -- "measures continuous on
electrical test: release" written as 「電測證實連續開路，則可視為 cosmetic
thinning 釋放」, which would release a confirmed open. Every word is sourced,
so all six checks pass it; it is the grounded-phrase-compared-wrongly
boundary with a count for the first time (2 of 60 before the note, 0 of 60
in English either time). The fix is a prompt wording or a fourth flag, and
neither is made or measured.
The Chinese rationales also cite a *range* (「WI-201 至 WI-206」) that sweeps
in four documents nothing retrieved; the corrected document check reads
numbers, not ranges, and passes it. And the sample is 41 of 60 `open`,
because `open` is what the router sends to investigation, so the Chinese
count rides on one class. The eval scores the prose against the prompt; it
does not ask whether the prompt was the right thing to show, and nothing
does.

Retraining from operator corrections -- now selectable by who made them, which
is what `reviewer_auth` bought -- deploying the quantised model, demo video.

**Shadow mode is refused, not forgotten.** Running the model beside the existing
process and recording what it *would* have decided is the right way to rebuild
the escape evidence on a line -- `scripts/prevalence_report.py` ends by arguing
for it, since a pilot seeing 100 defects cannot confirm a 0.5% budget and one
seeing 30 says nothing at all. The store already supports it: decisions
accumulate rather than overwrite, so a model row and a human row on one
candidate are already the pair a shadow report would read, and provenance
already says which weights and which thresholds produced the first.

What is missing is a line. Built here, `--shadow` would run against DeepPCB,
where "the operator" is the ground-truth annotation -- which is the
operating-point sweep that already exists, reached by a longer route -- and the
comparison report would read a store whose human decisions are seeded demo data
plus five real ones, four of which were deleted as unreliable. That is the shape
the detector project was killed for: a mechanism with nothing to run it on.

What would make it buildable, in order: a line, or a second dataset with a
different prevalence and its own registration problem. **HRIPCB is that
dataset, it was run on 2026-08-26, and the shipped pipeline does not survive
it -- one layer earlier than the question expected.** Ten photographed boards,
one template each, 693 images with defects drawn onto the template, and the
same 693 rotated by -10..+10 degrees. `data/hripcb.py` presents it as pairs
without touching DeepPCB or `CLASS_NAMES` (its `missing_hole` is not
`pin-hole` and has no work instruction). Three answers, all in
docs/benchmarks.md under "S0 gate on HRIPCB" and "Transfer":

- **The differencing front end does not transfer.** At the shipped grey
  threshold of 60 it flags 16.9% of defects on aligned photographs; the S0
  gate clears on no setting -- recall peaks at 92% with 1.8 false calls an
  image, and the gate's own perturbation produces 860. DeepPCB passed because
  binarisation makes a defect and a misaligned edge the same 255-level
  difference; on a photograph the defect is a 36-level one and every edge is a
  gradient. **The differencing stage's operating regime is binarised imagery**,
  and nothing in this project said so until it was measured.
- **The dismissal threshold does not transfer either, and it fails at the
  model.** With the grey threshold set where the gate found best recall, the
  stage flags 99.5% of defects and the re-verifier at `0.961` dismisses 1,387
  of 2,953 -- a 46.97% escape rate against a 0.50% budget, on a queue that is
  90% real defects. It dismisses defects at the rate it dismisses everything:
  a model trained on 255-level differences reads a faint one as a false call
  with confidence above 0.961, which is the class-escape section's finding
  again -- confident errors, which no cut on the model's own output separates.
- **The registration refusal holds, and now has a price.** Of 693 rotated
  pairs it refused 563 (525 low confidence, 38 already aligned -- 35 of them
  the dataset's own zero-angle images) and acted on 130 with a translation a
  rotation cannot be undone by. 312 candidates a board at the shipped
  threshold against 0.7 un-turned; 37% of defects never flagged.

What this does not establish: one checkpoint never trained on a photograph,
a grey threshold chosen after looking, and synthetic edits on a single
photograph per board -- a real second acquisition sits between `aligned` and
`aligned, perturbed` and is neither. What it does establish is that the next
step is not "fine-tune on HRIPCB": the layer that failed first failed for a
physical reason, and the detector front end the PCB-AoI inventory argued for
(SMT has no template to difference against) is the same conclusion from the
other side. `scripts/transfer_report.py` and `scripts/gate_check.py --dataset
hripcb` rebuild every number; `tests/test_hripcb.py` holds the adapter's
geometry and the class boundary.

**The detector front end exists, and it localises without discriminating.**
2026-08-26. YOLO26n on PCB-AoI -- real solder-paste images, no template, so a
detector is the only front end that can exist there -- trained 56 minutes on
the M5 Air, read the only way this project reads a model. It finds boxes:
91.6% of defects covered at the floor, validation mAP50 0.658 on a median
17 px target. It does not order them: at the ≤0.5% escape budget its
confidence removes **1.2%** of the queue, against 52.8% for differencing plus
the re-verifier on DeepPCB. Two lines, two prevalences, so not a ranking --
but the shape is the finding: a detector's confidence is a localisation score
with a class head attached, not a calibrated P(false_call), and this front
end therefore has **no re-verification stage**. The two front ends are not
interchangeable: one produces candidates for a re-verifier, the other
produces candidates and an uncalibrated score. `imgsz=1280` was the one
untried axis and it was tried on 2026-08-31: **validation mAP50 rose 0.658 ->
0.712 and review removed at the budget halved, 1.2% -> 0.6%**, because the
higher-resolution model is more precise (0.765) and less sensitive (0.595)
and S0 is where sensitivity is the whole job -- 36 defects covered by no box
against 28. That is the first invariant arriving as evidence: a reader with
only the validation figure would have shipped it. Nothing ships; the 640
checkpoint stays the default. `scripts/train_detector.py`
refuses to run beside a busy GPU; `scripts/detector_report.py` prints the
sixty-image basis in its first line. Held by `tests/test_detector.py` and
`tests/test_pcbaoi.py`.

**The crop re-verifier exists since 2026-08-28 and does not order either.**
`scripts/build_detector_patches.py` turns the detector's boxes into 64 px RGB
patches in `train.py`'s own schema (grouped by base stem, so the augmentation
set cannot leak across the by-image split; fragments held out; the detector's
own P(false call) kept in a sidecar), and `scripts/crop_reverifier_report.py`
reads both orderings over the identical 578 test candidates. Result: 2.8%
review removed at ≤0.5% against the detector's 0.3% on that basis, on a queue
that is 77.7% genuine defects with a 22.3% ceiling. Trained on CPU beside a
translation job holding the GPU, one seed, and on candidates from a detector
that had seen the training images -- all stated in the entry. What it
establishes is that on this data appearance alone does not separate a false
call from a defect at the budget, from either front end; the template channel
was never a convenience. Held by `tests/test_detector_patches.py`.
Re-run on 2026-09-16 over three seeds, same recipe and candidates, trained on
MPS: **0.3%, 1.2%, 2.8%, median 1.2%** -- the published 2.8% is the top of its
own range, not its level. Every seed's validation-chosen threshold also missed
the budget on test (worst 0.67%), and the standing explanation was that the
training crops are the detector's in-sample boxes and the test crops are not.
**Measured on 2026-09-20** (`scripts/detector_fold_report.py`): one detector on
half the boards, two arms of seventy boards each -- the boards it trained on
and boards it never saw -- one shared test set of 717 candidates. The
pre-registered rule returns "the in-sample boxes were the shift", and the
adjudication beside it says why that reading is mostly vacuous: two of the
out-of-sample arm's three seeds choose a threshold of 1.0 or within 0.00003 of
it, dismiss nothing, and hold the budget by declining to act, while two of
three in-sample seeds found no threshold inside the budget at all. What is left
is the oracle column, where every out-of-sample seed but one beats every
in-sample seed (median 2.0% against 0.7%) on the same candidates -- weak
evidence in the caveat's direction, on a column that is not an operating point.
The verdict this experiment needed and did not have is a floor on review
removed: a threshold that removes nothing is not an operating point.

**Stronger pretrained features were measured twice and did not help.**
2026-09-15/16, frozen DINOv2 ViT-S/14 with a linear probe, each verdict
committed before its run. On DeepPCB (`scripts/dinov2_probe.py`) it is
**worse**: median 42.47% review removed against the ResNet-18's five-seed
49.00%-55.59%, the oracle column agreeing (39.89%-42.04% against
49.73%-52.79%), at 41 ms a candidate on CPU against 2.5 ms. On the detector's
RGB crops (`scripts/dinov2_crops_report.py`), where the binarised-texture
excuse is gone, it is **indistinguishable**: paired bootstrap over the sixty
test images puts the difference at -0.2 points, 95% interval -1.7 to +1.5, and
its validation-chosen threshold escapes 12.25% on test, 24x the budget. What
neither run tests is adapting the backbone -- a linear probe is the weakest
way to use these features, and prompt tuning is the one the literature ranks
ahead of it.
**Adapting it closes the gap at the median.** 2026-09-17, `scripts/dinov2_vpt.py`: deep
visual prompt tuning, ten prompts before each of the twelve frozen blocks and a
linear head, 48,775 trainable parameters, trained by `train.fit` on DeepPCB.
Median oracle review removed **50.67%** over three seeds -- inside the
ResNet-18's five-seed 49.73%-52.79%, so the pre-registered verdict is
**matches**, and the probe's gap was the probe rather than the backbone. That
reading is the 2026-09-18 re-run's, not the first one's: the prompts and head
were drawn before `train.fit` seeded the generator, so the three rows published
that morning started from unseeded weights and spanned 41.20%-52.20%, a spread
that compared with nothing (the ResNet builds its model after the seed). With
seeded starts the same recipe reads **50.38%, 51.39%, 51.58%** -- 1.2 points
across three seeds against the ResNet's 3.1 -- so the spread was mostly the
initialisation, and the 41.20% row that made "the gap was the probe" arguable
was too. What it still does not buy is a deployable figure: the threshold
chosen on one validation split by the interval's upper bound removes
28.22%-40.90% at 0.13%-0.30% escape, a selection more conservative than the
ResNet's five-fold one, so the two deployable columns are not comparable.
Backbone hash unchanged on every seed, both runs.
**On the detector's crops the same prompts settle nothing, because the queue
cannot.** 2026-09-18, `scripts/dinov2_vpt_crops.py`: the same recipe on the
2026-09-16 candidates, against that entry's own ResNet-18 checkpoints and read
by its paired bootstrap. Oracle review removed 4.3%, 0.7%, 3.8% against the
ResNet's 0.3%, 1.2%, 2.8% -- medians 3.8% and 1.2%, and both above the linear
probe's 0.7%-1.0%, so adapting the backbone moves this ordering the way it moved
DeepPCB's. The verdict is still **indistinguishable**: the paired 95% interval
is -4.6 to +3.5 points, wider than the probe's -1.7 to +1.5 because these seeds
spread further, and sixty test images with about two escapes at the budget
resolve a large effect and nothing smaller. The seed that wins is not the seed
the ResNet wins on either. What this rules out is the reading the crops entry
left open -- that a linear probe was hiding what these features can do here --
and what it cannot rule out is anything the size of the difference it measured.

**The model is `gpt-oss:20b` because it is the one that fits the deadline, and
that is now measured.** 2026-09-17, against a rule committed first (`77ac71a`):
`qwen3:14b` could not plan inside the 60 s explanation deadline on the real
planning prompt, and `nemotron-3-nano:30b-a3b-q4_K_M` missed it on 17 of 20
Chinese calls and 20 of 20 English, with swap past 15 GB on this 32 GB machine,
so neither reached condition c or d. The same-night control answered 23/42 on
the independent seventy against 28/42 on 2026-08-30. **A third reading on
2026-09-19 answered 26/42 with nothing in the repository changed** -- same
registry, same 9-day store, `temperature` 0, the model's manifest dated
2026-08-16; that entry also said "same prompt, last touched 2026-08-30",
which was false and is corrected at the end of this paragraph -- so three
runs of one model span 23-28 and the ±3 drift baseline, taken from a single re-run, is too
narrow to call a five-question move a change. Two of the five come back on the
third run; the three that do not (A09, A23, A25) drop a named filter or pick
the wrong aggregate tool, and are not addressed. **Five more readings on
2026-09-22, run back to back, answer 25/42 every time and miss the same 28
questions identically** -- so the variance is between sessions and not between
runs, and two arms have to be measured inside one session to be comparable at
all. **What moves it is the runner, not the sequence** (2026-09-23,
`scripts/planner_session_probe.py`): inside one session, a straight repeat, a
fixed 200-question interposition and an unload-reload each move 0-1 of 70
planned calls on a batch-512 load; the server picks `n_batch` by free memory
at load time -- 1024 when 18.6 GiB is free, 512 otherwise, 136 loads without
exception -- and two independent loads forced to 1024 move **the same 22 of
70** against the 512 baseline while differing from each other on 0. Loading
without mmap moves 0. The checkpoint restores the runner logs go from 5-6 per
70 requests to 0 on the 1024 loads, which is the mechanism candidate, read
beside the plans and not manipulated. So `OllamaClient` carries
`num_batch`/`use_mmap` as `load_options` and the probe reads the load back out
of the server log and refuses a step that did not get what it asked for. Not
explained: the four historical readings were all 512 loads, so their 23-28
spread has a cause this run does not contain. **Nor is it the prompt.** The
09-19 entry's "same prompt" was wrong: `LANGUAGE_NOTE` gained the class list on
2026-09-02 (`bbe1df8`) and the planner appends that sentence, so the 28 was
read under one sentence and the 23/26/25 under another. Read on one 512 load
the same night (`scripts/language_note_arms.py`, ABAB, rule written first):
**27/42 under either sentence**, twice each, one question moved either way
(S03 gained, S17 timed out) and 12 of 70 planned differently without the
score moving -- `AOI_LANGUAGE_NOTE=2026-08-30` is the switch. That
same-sentence, same-batch 27 sits beside the previous day's five readings of
25, so what separates sessions is still not a prompt, a batch, a sequence or
a reload. **Closed 2026-09-25, cause bounded and not identified.** Read
across three sessions' per-question plans (23, 25, 27), the 13-14 of 70 that
move between sessions land 10-12 inside the 22 the forced-1024 loads moved
(p < 0.001 by chance), the load blocks of the 25 and the 27 differ in nothing
but durations, and three server processes on one day plan identically on
every 512 load; what pushes the knife-edge subset between days is below
what the log shows. The rule is what matters: two arms on one load, batch
pinned, load read back, 2 or fewer of 70 is the floor. Since 2026-09-25
`analysis_eval.py` pins `num_batch` 512 by default (`--num-batch 0` hands
the choice back to the server); every published planner reading was a 512
load, so the default is the setting they are comparable with. The
adjudication entry names what is not made.

**Every decision the store held before 2026-08-23 reads `unrecorded`** -- 9,140
of them, in `model_digest` and now in `reviewer_auth` too, stamped by the
migration rather than left `NULL`, because a decision whose provenance was
never captured must not be readable as one that has none, and because every one
of those rows has `reviewer` NULL as well -- so without the stamp "nobody
recorded who" and "nobody was involved" would be the same row.
They stay that way: the digest that produced them is not recoverable and
inventing one would be worse than the gap. Decisions written from here name the
checkpoint's SHA-256, the thresholds that routed them and the commit that ran
-- `store.boards.record_decision` raises on an automated row that does not, and
`tests/test_provenance.py` is the guard. Reseeding the store is what makes the
figure move; nothing else will.

INT8/ONNX is measured, and the answer was not a speed-up. **Both INT8 engines
hold the operating point on the shipped checkpoint** -- 52.5% (dynamic) and
53.0% (static) review removed at the ≤0.5% escape budget against FP32's 52.8%,
calibration on 512 trainval patches, never test -- and what they buy is
**memory: 389MB resident down to 74-81MB, around 5x**, because most of the
float32 process is the torch runtime rather than the weights. The report's
rule picks the survivor that saves the most disk, which is dynamic by 0.1MB: a
coin, and the two are 15 against 12 disagreements out of 7,322. **This
paragraph refused INT8 dynamic until 2026-08-26** -- on the previous
checkpoint it gave up 1.3 points, around eighty regions a shift -- and that
loss did not survive the 2026-08-24 retrain. A quantisation verdict is about
one set of weights, so `quantisation_report.py` is now in the retraining
chain, and `models/onnx/` had been the previous checkpoint's export for two
days before anyone noticed. Latency was never the problem -- 41ms of a
board's cycle before, 12ms after, on a budget of ten seconds. Nothing is
deployed on it; the station keeps the float32 checkpoint and its swept
threshold. See docs/benchmarks.md, and `scripts/quantisation_report.py`
rebuilds every artefact.

The re-verifier costs **2.5ms per candidate on CPU** (p50, single-shot) and the
43MB checkpoint fits anywhere. A re-verification station does not need a GPU --
and at a batch of one the GPU is 2.9x *slower* than the CPU, because dispatch
costs more than the forward on a model this small. MPS only overtakes at batch
8. See docs/benchmarks.md.

Two things that run counter to intuition and are easy to undo by accident:
sustained CPU inference throttles about 20% past the first 60 seconds on this
fanless chassis, and CPU per-candidate cost gets *worse* past batch 8 -- by
several-fold, independent of core count. Batch at 8 on CPU.

**The response budget and the client timeout are two numbers now, and were one
until 2026-08-23.** `RESPONSE_BUDGET_S` was WI-300's 10s promise about a verdict
*and* the httpx timeout, against a model whose measured service time had a
median of 12.5s. More than half of the station's explanations therefore failed
by construction, and after the LLM came off the decision path writing them was
its only remaining job. The queue held an escalation whose entire content was
`the model did not answer (ReadTimeout)`, and nothing counted how many others
there had been.

`RESPONSE_BUDGET_S` stays 10s, stays WI-300's, and moved to `graph/flow.py`: it
bounds the *verdict*, which is `classify_node` at 2.5ms. `EXPLANATION_DEADLINE_S`
is 60s in `llm/ollama.py` and bounds a wait nobody blocks on -- the disposition
is decided from `model_class` and `model_confidence`, both of which exist before
the reason node is entered. Re-measured at that deadline: median 8.6s, p90
11.1s, max 13.0s, **0 of 24 calls without an explanation**. That figure is
English; with the rationale written in Chinese (the default since 2026-08-29)
the same measurement read median 31.9s, p90 35.9s, max 36.8s until
2026-08-30, when the prompt was given the thresholds and asked for one
paragraph: Chinese median 16.7s / p90 19.7s and English 11.8s / 16.1s,
0 of 20 past the deadline in either. The 2026-09-17 control run read Chinese
9.0s / 10.8s and English 10.0s / 11.7s, again 0 of 20. A missing
explanation is now `explanation_status`, shown to the operator as a notice
rather than an exception name, and counted by `uv run python -m aoi_agent
explanations`. Do not re-merge the two constants;
`tests/test_response_budget.py` fails if the client timeout becomes the budget
again.

On the station itself:

- **The board browser is whole, and what it fixed was a reading of the whole
  system.** Closed 2026-08-25. `/board/<stem>` had rendered one board's record
  -- its disposition and every region's, with the weights, thresholds and code
  behind each -- since 2026-08-23, and the only route to it was a link on a
  *queued* region. So everything reachable from the front door was a region the
  agent could not settle: the 82% it did settle had no page, and a reviewer
  opening this station read the failures and took them for the system.
  `/boards` is the index, and since 2026-09-05 the front door is no longer
  the queue at all: `/` shows the six counts and three doors, the queue is
  one click in at `/queue`, and `/next` with nothing waiting lands on the
  queue rather than the door (`tests/test_home.py`). `/boards` carries
  the denominator the queue cannot --
  held and released as `COUNT(*)` over the standing rows, never the length of
  the page, which is the queue badge's own defect written down. A third count
  sits beside those two and not inside them, since 2026-08-26: **waiting**,
  the boards that have been run and have a region on either open queue, so no
  row yet. Fifty boards run showed twenty-nine on the index and nothing said
  where the rest were; "someone is still looking" is a state, and an index
  without it has no denominator. It reads `OPEN_STATUSES`, so a deferred
  region keeps its board waiting -- the same rule `assess` applies.
  **"This board's disposition" is a rule, not a column** -- rows accumulate, so
  a board held on Monday and released on Tuesday is two rows -- and the rule
  lives once, in `dispositions._standing_ids`, because two expressions of it
  would both return a real row and disagree only on the boards dispositioned
  twice, which are the boards an auditor asks about.
  `tests/test_board_index.py` holds `recent()` against `latest()` rather than
  the two being merged. An unknown `?status=` is refused rather than ignored:
  a filter that silently matches everything answers a typed URL with a
  plausible page. No `ground_truth` here either, at the same dict boundary.
- **A region re-run on a fresh path carries nothing from its last run.**
  2026-09-13, found by running the demo's CLI step before a presentation. The
  checkpointer is keyed by region, and `start_review` reset only the two
  channels that append (`trace`, `timings_ms`); the reason node's channels
  replace, so a run that never entered the reason node handed back whatever
  the last run wrote. The demo board's region 29 -- escalated at 0.961,
  dismissed by the classifier at 0.912 -- had 31 `model` rows since 09-05,
  each carrying the August rationale that quotes the 0.961 threshold, with
  `explanation_status` `ok`, for a decision no model was asked to explain.
  The CLI printed it under `classify -> dismiss`. Every reason-node channel is
  reset on invoke now, and a `model` row stores `None` rather than `""` for
  the status; `tests/test_rerun_state.py`. The 31 rows are a data fix on
  the local store (`source='model' and explanation_status='ok'` is exactly
  that set, since the code never writes a status on a `model` row).
- **A region someone handed back is not re-run either.** 2026-09-17.
  `start_review` skipped a region waiting on a person only when its status was
  `pending`, so a board re-run re-invoked the graph on a `deferred` one. The
  board was never released -- `assess` reads every open status first -- which
  is why nothing at the board level saw it; the damage was to the record the
  senior answers from. A re-escalation rewrote the reason the declines were
  about; a run that no longer escalated finished the thread and took the
  interrupt with it, so a later senior answer was recorded against the
  re-run's state; and the CLI without `--queue` then prompted at the terminal
  and answered the region as the host account, past the senior-only rule that
  lives in the station's route. The CLI prints `HANDED BACK` with the decline
  count now; `tests/test_deferred_rerun.py`, whose third test answered the
  region at the terminal before the fix. Seen and not changed:
  `pending_unexplained_count` and `explanation_counts` read `pending` only, so
  the CLI's "waiting on a person" explanation count leaves out handed-back
  regions.
- **The criteria index opens under a lock, because `/ask` opens it from six
  threads at once.** 2026-09-13, rehearsing the station's own example question
  "比較三條線的缺陷組成，並說明驗收規定" on a freshly started station: the plan fans
  out six `search_standards` branches, `lru_cache` does not serialise a miss,
  and each thread built its own `PersistentClient` while Chroma's shared system
  for the path was half-built -- "Could not connect to tenant default_tenant",
  `KeyError: 'data/chroma'`. All six failed and the answer said the criteria
  were unavailable. Eight concurrent first searches in a fresh interpreter
  failed 8/8, 3/8 and 8/8 before `standards._OPENING`, and 8/8 succeeded on five
  runs after it. Only the open is locked, never the query. Held by
  `test_concurrent_first_searches_all_answer`, which runs
  `tests/standards_race_in_another_process.py` because a process opens the
  index once.
- **Timestamps are stored UTC and displayed UTC**, and labelled `UTC` on the
  board record, the CLI, the corrections page, and -- closed 2026-08-25 -- the
  queue, which until then showed no clock at all: the ordering rule ("whoever
  waited longest goes next") ran on a timestamp the page never displayed. The
  remaining honest gap is that display is UTC rather than local; "store UTC,
  render local, say which" is still the right end state, and every page saying
  `UTC` explicitly is what makes that a rendering change rather than a data
  migration.
- **Authentication is done, and what it deliberately leaves out is not.**
  Closed on 2026-08-23. Both pages are behind a sign-in, `record_decision`
  refuses a `human` row that names nobody, and the station reads the name off
  the session rather than off the form. This was the item blocking the station
  from running anywhere but a laptop, and the thing it was really blocking was
  retraining: the five regions clicked through without domain knowledge, four
  of them wrong, were indistinguishable from an expert's labels and had to be
  deleted by hand -- and the five queue entries left closed behind them still
  read `resolved_unattributed`, which is the true statement about them.
  What is *not* here, on purpose, and should not be added without a reason
  that is written down: TLS termination, and a rate limit or lockout on
  `/login`. The scheme's limits are stated in `station/auth.py`; a shared
  passphrase still names one operator for two people, and nothing here can tell
  them apart.
- **Two roles, and the second one exists to answer one question.** Added
  2026-08-25 with the deferral path, because that path created the first
  permission this station actually needed: a region reaches `deferred` because
  a trained person could not read it, so handing it to the next ordinary
  operator hands it to the same judgement that already failed, and what comes
  out is a guess recorded as a training label. `senior` may answer a
  handed-back region; `operator` may not; **nothing else is gated**, because on
  a line every trained operator answers every ordinary region and a permission
  grid over defect classes would encode a policy no work instruction states.
  The role lives in a third field of the credential file, an old two-field line
  still parses, and silence about a role reads as `operator` -- reading it as
  `senior` would grant a permission nobody decided to give, on the file whose
  whole purpose is making that decidable. **Read from the file on every
  request, never off the session**: a role frozen at sign-in outlives the file
  that granted it, so revoking one would take effect whenever the operator next
  happened to log out, which is not a revocation. A role the vocabulary does
  not contain is tolerated on read (a typo must not lock out a station) and
  refused on write, and `add_operator.py --list` reports both any unknown role
  and the state where **no senior exists at all** -- in which the handed-back
  queue grows with nobody able to empty it and nothing else anywhere raises.
  Held by `tests/test_roles.py`.
- **An operator who cannot judge a region now has a button for it.** Closed on
  2026-08-25. The doctrine has always been that an uncertain region goes to a
  person; what the screen offered that person was seven certain answers, so
  "I don't know" could only be expressed by navigating away -- and a region
  navigated away from is indistinguishable from one nobody has reached, so the
  next operator met it and declined it again, and nothing counted how often.
  That absence has a price already paid: five regions clicked through without
  the domain knowledge to judge them, four of the five wrong, all five deleted
  by hand. A deferral is **not** a `ReviewDecision` -- that table is the next
  training round's labels and "unsure" is the one label that must never be one
  -- so it is its own table, its own route, and its own queue state
  (`deferred`). It does not resume the graph either: the interrupt is what
  keeps the region answerable, and consuming it would close a run nobody
  answered. Declines accumulate and the count is the ranking, because a region
  three people could not judge is a different object from one somebody skipped.
  Held by `tests/test_deferral.py`. **What it deliberately does not do is route
  anything to anyone** -- this station has no notion of who is more senior, so
  `/deferred` is a list and says on its face that it is not an assignment.
  Giving it teeth needs roles, which is the item below -- and once roles
  shipped, that standing note went false where it stood: it still read "the
  station has no notion of senior and ordinary, every operator can answer every
  region", one paragraph above the line telling an operator they may not answer
  these. Two adjacent paragraphs contradicting each other, and the test passed
  because it asserted only the half that stayed true. Fixed 2026-08-25, and the
  rule is structural rather than a corrected sentence: **the standing note may
  not name a role at all**, because anything it says about who may answer is a
  second source of truth beside the credential file.

  **The defect it shipped with, because it is the shape to watch for.**
  `dispositions.OPEN_STATUSES` listed only `pending`, so a deferral took the
  escalation out of the open set while the model's row stayed the standing
  decision, and a board went from **held to released** -- an operator refusing
  to guess was the act that shipped it. Nothing at the region level was wrong,
  which is why no region-level test saw it. A new queue state has to be checked
  against every rule that reads queue state, not only the ones that write it.
- **Three classes are judged by a ratio, and the ruler now exists.** WI-203's
  own "escalate for measurement" pointed at the operator, who is the last stop,
  so it had nowhere to go. `station/static/measure.js` measures a *ratio* --
  DeepPCB carries no mm-per-pixel and a ruler reporting millimetres would be
  inventing the only number that mattered, while all three criteria are already
  written as ratios and so need no calibration. It refuses what it cannot mean:
  a segment across the panel gap, and two segments taken on different panels
  (arithmetically fine, and a ratio of the template to the board under test).
  The reading is stored on the decision -- a measurement nobody records is a
  measurement that never happened -- and `NULL` there means nobody measured,
  which is a true statement about the decision rather than a missing field.
- **The criteria answer the wrong question for the operator.** For `open` the
  retrieved passage says any confirmed open is critical -- how to *disposition*
  one. It never says how to *confirm* one, which is what the person looking at
  the images actually needs. This is what is left of the item after the larger
  half of it turned out to be a defect rather than a gap: the retrieval was
  unscoped, WI-206's "inside a pad: reject" outranked WI-201 on the flow's own
  `open` query, and every escalation in the store told an operator to
  disposition an open by whether it sat inside a pad. 27.8% of retrieved
  passages came from the wrong class's document. Fixed at the retrieval
  boundary on 2026-08-23 -- see docs/benchmarks.md and
  `tests/test_standards_retrieval.py`. What remains is a documents problem:
  WI-201's re-verification notes are the closest thing to confirmation
  guidance, and they are advice about ambiguity, not a procedure.
