> Moved out of CLAUDE.md on 2026-09-25; before that date this section lived there.

## Invariants — do not quietly change these

Twenty of them, and `scripts/invariant_audit.py` says which ones would
actually fail a test if broken: **17 enforced, 2 partly enforced, 1
unenforceable**. Each
entry there names the tests that hold it and states what those tests do not
cover; adding an invariant here without an entry fails
`tests/test_invariant_audit.py`. The two that are only partly guarded are
named in docs/benchmarks.md rather than left for a reader to discover.

Three of these are scoped to the **disposition path** -- the flow that ends in a
board being dismissed, confirmed or held. That scope is not a loophole, it is
what the invariants are about: they all buy the same thing, which is that a
wrong call cannot ship a board or corrupt the next training round. The `/ask`
flow dispositions nothing. Nothing on that page holds or releases a part, no
label it produces reaches `review_decisions`, and its worst failure is a
supervisor reading a wrong sentence with the raw figures printed beside it. So
it has no checkpointer and no escalation queue, deliberately -- see
docs/architecture.md, "The analysis path". Anything that ever does disposition a
board is back under the unscoped reading.

- **Report an operating-point curve, never bare accuracy.** An escape ships a
  bad board; a false call costs seconds. Headline is "review removed at an
  escape budget" -- and since 2026-08-24 that headline carries an interval.
  Since 2026-08-31 it also carries a threshold that was not chosen on the split
  it is read from: **55.6% review removed at 0.66% escape** (20 of 3,018
  defect-labelled candidates, 95% interval to 1.02%), which **exceeds QP-110's
  0.5%**. Counting defects rather than candidates, the reading QP-110 is
  written in, the re-verifier escapes 0.35% and the whole line 0.51%. It read
  **52.8% at 0.50%** until that date, and the interval already said the budget
  was not established; what the interval could not say is that one split was
  doing two jobs -- choosing the operating point and reporting it. And since
  2026-08-31 the headline also names the seed, because every figure in this
  file is one: over five re-runs of the whole procedure the median is **50.9%
  at 0.50%**, the range 49.0%-55.6%, and the shipped checkpoint is the best of
  the five on both axes. Half that spread is the model and half is where the
  selection rule landed on the same curve -- each seed read at its own oracle
  spans 3.1 points against the deployed column's 6.6. Two intervals, and they
  are not the same interval: Wilson is sampling error on a fixed model, the
  seed spread is what a re-run lands on, and quoting only the first reads as
  though re-running would return the same number.
  `scripts/prevalence_report.py`, `scripts/seed_variance.py`.
- **The dismissal threshold is never chosen on the split it is reported
  against.** It is chosen out-of-fold over trainval -- `scripts/threshold_cv.py`,
  five folds by image, 6,569 defects behind the choice -- and by the *upper
  bound* of the interval rather than the point estimate, because a budget is a
  promise about defects nobody has seen. Both halves were measured before they
  were adopted: on the single validation split one defect is 0.10%, and the
  point-estimate rule there picks 0.610, which escapes 0.93% on test. What the
  honest choice costs is stated rather than hidden, and what it exposed is
  larger than the threshold: **the selection set does not predict the
  deployment set.** The procedure estimated 0.32% out-of-fold and this split
  measures 0.66%, about twice, at every threshold tried, with the same class
  mix on both sides and the excess in `open` and `short`. So no threshold on
  this model meets the budget honestly, and the thing that does is the same
  thing the class-escape section already names -- a second measurement, and
  the arithmetic is in the escape-accounting entry: with electrical test
  covering the 7 escaped opens the remaining 13 of 3,018 is 0.43%, inside
  QP-110. Held by `tests/test_threshold_selection.py` and
  `tests/test_threshold_citations.py`.
- **A published figure names the run it came from.** `docs/benchmarks.md` is
  append-only and newest-last, so the file already says which measurement is
  current; what nothing checked until 2026-08-26 was whether the documents
  quoting it had kept up. They had not. `README.md` opened with **56.2% review
  removed over 8,143 candidates** -- a table produced on 2026-08-22 by a tree
  whose own benchmarks header reads `commit uncommitted`, so unreproducible even
  in principle -- while the shipped checkpoint measured **52.8% over 7,322** at
  the threshold of that day, and **55.6% at 0.912** since 2026-08-31.
  The code was right the whole time: registration was turned on, the population
  fell, the model was retrained and the threshold re-swept from 0.915 to 0.961,
  exactly as `retraining-the-reverifier` requires. Only the prose was left
  behind, in the one place a reader looks first. A superseded figure may still
  appear -- "this read 56.2% until 2026-08-24" is the sentence a reader needs --
  but **only in a paragraph that dates it**, which is the rule
  `tests/test_published_figures.py` holds over README.md, README.zh-TW.md,
  CLAUDE.md and docs/architecture.md.
- **The LLM explains; it does not decide.** Measured, its verdict was worse
  than the classifier's (12 overrides, 1 right) and its `confident` flag was
  worse at selecting who needs a person than a plain threshold on the
  classifier's own number. `route_after_reason` routes on `ESCALATE_BELOW`,
  `decide_node` takes `model_class`, and what the LLM writes is what the
  operator reads. Do not put it back on the decision path without a measurement
  that says to. On the analysis path the same rule holds in its own terms: the
  planner chooses which *lookups* to make, every one of them is validated
  against the real signatures and value domains before it runs, and the chart is
  derived from the result shape rather than chosen by the model. What the LLM
  contributes at either end is language.
- **On the disposition path, every failure escalates to a human, except the
  LLM's own.** `open` below the threshold, anything under `ESCALATE_BELOW` --
  all route to a person. The LLM's two failures no longer do, because the
  decision no longer depends on it: an unreachable model and a response that
  will not parse both fall back to the classifier's class and confidence and
  route on those, so the operator loses an explanation, not a verdict, and the
  queue does not fill with every candidate on the line. This bullet said
  "unparseable verdict" escalates until 2026-08-23, which had been false since
  `route_after_reason` stopped reading the LLM -- above `ESCALATE_BELOW` an
  unparseable response is dispositioned like any other. What the failure costs
  is the explanation, and that absence is now a first-class state --
  `explanation_status`, one of `timed_out`, `unreachable` or `unparsed` -- shown
  to the operator as a notice and counted by
  `uv run python -m aoi_agent explanations`. It used to be an error string in
  the rationale field, which read like something the model had concluded and
  which nothing counted. Pinned by
  `test_an_unparseable_verdict_in_the_explanation_band_is_decided_anyway` and
  `tests/test_explanation_status.py`. The analysis flow's failures terminate in a message on the page
  instead -- an unplanned question, a rejected plan, a branch whose tool raised.
  There is no disposition waiting on any of them and nothing for a person to
  answer, so a queue entry would be a task nobody can close.
- **On the disposition path, an escalation must outlive the process.** The
  checkpointer is a SQLite file, not `InMemorySaver`. An escalation that dies
  with the CLI run is a prompt wearing a graph's clothes. The analysis flow has
  no checkpointer because nothing in it suspends -- adopting the feature anyway
  is what this project spent a day removing from the other graph. What that
  flow needs instead is reproducibility, and the `analysis_runs` table serves
  it: a chart is redrawn from stored data rather than by re-running a plan the
  model may not reproduce.
- **The review station never shows `ground_truth`.** The operator's answer is
  the next training round's label; showing them the answer key first collects an
  echo, not a judgement. Enforced at the dict boundary, not by grepping HTML --
  `store.boards.resolve_candidate` for the queue, and on the analysis page both
  routes out of a tool's payload: `result_view.readable_rows` for the table and
  `result_view.strip_hidden` for what the synthesis prompt is shown. A boundary
  with a second door is not a boundary.
- **Free-form SQL reaches the store only through the read-only guard, and
  whether it stays is a measurement.** Until 2026-08-29 this read *No
  free-form text-to-SQL. Typed parameters over a fixed query set.* The typed
  tools are still the rule, for the reason that bullet gave: a valid but
  semantically wrong query returns a plausible number and gets acted on, which
  is why `/ask` validates a plan's argument *values* against the store's real
  domains and refuses rather than retrying -- `line_id="L4"` raises nothing
  and returns nothing, and the missing series reads as a finding. What changed
  is what the independent seventy showed: six supervisor questions sat on
  dimensions no typed tool combines -- a shift on one machine, a lot's
  machines, a count of flagged regions -- and every one was refused. `run_sql`
  takes one SELECT for those, and the model's text is handed to
  `analysis/sql_guard.guarded_select` and to nothing else: the registry's
  `sql` account reads the tool's body for that, the way the earlier gate read
  it for `text()`. The guard is structural, not a prompt asking the model to
  be careful: the SQL runs against an in-memory copy of the store holding only
  listed columns, so `ground_truth` is not filtered on the way out but absent
  on the way in; the connection is `query_only`; one statement, parsed, tables
  allowlisted, filesystem functions refused; 200 rows and two seconds; the SQL
  as run is stored on the run and printed beside its rows. What no guard holds
  is *meaning*, so the planner is told it is the last resort, and
  `AOI_SQL_TOOL=0` is the control arm of the eval that decides whether it
  stays. Measured the same day, both arms on the independent seventy:
  adjudicated refusals **22/28 with the tool against 25/28 without**, answered
  24/42 against 26/42 -- five questions answered that had no route before
  (S02, S06, S11, S12, A04), one composition lost (S25 wrote a SELECT where
  the event tools were the answer) and three reaches into questions naming no
  entity (S20, S23, A22). It stays registered; the next prompt change is named
  in the adjudication entry and is not yet made or measured. Held by
  `tests/test_sql_guard.py` and `tests/test_analysis_registry.py`.
- **The fan-out is the shape of the work, not a latency optimisation.** The plan
  expands into `Send` branches because the facts are independent. The tools
  cost milliseconds either side of two model calls costing around 25 seconds,
  so the saving is noise. Every run records `tools_wall` against
  `tools_longest_branch` in `analysis_runs`, which is where that comparison
  lives -- do not quote a figure that is not in `docs/benchmarks.md`. Nothing
  in the code, the docs or the page may present the fan-out as a speed-up.
- **Criteria retrieved about a class come from that class's document.** Every
  standards document declares the class it governs and `search_standards` takes
  a `defect_class` scope; the disposition path always passes the class it is
  reasoning about. Unscoped, WI-206's "inside a pad: reject" outranked WI-201
  on the flow's own `open` query and the model handed operators an acceptance
  limit for opens that no document contains -- 27.8% of retrieved passages came
  from the wrong class's document. A prompt telling the model to ignore
  irrelevant passages is not the fix; the passage must not reach the prompt.
  Leaving the scope unset is for questions that genuinely span classes, which
  is `/ask`'s case and never the queue's.
- **Thresholds come from the sweep or the work instructions**, not from hand
  tuning against the test set, and every one of them cites a file a reader can
  open. Two did not until 2026-08-23: `CONFIDENT` was cited to a clause of
  WI-300 that names no number, and `ESCALATE_BELOW` to a sweep nobody had run.
  `scripts/threshold_sweep.py` is that sweep and
  `tests/test_threshold_citations.py` is what stops the citations rotting
  again -- it fails on a value that drifts from its source, and on a threshold
  that reaches the code with no row in the table at all. See
  docs/architecture.md.
- **Registration recovers a translation and nothing else, and it must be able
  to refuse.** There was no registration stage at all until 2026-08-24 -- the
  detector inherited pre-registered pairs from DeepPCB, and misalignment cost
  4.6 points of AOI recall (95.0% -> 90.4% at 4 px), which is a loss *before*
  the escape budget where no threshold reaches it. `aoi/registration.py` is
  phase correlation and it takes a 4 px shift back to 20.7 candidates a board
  from 58.1, recall 90.8% -> 97.6%. **It does not recover rotation** -- at 1.0
  degrees the queue stays at roughly twice an aligned pair's -- and it is not
  allowed to pretend otherwise. Three refusals, each measured rather than
  assumed: a sub-pixel correction is declined, because warping a binarised
  image writes grey along every edge and registering already-aligned pairs made
  17 of 60 *worse*; an estimate over 5% of the frame is declined, because four
  of those 60 produced 240-355 px estimates on a 640 px frame; and a low peak is
  declined. **Confidence alone is not the guard** -- two of those four came back
  at 0.134 and 0.076, and the first draft's docstring claiming it sufficient was
  wrong. Held by `tests/test_registration_stage.py`.
- **The escape budget is one number over six classes the work instructions do
  not treat alike.** WI-201 and WI-202 admit no acceptable open or short; the
  other four are conditional on a measurement. At the shipped threshold
  (0.912, since 2026-08-31) **`short` escapes at 1.77% and `open` at 1.16%
  against an aggregate of 0.66%** -- the classes that may never ship are the
  ones furthest over the budget they are averaged into. It read "the aggregate
  meets QP-110 at 0.50% and `short` escapes at 1.55%, 3.1x that" until
  2026-09-08, which was the 0.961 oracle's reading, not the deployed one. Which class it is moves with the checkpoint (it was `open` at 1.35%
  before the 2026-08-24 retrain), which is why the guard reads `GOVERNS` rather
  than naming a class. A class-aware veto is the obvious fix and **it does not
  exist**: on the opens this model dismisses, `P(open)` is 0.00007-0.00589, and
  on the ones it keeps the median is 1.000. They are confident errors, not uncertain ones, so
  no cut on the model's own output separates them. What closes them is a second
  measurement that does not share the failure -- electrical test, which WI-201
  already names. Do not present the aggregate figure without this split.
  `scripts/class_escape_report.py`, held by `tests/test_class_escape.py`.
- **Say what the prevalence is, and what it costs.** The split is **41.2%
  genuine defects** (3,018 of 7,322 candidates); no line is, and an AOI tuned
  for recall over-calls by one to two orders of magnitude. Measured rather than
  hedged: the escape rate is prevalence-*invariant* (it divides by
  `defects_total`, so re-weighting cancels), review reduction *rises* on a
  cleaner line (52.8% is a floor, not a ceiling -- 89.0% at 0.5% prevalence),
  and what does not transfer is the **verification**: a pilot seeing 100
  defects cannot confirm a 0.5% budget and one seeing 30 says nothing at all.
  Do not quote a review-reduction figure without the prevalence it assumes.
- **Use the official DeepPCB split.** Do not re-split; comparability matters.
- Split train/val **by image**, never by patch -- patches from one board leak.
- **An automated disposition names what produced it.** A `model` or `agent`
  decision carries the checkpoint's SHA-256, the thresholds that routed it and
  the commit that ran; `store.boards.record_decision` raises on one that does
  not. Derived, never declared -- a `model_version` somebody must bump is worse
  than no field, because the release it is forgotten on is the release where it
  is wrong and looks right. The three absences (`unrecorded`, `unavailable`,
  `unknown`) are three words and none of them is `NULL`.
- **A human disposition names who made it.** An operator's answer is the next
  training round's label, so the row carries the name *and* how that name was
  established -- `signed_in` off the station's session, `host_account` off the
  CLI's OS account. `store.boards.record_decision` raises on a `human` row that
  is not attributable, which is the mirror of the rule above, and the station
  reads the name off the session and never off the form: a field a browser can
  fill names whoever it likes. Not an access-control rule dressed up. Five
  regions were clicked through without the domain knowledge to judge them, four
  of the five wrong, and nothing could tell those labels from an expert's, so
  all five had to be deleted by hand. The absences are named here too --
  `automated` where no person was involved, `unrecorded` where the row predates
  the column -- and neither of them is `NULL`. What this buys is a mechanism
  behind a name, not proof of who was at the keyboard; a shared passphrase
  still names one operator for two people, and `station/auth.py` says so.
- **Language is a rendering, not a record.** The station reads in Traditional
  Chinese or English and a switch moves between them, but only *chrome* moves:
  the question a supervisor typed, and everything the planning call wrote about
  it -- `interpretation`, the assumptions, each call's `why` -- keep the
  language they were made in and are labelled as records rather than rewritten.
  The planning call is never made again, so what it wrote is what happened.
  The one re-derivable thing is the synthesised answer, and it is *written
  again* from the stored `results_json` down the same measured path -- never
  translated. A translation is a third artefact, produced from prose rather
  than from results, and nothing here measures it. The rewrite is a button
  (`POST /ask/{id}/answer`), not a side effect of the switch: opening a
  stored run must not cost a model call. **Nothing called that path until
  2026-09-02** -- the column, `add_answer` and the docs existed, and the page
  showed the original answer under whichever heading was set; found by the
  close-out check before the demo, and held by the rewrite tests in
  `tests/test_i18n.py`. Two languages over one
  payload is a cross-check the single-language system never had, so
  `synthesis_eval.py` scores both and **refuses to publish a single-language
  report**: a figure written from one surface reads as a claim about the system
  and is a claim about half of it. Held by `tests/test_i18n.py` and
  `tests/test_analysis_run_languages.py`. The disposition path's rationale is
  a record of the same kind and, since 2026-08-29, is written in the *line's*
  language (`AOI_LINE_LANGUAGE`, default the station's) rather than in
  English regardless -- the queue then shows it as written, whichever way the
  switch is set. Measured in Chinese that day at median 31.9 s / p90 35.9 s
  against 8.6 s / 11.1 s in English, 3.7x -- and that gap was mostly the
  prompt: the model was never told the station's thresholds, so every Chinese
  rationale wrote an outline of both cases. Since 2026-08-30 the prompt states
  the two cuts and asks for one plain paragraph, and the same measurement
  reads **Chinese median 16.7 s, p90 19.7 s; English median 11.8 s, p90
  16.1 s; 0 of 20 past the 60 s deadline in either** -- 1.4x, all of it
  generation. `scripts/latency_report.py` names the language it ran in;
  neither figure applies to the other language.
  The same day's first Chinese rationale cited "a 0.85 threshold" that no
  document and no line of its prompt contains, so since 2026-08-30 the reason
  node checks every figure in the rationale against the prompt it composed
  (`graph/rationale_check.py`), stores the unsourced ones on the queue row and
  the decision (`rationale_flags`), and the queue and region page show them
  as a warning -- the analysis page's figure check, one path over. It cannot
  see a real figure compared wrongly; it says so. Held by
  `tests/test_rationale_check.py`. Read over a population for the first time
  on 2026-09-01 (`scripts/rationale_eval.py`, the same 60 candidates in both
  languages, no model judging a model): **`unsourced_figure` 0 of 120 and
  `no_explanation` 0 of 120**, which is what makes the 0.85 incident a fixed
  defect rather than a standing one. What the two-language reading found is
  that the rationale is not the same record in both: clean **20/60 in Chinese
  against 50/60 in English**, and the whole gap is one defect -- the model
  translates a class exactly when Chinese has an idiomatic word for it
  (`open` -> 開路 on 24 of 41, `short` -> 短路 on 3 of 3, every other class
  0) and not on every row, so the queue's `defect_class` column and the
  sentence beside it name the class in two vocabularies and neither is
  stable. A note asking for identifiers to be kept was not enough; since
  2026-09-02 `LANGUAGE_NOTE` lists the six class names as a closed set and
  forbids the two renderings by name, and the same 60 re-run that day read
  **`class_not_named` 0 of 60 and clean 45/60 in Chinese, 52/60 in
  English** -- the word 開路 survives in 9 rows as the physical event beside
  the identifier, never in place of it. What the re-read found instead is
  the thing no exact check counts: 5 of 60 Chinese rationales write WI-201's
  continuity clause backwards (「電測證實連續開路 ... cosmetic thinning 釋放」),
  every word sourced, the condition reversed. Counted since 2026-09-22 by a
  fourth flag, `release_on_confirmed_open`, which reproduces the hand count
  on the stored runs (5 of 60 at `64d363a`, 3 of 60 at `6c09a60`, 0 of 60 in
  English both times), and addressed by `i18n.RATIONALE_GLOSSARY` -- one
  sentence naming 連續開路 and asking for 電測導通, appended to the
  disposition path's prompt only, because `LANGUAGE_NOTE` is also the
  planner's and the synthesis prompt's and their figures were measured under
  it. `AOI_RATIONALE_GLOSSARY=0` is the control arm, and the two-arm run on
  2026-09-22 reads **9 of 60 without the sentence and 0 of 60 with it**,
  zh-TW clean 37 -> 55, English unchanged at 50 of 60 on both sides and both
  guardrails at 0 of 120, so the verdict written first is **fixed**. The
  control's nine is higher than the 5 and 3 the flag counts on the stored
  runs, so the defect was not shrinking on its own; and `clean` now counts a
  fourth flag, so its figures are not comparable with the entries before
  2026-09-22. Nothing vetoes the sentence at run time: the glossary prevents
  it, the flag counts it. Three of the eval's own six rows did not survive
  adjudication and two of the three failures were the checker's: a document
  check comparing WI numbers against file slugs, so it could never return
  zero; a class name typeset with U+2011 counted as unnamed. Both are fixed
  in `64d363a`; the pre-fix counts stay in the 09-01 entry under the commit
  that produced them.
- **Say what is simulated.** Production metadata is generated with **two
  planted signals, both by assignment** -- the seeder never writes a defect,
  it decides which DeepPCB board went to which machine. The first is M22,
  which receives the open-heaviest fifth of boards. The second, added
  2026-08-26, is a `parameter_change` on M32: before it M32 receives the
  open-heavy fifth of the remaining boards, after it the open-light fifth.
  Three other machines carry events with **no** effect, because a tool read
  against a seed with no controls is scored on "is there an event" and not
  on "did it matter"; and one machine, M11, carries the effect's mirror
  image with no event at all -- boards are conserved, so the mirror has to
  land somewhere, and it lands on one named machine rather than smeared
  over the controls. `tests/test_machine_events.py` holds all four
  properties on the assignment planner. Acceptance criteria are original
  documents, not IPC-A-610 (copyrighted, must stay out).

