# AOI-Agent

A PCB AOI re-verification system. Production optical inspection is tuned for
recall and over-flags; every flagged region goes to a human today. This puts a
vision model in front of that queue and an agent behind it for the cases the
model cannot settle.

Started 2026-08-22. Portfolio project aimed at smart-manufacturing AI roles, so
the engineering has to hold up to an interviewer's questions, not just run.

## Layout

```
src/aoi_agent/
    data/deeppcb.py         dataset access, official splits
    data/pcbaoi.py          PCB-AoI, the detector's dataset: VOC boxes, a split
                            by base stem so no transform leaks, YOLO export
    data/hripcb.py          the second dataset, as a view: same pair interface,
                            its own class table, the rotated subset padded onto
                            the template's frame
    aoi/registration.py     phase correlation, and the three things it refuses
    aoi/simulator.py        template differencing -- the "AOI"
    aoi/matching.py         label candidates against ground truth
    vision/                 patches, dataset, ResNet-18, inference, operating
                            point, ONNX export and INT8 quantisation;
                            detector.py is the second front end -- a YOLO box
                            with P(false_call) = 1 - confidence, no template
    store/                  SQLAlchemy models, seeding, standards retrieval,
                            machine_events (the store's first non-board entity,
                            and its only writer, events.py),
                            the analysis_runs log, and the provenance every
                            automated decision carries -- dispositions.py is
                            the board-level record
    mcp_servers/            four MCP servers (classify, production, standards,
                            sql_readonly -- the last is the experiment)
    graph/                  LangGraph flow with the human escalation,
                            durable SQLite checkpointer
    analysis/               the /ask flow -- typed plan, validator, Send
                            fan-out, chart derived from the result shape;
                            sql_guard.py is the one door model-written SQL
                            may go through (a copy without the answer key,
                            one SELECT, capped)
    i18n.py                 the two string tables, and the rule that the
                            switch renders chrome and never rewrites a record
    station/                the review station -- FastAPI + Jinja, the
                            escalation queue, /ask and its SSE progress
                            stream, and service.py -- the review layer the
                            CLI shares with it. /ask has its own writer in
                            analysis/service.py.
                            result_view.py is the ground_truth boundary and
                            prose.py is the model's-Markdown one -- it returns
                            structure, never markup, so no template needs
                            `|safe`;
                            chart_svg.py renders a chart spec server-side;
                            auth.py is the sign-in, and states what the
                            scheme does not protect against
    cli.py
scripts/                    gate_check, build_patches, train, report, seed_store,
                            analysis_eval, add_operator, render_diagrams, demo_record,
                            build_detector_patches, crop_reverifier_report,
                            mark_unattributed_resolutions, ...
tests/                      1,382 tests; dataset-dependent ones behind `-m dataset`
docs/benchmarks.md          every measurement run, newest last
docs/deck/                  the project-journey deck (pptx, html with a self-test
                            mode, study guide) -- built from scripts/deck_content.py,
                            whose figures must exist in a published document;
                            planner-reach.zh-TW.html is a hand-authored diagram
                            page the build screenshots onto its slide
docs/architecture.md        layers, thresholds and where they come from
.claude/skills/             project skills -- procedures with gates, not notes
```

Invoke the project skills; do not just read past them. `retraining-the-reverifier`
before touching the training pipeline or regenerating a benchmark, and
`measuring-llm-latency` before quoting any timing number. Both encode a failure
that is silent -- a stale threshold, a contended GPU -- and neither shows up as
an error.

## Commands

```bash
uv run pytest                                    # 1,382 tests, no GPU needed, no model called
uv run python scripts/gate_check.py              # S0: does differencing make false calls?
uv run python scripts/gate_check.py --dataset hripcb --split aligned --limit 693 --thresholds 10 15 20 30 45 60 \
    --out eval/results/gate_check_hripcb_aligned.json   # the same gate on photographs (~2 min)
uv run python scripts/transfer_report.py         # the shipped pipeline on HRIPCB, unchanged (~10 min)
uv run python scripts/train_detector.py          # YOLO26n on PCB-AoI, refuses beside a busy GPU (~30 min)
uv run python scripts/detector_report.py         # the detector at the escape budget, on 60 images
uv run python scripts/build_detector_patches.py  # RGB crops of the detector's boxes, in train.py's schema
uv run python scripts/train.py --patches data/patches_pcbaoi --out models/pcbaoi_reverifier
                                                 # the same ResNet-18, minus the template channel
uv run python scripts/crop_reverifier_report.py  # its ordering against the detector's, same boxes
uv run python scripts/render_diagrams.py         # the two README flow diagrams, from the graphs' constants
uv run --with python-pptx --with playwright python scripts/build_deck.py [--embed-video]
                                                 # the journey deck: docs/deck/*.pptx, .html, study guide
uv run --with playwright python scripts/demo_record.py --lang zh-TW --stem <stem>
                                                 # the demo video: Playwright + say + ffmpeg, ~5 min, macOS only
uv run python scripts/train.py                   # ~4 min on the M5 Air (MPS)
uv run python scripts/report.py                  # operating-point table -> docs/benchmarks.md
uv run python scripts/routing_report.py          # how much never reaches the LLM
uv run python scripts/retrieval_report.py        # can a class's criteria come back as another's?
uv run python scripts/threshold_sweep.py         # what each graph threshold costs and buys
uv run python scripts/latency_report.py          # does the reason node fit the response budget?
uv run python scripts/reverifier_latency.py      # what one candidate costs: MPS vs CPU, cold vs warm (~7 min)
uv run python scripts/quantisation_report.py     # what INT8 costs at the escape budget (~20 min)
uv run python scripts/agent_eval.py              # does the agent layer beat the classifier? (~9 min)
uv run python scripts/analysis_eval.py           # does the planner plan the right lookups, and refuse the rest?
uv run python scripts/analysis_eval.py --plan-only  # the same score, without the tools and the prose nobody scores
uv run python scripts/analysis_eval.py --questions tests/fixtures/analysis_questions_independent.json
                                                 # the same scorer on seventy questions whose authors never saw the prompt
AOI_SQL_TOOL=0 uv run python scripts/analysis_eval.py --plan-only --questions tests/fixtures/analysis_questions_independent.json
                                                 # the control arm: the same seventy with run_sql out of the registry
uv run python scripts/synthesis_eval.py          # is the prose true of the results it was written from,
                                                 # in both languages? (~50 min; --lang both is the default
                                                 # and the only value it will publish)
uv run python scripts/check_mcp_servers.py       # servers start and advertise tools
uv run python scripts/invariant_audit.py         # which invariants below would fail a test if broken
uv run python scripts/invariant_audit.py --collect  # the same, checking pytest really collects each one
uv run python -m aoi_agent board 20085294 --queue  # run a board, queue what it cannot settle
uv run python -m aoi_agent station               # review station on :8110 -- the queue, /boards and /ask
uv run python -m aoi_agent queue                 # what is waiting on a person
uv run python -m aoi_agent explanations          # how many dispositions carry no rationale, and why
uv run python -m aoi_agent provenance 20085294   # who decided this board, when, and under which model
uv run python scripts/seed_store.py --migrate-only  # add missing columns to an existing store
uv run python scripts/add_operator.py <name> [--role senior]
                                                 # who may answer the queue, and who
                                                 # may answer what others handed back
uv run python scripts/mark_unattributed_resolutions.py --dry-run
                                                 # queue entries closed with no human decision behind them

docker build -t aoi-agent .                      # CPU torch; nothing heavy baked in
docker run --rm -p 8110:8110 \
  -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" aoi-agent
                                                 # the station, containerised
```

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
  doing two jobs -- choosing the operating point and reporting it. `scripts/prevalence_report.py`.
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
  other four are conditional on a measurement. At the shipped threshold the
  aggregate meets QP-110 at 0.50% and **`short` escapes at 1.55%, 3.1x that** --
  the class that may never ship is the one exceeding the budget it is averaged
  into. Which class it is moves with the checkpoint (it was `open` at 1.35%
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
  than from results, and nothing here measures it. Two languages over one
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
  `tests/test_rationale_check.py`.
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

## Environment gotchas

- MacBook Air M5, 32GB, **fanless** -- sustained load throttles. Report "first
  60s" and "steady state" separately.
- **Ollama contention is the big one.** A translation job in
  `~/Projects/video_transfer` periodically holds `gpt-oss:20b` and 12GB of GPU.
  When it runs, an LLM call's wall time can be 25x its inference time. Always
  check `ollama ps` before believing a latency number, and report
  `eval_duration`, never `total_duration`.
- **`ollama ps` is not a GPU check.** It reports Ollama's own resident models
  and nothing else, so a torch/MPS job in another shell saturates the same
  silicon while that check comes back clean. It has already cost one run: an
  MPS benchmark taken beside a concurrent detector training job, with a clean
  residency check the whole way. Sweep the process table too --
  `scripts/reverifier_latency.py` does both and refuses to publish when either
  fires.
- MCP SDK is **2.0**: `MCPServer` not `FastMCP`, `server_info` not `serverInfo`.
- torch runs on **MPS**. Python is pinned to 3.12 for torch compatibility.
- Dataset (231MB), patches, models, the SQLite db and the Chroma index are all
  gitignored and rebuilt by scripts.

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
paragraph: now Chinese median 16.7s / p90 19.7s and English 11.8s / 16.1s,
0 of 20 past the deadline in either. A missing
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
  `/boards` is the index, and it carries the denominator the queue cannot --
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
