> Moved out of CLAUDE.md on 2026-09-25; before that date this section lived there.

## Commands

```bash
uv run pytest                                    # 1,655 tests, no GPU needed, no model called
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
uv run python scripts/detector_fold_report.py    # is the crop threshold's shift the in-sample boxes?
                                                 # one detector on half the boards, two arms of equal
                                                 # size -- seen and unseen -- one test set (~1 h)
uv run python scripts/render_diagrams.py         # the two README flow diagrams, from the graphs' constants
uv run --with python-pptx --with playwright python scripts/build_deck.py [--embed-video]
                                                 # the journey deck: docs/deck/*.pptx, .html, study guide
uv run python scripts/demo_script.py             # the shot list -> docs/demo-script.md (one row per cue)
uv run --with playwright python scripts/demo_record.py --lang zh-TW --stem <stem> --index <n> --check --base http://127.0.0.1:8111
                                                 # every element a cue frames exists on the station's pages
uv run --with playwright python scripts/demo_record.py --lang zh-TW --stem <stem> --index <n> --silent --base http://127.0.0.1:8111
                                                 # the demo take: Playwright cue by cue, subtitles in a band under
                                                 # the page, no audio (~6 min, macOS only); --narrate-existing dubs it
uv run python scripts/train.py                   # ~4 min on the M5 Air (MPS)
uv run python scripts/report.py                  # operating-point table -> docs/benchmarks.md
uv run python scripts/routing_report.py          # how much never reaches the LLM
uv run python scripts/retrieval_report.py        # can a class's criteria come back as another's?
uv run python scripts/threshold_sweep.py         # what each graph threshold costs and buys
uv run python scripts/latency_report.py          # does the reason node fit the response budget?
uv run python scripts/reverifier_latency.py      # what one candidate costs: MPS vs CPU, cold vs warm (~7 min)
uv run python scripts/quantisation_report.py     # what INT8 costs at the escape budget (~20 min)
uv run python scripts/pretrained_ablation.py     # is the ImageNet init worth anything on binarised scans? (~12 min)
uv run python scripts/seed_variance.py           # what a re-run of the whole procedure lands on (~48 min)
uv run python scripts/model_free_baseline.py     # the floor: the same job with no network at all
uv run python scripts/feature_baseline.py        # the floor with shape in it: hand features + a tree (~40 s)
uv run python scripts/dinov2_probe.py            # stronger pretrained features: DINOv2 ViT-S/14, frozen,
                                                 # with a linear probe on the same curve; verdict written first
uv run python scripts/dinov2_latency.py          # what that probe costs per candidate, beside a same-session ResNet-18
uv run python scripts/dinov2_crops_report.py     # the same question where the RGB crops are photographs:
                                                 # DINOv2 against the ResNet-18 on the detector's boxes,
                                                 # verdict a paired bootstrap over the test images (~2.5 h)
uv run python scripts/dinov2_vpt.py --max-steps 20  # time real deep-VPT steps and project the run first
uv run python scripts/dinov2_vpt.py              # the probe's gap, adapted: deep prompts on frozen DINOv2,
                                                 # trained by train.fit, three seeds (~2.5 h on MPS)
uv run python scripts/dinov2_vpt_crops.py        # the same prompts where the crops are photographs:
                                                 # against the 2026-09-16 ResNet-18 checkpoints on the
                                                 # detector's boxes, verdict a paired bootstrap (~2 h)
uv run python scripts/calibration_report.py      # does calibrating let a threshold survive a retrain? (~3 min)
uv run python scripts/gap_decomposition.py       # why are the trainval boards easier? (~6 min)
uv run python scripts/rationale_eval.py          # is the operator's rationale true of what the model was shown?
                                                 # (~40 min, both languages, refuses to publish one)
uv run python scripts/agent_eval.py              # does the agent layer beat the classifier? (~9 min)
uv run python scripts/analysis_eval.py           # does the planner plan the right lookups, and refuse the rest?
uv run python scripts/analysis_eval.py --plan-only  # the same score, without the tools and the prose nobody scores
uv run python scripts/planner_session_probe.py   # what moves the planner between sessions: four
                                                 # conditions in one session, plans compared call
                                                 # by call, C2 the noise floor (~1 h)
uv run python scripts/language_note_arms.py --runs 2026-08-30:A.json current:B.json 2026-08-30:C.json current:D.json --day D --start T --end T
                                                 # four `analysis_eval --raw` files from one load, ABAB: did
                                                 # the 2026-09-02 language note move the planner? (reads only)
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

