---
name: retraining-the-reverifier
description: Use when retraining or re-evaluating the ResNet-18 re-verifier — changing epochs, patch size, differencing threshold, the model architecture, or the dataset — and whenever a benchmark number in docs/benchmarks.md needs to be regenerated.
---

# Retraining the Re-verifier

## Overview

The pipeline has one silent failure: **the dismissal threshold does not follow
the model.** `DEFAULT_DISMISS_THRESHOLD = 0.915` is a literal in
`src/aoi_agent/vision/inference.py`. Retraining rewrites the checkpoint but
leaves that constant alone, so the new model runs on the old model's operating
point. Nothing errors. The routing report and the graph flow just quietly
report the wrong escape rate.

Every retrain must end by carrying the threshold across by hand.

## Order

Each step consumes the previous step's artefact. Skipping one means the next
step reads a stale file.

```bash
uv run python scripts/build_patches.py --split trainval   # -> data/patches/trainval.npz
uv run python scripts/build_patches.py --split test       # -> data/patches/test.npz
uv run python scripts/train.py                            # -> models/reverifier.pt,
                                                          #    models/test_predictions.npz,
                                                          #    models/history.json  (~4 min, MPS)
uv run python scripts/report.py                           # reads test_predictions.npz
                                                          # -> docs/benchmarks.md
# --- THRESHOLD GATE: see below, do not skip ---
uv run python scripts/routing_report.py                   # imports the threshold constant
uv run python scripts/escape_accounting.py                # the whole-line escape rate
uv run pytest
```

`escape_accounting.py` is in the chain because it runs the model: it re-derives
the whole-line escape rate by asking, per defect, whether anything was flagged
on it and what the re-verifier then did with every candidate covering it. A new
checkpoint moves that number and nothing else recomputes it.

**`report.py` does not produce a whole-line figure and must not be made to.** It
had a `--aoi-escape-rate` argument until 2026-08-23, defaulting to 0.050, into
which the operator was expected to hand-carry `build_patches.py`'s
unmatched-at-IoU-0.33 print. That print is a box-tightness statistic — 150 of
the 157 "missed" defects on the test split have a candidate sitting on them —
and composing it with the model's escape rate published 5.4% for a line that
escapes 0.61%. Anything reading `test_predictions.npz` is working from
candidates with every `fragment` already dropped, and cannot see whether a
defect was flagged at all.

**Both splits or neither.** `train.py` loads `trainval.npz` and `test.npz`
together. Rebuilding one and not the other trains a new model against old test
patches, and the benchmark is then meaningless.

## The Threshold Gate

`scripts/report.py` prints the threshold for the ≤0.5% escape budget alongside
the operating-point table. After it runs:

1. Read the new threshold out of the report output / `docs/benchmarks.md`.
2. Update `DEFAULT_DISMISS_THRESHOLD` in `src/aoi_agent/vision/inference.py`.
3. Update the threshold row in `docs/architecture.md` (the table that records
   where each threshold came from). `tests/test_threshold_citations.py` fails
   until you do.
4. Only then run `scripts/routing_report.py` — it imports the constant.

`ESCALATE_BELOW` needs no separate carry-across: it *is*
`DEFAULT_DISMISS_THRESHOLD`, which is the point of setting it that way. A
second swept number here would be a second silent failure of exactly this
shape. Re-run `scripts/threshold_sweep.py` after a retrain anyway — it
re-derives both graph thresholds from the new predictions, and
`tests/test_threshold_citations.py -m dataset` asserts the agent branch still
dismisses nothing.

The number comes from the sweep. Never hand-tune it to make a routing number
look better; that is fitting the test set through the back door.

## When gate_check Applies

`scripts/gate_check.py` is the S0 gate on the differencing simulator, not part
of the training chain. Run it when the simulator changes — `--shift`,
`--noise`, `--gain`, or the differencing threshold. It does not need to be
re-run for a plain retrain.

Note `build_patches.py --threshold` defaults to 60 and belongs to the same
differencing stage: changing it invalidates the gate check too.

## Invariants This Touches

- **Official DeepPCB split only.** Do not re-split; comparability matters.
- **Val split is by image, not by patch.** Enforced in `scripts/train.py`;
  patches from one board leak across the boundary otherwise.
- **Report the operating-point curve, never bare accuracy.** The headline is
  review removed at an escape budget. Accuracy weighs an escape the same as a
  false call, which is wrong for this line.
- **`docs/benchmarks.md` is append-only, newest last.** Do not overwrite an
  earlier run's numbers.

## Common Mistakes

| Mistake | Result |
|---|---|
| Retrain, skip the threshold gate | New model, old operating point. Routing numbers are wrong and nothing errors. |
| Rebuild only `trainval` | New model scored against stale test patches. |
| Run `routing_report.py` before updating the constant | Reports the previous model's routing split. |
| Quote "96.5% accuracy" as the headline | Violates the operating-point invariant. |
| Benchmark LLM latency while training runs | `train.py` holds MPS. See the `measuring-llm-latency` skill. |
| Quote an IoU miss rate as an escape rate | Off by 8x. A defect whose candidate drew a looser box is reviewed, not escaped. |
