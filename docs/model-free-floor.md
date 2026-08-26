# The floor under 50.2%

Run: `uv run python scripts/model_free_baseline.py` · 2026-08-26

## Why

"Removes 50.2% of manual review at a 0.25% escape budget" has no floor under it.
The first question anyone should ask is what the queue looks like with **no model
at all** — just a threshold on the difference image the AOI stage has already
computed by the time a candidate exists. Without that number, 50.2% is a
quantity, not evidence.

## What was run

`data/patches/test.npz`, channel 2 (`|test - template|`, from
`vision/patches.build_patch`), reduced to one scalar per candidate, mapped
monotonically into [0,1] so a *small* difference reads as high P(false_call),
then handed to the same `operating_point.sweep` the model is scored with. Same
split, same labels, same metric definitions, same budgets. No weights, no
training, no calibration set.

The model was re-scored on the identical array in the same run, because two
numbers from two bases cannot be compared.

## Result

Split: 7,322 candidates, 3,018 real defects, 4,304 false calls.
Cells are review reduction (escape rate, escapes/defects).

| Scorer | ≤0.25% | ≤0.50% | ≤1.00% |
|---|---|---|---|
| mean \|test−template\| | 0.8% (0.23%) | 1.0% (0.50%) | 1.5% (0.99%) |
| max \|test−template\| | 0.1% | 0.1% | 0.1% |
| sum of top 32 pixels | 1.0% (0.23%) | 1.3% (0.50%) | 1.8% (0.99%) |
| sum of top 64 / 128 pixels | 0.9% | 1.2% | 1.6–1.7% |
| count of pixels > 32 / 64 / 96 / 128 | 0.9% | 1.2% | 1.6–1.7% |
| constant score (no signal) | 0.0% | 0.0% | 0.0% |
| **ResNet-18 re-verifier (ONNX fp32)** | **38.9%** (0.23%) | **51.3%** (0.50%) | **56.3%** (0.99%) |

**The floor is ~1%. The model is ~40× above it** at the 0.25% budget. The
difference image alone is close to worthless here, which is the finding: the
candidates the AOI stage emits are *already* the ones with a large difference,
so difference magnitude no longer separates them. Whatever the re-verifier
learned, it is not "look for a big difference."

Every value is the best threshold **on this split**, the same overfit the
model's own headline carries. The comparison is fair because both sides
inherit it.

## The basis problem this surfaced

*Reconciled 2026-08-26, after this was written.* Two runs are in play, not
three -- 8,031 and 7,322 are the same run counted two ways (709 `fragment`
candidates are stored but held out of every sweep), and they match the
shipped `test.npz` exactly. The 8,143 / 2,997 / 5,146 row is the earlier run,
dated 2026-08-22 under a benchmarks header reading `commit uncommitted`, taken
before the registration stage existed; the published headline was quoting it
two days after the retrain replaced it. README, CLAUDE.md and
docs/architecture.md now carry the 7,322 basis, and
`tests/test_published_figures.py` holds them to whichever run
`docs/benchmarks.md` records last.

| Source | Candidates | Defects | False calls |
|---|---|---|---|
| `docs/benchmarks.md` 2026-08-22, `commit uncommitted` -- superseded | 8,143 | 2,997 | 5,146 |
| `data/aoi_agent.db` `candidates` table, including 709 `fragment` | 8,031 | -- | -- |
| `docs/benchmarks.md` 2026-08-24 and `data/patches/test.npz` -- current | 7,322 | 3,018 | 4,304 |

**The 38.9% / 51.3% this run scored for the ONNX model is itself an artefact
of the same problem.** `models/onnx/` was dated 2026-08-23 14:54 -- exported
from the checkpoint *before* the 2026-08-24 retrain -- so this scored the old
weights' export against the new population's patches. Re-exported the same
morning, FP32 ONNX reads 40.2% at ≤0.25% and 52.8% at ≤0.50%, identical to
the torch checkpoint. The retraining chain did not include re-export; it does
now. The floor comparison above is unaffected: every scorer in that table,
including the model, was run on the identical array in one process, which is
what the script was built to guarantee.
