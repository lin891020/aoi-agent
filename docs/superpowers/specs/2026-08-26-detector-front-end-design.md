# A second front end: a detector, on a dataset that has no template

**Status:** approved in conversation 2026-08-26 ("兩個都做"), after the dataset
inventory and the HRIPCB transfer run. This document is the design.

## Why a second front end, and why now

Two measurements this week point at the same place from opposite sides.

- **HRIPCB** (2026-08-26): the shipped pipeline, run unchanged on photographed
  boards, fails at the *differencing stage* before the re-verifier is asked.
  Template subtraction needs two images that agree everywhere except the
  defect; a photograph gives two that disagree faintly everywhere. The S0 gate
  clears on no setting. The stage's operating regime is binarised imagery.
- **PCB-AoI** (inventory, same day): real SMT solder-paste inspection images
  from a line, 2016–2018. There is **no template at all** — placement has
  tolerance, so pixel differencing would flag every component. On this data a
  detector is not a preference; it is the only front end that can exist.

So the claim this builds toward is not "the model works on a second dataset".
It is: **the pipeline's front end is replaceable, and the replacement was
proven on a domain where the original front end is physically impossible.**
Everything downstream that does not depend on a template — the escape budget,
the operating-point sweep, the queue, provenance, the interval arithmetic —
should run unchanged over the new front end's output. What *does* depend on a
template (the 3-channel re-verifier, the station's triptych) is out of scope
for this front end and said so.

## What "candidate" means without a template

On DeepPCB a candidate is a differencing blob with no score of its own; the
re-verifier supplies the score. On PCB-AoI the detector supplies both the box
and the score in one step. So:

- a **candidate** is a detector box above a low floor confidence (`conf ≥
  0.01`, i.e. essentially everything the NMS-free head emits);
- its **P(false_call)** is `1 − max class confidence`;
- the **operating point** is swept over that score with the existing
  `vision.operating_point.sweep`, against ground truth via the existing
  `aoi.matching.match`;
- a defect no box covers at the floor is **unflagged** — the same outcome the
  escape accounting already names, and the detector's S0.

This is the same accounting the project already does, with the detector
standing in for *both* stages. The headline stays what it has always been:
**manual review removed at an escape budget**, never mAP. mAP is reported once,
for reference, the way accuracy is.

## The model

**YOLO26n** (`ultralytics` 8.4.x, weights `yolo26n.pt` from release v8.4.0).
Chosen for two measured reasons and one deliberate one:

- **Native NMS-free.** The existing pipeline is "boxes → IoU match → per-box
  re-verify". An end-to-end head removes a post-processing stage whose
  threshold would otherwise be one more number to sweep and cite.
- **STAL (small-target-aware label assignment).** PCB-AoI's median box is
  **17 px** on a 600 px frame. That is the regime the architecture claims to
  target, and this dataset is a direct test of the claim.
- **The `n` size, on purpose.** The project's own finding is that a
  re-verification station needs no GPU (2.5 ms/candidate on CPU). The smallest
  variant keeps that story honest; a larger one is a separate measurement.

Training runs on the M5 Air (MPS), `imgsz=640`, and the budget is stated in
the benchmarks entry with the machine-quiet check the latency skill requires.
**Only the MPS availability and the wall time are reported from this
machine; nothing about inference speed is claimed until measured on CPU the
way the re-verifier was.**

## Data and split

PCB-AoI ships `train_data` (173), `train_data_augmentation` (the same 173 × 6
transforms + originals = 1,211) and `test_data` (60). Verified in the
inventory: no augmentation stem leaks into test.

- **Train:** `train_data_augmentation`. Validation: a by-*stem* split of the
  173 originals (every transform of a stem goes with it) — the split-by-image
  invariant, applied to its augmentations.
- **Test:** the 60, touched once, for the benchmarks entry.
- **Classes:** the dataset's two — `Bad_podu` (paste offset), `Bad_qiaojiao`
  (lifted lead). **Not mapped onto DeepPCB's six.** They are different
  defects on a different process step, and the standards retrieval has no
  document for either; the escape accounting is per class, so they report
  under their own names.

**The 60-image test set is small and the entry says so.** 332 boxes, of which
37 are `Bad_qiaojiao`. Every escape figure carries its Wilson interval and
the interval will be wide — that is the honest shape of the result, not a
defect of the method. A 0.5% budget cannot be *confirmed* on 332 defects
(prevalence section, "what does not transfer is the verification"); what can
be measured is where this detector sits relative to it.

## Comparison

The plan file's win condition, verbatim: *同一 escape budget 下多移除多少人工複判.*

There is nothing to compare *against* on PCB-AoI — the differencing front end
cannot run there. So the comparison is the one that can be made honestly:

| | DeepPCB, differencing + re-verifier | PCB-AoI, YOLO26n |
|---|---|---|
| review removed @ ≤0.5% | 52.8% (7,322 candidates, 41.2% prevalence) | *measured here* |
| prevalence | detector-manufactured | detector-manufactured |
| S0 recall | 99.5% (registration on) | *measured here* |

The two rows are **different populations** and the entry says so above the
table, in the same words the prevalence invariant uses. What transfers between
them is the *method of reading* — a curve against a budget — not the number.

A second, cheaper comparison is possible and is included: YOLO26n **on
DeepPCB**, trained on the same trainval split, read on the same 7,322
candidates' ground truth. That is the "single vs paired input" milestone
from the plan file, and it is the only row where the two front ends can be
scored on one population. It is a separate entry and a separate run.

## Files

| file | change |
|---|---|
| `pyproject.toml` | `ultralytics` dependency |
| `src/aoi_agent/data/pcbaoi.py` | new: VOC XML → boxes, stem-aware split, YOLO-format export to `data/pcbaoi_yolo/` (gitignored) |
| `src/aoi_agent/vision/detector.py` | new: thin wrapper — load weights, `detect(image) -> list[ScoredCandidate]`, P(false_call) as defined above |
| `scripts/train_detector.py` | new: trains YOLO26n; writes `models/detector_pcbaoi.pt` and `models/detector_history.json`; refuses to run beside a busy GPU (the latency skill's check) |
| `scripts/detector_report.py` | new: operating-point table for the detector on the 60; appends to `docs/benchmarks.md`; mAP once, for reference |
| `.claude/skills/retraining-the-reverifier/SKILL.md` | one paragraph: this chain does not include the detector; `train_detector.py` has its own gate |
| `CLAUDE.md` | Layout, Commands, a "second front end" paragraph under Still open |
| `tests/test_pcbaoi.py` | new: split-by-stem, no test leakage, class table untouched, YOLO label round-trip |
| `tests/test_detector.py` | new: P(false_call) definition, unflagged accounting, sweep on a synthetic result |

## Out of scope, and why

- **Wiring the detector into the store or the station.** `CandidateRecord`
  assumes a board pair and the station renders a triptych. That is a UI and
  store change with its own design; this spec proves the front end on the
  benchmark first, which is the order the project has used for everything
  else.
- **The re-verifier on PCB-AoI.** It needs a template. Nothing to measure.
- **SolDef_AI.** Classification-shaped; deferred per the inventory.
- **Any claim about a line.** Sixty test images from one dataset.

## Gate before the benchmarks entry

1. `ollama ps` empty and no torch process besides this one (the latency skill).
2. Train; record wall time and epochs.
3. `detector_report.py --dry-run`; read the intervals; then append.
4. The entry names the 60-image basis in its first line, per the published-
   figures invariant.
