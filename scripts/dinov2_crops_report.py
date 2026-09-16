"""Stronger pretrained features on the detector's crops: DINOv2 ViT-S/14 against the ResNet-18.

`scripts/dinov2_probe.py` asked this on DeepPCB and the answer was no: frozen
DINOv2 features with a linear probe removed less review than the ResNet-18 and
cost 17x more per candidate. Two things could explain that, and only one of them
transfers. The patches there are binarised 64 px differences upsampled 3.5x,
which is not imagery this backbone has seen; and the backbone is frozen rather
than fine-tuned. On PCB-AoI the first reason is gone -- a patch is a 64 px RGB
window of a photographed board, which is what natural-image pretraining is for.
So the DeepPCB verdict does not carry over and this measures it.

The candidates are the 2026-08-28 entry's, unchanged: the 600 px detector at its
confidence floor, `data/patches_pcbaoi`. Rebuilding them with the 1280 px
detector would produce a different queue that nothing published can be read
against, so this script refuses to run on a set whose counts differ from that
entry's.

**The verdict, written before the run.**

578 candidates, 449 of them defects, and a 0.5% budget is about two escapes.
Comparing medians at that scale reads sampling noise as a finding, so the
verdict is a *paired* bootstrap over the 60 test images: resample images with
replacement, recompute both orderings on the same resampled candidates, and
take the 95% interval of the difference in review removed (DINOv2 - ResNet).
Pairing removes "these boards happened to be easy", which is the variance that
would otherwise swamp the comparison.

* **helps** -- the interval lies entirely above zero.
* **indistinguishable** -- the interval contains zero. Sixty images resolve only
  so much, and this is the expected outcome.
* **worse** -- the interval lies entirely below zero.
* Whatever the interval says, if both medians are under 1% the entry leads with
  the sentence that neither ordering is one: the detector's own confidence
  removes 0.3% on this queue, and a re-verifier that cannot beat that is not
  ordering anything.

Two threshold columns, the same rule for both models:

* **oracle** -- the best threshold at the budget read on test. This is how the
  published 2.8% was produced (`crop_reverifier_report.py`), so it is the
  like-for-like column and **the verdict is read from it**.
* **deployable** -- chosen on this seed's own by-image validation split by the
  upper bound of the interval (`threshold_cv.choose`), test read once. Weaker
  than the five-fold procedure DeepPCB uses, because five folds of this ResNet
  cost 2.6 h a seed; symmetric between the two models, which is what the
  comparison needs.

Which of the eight DINOv2 configurations becomes its result is chosen on
five-fold out-of-fold trainval alone, never on test. Three seeds for both
models; the published 2.8% is one seed and is printed beside them, not as a
comparator.

Nothing here may be compared with the DeepPCB figures: different front end,
different prevalence, and a 22.3% ceiling against that curve's 100%.

    uv run python scripts/dinov2_crops_report.py --dry-run --seeds 0
    uv run python scripts/dinov2_crops_report.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dinov2_probe as probe  # noqa: E402
from threshold_cv import choose  # noqa: E402
from train import DEFAULTS, predict, split_by_image  # noqa: E402

from aoi_agent.stats import wilson  # noqa: E402
from aoi_agent.vision.operating_point import best_at_escape_budget, sweep  # noqa: E402
from aoi_agent.vision.patches import PatchSet  # noqa: E402

#: A separate cache directory, because `probe.cached_features` names its files
#: after the model, the input mode and the split -- not the dataset. A run
#: pointed at `data/features` would rebuild DeepPCB's caches in place with this
#: dataset's features, and the only guard is a candidate count that happens to
#: differ. Nothing would report an error.
FEATURES = ROOT / "data" / "features_pcbaoi"
PATCHES = ROOT / "data" / "patches_pcbaoi"
#: Not `models/dinov2_probe.json`: `dinov2_latency.py` reads that file to decide
#: which pooling it times.
OUT_JSON = ROOT / "models" / "dinov2_crops.json"
RESNET_ROOT = ROOT / "models"
RESNET_DIR = "pcbaoi_reverifier_seed{seed}"

PUBLISHED_SEEDS = [0, 1, 2]
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
BOOTSTRAP_INTERVAL = (2.5, 97.5)
#: Under this, an ordering is not one. The detector's own confidence is 0.3%.
NOT_AN_ORDERING = 0.01

#: The 2026-08-28 entry, as published. `tests/test_dinov2_crops.py` holds these
#: against docs/benchmarks.md, and `check_basis` holds the candidate set against
#: them, so neither the prose nor the data can drift from the run being compared.
PUBLISHED_IMAGES = 60
PUBLISHED_DEFECTS_ANNOTATED = 332
PUBLISHED_UNFLAGGED = 28
PUBLISHED_CANDIDATES = 578
PUBLISHED_FLAGGED_DEFECTS = 449
PUBLISHED_FALSE_CALLS = 129
PUBLISHED_PREVALENCE = 0.777
PUBLISHED_CEILING = 0.223
PUBLISHED_DETECTOR_REVIEW = 0.003
PUBLISHED_RESNET_REVIEW = 0.028


# --------------------------------------------------------------------------
# the candidate set is the published one


def shares_deeppcb_cache(features_dir: Path) -> bool:
    """True when this run would write its features into DeepPCB's cache directory.

    Resolved on both sides: the first version of this guard compared the argument
    as given against an absolute path, so `--features-dir data/features` walked
    straight past it and began a run that would have rebuilt DeepPCB's caches
    with this dataset's features.
    """
    return features_dir.resolve() == (ROOT / "data" / "features").resolve()


def check_basis(test: PatchSet) -> dict:
    """Refuse a candidate set that is not the one the published figures describe."""
    names = list(test.label_names)
    false_call_index = names.index("false_call")
    candidates = len(test.labels)
    false_calls = int((test.labels == false_call_index).sum())
    defects = candidates - false_calls
    images = int(len(np.unique(test.image_index)))
    actual = (images, candidates, defects, false_calls)
    expected = (PUBLISHED_IMAGES, PUBLISHED_CANDIDATES, PUBLISHED_FLAGGED_DEFECTS,
                PUBLISHED_FALSE_CALLS)
    if actual != expected:
        raise SystemExit(
            f"this candidate set is {actual} (images, candidates, defects, false calls) "
            f"and the published entry's is {expected}. Rebuilding the patches makes a "
            "different queue, which no published figure can be read against."
        )
    return {"images": images, "candidates": candidates, "defects": defects,
            "false_calls": false_calls, "prevalence": defects / candidates}


def aligned_scores(predictions: Path, test: PatchSet, false_call_index: int) -> np.ndarray:
    """`test_predictions.npz` P(false_call), checked row by row against the patches.

    `train.py` writes probabilities and labels and no image index, so every join
    downstream is by row order. The bootstrap resamples *images*, which it can
    only do by reading `image_index` off the PatchSet and trusting that order.
    Comparing the labels is what makes that trust checkable.
    """
    data = np.load(predictions, allow_pickle=False)
    names = [str(n) for n in data["label_names"]]
    if names != list(test.label_names):
        raise SystemExit(f"{predictions} has classes {names}, the patches have "
                         f"{list(test.label_names)}")
    labels = data["labels"]
    if len(labels) != len(test.labels) or not np.array_equal(labels, test.labels):
        raise SystemExit(f"{predictions} is not row-aligned with the test patches")
    return data["probabilities"][:, false_call_index]


# --------------------------------------------------------------------------
# the two columns, one rule for both models


def deployable_point(val_scores: np.ndarray, val_labels: np.ndarray, false_call_index: int,
                     budget: float):
    """The threshold this model would ship, chosen on validation and nothing else.

    By signature this cannot see test. The upper bound rather than the point
    estimate, for the reason `threshold_cv.choose` gives.
    """
    return choose(sweep(val_scores, val_labels, false_call_index), budget)[1]


def read_columns(val_scores: np.ndarray, val_labels: np.ndarray, test_scores: np.ndarray,
                 test_labels: np.ndarray, false_call_index: int, budget: float) -> dict:
    """Both readings of one ordering: deployable threshold, and the oracle at the budget."""
    deployable = deployable_point(val_scores, val_labels, false_call_index, budget)
    oracle = best_at_escape_budget(sweep(test_scores, test_labels, false_call_index), budget)
    row: dict = {
        "deployable_threshold": None if deployable is None else deployable.threshold,
        "oracle_threshold": None if oracle is None else oracle.threshold,
        "oracle_review_reduction": None if oracle is None else oracle.review_reduction,
        "oracle_escape_rate": None if oracle is None else oracle.escape_rate,
    }
    if oracle is not None:
        low, high = wilson(oracle.escapes, oracle.defects_total)
        row["oracle_escape_ci"] = [low, high]
        row["oracle_escapes"] = int(oracle.escapes)
        row["defects_total"] = int(oracle.defects_total)
    if deployable is None:
        row.update({"test_review_reduction": None, "test_escape_rate": None,
                    "test_escape_ci": None, "class_escapes": None})
        return row
    point = sweep(test_scores, test_labels, false_call_index,
                  thresholds=np.array([deployable.threshold]))[0]
    low, high = wilson(point.escapes, point.defects_total)
    row.update({
        "test_review_reduction": point.review_reduction,
        "test_escape_rate": point.escape_rate,
        "test_escapes": int(point.escapes),
        "test_escape_ci": [low, high],
    })
    return row


# --------------------------------------------------------------------------
# the ResNet-18 arm


def train_resnet(seed: int, patches: Path, out_dir: Path, device, log=print) -> None:
    """`scripts/train.py` unchanged -- a re-implementation here would not be the same model."""
    if (out_dir / "reverifier.pt").exists() and (out_dir / "test_predictions.npz").exists():
        log(f"  {out_dir.relative_to(ROOT)} already trained")
        return
    command = [sys.executable, str(ROOT / "scripts" / "train.py"),
               "--patches", str(patches), "--seed", str(seed), "--out", str(out_dir)]
    if device is not None:
        command += ["--device", str(device)]
    log(f"  training seed {seed} -> {out_dir.relative_to(ROOT)}")
    began = time.perf_counter()
    if subprocess.run(command, check=False).returncode != 0:
        raise SystemExit(f"scripts/train.py failed for seed {seed}")
    log(f"  trained in {time.perf_counter() - began:.0f}s")


def resnet_scores(seed: int, trainval: PatchSet, test: PatchSet, out_dir: Path,
                  false_call_index: int, device, batch_size: int = 256):
    """(validation scores, validation labels, test scores) for the trained checkpoint."""
    import torch
    from torch.utils.data import DataLoader, Subset

    from aoi_agent.vision.dataset import CandidateDataset
    from aoi_agent.vision.model import build_model

    payload = torch.load(out_dir / "reverifier.pt", map_location="cpu")
    model = build_model(len(trainval.label_names), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    _train_idx, val_idx = split_by_image(trainval, DEFAULTS["val_fraction"], seed)
    loader = DataLoader(Subset(CandidateDataset(trainval, augment=False), val_idx),
                        batch_size=batch_size)
    val_probabilities, val_labels = predict(model, loader, device)
    test_scores = aligned_scores(out_dir / "test_predictions.npz", test, false_call_index)
    return val_probabilities[:, false_call_index], val_labels, test_scores


# --------------------------------------------------------------------------
# the DINOv2 arm


def probe_scores(x_trainval: np.ndarray, trainval: PatchSet, x_test: np.ndarray,
                 false_call_index: int, seed: int, budget: float):
    """(validation scores, validation labels, test scores, C) for one configuration."""
    n_classes = len(trainval.label_names)
    train_idx, val_idx = split_by_image(trainval, DEFAULTS["val_fraction"], seed)
    c, model = probe.choose_c(x_trainval, trainval.labels, train_idx, val_idx,
                              false_call_index=false_call_index, budget=budget,
                              seed=seed, n_classes=n_classes)
    val_scores = probe.probabilities(model, x_trainval[val_idx], n_classes)[:, false_call_index]
    test_scores = probe.probabilities(model, x_test, n_classes)[:, false_call_index]
    return val_scores, trainval.labels[val_idx], test_scores, c


def probe_seed(features: dict, trainval: PatchSet, test: PatchSet, seed: int, args,
               log=print) -> dict:
    """One seed: select a configuration out-of-fold, then read every configuration on test."""
    names = list(trainval.label_names)
    false_call_index = names.index("false_call")
    log(f"\n=== DINOv2 seed {seed} ===")
    guarded_by_key: dict[tuple[str, str], object] = {}
    dims: dict[tuple[str, str], int] = {}
    for mode in args.inputs:
        for pooling in args.poolings:
            x = features[mode]["trainval"][pooling]
            dims[(mode, pooling)] = int(x.shape[1])
            log(f"  {mode}/{pooling}: out-of-fold over {args.folds} folds, {x.shape[1]} dims")
            _oof, _optimistic, guarded, _cs = probe.oof_select(
                x, trainval, folds=args.folds, budget=args.budget, seed=seed,
                log=lambda line: log("  " + line))
            guarded_by_key[(mode, pooling)] = guarded

    chosen = probe.select_configuration(guarded_by_key)
    rows = []
    for key in guarded_by_key:
        mode, pooling = key
        val_scores, val_labels, test_scores, c = probe_scores(
            features[mode]["trainval"][pooling], trainval, features[mode]["test"][pooling],
            false_call_index, seed, args.budget)
        row = {"input": mode, "pooling": pooling, "dims": dims[key], "final_c": c,
               "oof_review_reduction": None if guarded_by_key[key] is None
               else guarded_by_key[key].review_reduction,
               "chosen": key == chosen,
               **read_columns(val_scores, val_labels, test_scores, test.labels,
                              false_call_index, args.budget)}
        if row["chosen"]:
            row["class_escapes"] = None if row["deployable_threshold"] is None else \
                probe.class_escapes(test_scores, test.labels, names, row["deployable_threshold"])
            row["test_scores"] = test_scores
        rows.append(row)
        log(f"  {mode}/{pooling}: oracle {_percent(row['oracle_review_reduction'])} "
            f"deployable {_percent(row['test_review_reduction'])}"
            f"{'  <- chosen' if row['chosen'] else ''}")
    return {"seed": seed, "chosen": None if chosen is None else "/".join(chosen), "configs": rows}


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


# --------------------------------------------------------------------------
# the verdict


def review_at_budget(scores: np.ndarray, labels: np.ndarray, false_call_index: int,
                     budget: float) -> float:
    """Review removed at the oracle threshold, or zero where no threshold fits the budget."""
    point = best_at_escape_budget(sweep(scores, labels, false_call_index), budget)
    return 0.0 if point is None else point.review_reduction


def paired_bootstrap(first: np.ndarray, second: np.ndarray, labels: np.ndarray,
                     image_index: np.ndarray, false_call_index: int, budget: float,
                     resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    """The 95% interval of (first - second) review removed, resampling whole images.

    Both orderings are recomputed on the same resampled candidates, so the
    difference is paired: board-to-board difficulty cancels instead of being
    counted twice as noise. Images, not candidates, because candidates from one
    board are not independent of each other.
    """
    images = np.unique(image_index)
    rows_for = {image: np.flatnonzero(image_index == image) for image in images}
    rng = np.random.default_rng(seed)
    differences, skipped = [], 0
    for _ in range(resamples):
        drawn = rng.choice(images, size=len(images), replace=True)
        rows = np.concatenate([rows_for[image] for image in drawn])
        resampled = labels[rows]
        if not (resampled != false_call_index).any():
            skipped += 1     # no defect in this resample, so no escape rate exists
            continue
        differences.append(
            review_at_budget(first[rows], resampled, false_call_index, budget)
            - review_at_budget(second[rows], resampled, false_call_index, budget))
    low, high = np.percentile(differences, BOOTSTRAP_INTERVAL)
    return {"median": float(np.median(differences)), "low": float(low), "high": float(high),
            "resamples": len(differences), "skipped": skipped}


def verdict(interval: dict, dinov2_median: float, resnet_median: float) -> tuple[str, str]:
    """The pre-registered reading: the sign of the paired interval, and nothing else.

    Review reduction is a fraction, so every figure quoted here is scaled to
    percentage points -- the first draft printed the fractions and called them
    points, which rendered a real -1.7 to +1.5 interval as "-0.0 to +0.0".
    """
    median, low, high = (interval["median"] * 100, interval["low"] * 100,
                         interval["high"] * 100)
    if interval["low"] > 0:
        label, sentence = "helps", (
            f"Helps: DINOv2 removes {median:+.1f} points more review than the "
            f"ResNet-18 on the same candidates, 95% interval "
            f"{low:+.1f} to {high:+.1f} points, entirely above zero.")
    elif interval["high"] < 0:
        label, sentence = "worse", (
            f"Worse: DINOv2 removes {median:+.1f} points of review against the "
            f"ResNet-18, 95% interval {low:+.1f} to {high:+.1f} "
            "points, entirely below zero.")
    else:
        label, sentence = "indistinguishable", (
            f"Indistinguishable: the paired 95% interval on the difference is "
            f"{low:+.1f} to {high:+.1f} points and contains zero, "
            f"so sixty test images cannot separate these two orderings.")
    if max(dinov2_median, resnet_median) < NOT_AN_ORDERING:
        sentence += (
            f" Both are under {NOT_AN_ORDERING:.0%} review removed, against the detector's "
            f"own {PUBLISHED_DETECTOR_REVIEW:.1%} on this queue: neither ordering is one.")
    return label, sentence


def publishable(seeds: list[int], folds: int, inputs, poolings) -> bool:
    """Only the full pre-registered run may append; anything shorter is a read."""
    return (sorted(seeds) == PUBLISHED_SEEDS and folds == 5
            and tuple(inputs) == probe.INPUTS and tuple(poolings) == probe.POOLINGS)


# --------------------------------------------------------------------------
# the entry


def _median(rows: list[dict], key: str) -> float:
    values = [r[key] for r in rows if r[key] is not None]
    return statistics.median(values) if values else 0.0


def _range(rows: list[dict], key: str) -> str:
    values = [r[key] for r in rows if r[key] is not None]
    if not values:
        return "—"
    return f"{min(values):.1%}–{max(values):.1%}"


def _worst_escape(rows: list[dict]) -> float:
    """The largest test escape any seed's deployable threshold produced."""
    return max((r["test_escape_rate"] for r in rows if r["test_escape_rate"] is not None),
               default=0.0)


def _range_raw(rows: list[dict], key: str) -> str:
    values = [r[key] for r in rows if r[key] is not None]
    return "—" if not values else f"{min(values):.3f}–{max(values):.3f}"


def render(record: dict) -> str:
    basis = record["basis"]
    dinov2 = record["dinov2_rows"]
    resnet = record["resnet_rows"]
    interval = record["bootstrap"]
    lines = [
        f"## {record['date']} · commit {record['commit']}",
        "",
        "### Stronger pretrained features on the detector's crops: DINOv2 ViT-S/14 "
        "against the ResNet-18",
        "",
        f"The question `scripts/dinov2_probe.py` asked on DeepPCB, asked again where the "
        f"reason it lost there does not apply: these patches are 64 px RGB windows of "
        f"photographed boards, which is the imagery this backbone was pretrained on. "
        f"`dinov2_vits14` at `{record['hub_ref'][:12]}`, frozen, a 7-class logistic "
        f"regression replaced by a {basis['classes']}-class one on standardised features. "
        f"The candidate set is the 2026-08-28 entry's, unchanged: "
        f"**{basis['images']} test images, {PUBLISHED_DEFECTS_ANNOTATED} annotated defects, "
        f"{PUBLISHED_UNFLAGGED} of them never boxed by the detector and outside every figure "
        f"here**; {basis['candidates']} candidates ({basis['defects']} covering a defect, "
        f"{basis['false_calls']} false calls), **{basis['prevalence']:.1%} genuine defects**, "
        f"so review removed cannot exceed {1 - basis['prevalence']:.1%} on any ordering. "
        f"Three seeds for both models. {record['wall_minutes']:.0f} min. "
        "`scripts/dinov2_crops_report.py`.",
        "",
        f"**Verdict, by the rule written before the run: {record['verdict_sentence']}**",
        "",
        "Two readings of every ordering, the same rule for both models. **oracle** is the "
        "best threshold at the ≤0.50% budget read on test -- how the published 2.8% was "
        "produced, so it is the like-for-like column and the verdict is read from it. "
        "**deployable** is chosen on each seed's own by-image validation split by the "
        "interval's upper bound (`threshold_cv.choose`), test read once.",
        "",
        "| model | seed | configuration | oracle review removed | oracle escape | "
        "deployable review removed | deployable escape |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in resnet:
        lines.append(
            f"| ResNet-18 | {row['seed']} | RGB crops | "
            f"**{_percent(row['oracle_review_reduction'])}** | "
            f"{_rate(row['oracle_escape_rate'])} | {_percent(row['test_review_reduction'])} | "
            f"{_rate(row['test_escape_rate'])} |")
    for row in dinov2:
        lines.append(
            f"| DINOv2 probe | {row['seed']} | {row['input']}/{row['pooling']} | "
            f"**{_percent(row['oracle_review_reduction'])}** | "
            f"{_rate(row['oracle_escape_rate'])} | {_percent(row['test_review_reduction'])} | "
            f"{_rate(row['test_escape_rate'])} |")
    lines += [
        f"| ResNet-18 | median | — | **{_median(resnet, 'oracle_review_reduction'):.1%}** | — | "
        f"{_median(resnet, 'test_review_reduction'):.1%} | — |",
        f"| DINOv2 probe | median | — | **{_median(dinov2, 'oracle_review_reduction'):.1%}** | "
        f"— | {_median(dinov2, 'test_review_reduction'):.1%} | — |",
        f"| ResNet-18, one seed (2026-08-28) | — | RGB crops | "
        f"**{PUBLISHED_RESNET_REVIEW:.1%}** | 0.45% | — | — |",
        f"| detector `1 − confidence`, one seed (2026-08-28) | — | — | "
        f"**{PUBLISHED_DETECTOR_REVIEW:.1%}** | 0.45% | — | — |",
        "",
        f"Ranges on the oracle column: ResNet-18 {_range(resnet, 'oracle_review_reduction')}, "
        f"DINOv2 {_range(dinov2, 'oracle_review_reduction')}.",
        "",
        f"**Neither deployable threshold holds the budget on test, and the probe's misses by "
        f"{_worst_escape(dinov2) / 0.005:.0f}x.** A threshold chosen on the validation split "
        f"escapes {_worst_escape(resnet):.2%} on test for the ResNet-18 and "
        f"{_worst_escape(dinov2):.2%} for the probe, against the 0.50% budget that chose "
        f"both; the probe's deployable thresholds sit at "
        f"{_range_raw(dinov2, 'deployable_threshold')} where the ResNet's sit at "
        f"{_range_raw(resnet, 'deployable_threshold')}, so a cut that removes almost nothing "
        "on validation removes a sixth of the queue on test. The training crops are the "
        "detector's boxes on images it was trained on and the test crops are not, which is "
        "the shift `build_detector_patches.py` names; the probe reads it worse than the "
        "fine-tuned network does. On this queue the oracle column is the only one either "
        "model can be read from, which is a statement about the evidence and not a threshold "
        "anyone could ship.",
        "",
        f"**The paired bootstrap.** {interval['resamples']} resamples of the "
        f"{basis['images']} test images with replacement, both orderings recomputed on the "
        f"same resampled candidates, difference in review removed at the budget "
        f"(DINOv2 − ResNet-18): median {interval['median'] * 100:+.1f} points, 95% interval "
        f"{interval['low'] * 100:+.1f} to {interval['high'] * 100:+.1f}. "
        f"Seed {BOOTSTRAP_SEED}, fixed "
        "before the run; pairing is what removes board-to-board difficulty from the "
        "comparison.",
        "",
        "**Every DINOv2 configuration, read on test** -- exploration, not the result. The "
        "configuration that becomes DINOv2's result is chosen on out-of-fold trainval alone; "
        "choosing among these rows by their test column would be choosing on test. "
        "`per_channel` here means the three colour planes each read as a grey image, three "
        "forwards a candidate -- not the template/test/difference it means on DeepPCB.",
        "",
        "| configuration | dims | out-of-fold review removed, per seed | "
        "oracle review removed on test, per seed |",
        "|---|---|---|---|",
    ]
    for mode in probe.INPUTS:
        for pooling in probe.POOLINGS:
            matching = [r for record_seed in record["seeds"] for r in record_seed["configs"]
                        if r["input"] == mode and r["pooling"] == pooling]
            if not matching:
                continue
            lines.append(
                f"| {mode}/{pooling} | {matching[0]['dims']} | "
                + " / ".join(_percent(r["oof_review_reduction"]) for r in matching) + " | "
                + " / ".join(_percent(r["oracle_review_reduction"]) for r in matching) + " |")
    escapes = [r["class_escapes"] for r in dinov2 if r.get("class_escapes")]
    if escapes:
        lines += [
            "",
            "Per-class escapes at the deployable threshold, chosen seed: "
            + "; ".join(f"`{name}` " + " / ".join(f"{e[name][0]}/{e[name][1]}" for e in escapes)
                        for name in escapes[0]) + ".",
        ]
    lines += [
        "",
        "**What this does not establish.** One model size, the smallest, frozen rather than "
        "fine-tuned -- and the literature's own reading is that a linear probe is the "
        "weakest way to adapt these features, with prompt tuning well ahead of it, so this "
        "bounds frozen features and not the backbone. Sixty test images and about two "
        "escapes at the budget, which is why the verdict is an interval and not a "
        "difference of medians. The candidates come from the 600 px detector at its "
        "confidence floor and the training crops are its in-sample boxes, both as in the "
        "2026-08-28 entry. The ResNet arm keeps that entry's recipe, including a checkpoint "
        "chosen on a single by-image validation split of 1,460 to 1,775 patches depending on "
        "the seed, where one patch is 0.06 to 0.07 points -- its seed spread here is partly "
        "that granularity. **None of these figures may be read against the "
        "DeepPCB curve**: different front end, different prevalence, and a "
        f"{1 - basis['prevalence']:.1%} ceiling against that curve's 100%.",
        "",
    ]
    return "\n".join(lines)


def _rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patches", type=Path, default=PATCHES)
    parser.add_argument("--features-dir", type=Path, default=FEATURES)
    parser.add_argument("--resnet-root", type=Path, default=RESNET_ROOT)
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "docs" / "benchmarks.md")
    parser.add_argument("--seeds", type=int, nargs="+", default=PUBLISHED_SEEDS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--budget", type=float, default=DEFAULTS["escape_budget"])
    parser.add_argument("--inputs", nargs="+", default=list(probe.INPUTS), choices=probe.INPUTS)
    parser.add_argument("--poolings", nargs="+", default=list(probe.POOLINGS),
                        choices=probe.POOLINGS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not publishable(args.seeds, args.folds, args.inputs, args.poolings):
        print("only the pre-registered run (seeds 0 1 2, five folds, every input and "
              "pooling) may append to docs/benchmarks.md; pass --dry-run", file=sys.stderr)
        return 2
    if shares_deeppcb_cache(args.features_dir):
        print("refusing to share DeepPCB's feature cache: the cache is keyed on model, "
              "input mode and split, not on the dataset", file=sys.stderr)
        return 2

    from aoi_agent.vision.model import select_device

    trainval = PatchSet.load(args.patches / "trainval.npz")
    test = PatchSet.load(args.patches / "test.npz")
    basis = check_basis(test)
    basis["classes"] = len(test.label_names)
    names = list(test.label_names)
    false_call_index = names.index("false_call")
    device = select_device(args.device)
    train_device = args.train_device or (None if args.device is None else args.device)
    started = time.perf_counter()

    print(f"classes: {names}; {basis['candidates']} test candidates, "
          f"{basis['prevalence']:.1%} defects, ceiling {1 - basis['prevalence']:.1%}")

    print("\nResNet-18 arm")
    resnet_rows, resnet_test_scores = [], {}
    for seed in args.seeds:
        out_dir = args.resnet_root / RESNET_DIR.format(seed=seed)
        train_resnet(seed, args.patches, out_dir, train_device)
        val_scores, val_labels, test_scores = resnet_scores(
            seed, trainval, test, out_dir, false_call_index, device)
        row = {"seed": seed, **read_columns(val_scores, val_labels, test_scores, test.labels,
                                            false_call_index, args.budget)}
        resnet_test_scores[seed] = test_scores
        resnet_rows.append(row)
        print(f"  seed {seed}: oracle {_percent(row['oracle_review_reduction'])} "
              f"deployable {_percent(row['test_review_reduction'])}")

    features: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    extraction: dict[str, float] = {}
    for mode in args.inputs:
        features[mode] = {}
        for split, patch_set in (("trainval", trainval), ("test", test)):
            print(f"features: {mode} {split}")
            features[mode][split], seconds = probe.cached_features(
                split, patch_set, mode, args.features_dir, device, args.batch_size)
            extraction[f"{mode} {split}"] = round(seconds, 1)

    seeds = [probe_seed(features, trainval, test, seed, args) for seed in args.seeds]
    dinov2_rows = []
    for record_seed in seeds:
        for row in record_seed["configs"]:
            if row["chosen"]:
                dinov2_rows.append({"seed": record_seed["seed"], **row})
    if not dinov2_rows:
        print("no configuration produced a threshold within the budget on any seed",
              file=sys.stderr)
        return 1

    median_seed = sorted(dinov2_rows, key=lambda r: r["oracle_review_reduction"] or 0.0)[
        len(dinov2_rows) // 2]["seed"]
    interval = paired_bootstrap(
        [r for r in dinov2_rows if r["seed"] == median_seed][0]["test_scores"],
        resnet_test_scores[median_seed], test.labels, test.image_index,
        false_call_index, args.budget)
    label, sentence = verdict(interval,
                              _median(dinov2_rows, "oracle_review_reduction"),
                              _median(resnet_rows, "oracle_review_reduction"))

    for row in dinov2_rows:
        row.pop("test_scores", None)
    for record_seed in seeds:
        for row in record_seed["configs"]:
            row.pop("test_scores", None)

    record = {
        "date": datetime.now(UTC).date().isoformat(), "commit": probe.commit(),
        "seed_list": args.seeds, "folds": args.folds, "budget": args.budget,
        "device": str(device), "hub_ref": probe.HUB_REF,
        "weights_sha256": probe.sha256(probe.weights_path()), "input_size": probe.INPUT_SIZE,
        "basis": basis, "bootstrap": interval, "bootstrap_seed_read": median_seed,
        "verdict": label, "verdict_sentence": sentence,
        "resnet_rows": resnet_rows, "dinov2_rows": dinov2_rows, "seeds": seeds,
        "extraction_seconds": extraction,
        "wall_minutes": (time.perf_counter() - started) / 60,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, default=float) + "\n")

    body = render(record)
    print("\n" + body)
    if args.dry_run:
        print("(dry run -- nothing appended)")
        return 0
    existing = args.benchmarks.read_text() if args.benchmarks.exists() else "# Benchmarks\n"
    args.benchmarks.write_text(existing.rstrip() + "\n\n" + body.rstrip() + "\n")
    print(f"appended to {args.benchmarks.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
