"""Deep prompt tuning on the detector's RGB crops: DINOv2 adapted, against the ResNet-18.

Three readings of these features exist and none of them adapted the backbone.
On DeepPCB a linear probe was worse than the ResNet-18 and deep prompts then
matched it (`scripts/dinov2_vpt.py`); on these crops -- 64 px RGB windows of
photographed boards, the imagery this backbone was pretrained on -- the probe
was indistinguishable from the ResNet-18 (`scripts/dinov2_crops_report.py`,
2026-09-16). That entry's own closing sentence is what this run answers: a
linear probe bounds frozen features, not the backbone, and prompt tuning is
what the literature ranks ahead of it.

The candidate set, the ResNet-18 arm and the reading are the 2026-09-16 entry's,
unchanged. The ResNet checkpoints are the ones that entry trained -- retraining
them would compare against a different model than the published rows. Nothing
here may be read against the DeepPCB curve: different front end, different
prevalence, and a 22.3% ceiling against that curve's 100%.

**The verdict, written before the run.**

578 candidates, 449 of them defects, and a 0.5% budget is about two escapes, so
a difference of medians reads sampling noise as a finding. The verdict is the
crops entry's *paired* bootstrap over the 60 test images: resample images with
replacement, recompute both orderings on the same resampled candidates, and take
the 95% interval of the difference in review removed (VPT - ResNet-18), read on
the seed whose oracle is the median of the three.

* **helps** -- the interval lies entirely above zero.
* **indistinguishable** -- the interval contains zero. Sixty images resolve only
  so much, and this is the expected outcome.
* **worse** -- the interval lies entirely below zero.
* Whatever the interval says, if both medians are under 1% the entry leads with
  the sentence that neither ordering is one: the detector's own confidence
  removes 0.3% on this queue.

The linear probe's three seeds (0.7%-1.0%) are printed beside the result and
decide nothing; the comparison the verdict is read from is against the ResNet-18,
because that is the ordering a station would otherwise run.

Two columns, the same rule for both models, as in that entry: **oracle** is the
best threshold at the budget read on test, the like-for-like column the verdict
uses; **deployable** is chosen on the seed's own by-image validation split by the
interval's upper bound, test read once, and decides nothing.

The recipe is `dinov2_vpt.py`'s, unchanged and unsearched: ten prompts before
each of the twelve frozen blocks, a linear head on the class token, AdamW at
1e-3, batch 64, ten epochs, `train.fit` with the ResNet's split, augmentation,
class weights and epoch rule. The head is three-class here because this dataset
has three classes. The probe's arm chose `centre` pooling on out-of-fold
trainval; this reads the class token, as prompt tuning does.

    uv run python scripts/dinov2_vpt_crops.py --max-steps 20     # project the run first
    uv run python scripts/dinov2_vpt_crops.py --seeds 0 --epochs 1 --dry-run
    uv run python scripts/dinov2_vpt_crops.py                    # three seeds, appends
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dinov2_crops_report as crops  # noqa: E402
import dinov2_probe as probe  # noqa: E402
import dinov2_vpt as vpt  # noqa: E402
from train import DEFAULTS, fit, predict  # noqa: E402

from aoi_agent.vision.dataset import CandidateDataset  # noqa: E402
from aoi_agent.vision.patches import PatchSet  # noqa: E402

PATCHES = ROOT / "data" / "patches_pcbaoi"
OUT_JSON = ROOT / "models" / "dinov2_vpt_crops.json"

#: The comparators on this queue, as published on 2026-09-16. The ResNet-18 arm
#: is re-read from its own checkpoints rather than quoted, so these are the
#: numbers the entry prints beside the result; `tests/test_dinov2_vpt_crops.py`
#: holds them against docs/benchmarks.md.
RESNET_ORACLE_MIN = 0.003
RESNET_ORACLE_MAX = 0.028
PROBE_ORACLE_MIN = 0.007
PROBE_ORACLE_MAX = 0.010
PROBE_ORACLE_MEDIAN = 0.010


def publishable(seeds: list[int], epochs: int, prompts: int, lr: float, batch: int) -> bool:
    """Only the pre-registered configuration may append; anything else is a read."""
    return vpt.publishable(seeds, epochs, prompts, lr, batch)


def median_seed(rows: list[dict]) -> int:
    """The seed whose oracle review removed is the median of the three.

    The bootstrap is paired, so it reads one seed of each model and they must be
    the *same* seed: the two arms share that seed's by-image validation split,
    and pairing a VPT seed with another seed's ResNet would put a different
    training split on each side of the difference. A seed that found no
    threshold inside the budget sorts as zero rather than dropping out, because
    dropping it would move which seed is the median.
    """
    ordered = sorted(rows, key=lambda r: (r["oracle_review_reduction"] or 0.0, r["seed"]))
    return int(ordered[len(ordered) // 2]["seed"])


def verdict(interval: dict, vpt_median: float, resnet_median: float) -> tuple[str, str]:
    """The pre-registered reading: the sign of the paired interval, and nothing else."""
    median, low, high = (interval["median"] * 100, interval["low"] * 100,
                         interval["high"] * 100)
    if interval["low"] > 0:
        label, sentence = "helps", (
            f"Helps: deep prompts remove {median:+.1f} points more review than the ResNet-18 "
            f"on the same candidates, 95% interval {low:+.1f} to {high:+.1f} points, entirely "
            "above zero.")
    elif interval["high"] < 0:
        label, sentence = "worse", (
            f"Worse: deep prompts remove {median:+.1f} points of review against the ResNet-18, "
            f"95% interval {low:+.1f} to {high:+.1f} points, entirely below zero.")
    else:
        label, sentence = "indistinguishable", (
            f"Indistinguishable: the paired 95% interval on the difference is {low:+.1f} to "
            f"{high:+.1f} points and contains zero, so sixty test images cannot separate "
            "adapting this backbone from the ResNet-18.")
    if max(vpt_median, resnet_median) < crops.NOT_AN_ORDERING:
        sentence += (
            f" Both are under {crops.NOT_AN_ORDERING:.0%} review removed, against the detector's "
            f"own {crops.PUBLISHED_DETECTOR_REVIEW:.1%} on this queue: neither ordering is one.")
    return label, sentence


def one_seed(trainval: PatchSet, test: PatchSet, seed: int, args, device, log=print) -> dict:
    """One VPT seed, read in both columns, with its test scores kept for the bootstrap."""
    from torch.utils.data import DataLoader

    names = list(trainval.label_names)
    false_call_index = names.index("false_call")
    train_data, val_data = vpt._datasets(trainval, seed)
    backbone = probe.load_backbone(device)
    before = vpt.backbone_digest(backbone)
    model = vpt.seeded_model(backbone, len(names), args.prompts, seed, log_every=50, log=log)
    log(f"\n=== seed {seed}: {vpt.trainable_parameters(model):,} trainable parameters ===")
    began = time.perf_counter()
    model, history, _best = fit(
        train_data, val_data, trainval, names, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, seed=seed, device=device, escape_budget=args.budget, model=model, log=log)
    minutes = (time.perf_counter() - began) / 60

    val_p, val_y = predict(model, DataLoader(val_data, batch_size=args.batch_size), device)
    test_p, test_y = predict(model, DataLoader(CandidateDataset(test, augment=False),
                                               batch_size=args.batch_size), device)
    if not np.array_equal(test_y, test.labels):
        raise SystemExit("the test scores are not row-aligned with the test patches; the "
                         "bootstrap pairs by row")
    row = {
        "seed": seed, "minutes": round(minutes, 1),
        "trainable_parameters": vpt.trainable_parameters(model),
        "backbone_unchanged": vpt.backbone_digest(model.backbone) == before,
        "best_epoch": max(history, key=lambda h: h["val_review_reduction"])["epoch"],
        "history": history,
        **crops.read_columns(val_p[:, false_call_index], val_y, test_p[:, false_call_index],
                             test_y, false_call_index, args.budget),
    }
    row["test_scores"] = test_p[:, false_call_index]
    log(f"  seed {seed}: oracle {crops._percent(row['oracle_review_reduction'])}  "
        f"deployable {crops._percent(row['test_review_reduction'])}  ({minutes:.0f} min)")
    return row


def render(record: dict) -> str:
    basis = record["basis"]
    rows, resnet_rows = record["seeds"], record["resnet_rows"]
    interval = record["bootstrap"]
    lines = [
        f"## {record['date']} · commit {record['commit']}",
        "",
        "### The detector's crops, adapted: deep visual prompt tuning on DINOv2 ViT-S/14",
        "",
        f"The sentence the 2026-09-16 crops entry closed on: a linear probe bounds frozen "
        f"features and not the backbone. Same candidates, same ResNet-18 checkpoints, same two "
        f"columns and the same paired bootstrap; what changes is that the backbone is adapted -- "
        f"{vpt.PROMPTS} prompt tokens before each of its 12 frozen blocks and a linear head on "
        f"the class token, {rows[0]['trainable_parameters']:,} trainable parameters, trained by "
        f"`train.fit` at AdamW {vpt.LR:g}, batch {vpt.BATCH}, {record['epochs']} epochs, all "
        f"fixed before the run. **{basis['images']} test images, {basis['candidates']} "
        f"candidates ({basis['defects']} covering a defect, {basis['false_calls']} false calls), "
        f"{basis['prevalence']:.1%} genuine defects**, so review removed cannot exceed "
        f"{1 - basis['prevalence']:.1%} on any ordering. Seeds "
        f"{', '.join(str(r['seed']) for r in rows)}, {record['wall_minutes']:.0f} min on "
        f"`{record['device']}`. `scripts/dinov2_vpt_crops.py`.",
        "",
        f"**Verdict, by the rule written before the run: {record['verdict_sentence']}**",
        "",
        "| model | seed | best epoch | oracle review removed | oracle escape | deployable review "
        "removed | deployable escape | backbone unchanged |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| DINOv2 + deep prompts | {r['seed']} | {r['best_epoch']} | "
            f"**{crops._percent(r['oracle_review_reduction'])}** | "
            f"{crops._percent(r['oracle_escape_rate'])} | "
            f"{crops._percent(r['test_review_reduction'])} | "
            f"{crops._percent(r['test_escape_rate'])} | "
            f"{'yes' if r['backbone_unchanged'] else '**no**'} |")
    for r in resnet_rows:
        lines.append(
            f"| ResNet-18 (2026-09-16 checkpoints) | {r['seed']} | — | "
            f"**{crops._percent(r['oracle_review_reduction'])}** | "
            f"{crops._percent(r['oracle_escape_rate'])} | "
            f"{crops._percent(r['test_review_reduction'])} | "
            f"{crops._percent(r['test_escape_rate'])} | — |")
    lines += [
        f"| DINOv2 + deep prompts | median | — | "
        f"**{record['vpt_median_oracle']:.1%}** | — | — | — | — |",
        f"| ResNet-18 | median | — | **{record['resnet_median_oracle']:.1%}** | — | — | — | — |",
        f"| DINOv2 linear probe, three seeds (2026-09-16) | — | — | "
        f"**{PROBE_ORACLE_MIN:.1%}–{PROBE_ORACLE_MAX:.1%}** | — | — | — | — |",
        "",
        f"**The paired bootstrap.** {interval['resamples']:,} resamples of the "
        f"{basis['images']} test images with replacement, both orderings recomputed on the same "
        f"resampled candidates, difference in review removed at the budget (deep prompts − "
        f"ResNet-18): median {interval['median'] * 100:+.1f} points, 95% interval "
        f"{interval['low'] * 100:+.1f} to {interval['high'] * 100:+.1f}. Read on seed "
        f"{record['bootstrap_seed_read']}, whose oracle is the median of the seeds, against that "
        f"same seed's ResNet-18 -- the two arms share its by-image validation split.",
        "",
        "**What this does not establish.** Deep prompts only, at one prompt length, one learning "
        "rate and one batch size, none searched; shallow prompts, LoRA and full fine-tuning are "
        "untested. The smallest backbone. Three seeds. And the evidence itself is thin: sixty "
        "test images and about two escapes at the budget, which is why the verdict is an "
        "interval rather than a difference of medians, and why it can rule out a large effect "
        "and nothing smaller. The candidates come from the 600 px detector at its confidence "
        "floor and the training crops are its in-sample boxes, as in the 2026-08-28 entry. "
        "**None of these figures may be read against the DeepPCB curve**: different front end, "
        "different prevalence, and a ceiling of "
        f"{1 - basis['prevalence']:.1%} against that curve's 100%.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patches", type=Path, default=PATCHES)
    parser.add_argument("--resnet-root", type=Path, default=crops.RESNET_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=vpt.PUBLISHED_SEEDS)
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=vpt.BATCH)
    parser.add_argument("--lr", type=float, default=vpt.LR)
    parser.add_argument("--prompts", type=int, default=vpt.PROMPTS)
    parser.add_argument("--budget", type=float, default=DEFAULTS["escape_budget"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="time this many training steps, project the run, and stop")
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "docs" / "benchmarks.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from aoi_agent.vision.model import select_device

    device = select_device(args.device)
    trainval = PatchSet.load(args.patches / "trainval.npz")

    if args.max_steps:
        projection = vpt.speed_test(trainval, args, device)
        print(json.dumps(projection))
        return 0 if projection["fits"] else 3

    if not args.dry_run and not publishable(args.seeds, args.epochs, args.prompts, args.lr,
                                            args.batch_size):
        print("only the pre-registered configuration (seeds 0 1 2, ten epochs, 10 prompts, "
              "lr 1e-3, batch 64) may append; pass --dry-run", file=sys.stderr)
        return 2

    test = PatchSet.load(args.patches / "test.npz")
    basis = crops.check_basis(test)
    names = list(test.label_names)
    false_call_index = names.index("false_call")
    started = time.perf_counter()

    print(f"classes: {names}; {basis['candidates']} test candidates, "
          f"{basis['prevalence']:.1%} defects, ceiling {1 - basis['prevalence']:.1%}")

    rows = [one_seed(trainval, test, seed, args, device) for seed in args.seeds]
    if not all(r["backbone_unchanged"] for r in rows):
        print("a backbone tensor changed during training; this is not frozen VPT", file=sys.stderr)
        return 4

    print("\nResNet-18 arm, from the 2026-09-16 checkpoints")
    resnet_rows, resnet_test_scores = [], {}
    for seed in args.seeds:
        out_dir = args.resnet_root / crops.RESNET_DIR.format(seed=seed)
        if not (out_dir / "reverifier.pt").exists():
            raise SystemExit(f"{out_dir.relative_to(ROOT)} does not exist; this run reads the "
                             "published checkpoints and does not retrain them")
        val_scores, val_labels, test_scores = crops.resnet_scores(
            seed, trainval, test, out_dir, false_call_index, device)
        resnet_test_scores[seed] = test_scores
        resnet_rows.append({"seed": seed, **crops.read_columns(
            val_scores, val_labels, test_scores, test.labels, false_call_index, args.budget)})
        print(f"  seed {seed}: oracle {crops._percent(resnet_rows[-1]['oracle_review_reduction'])}")

    read_seed = median_seed(rows)
    interval = crops.paired_bootstrap(
        [r for r in rows if r["seed"] == read_seed][0]["test_scores"],
        resnet_test_scores[read_seed], test.labels, test.image_index, false_call_index,
        args.budget)
    vpt_median = crops._median(rows, "oracle_review_reduction")
    resnet_median = crops._median(resnet_rows, "oracle_review_reduction")
    label, sentence = verdict(interval, vpt_median, resnet_median)

    for row in rows:
        row.pop("test_scores", None)

    record = {
        "date": datetime.now(UTC).date().isoformat(), "commit": probe.commit(),
        "device": str(device), "hub_ref": probe.HUB_REF,
        "weights_sha256": probe.sha256(probe.weights_path()),
        "prompts": args.prompts, "lr": args.lr, "batch": args.batch_size, "epochs": args.epochs,
        "budget": args.budget, "basis": basis,
        "wall_minutes": (time.perf_counter() - started) / 60,
        "bootstrap": interval, "bootstrap_seed_read": read_seed,
        "vpt_median_oracle": vpt_median, "resnet_median_oracle": resnet_median,
        "verdict": label, "verdict_sentence": sentence,
        "seeds": rows, "resnet_rows": resnet_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, default=float) + "\n")

    body = render(record)
    print("\n" + body)
    if args.dry_run:
        print("(dry run -- nothing appended)")
        return 0
    existing = args.benchmarks.read_text()
    args.benchmarks.write_text(existing.rstrip() + "\n\n" + body.rstrip() + "\n")
    print(f"appended to {args.benchmarks.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
