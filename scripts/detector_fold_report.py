"""Does the crop re-verifier's threshold miss the budget because its boxes are in-sample?

Every crop entry so far carries the same caveat: the training crops are the
detector's boxes on images the detector was trained on, and the test crops are
not. On 2026-09-16 all three seeds' validation-chosen thresholds missed the
≤0.50% escape budget on test, worst 0.67%, and that caveat is the standing
explanation. It has never been measured, because there was one detector and it
had seen everything it drew on.

This measures it with the variable on its own. One detector is trained on half
the boards (`--base-stems`, the split here is by base stem and half by
*boards*, not by images), and then draws boxes three times:

* **in-sample** -- the boards it trained on, which is the published recipe;
* **out-of-sample** -- the other half, which it has never seen;
* **test** -- the 60 test images, unseen by it either way, and the *same*
  candidates for both arms.

Two arms of three seeds each, `scripts/train.py` unchanged, the two columns the
crop entries already use. The only difference between the arms is whose boards
the training boxes were drawn on. The detector has half the usual training data,
which makes it a weaker detector than the published one -- equally so for both
arms, and a reason none of these figures may be read against the published
tables.

**The verdict, written before the run.** Read on the deployable column -- the
threshold chosen on each seed's own by-image validation split by the interval's
upper bound (`threshold_cv.choose`), test read once -- as the median of three
seeds' escape rate on test:

* **the in-sample boxes were the shift** -- out-of-sample holds the budget
  (median escape ≤ 0.50%) and in-sample does not;
* **part of it** -- out-of-sample's median escape is lower than in-sample's but
  still over the budget;
* **not the shift** -- out-of-sample is not lower.

Expected before the run: *part of it*, or three seeds spread too widely to read,
because the budget is about two escapes on this queue and the arms differ in
size as well. Both are reported as they come.

    uv run python scripts/detector_fold_report.py --dry-run --seeds 0 --epochs 1 --detector-epochs 1
    uv run python scripts/detector_fold_report.py            # the pre-registered run (~1.2 h)
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_detector_patches as crops_build  # noqa: E402
import dinov2_crops_report as crops  # noqa: E402
import dinov2_probe as probe  # noqa: E402
from train import DEFAULTS  # noqa: E402

from aoi_agent.data import pcbaoi  # noqa: E402
from aoi_agent.vision.detector import CONF_FLOOR  # noqa: E402
from aoi_agent.vision.patches import PatchSet  # noqa: E402

PATCHES = ROOT / "data" / "patches_pcbaoi_fold"
CHECKPOINT = ROOT / "models" / "detector_pcbaoi_fold.pt"
OUT_JSON = ROOT / "models" / "detector_fold.json"
RESNET_DIR = "fold_{arm}_seed{seed}"
PUBLISHED_SEEDS = [0, 1, 2]
ARMS = ("in_sample", "out_of_sample")

#: Half the boards, so the two arms are comparable in size. The detector's own
#: 80/20 split then runs inside the half it is given, which is why the in-sample
#: arm is the boards it actually trained on rather than everything it was shown.
FOLD_SHARE = 0.5
FOLD_SEED = 20260920
DETECTOR_SEED = 20260826

#: The worst escape any 2026-09-16 seed's deployable threshold produced on test
#: -- the row this run exists to explain -- and the budget both are read
#: against. `tests/test_detector_fold.py` holds them against docs/benchmarks.md.
PUBLISHED_WORST_ESCAPE = 0.0067
BUDGET = DEFAULTS["escape_budget"]

#: Past this, the arms are not the same experiment with one variable changed.
SIZE_TOLERANCE = 0.20


def fold_stems(seed: int = FOLD_SEED, share: float = FOLD_SHARE) -> tuple[list[str], list[str]]:
    """The two halves of the training boards, by base stem."""
    items = pcbaoi.load("train_data_augmentation")
    first, second = pcbaoi.split_by_stem(items, validation_share=share, seed=seed)
    return (sorted({i.base_stem for i in first}), sorted({i.base_stem for i in second}))


def matched(unseen: list[str], to: int, seed: int = FOLD_SEED) -> list[str]:
    """As many unseen boards as the in-sample arm has, drawn deterministically.

    The detector keeps a fifth of the half it is given for its own validation,
    so the boards it trained on are fewer than the boards it never saw. Handing
    the out-of-sample arm all of them makes the arms differ in size as well as
    in provenance, and this experiment has exactly one variable to spend.
    """
    if to >= len(unseen):
        return sorted(unseen)
    shuffled = sorted(unseen)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:to])


def trained_on(fold_one: list[str], seed: int = DETECTOR_SEED) -> list[str]:
    """Of the half the detector was given, the boards it actually trained on.

    `train_detector.py` splits what it is handed 80/20 by stem for validation,
    so the in-sample arm is that 80 -- the 20 it scored against is neither
    in-sample nor unseen, and belongs in neither arm.
    """
    items = [i for i in pcbaoi.load("train_data_augmentation") if i.base_stem in set(fold_one)]
    train, _val = pcbaoi.split_by_stem(items, seed=seed)
    return sorted({i.base_stem for i in train})


def train_detector(stems: list[str], checkpoint: Path, epochs: int, device: str | None,
                   log=print) -> None:
    """`scripts/train_detector.py` on those boards only; it refuses a busy GPU itself."""
    if checkpoint.exists():
        log(f"  {checkpoint} already trained")
        return
    listing = checkpoint.with_suffix(".stems.txt")
    listing.parent.mkdir(parents=True, exist_ok=True)
    listing.write_text("\n".join(stems) + "\n")
    command = [sys.executable, str(ROOT / "scripts" / "train_detector.py"),
               "--base-stems", str(listing), "--out", str(checkpoint), "--epochs", str(epochs)]
    if device:
        command += ["--device", device]
    log(f"  training the fold detector on {len(stems)} boards")
    began = time.perf_counter()
    if subprocess.run(command, check=False).returncode != 0:
        raise SystemExit("scripts/train_detector.py failed")
    log(f"  trained in {(time.perf_counter() - began) / 60:.1f} min")


def items_for(stems: list[str]) -> list[pcbaoi.Item]:
    """Every capture and transform of those boards, as the published build reads them."""
    wanted = set(stems)
    everything = pcbaoi.load("train_data") + pcbaoi.load("train_data_augmentation")
    return [item for item in everything if item.base_stem in wanted]


def build_arms(detector, stems: dict[str, list[str]], out: Path, floor: float,
               log=print) -> dict[str, PatchSet]:
    """One test set shared by both arms, and one training set each.

    The test crops are built once and written into both arms' directories, so
    `train.py` reads the same rows on either side and the two orderings are
    comparable candidate by candidate.
    """
    test_items = pcbaoi.load("test_data")
    for arm, arm_stems in stems.items():
        if pcbaoi.leaks(items_for(arm_stems), test_items):
            raise SystemExit(f"{arm}: a test board appears in the training fold")
    if set(stems["in_sample"]) & set(stems["out_of_sample"]):
        raise SystemExit("the two arms share a board; they would not differ in the one variable")

    def built_or_refuse(name: str, items: list[pcbaoi.Item]):
        """A set with no candidates is a detector that drew nothing, not an empty queue.

        `build` stacks the patches, so an empty arm surfaces as a numpy error
        three frames down; this says which set was empty and that the detector
        is the thing to look at.
        """
        patch_set, sidecar = crops_build.build(detector, items, floor)
        if not len(patch_set):
            raise SystemExit(f"{name}: the detector drew no box above floor {floor} on any of "
                             f"{len(items)} images, so there is nothing to train or read")
        return patch_set, sidecar

    log("  test crops (shared by both arms)")
    test_set, test_sidecar = built_or_refuse("test", test_items)
    built = {"test": test_set}
    for arm, arm_stems in stems.items():
        log(f"  {arm} crops from {len(arm_stems)} boards")
        patch_set, _sidecar = built_or_refuse(arm, items_for(arm_stems))
        (out / arm).mkdir(parents=True, exist_ok=True)
        patch_set.save(out / arm / "trainval.npz")
        test_set.save(out / arm / "test.npz")
        built[arm] = patch_set
    np.savez_compressed(out / "test_detector.npz", **test_sidecar)
    return built


def arm_rows(arm: str, trainval: PatchSet, test: PatchSet, seeds: list[int], patches: Path,
             resnet_root: Path, false_call_index: int, device, budget: float,
             log=print) -> list[dict]:
    """Three seeds of the published ResNet-18 recipe, read in both columns."""
    rows = []
    for seed in seeds:
        out_dir = resnet_root / RESNET_DIR.format(arm=arm, seed=seed)
        crops.train_resnet(seed, patches / arm, out_dir, None, log=log)
        val_scores, val_labels, test_scores = crops.resnet_scores(
            seed, trainval, test, out_dir, false_call_index, device)
        row = {"arm": arm, "seed": seed, **crops.read_columns(
            val_scores, val_labels, test_scores, test.labels, false_call_index, budget)}
        rows.append(row)
        log(f"  {arm} seed {seed}: oracle {crops._percent(row['oracle_review_reduction'])} "
            f"deployable {crops._percent(row['test_review_reduction'])} at "
            f"{crops._percent(row['test_escape_rate'])} escape")
    return rows


def size_gap(sizes: dict[str, int]) -> float:
    """How far apart the two arms' training sets are, as a share of the larger."""
    larger = max(sizes.values())
    return 0.0 if not larger else (larger - min(sizes.values())) / larger


def verdict(in_sample: float | None, out_of_sample: float | None,
            budget: float = BUDGET) -> tuple[str, str]:
    """The pre-registered reading of the two arms' median escape on test."""
    if in_sample is None or out_of_sample is None:
        return "unreadable", (
            "Unreadable: an arm found no threshold inside the budget on its validation split, "
            "so the two arms have no common column to be compared in.")
    if in_sample <= budget:
        return "nothing_to_attribute", (
            f"Nothing to attribute: the in-sample arm already holds the {budget:.2%} budget on "
            f"test at {in_sample:.2%}, so this detector's queue does not reproduce the shift "
            f"the run was built to explain, and the out-of-sample arm's {out_of_sample:.2%} "
            "says nothing about it.")
    if out_of_sample <= budget < in_sample:
        return "the_shift", (
            f"The in-sample boxes were the shift: trained on crops the detector had not seen, "
            f"the validation-chosen threshold escapes {out_of_sample:.2%} on test, inside the "
            f"{budget:.2%} budget, against {in_sample:.2%} trained on its own boards.")
    if out_of_sample < in_sample:
        return "part_of_it", (
            f"Part of it: the out-of-sample arm escapes {out_of_sample:.2%} on test against "
            f"{in_sample:.2%} for the in-sample arm, lower but still over the {budget:.2%} "
            "budget, so the boxes' provenance moves this threshold without fixing it.")
    return "not_the_shift", (
        f"Not the shift: the out-of-sample arm escapes {out_of_sample:.2%} on test against "
        f"{in_sample:.2%} for the in-sample arm, so crops from boards the detector had never "
        "seen do not carry their threshold across any better.")


def publishable(seeds: list[int], epochs: int, detector_epochs: int) -> bool:
    """Only the pre-registered configuration may append; anything else is a read."""
    return (sorted(seeds) == PUBLISHED_SEEDS and epochs == DEFAULTS["epochs"]
            and detector_epochs == 60)


def render(record: dict) -> str:
    medians = record["median_escape"]
    lines = [
        f"## {record['date']} · commit {record['commit']}",
        "",
        "### Are the crop re-verifier's thresholds shifted by boxes the detector had seen?",
        "",
        f"Every crop entry carries the caveat and none of them measured it: the training crops "
        f"are the detector's boxes on its own training images, the test crops are not, and on "
        f"2026-09-16 all three seeds' validation-chosen thresholds missed the "
        f"{record['budget']:.2%} budget on test, worst {PUBLISHED_WORST_ESCAPE:.2%}. Here one "
        f"detector is trained on **half the boards** and draws the boxes for both arms: "
        f"in-sample on the {record['boards']['in_sample']} boards it trained on, out-of-sample "
        f"on the {record['boards']['out_of_sample']} it has never seen, and the same "
        f"{len(record['test_labels'])} test candidates from the 60 test images for both. "
        f"`scripts/train.py` unchanged, three seeds an arm, {record['wall_minutes']:.0f} min. "
        f"`scripts/detector_fold_report.py`.",
        "",
        f"**Verdict, by the rule written before the run: {record['verdict_sentence']}**",
        "",
        "| arm | seed | oracle review removed | oracle escape | deployable review removed "
        "| deployable escape |",
        "|---|---|---|---|---|---|",
    ]
    for row in record["rows"]:
        flag = "" if (row["test_escape_rate"] or 0) <= record["budget"] else " (over budget)"
        lines.append(
            f"| {row['arm'].replace('_', '-')} | {row['seed']} | "
            f"**{crops._percent(row['oracle_review_reduction'])}** | "
            f"{crops._percent(row['oracle_escape_rate'])} | "
            f"{crops._percent(row['test_review_reduction'])} | "
            f"{crops._percent(row['test_escape_rate'])}{flag} |")
    for arm in ARMS:
        lines.append(f"| {arm.replace('_', '-')} | median | — | — | — | "
                     f"{crops._percent(medians[arm])} |")
    lines += [
        "",
        f"**The two arms' training sets.** in-sample {record['patches']['in_sample']:,} crops "
        f"from {record['boards']['in_sample']} boards, out-of-sample "
        f"{record['patches']['out_of_sample']:,} from {record['boards']['out_of_sample']}.",
    ]
    if record["size_gap"] > SIZE_TOLERANCE:
        lines.append(
            f"**They differ by {record['size_gap']:.0%}, past the {SIZE_TOLERANCE:.0%} this run "
            "set as the point where size is a second variable: the comparison above is not one "
            "variable changed, and the verdict is to be read with that in it.**")
    lines += [
        "",
        "**What this does not establish.** The detector is trained on half the boards, so it is "
        "weaker than the published one and its queue is its own: **none of these figures may be "
        "read against the 2026-08-28 or 2026-09-16 tables**, and none of them against the "
        "DeepPCB curve. One detector, one fold split, one seed for it. Three seeds an arm on a "
        "queue where the budget is about two escapes, which is the same thinness those entries "
        "name. And the arms differ in what the detector had seen, not in how the crops were "
        "cut: nothing here says a second acquisition of the same board would not do more.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patches", type=Path, default=PATCHES)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--resnet-root", type=Path, default=crops.RESNET_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=PUBLISHED_SEEDS)
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--detector-epochs", type=int, default=60)
    parser.add_argument("--floor", type=float, default=CONF_FLOOR)
    parser.add_argument("--budget", type=float, default=BUDGET)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "docs" / "benchmarks.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.patches.resolve() == (ROOT / "data" / "patches_pcbaoi").resolve():
        print("refusing to write over data/patches_pcbaoi: the published entries and "
              "dinov2_crops_report.py read it", file=sys.stderr)
        return 2
    if not args.dry_run and not publishable(args.seeds, args.epochs, args.detector_epochs):
        print("only the pre-registered configuration (seeds 0 1 2, ten epochs, a 60-epoch "
              "detector) may append; pass --dry-run", file=sys.stderr)
        return 2

    from aoi_agent.vision.detector import Detector
    from aoi_agent.vision.model import select_device

    device = select_device(args.device)
    started = time.perf_counter()

    fold_one, fold_two = fold_stems()
    in_sample = trained_on(fold_one)
    stems = {"in_sample": in_sample, "out_of_sample": matched(fold_two, len(in_sample))}
    print(f"fold: {len(fold_one)} boards to the detector, {len(fold_two)} held out; "
          f"arms of {len(stems['in_sample'])} and {len(stems['out_of_sample'])} boards")

    train_detector(fold_one, args.checkpoint, args.detector_epochs, args.device)
    detector = Detector(args.checkpoint, device=args.device)
    args.patches.mkdir(parents=True, exist_ok=True)
    built = build_arms(detector, stems, args.patches, args.floor)
    test = built["test"]
    names = list(test.label_names)
    false_call_index = names.index(crops_build.FALSE_CALL)

    rows: list[dict] = []
    for arm in ARMS:
        rows += arm_rows(arm, built[arm], test, args.seeds, args.patches, args.resnet_root,
                         false_call_index, device, args.budget)

    medians = {arm: crops._median([r for r in rows if r["arm"] == arm], "test_escape_rate")
               if any(r["test_escape_rate"] is not None for r in rows if r["arm"] == arm)
               else None
               for arm in ARMS}
    label, sentence = verdict(medians["in_sample"], medians["out_of_sample"], args.budget)
    patches = {arm: len(built[arm]) for arm in ARMS}

    record = {
        "date": datetime.now(UTC).date().isoformat(), "commit": probe.commit(),
        "device": str(device), "budget": args.budget, "floor": args.floor,
        "fold_seed": FOLD_SEED, "fold_share": FOLD_SHARE, "detector_epochs": args.detector_epochs,
        "boards": {arm: len(stems[arm]) for arm in ARMS},
        "patches": patches, "size_gap": size_gap(patches),
        "test_labels": test.labels.tolist(),
        "median_escape": medians, "verdict": label, "verdict_sentence": sentence, "rows": rows,
        "wall_minutes": (time.perf_counter() - started) / 60,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, default=float) + "\n")

    body = render(record)
    print("\n" + body)
    if args.dry_run:
        print("(dry run -- nothing appended)")
        return 0
    args.benchmarks.write_text(args.benchmarks.read_text().rstrip() + "\n\n" + body.rstrip() + "\n")
    print(f"appended to {args.benchmarks.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
