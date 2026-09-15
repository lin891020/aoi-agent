"""Stronger pretrained features: DINOv2 ViT-S/14, frozen, with a linear probe.

The re-verifier is an ImageNet ResNet-18, fine-tuned end to end. This asks the
next question a reader asks about any backbone choice: with pretrained features
stronger than ImageNet's, does re-verification remove more of the review queue
at the same escape budget? It is read on the same curve as the ResNet, by the
same procedure, so the answer is a row beside the 2026-08-31 seed-variance
table and not a separate claim.

**Same procedure, piece by piece.** The same 12,634 trainval and 7,322 test
candidates from `data/patches/`. The threshold is chosen out-of-fold over
trainval with `threshold_cv`'s own `folds_by_image`, `inner_split` and
`choose` (the Wilson upper bound at the budget), then read once on test. The
probe reads `P(false_call)` from a 7-class softmax, as the ResNet does, rather
than being given the easier binary problem. Its one hyper-parameter, the
inverse regularisation `C`, is chosen on the by-image inner validation split by
review removed at the budget, the way the network's epoch and the tree's
iteration count are. Three seeds.

**Inputs.** A patch is (template, test, |difference|), 64 px, binarised. Two
ways to show it to a network pretrained on natural images, both upsampled to
224 (its training resolution; 64 px would be 4x4 tokens) with bilinear
interpolation, fixed here and not swept:

    per_channel  each of the three as a grey image, features concatenated
    stacked      the three as one RGB image, exactly as the ResNet eats them

**Poolings**, all from the same forward: the CLS token, the mean of the patch
tokens, the two concatenated (DINOv2's own linear evaluation), and the mean of
the centre 4x4 tokens (the candidate is centred in its patch, and frozen ViT
CLS tokens are known to lose small low-contrast signal that local tokens keep).
Eight configurations. **Which one is DINOv2's result is chosen on out-of-fold
results only**; the other seven are read on test and printed as exploration.

**Seeds move less here than for the ResNet.** Frozen features are
deterministic and the probe is convex, so a seed moves only the folds and the
split. A three-seed range here and the ResNet's five-seed range are not the
same kind of interval.

**Pre-registered verdict** -- written before the first run and not edited
after it. Comparators are the published figures: the ResNet's deployed column
over five seeds, 49.00%-55.59% review removed with test escapes
0.331%-0.663% (2026-08-31 seed variance entry), and the feature floor's
16.70% (2026-09-01).

    helps        median review removed > 55.59% AND median test escape <= 0.663%
    no help      median review removed within 49.00%-55.59%, or above it with
                 median test escape > 0.663%
    worse        median review removed < 49.00%
    below floor  median review removed < 16.70%

The oracle column answers whether the probe is a better model; it does not
move the verdict.

    uv run python scripts/dinov2_probe.py                      # three seeds, appends
    uv run python scripts/dinov2_probe.py --seeds 0 --dry-run  # a shorter read

Caches features under `data/features/`; writes `models/dinov2_probe.json`;
appends to docs/benchmarks.md unless `--dry-run`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from threshold_cv import choose, folds_by_image, indices_for, inner_split  # noqa: E402
from train import DEFAULTS, split_by_image  # noqa: E402

from aoi_agent.stats import wilson  # noqa: E402
from aoi_agent.vision.operating_point import (  # noqa: E402
    OperatingPoint,
    best_at_escape_budget,
    sweep,
)
from aoi_agent.vision.patches import PatchSet  # noqa: E402

HUB_REPO = "facebookresearch/dinov2"
#: Pinned so the code that builds the model is the code this entry was run on.
HUB_REF = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
MODEL_NAME = "dinov2_vits14"
WEIGHTS_FILE = "dinov2_vits14_pretrain.pth"
INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

INPUTS = ("per_channel", "stacked")
POOLINGS = ("cls", "mean", "cls_mean", "centre")
CENTRE = 4
C_GRID = (0.01, 0.1, 1.0, 10.0)
MAX_ITER = 3000

#: The comparators, as published. `tests/test_dinov2_probe.py` holds them
#: against docs/benchmarks.md so they cannot drift from the runs they name.
RESNET_DEPLOYED_MIN = 0.4900
RESNET_DEPLOYED_MAX = 0.5559
RESNET_DEPLOYED_MEDIAN = 0.5090
RESNET_ORACLE_MIN = 0.4973
RESNET_ORACLE_MAX = 0.5279
RESNET_ESCAPE_MIN = 0.00331
RESNET_ESCAPE_MAX = 0.00663
FEATURE_FLOOR = 0.1670
PUBLISHED_SEEDS = [0, 1, 2]


# --------------------------------------------------------------------------
# input and pooling


def prepare(batch: np.ndarray, mode: str):
    """uint8 (n, 3, 64, 64) patches -> the normalised 224 px tensor DINOv2 reads.

    ``per_channel`` returns 3n images in the order template, test, difference
    for each patch; ``stacked`` returns n.
    """
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.ascontiguousarray(batch)).float().div_(255.0)
    if mode == "per_channel":
        n, _, h, w = x.shape
        x = x.reshape(n * 3, 1, h, w).expand(-1, 3, -1, -1)
    elif mode != "stacked":
        raise ValueError(f"unknown input mode {mode!r}")
    x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear",
                      align_corners=False, antialias=False)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def _centre_slice(tokens_per_side: int) -> slice:
    start = (tokens_per_side - CENTRE) // 2
    return slice(start, start + CENTRE)


def pool(cls: np.ndarray, patch_tokens: np.ndarray) -> dict[str, np.ndarray]:
    """The four poolings of one forward. ``patch_tokens`` is (m, side*side, d)."""
    m, count, d = patch_tokens.shape
    side = int(round(count ** 0.5))
    if side * side != count:
        raise ValueError(f"{count} patch tokens is not a square grid")
    grid = patch_tokens.reshape(m, side, side, d)
    centre = _centre_slice(side)
    mean = patch_tokens.mean(axis=1)
    return {
        "cls": cls,
        "mean": mean,
        "cls_mean": np.concatenate([cls, mean], axis=1),
        "centre": grid[:, centre, centre, :].mean(axis=(1, 2)),
    }


def pool_torch(cls, patch_tokens, pooling: str):
    """One pooling on the device, for the latency path. Must equal `pool`."""
    import torch

    m, count, d = patch_tokens.shape
    side = int(round(count ** 0.5))
    if pooling == "cls":
        return cls
    if pooling == "mean":
        return patch_tokens.mean(dim=1)
    if pooling == "cls_mean":
        return torch.cat([cls, patch_tokens.mean(dim=1)], dim=1)
    if pooling == "centre":
        centre = _centre_slice(side)
        return patch_tokens.reshape(m, side, side, d)[:, centre, centre, :].mean(dim=(1, 2))
    raise ValueError(f"unknown pooling {pooling!r}")


def extract(patches: np.ndarray, backbone, mode: str, device, batch_size: int = 64,
            log=print) -> dict[str, np.ndarray]:
    """Every pooling of every patch, as float32 (n, k*d) -- k=3 for per_channel."""
    import torch

    per_patch = 3 if mode == "per_channel" else 1
    out: dict[str, list[np.ndarray]] = {name: [] for name in POOLINGS}
    total = len(patches)
    for start in range(0, total, batch_size):
        chunk = patches[start:start + batch_size]
        x = prepare(chunk, mode).to(device)
        with torch.no_grad():
            features = backbone.forward_features(x)
        pooled = pool(features["x_norm_clstoken"].float().cpu().numpy(),
                      features["x_norm_patchtokens"].float().cpu().numpy())
        for name, values in pooled.items():
            out[name].append(values.reshape(len(chunk), per_patch * values.shape[1]))
        done = min(start + batch_size, total)
        if done % (batch_size * 40) == 0 or done == total:
            log(f"  {mode}: {done}/{total}")
    return {name: np.concatenate(parts).astype(np.float32) for name, parts in out.items()}


def load_backbone(device):
    import torch

    backbone = torch.hub.load(f"{HUB_REPO}:{HUB_REF}", MODEL_NAME, trust_repo=True,
                              verbose=False)
    return backbone.eval().to(device)


def weights_path() -> Path:
    import torch

    return Path(torch.hub.get_dir()) / "checkpoints" / WEIGHTS_FILE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_features(split: str, patch_set: PatchSet, mode: str, cache_dir: Path,
                    device, batch_size: int, log=print) -> tuple[dict[str, np.ndarray], float]:
    """Features for one split and input mode, extracted once and cached.

    The cache is refused and rebuilt when it was made from a different number
    of candidates, a different hub commit or a different input size -- a stale
    cache would be a different experiment reported as this one.
    """
    path = cache_dir / f"{MODEL_NAME}_{mode}_{split}.npz"
    if path.exists():
        data = np.load(path, allow_pickle=False)
        if (int(data["n"]) == len(patch_set) and str(data["hub_ref"]) == HUB_REF
                and int(data["input_size"]) == INPUT_SIZE):
            log(f"  {path.relative_to(ROOT)} (cached)")
            return {name: data[name].astype(np.float32) for name in POOLINGS}, 0.0
        log(f"  {path.relative_to(ROOT)} does not match this run; rebuilding")
    backbone = load_backbone(device)
    began = time.perf_counter()
    features = extract(patch_set.patches, backbone, mode, device, batch_size, log)
    wall = time.perf_counter() - began
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, n=len(patch_set), hub_ref=HUB_REF, input_size=INPUT_SIZE,
             **{name: values.astype(np.float16) for name, values in features.items()})
    return features, wall


# --------------------------------------------------------------------------
# the probe and the procedure


def inverse_frequency(labels: np.ndarray, n_classes: int) -> dict[int, float]:
    """The weights `vision.dataset.class_weights` gives the network, as sklearn wants them."""
    counts = np.bincount(labels, minlength=n_classes)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return {i: float(w) for i, w in enumerate(weights) if counts[i]}


def fit_probe(x: np.ndarray, y: np.ndarray, c: float, seed: int, n_classes: int):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight=inverse_frequency(y, n_classes),
                           max_iter=MAX_ITER, random_state=seed),
    )
    model.fit(x, y)
    return model


def probabilities(model, x: np.ndarray, n_classes: int) -> np.ndarray:
    """(n, n_classes), with a zero column for any class the fit never saw."""
    partial = model.predict_proba(x)
    full = np.zeros((len(x), n_classes), dtype=np.float32)
    full[:, model.classes_] = partial
    return full


def choose_c(x: np.ndarray, labels: np.ndarray, train_idx, val_idx, *, false_call_index: int,
             budget: float, seed: int, n_classes: int):
    """(C, the probe fitted with it) -- the C that removes most review on validation.

    A C with no threshold inside the budget on validation scores below every C
    that has one. Ties go to the smaller C, the more regularised probe.
    """
    best: tuple[float, float, object] | None = None
    for c in C_GRID:
        model = fit_probe(x[train_idx], labels[train_idx], c, seed, n_classes)
        scores = probabilities(model, x[val_idx], n_classes)[:, false_call_index]
        point = best_at_escape_budget(sweep(scores, labels[val_idx], false_call_index), budget)
        review = -1.0 if point is None else point.review_reduction
        if best is None or review > best[0]:
            best = (review, c, model)
    return best[1], best[2]


def oof_select(x: np.ndarray, patch_set: PatchSet, *, folds: int, budget: float, seed: int,
               log=print):
    """`threshold_cv.select_threshold`'s shape, with the probe where the network was.

    Returns (out-of-fold probabilities, point-estimate choice, upper-bound
    choice, the C chosen in each fold).
    """
    labels = patch_set.labels
    names = patch_set.label_names
    false_call_index = names.index("false_call")
    n_classes = len(names)
    oof = np.zeros((len(labels), n_classes), dtype=np.float32)
    covered = np.zeros(len(labels), dtype=bool)
    chosen_c = []

    for k, held_out in enumerate(folds_by_image(patch_set, folds, seed), 1):
        held = set(held_out)
        rest = np.array([i for i in np.unique(patch_set.image_index) if i not in held])
        inner_train, inner_val = inner_split(rest, DEFAULTS["val_fraction"], seed)
        train_idx = indices_for(patch_set, inner_train)
        val_idx = indices_for(patch_set, inner_val)
        out_idx = indices_for(patch_set, held)
        c, model = choose_c(x, labels, train_idx, val_idx, false_call_index=false_call_index,
                            budget=budget, seed=seed, n_classes=n_classes)
        oof[out_idx] = probabilities(model, x[out_idx], n_classes)
        covered[out_idx] = True
        chosen_c.append(c)
        log(f"    fold {k}/{folds}: C={c}  held out {len(out_idx)}")

    assert covered.all(), "every candidate must be predicted by a probe that did not train on it"
    optimistic, guarded = choose(sweep(oof[:, false_call_index], labels, false_call_index), budget)
    return oof, optimistic, guarded, chosen_c


def select_configuration(oof_guarded: dict[tuple[str, str], OperatingPoint | None]):
    """The configuration that is DINOv2's result, from out-of-fold choices alone.

    Takes nothing from test, by signature. The most review removed at the
    upper-bound threshold wins; ties keep the earlier configuration in
    `INPUTS` x `POOLINGS` order.
    """
    best_key, best_review = None, -1.0
    for key, point in oof_guarded.items():
        if point is not None and point.review_reduction > best_review:
            best_key, best_review = key, point.review_reduction
    return best_key


def class_escapes(scores: np.ndarray, labels: np.ndarray, names: list[str],
                  threshold: float) -> dict[str, list[int]]:
    """[escaped, total] per defect class at one threshold, as `class_escape_report` counts."""
    dismissed = scores >= threshold
    out = {}
    for index, name in enumerate(names):
        if name == "false_call":
            continue
        mask = labels == index
        out[name] = [int((dismissed & mask).sum()), int(mask.sum())]
    return out


def read_test(x_trainval: np.ndarray, trainval: PatchSet, x_test: np.ndarray, test: PatchSet,
              threshold: float, *, seed: int, budget: float) -> dict:
    """The final probe on this seed's own by-image split, read once at `threshold`."""
    names = trainval.label_names
    false_call_index = names.index("false_call")
    n_classes = len(names)
    train_idx, val_idx = split_by_image(trainval, DEFAULTS["val_fraction"], seed)
    c, model = choose_c(x_trainval, trainval.labels, train_idx, val_idx,
                        false_call_index=false_call_index, budget=budget, seed=seed,
                        n_classes=n_classes)
    scores = probabilities(model, x_test, n_classes)[:, false_call_index]
    point = sweep(scores, test.labels, false_call_index, thresholds=np.array([threshold]))[0]
    low, high = wilson(point.escapes, point.defects_total)
    oracle = best_at_escape_budget(sweep(scores, test.labels, false_call_index), budget)
    return {
        "final_c": c,
        "test_escape_rate": point.escape_rate,
        "test_escapes": int(point.escapes),
        "test_defects": int(point.defects_total),
        "test_escape_ci": [low, high],
        "test_review_reduction": point.review_reduction,
        "oracle_threshold": None if oracle is None else oracle.threshold,
        "oracle_review_reduction": None if oracle is None else oracle.review_reduction,
        "class_escapes": class_escapes(scores, test.labels, names, threshold),
    }


def one_seed(features: dict, trainval: PatchSet, test: PatchSet, seed: int, args,
             log=print) -> dict:
    log(f"\n=== seed {seed} ===")
    configs: dict[tuple[str, str], dict] = {}
    for mode in args.inputs:
        for pooling in args.poolings:
            x = features[mode]["trainval"][pooling]
            log(f"  {mode}/{pooling}: out-of-fold over {args.folds} folds, {x.shape[1]} dims")
            began = time.perf_counter()
            _oof, optimistic, guarded, cs = oof_select(
                x, trainval, folds=args.folds, budget=args.budget, seed=seed,
                log=lambda line: log("  " + line))
            configs[(mode, pooling)] = {
                "input": mode, "pooling": pooling, "dims": int(x.shape[1]), "fold_c": cs,
                "guarded": guarded, "optimistic": optimistic,
                "oof_seconds": round(time.perf_counter() - began, 1),
            }

    chosen = select_configuration({key: value["guarded"] for key, value in configs.items()})
    rows = []
    for key, config in configs.items():
        guarded = config.pop("guarded")
        config.pop("optimistic")
        if guarded is None:
            rows.append({**config, "threshold": None, "chosen": key == chosen})
            continue
        read = read_test(features[key[0]]["trainval"][key[1]], trainval,
                         features[key[0]]["test"][key[1]], test, guarded.threshold,
                         seed=seed, budget=args.budget)
        rows.append({**config, **read, "chosen": key == chosen,
                     "threshold": guarded.threshold,
                     "oof_escape_rate": guarded.escape_rate,
                     "oof_review_reduction": guarded.review_reduction})
        log(f"  {key[0]}/{key[1]}: thr {guarded.threshold:.4f}  oof review "
            f"{guarded.review_reduction:.2%}  test review {read['test_review_reduction']:.2%} "
            f"at {read['test_escape_rate']:.3%}{'  <- chosen' if key == chosen else ''}")
    return {"seed": seed, "chosen": None if chosen is None else "/".join(chosen), "configs": rows}


def chosen_rows(seeds: list[dict]) -> list[dict]:
    out = []
    for record in seeds:
        for row in record["configs"]:
            if row["chosen"]:
                out.append({"seed": record["seed"], **row})
    return out


def verdict(rows: list[dict]) -> tuple[str, str]:
    """The pre-registered reading of the chosen configuration's medians."""
    review = statistics.median(r["test_review_reduction"] for r in rows)
    escape = statistics.median(r["test_escape_rate"] for r in rows)
    if review < FEATURE_FLOOR:
        return "below_floor", (
            f"Below the feature floor: median review removed {review:.2%} is under "
            f"the hand-feature tree's {FEATURE_FLOOR:.2%}.")
    if review < RESNET_DEPLOYED_MIN:
        return "worse", (
            f"Worse than the ResNet-18: median review removed {review:.2%} is below "
            f"the lowest of its five seeds, {RESNET_DEPLOYED_MIN:.2%}.")
    if review > RESNET_DEPLOYED_MAX and escape <= RESNET_ESCAPE_MAX:
        return "helps", (
            f"Helps: median review removed {review:.2%} is above the ResNet-18's best "
            f"seed ({RESNET_DEPLOYED_MAX:.2%}), at a median test escape of {escape:.3%}, "
            f"within its worst ({RESNET_ESCAPE_MAX:.3%}).")
    if review > RESNET_DEPLOYED_MAX:
        return "no_help", (
            f"No help: median review removed {review:.2%} is above the ResNet-18's best "
            f"seed, but the median test escape {escape:.3%} is above its worst "
            f"({RESNET_ESCAPE_MAX:.3%}), so the queue it removes is bought with escapes.")
    return "no_help", (
        f"No help: median review removed {review:.2%} is inside the ResNet-18's "
        f"five-seed range, {RESNET_DEPLOYED_MIN:.2%}-{RESNET_DEPLOYED_MAX:.2%}.")


def publishable(seeds: list[int], folds: int, inputs, poolings) -> bool:
    """Only the full pre-registered run may append; anything shorter is a read."""
    return (sorted(seeds) == PUBLISHED_SEEDS and folds == 5
            and tuple(inputs) == INPUTS and tuple(poolings) == POOLINGS)


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _range(values: list[float], fmt: str) -> str:
    return f"{format(min(values), fmt)}–{format(max(values), fmt)}"


def render(record: dict) -> str:
    rows = chosen_rows(record["seeds"])
    _label, sentence = verdict(rows)
    reviews = [r["test_review_reduction"] for r in rows]
    escapes = [r["test_escape_rate"] for r in rows]
    oracles = [r["oracle_review_reduction"] for r in rows if r["oracle_review_reduction"] is not None]
    lines = [
        f"## {record['date']} · commit {record['commit']}",
        "",
        "### Stronger pretrained features: DINOv2 ViT-S/14, frozen, with a linear probe",
        "",
        "Testing stronger pretrained features on the same curve as the re-verifier. "
        f"`{MODEL_NAME}` from `{HUB_REPO}` at `{HUB_REF[:12]}`, weights sha256 "
        f"`{record['weights_sha256'][:16]}`, frozen; a 7-class logistic regression on "
        "standardised features reads `P(false_call)`. Same 12,634 trainval and 7,322 "
        "test candidates, threshold chosen out-of-fold over trainval by the Wilson upper "
        "bound (`threshold_cv.choose`), test read once. The probe's `C` is chosen on the "
        "by-image inner validation split. Patches upsampled 64 -> 224 bilinear. "
        f"Seeds {', '.join(str(s) for s in record['seed_list'])}. "
        f"{record['wall_seconds'] / 60:.0f} min. `scripts/dinov2_probe.py`.",
        "",
        f"**Verdict, by the rule written before the run: {sentence}**",
        "",
        "Eight configurations -- two inputs (`per_channel`: template, test and difference "
        "each as a grey image, features concatenated; `stacked`: the three as one RGB "
        "image, as the ResNet reads them) by four poolings (CLS, patch mean, both, "
        "centre 4x4 tokens). Which one is DINOv2's result was chosen per seed on "
        "out-of-fold review removed, never on test.",
        "",
        "| seed | chosen configuration | threshold | out-of-fold escape | test escape | "
        "95% interval | test review removed | oracle on this split | short | open |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        short = r["class_escapes"]["short"]
        opened = r["class_escapes"]["open"]
        lines.append(
            f"| {r['seed']} | {r['input']}/{r['pooling']} | {r['threshold']:.4f} | "
            f"{r['oof_escape_rate']:.3%} | {r['test_escape_rate']:.3%} "
            f"({r['test_escapes']}/{r['test_defects']}) | "
            f"{r['test_escape_ci'][0]:.2%}–{r['test_escape_ci'][1]:.2%} | "
            f"**{r['test_review_reduction']:.2%}** | "
            f"{'—' if r['oracle_review_reduction'] is None else format(r['oracle_review_reduction'], '.2%')} | "
            f"{short[0] / short[1]:.2%} | {opened[0] / opened[1]:.2%} |")
    lines += [
        f"| **median** | — | — | — | **{statistics.median(escapes):.3%}** | — | "
        f"**{statistics.median(reviews):.2%}** | "
        f"{statistics.median(oracles):.2%} | — | — |" if oracles else
        f"| **median** | — | — | — | **{statistics.median(escapes):.3%}** | — | "
        f"**{statistics.median(reviews):.2%}** | — | — | — |",
        f"| range | — | — | — | {_range(escapes, '.3%')} | — | {_range(reviews, '.2%')} | "
        f"{_range(oracles, '.2%') if oracles else '—'} | — | — |",
        f"| ResNet-18, five seeds (2026-08-31) | — | 0.9120–0.9778 | 0.320% | "
        f"{RESNET_ESCAPE_MIN:.3%}–{RESNET_ESCAPE_MAX:.3%} | — | "
        f"{RESNET_DEPLOYED_MIN:.2%}–{RESNET_DEPLOYED_MAX:.2%} (median {RESNET_DEPLOYED_MEDIAN:.2%}) | "
        f"{RESNET_ORACLE_MIN:.2%}–{RESNET_ORACLE_MAX:.2%} | 1.77% | 1.16% |",
        f"| hand features + tree, three seeds (2026-09-01) | — | — | — | — | — | — | "
        f"{FEATURE_FLOOR:.2%} (median, oracle read) | — | — |",
        "",
        "The ResNet's short and open figures are seed 0 at its shipped threshold "
        "(per-class escape entry); the probe's are each seed at its own threshold.",
        "",
        "**Every configuration, read on test** -- exploration, not the result. Choosing "
        "among these rows by their test column would be choosing on test.",
        "",
        "| configuration | dims | out-of-fold review removed, per seed | "
        "test review removed at its threshold, per seed | test escape, per seed |",
        "|---|---|---|---|---|",
    ]
    by_key: dict[str, list[dict]] = {}
    for record_seed in record["seeds"]:
        for row in record_seed["configs"]:
            by_key.setdefault(f"{row['input']}/{row['pooling']}", []).append(row)
    for key, seed_rows in by_key.items():
        def cells(field, fmt, rows=seed_rows):
            return " / ".join("—" if r.get("threshold") is None else format(r[field], fmt)
                              for r in rows)
        lines.append(f"| {key} | {seed_rows[0]['dims']} | {cells('oof_review_reduction', '.2%')} | "
                     f"{cells('test_review_reduction', '.2%')} | {cells('test_escape_rate', '.3%')} |")
    extraction = record["extraction_seconds"]
    lines += [
        "",
        f"**The seed moves less here than it does for the ResNet.** Frozen features are "
        f"deterministic and the probe is convex, so a seed moves only the folds and the "
        f"final split. The range above over {len(rows)} seeds and the ResNet's over five "
        f"are not the same kind of interval.",
        "",
        f"**Cost of the experiment.** Feature extraction "
        f"{sum(extraction.values()) / 60:.1f} min on `{record['device']}` "
        f"({', '.join(f'{k} {v / 60:.1f}' for k, v in extraction.items())}; 0 means cached). "
        f"What one candidate costs at inference is a separate entry, "
        f"`scripts/dinov2_latency.py`, because it has to be measured on a quiet machine.",
        "",
        "**What this does not establish.** One model size, the smallest. Frozen, not "
        "fine-tuned: a fine-tuned DINOv2 is a different question. One upsampling, "
        "bilinear, fixed before the run -- binarised 64 px patches upsampled 3.5x are "
        "not images this backbone was trained on, and no other interpolation was tried "
        "because trying them would add an axis chosen after looking. One linear head, "
        "and no augmentation, where the ResNet trained with flips and quarter turns. The "
        "same 64 px window, so nothing outside it. And DeepPCB only.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patches", type=Path, default=ROOT / "data" / "patches")
    parser.add_argument("--features-dir", type=Path, default=ROOT / "data" / "features")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "dinov2_probe.json")
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "docs" / "benchmarks.md")
    parser.add_argument("--seeds", type=int, nargs="+", default=PUBLISHED_SEEDS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--budget", type=float, default=DEFAULTS["escape_budget"])
    parser.add_argument("--inputs", nargs="+", default=list(INPUTS), choices=INPUTS)
    parser.add_argument("--poolings", nargs="+", default=list(POOLINGS), choices=POOLINGS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not publishable(args.seeds, args.folds, args.inputs, args.poolings):
        print("only the pre-registered run (seeds 0 1 2, five folds, every input and "
              "pooling) may append to docs/benchmarks.md; pass --dry-run", file=sys.stderr)
        return 2

    from aoi_agent.vision.model import select_device

    trainval = PatchSet.load(args.patches / "trainval.npz")
    test = PatchSet.load(args.patches / "test.npz")
    device = select_device(args.device)
    started = time.perf_counter()

    features: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    extraction: dict[str, float] = {}
    for mode in args.inputs:
        features[mode] = {}
        for split, patch_set in (("trainval", trainval), ("test", test)):
            print(f"features: {mode} {split}")
            features[mode][split], seconds = cached_features(
                split, patch_set, mode, args.features_dir, device, args.batch_size)
            extraction[f"{mode} {split}"] = round(seconds, 1)

    seeds = [one_seed(features, trainval, test, seed, args) for seed in args.seeds]
    wall = time.perf_counter() - started
    rows = chosen_rows(seeds)
    if not rows:
        print("no configuration produced a threshold within the budget on any seed",
              file=sys.stderr)
        return 1

    record = {
        "date": datetime.now(UTC).date().isoformat(),
        "commit": commit(), "seed_list": args.seeds, "folds": args.folds,
        "budget": args.budget, "device": str(device), "hub_ref": HUB_REF,
        "weights_sha256": sha256(weights_path()), "input_size": INPUT_SIZE,
        "wall_seconds": round(wall, 1), "extraction_seconds": extraction,
        "seeds": seeds,
    }
    label, _sentence = verdict(rows)
    record["verdict"] = label
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")

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
