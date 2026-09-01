"""The floor a reviewer would actually ask for: geometry plus a tree, no network.

`scripts/model_free_baseline.py` reduces the difference patch to one scalar and
gets 1.3% of the review queue at the escape budget against the re-verifier's
52.8%. That is a real floor but a weak one, and its own entry says so: the
candidates were selected by the AOI stage for *having* a large difference, so
re-ranking them by the size of that difference is asking a question already
answered upstream.

The honest floor uses the one thing the scalar throws away -- shape, and the
relationship between the two images. A residual from imperfect registration is
a thin sliver lying along a trace edge; a defect is compact and sits where the
template says copper should or should not be. Both are computable from the same
64 px patch with no weights at all:

    open   -> the test image is missing foreground the template has
    short  -> the test image has foreground the template does not
    sliver -> long, thin, low solidity, hugging a template edge

So: ~30 hand-computed features per candidate and a gradient-boosted tree. Same
7,322 test candidates, same by-image split, same `operating_point.sweep`, same
budget. The tree gets the *easier* problem on purpose -- binary false-call vs
defect, where the network solves seven classes and is read on one of them --
because a floor should be given its best shot, not a handicap.

Read at its own best threshold on the test split, like `model_free_baseline.py`
and for the same reason: both sides of a comparison between engines inherit the
same optimism, and correcting one side only flatters the other. Three seeds,
because the 2026-08-31 seed entry measured this recipe spanning 3.1 points with
nothing changed.

    uv run python scripts/feature_baseline.py              # ~10 min
    uv run python scripts/feature_baseline.py --seeds 0    # a shorter read

Writes `models/feature_baseline.json`; appends to docs/benchmarks.md unless
`--dry-run`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aoi_agent.vision.operating_point import best_at_escape_budget, sweep  # noqa: E402
from aoi_agent.vision.patches import PatchSet  # noqa: E402
from train import DEFAULTS, split_by_image  # noqa: E402

# The grey threshold the AOI stage itself used to make these candidates. The
# blob measured here is therefore the blob the detector drew the box around,
# not a different one at a different setting.
DIFF_THRESHOLD = 60

NAMES = [
    "blob_area", "blob_area_fraction", "n_components", "total_diff_area",
    "bbox_h", "bbox_w", "aspect", "extent", "solidity", "circularity",
    "elongation", "centroid_offset",
    "diff_mean", "diff_max", "diff_p90", "diff_mean_in_blob",
    "template_mean", "test_mean", "delta_foreground",
    "template_mean_in_blob", "test_mean_in_blob", "delta_foreground_in_blob",
    "blob_on_template_foreground", "template_edge_density",
    "blob_edge_overlap", "blob_distance_to_edge",
    "row_span", "col_span", "touches_border", "fill_ratio",
]


def _blob_shape(mask: np.ndarray) -> dict:
    """Geometry of the largest connected component of the thresholded difference."""
    labelled, count = ndimage.label(mask)
    if count == 0:
        return {"count": 0}
    sizes = ndimage.sum_labels(mask, labelled, range(1, count + 1))
    biggest = int(np.argmax(sizes)) + 1
    blob = labelled == biggest
    rows, cols = np.nonzero(blob)
    h = rows.max() - rows.min() + 1
    w = cols.max() - cols.min() + 1
    area = float(blob.sum())

    # Second moments give elongation without a convex hull; the hull gives
    # solidity, which is what separates a sliver from a compact blob when both
    # happen to be long.
    coords = np.column_stack([rows, cols]).astype(float)
    centred = coords - coords.mean(0)
    if len(coords) >= 2:
        eigenvalues = np.linalg.eigvalsh(np.cov(centred, rowvar=False) + 1e-9 * np.eye(2))
        elongation = float(np.sqrt(max(eigenvalues) / max(min(eigenvalues), 1e-9)))
    else:
        elongation = 1.0
    try:
        hull_area = float(ConvexHull(coords).volume) if len(coords) >= 3 else area
    except (QhullError, ValueError):
        hull_area = area
    perimeter = float((blob & ~ndimage.binary_erosion(blob)).sum())

    return {
        "count": int(count), "blob": blob, "area": area, "h": float(h), "w": float(w),
        "extent": area / float(h * w),
        "solidity": area / max(hull_area, 1.0),
        "circularity": 4 * np.pi * area / max(perimeter ** 2, 1.0),
        "elongation": elongation,
        "centroid_offset": float(np.hypot(rows.mean() - 31.5, cols.mean() - 31.5)),
        "touches_border": float(rows.min() == 0 or cols.min() == 0
                                or rows.max() == 63 or cols.max() == 63),
    }


def features(patch: np.ndarray) -> list[float]:
    """One candidate's ~30 numbers. `patch` is (3, 64, 64) uint8: template, test, |diff|."""
    template = patch[0].astype(np.float32) / 255.0
    test = patch[1].astype(np.float32) / 255.0
    diff = patch[2].astype(np.float32)
    mask = diff > DIFF_THRESHOLD
    pixels = float(template.size)

    shape = _blob_shape(mask)
    if shape["count"] == 0:
        blob = np.zeros_like(mask)
        geom = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        row_span = col_span = touches = 0.0
    else:
        blob = shape["blob"]
        geom = [shape["area"], shape["area"] / pixels, float(shape["count"]),
                float(mask.sum()), shape["h"], shape["w"],
                max(shape["h"], shape["w"]) / max(min(shape["h"], shape["w"]), 1.0),
                shape["extent"], shape["solidity"], shape["circularity"]]
        row_span, col_span = shape["h"] / 64.0, shape["w"] / 64.0
        touches = shape["touches_border"]

    inside = blob.sum()
    take = (lambda a: float(a[blob].mean()) if inside else 0.0)

    # Where the template says foreground, and how close the blob sits to a
    # template edge -- a registration residual lies *on* an edge, a defect
    # usually does not.
    edges = template != ndimage.binary_erosion(template > 0.5).astype(np.float32)
    edge_distance = ndimage.distance_transform_edt(~(edges > 0)) if edges.any() else np.full_like(template, 64.0)

    return geom[:4] + geom[4:] + [
        float(shape.get("elongation", 1.0)), float(shape.get("centroid_offset", 0.0)),
        float(diff.mean()), float(diff.max()), float(np.percentile(diff, 90)), take(diff),
        float(template.mean()), float(test.mean()), float(test.mean() - template.mean()),
        take(template), take(test), take(test) - take(template),
        take(template > 0.5) if inside else 0.0,
        float((edges > 0).mean()),
        float((blob & (edges > 0)).sum() / max(inside, 1)),
        float(edge_distance[blob].mean()) if inside else 64.0,
        row_span, col_span, touches,
        float(mask.mean()),
    ]


def feature_table(patches: np.ndarray, log) -> np.ndarray:
    rows = []
    for i, patch in enumerate(patches):
        rows.append(features(patch))
        if (i + 1) % 2500 == 0:
            log(f"  features {i + 1}/{len(patches)}")
    table = np.asarray(rows, dtype=np.float32)
    assert table.shape[1] == len(NAMES), (table.shape[1], len(NAMES))
    return np.nan_to_num(table, nan=0.0, posinf=0.0, neginf=0.0)


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def one_seed(x_trainval, y_trainval, x_test, y_test, trainval, seed, args, log) -> dict:
    """Train the tree on this seed's own by-image split, read test at its oracle.

    The number of boosting iterations is a hyper-parameter and gets chosen the
    way the network's epoch does: on the validation half of a *by-image* split,
    by review removed at the budget, never on test. sklearn's own early stopping
    is not used because it would split by row -- patches from one board on both
    sides, which is the leak `split_by_image` exists to prevent, and which the
    first version of this script quietly had (all three seeds returned the same
    number, because with `validation_fraction=None` the split never reached the
    model at all).
    """
    train_idx, val_idx = split_by_image(trainval, DEFAULTS["val_fraction"], seed)
    model = HistGradientBoostingClassifier(
        max_iter=args.max_iter, learning_rate=0.06, early_stopping=False,
        random_state=seed,
    )
    # The tree gets false-call vs defect, the easier framing, on purpose.
    model.fit(x_trainval[train_idx], y_trainval[train_idx])

    y_val = y_trainval[val_idx]
    best_iteration, best_reduction = 0, -1.0
    for iteration, probabilities in enumerate(model.staged_predict_proba(x_trainval[val_idx]), 1):
        point = best_at_escape_budget(sweep(probabilities[:, 1], y_val, 1), args.budget)
        reduction = point.review_reduction if point else 0.0
        if reduction > best_reduction:
            best_iteration, best_reduction = iteration, reduction
    log(f"  seed {seed}: {best_iteration}/{args.max_iter} iterations "
        f"(val review {best_reduction:.2%})")

    scores = None
    for iteration, probabilities in enumerate(model.staged_predict_proba(x_test), 1):
        if iteration == best_iteration:
            scores = probabilities[:, 1]
            break
    assert scores is not None

    # `sweep` wants a label vector whose false-call index it can count against;
    # 1 here is false_call, so the defect-labelled candidates are the zeros.
    oracle = best_at_escape_budget(sweep(scores, y_test.astype(np.int64), 1), args.budget)
    accuracy = float(((scores > 0.5).astype(int) == y_test).mean())

    importance = permutation_importance(
        model, x_trainval[val_idx], y_val,
        n_repeats=5, random_state=seed, scoring="roc_auc")
    order = np.argsort(importance.importances_mean)[::-1][:8]
    top = [(NAMES[i], float(importance.importances_mean[i])) for i in order]

    row = {
        "seed": seed,
        "iterations": best_iteration,
        "val_review_reduction": best_reduction,
        "test_accuracy": accuracy,
        "oracle_threshold": None if oracle is None else oracle.threshold,
        "oracle_review_reduction": None if oracle is None else oracle.review_reduction,
        "oracle_escape_rate": None if oracle is None else oracle.escape_rate,
        "top_features": top,
    }
    if oracle is not None:
        log(f"\nseed {seed}: review {oracle.review_reduction:.2%} @ {oracle.escape_rate:.3%}  "
            f"(acc {accuracy:.1%}, {best_iteration} iterations)")
        log("  top features: " + ", ".join(f"{n} {v:+.3f}" for n, v in top[:5]))
    return row


def spread(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values),
            "n": len(values)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=Path, default=ROOT / "data" / "patches")
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "models" / "test_predictions.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "feature_baseline.json")
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "docs" / "benchmarks.md")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--budget", type=float, default=DEFAULTS["escape_budget"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    trainval = PatchSet.load(args.patches / "trainval.npz")
    test = PatchSet.load(args.patches / "test.npz")
    false_call_index = list(trainval.label_names).index("false_call")
    print(f"trainval {len(trainval.labels)} / test {len(test.labels)} candidates")

    started = time.perf_counter()
    print("extracting features (no weights involved)")
    x_trainval = feature_table(trainval.patches, print)
    x_test = feature_table(test.patches, print)
    y_trainval = (trainval.labels == false_call_index).astype(np.int64)
    y_test = (test.labels == false_call_index).astype(np.int64)
    print(f"features done in {time.perf_counter() - started:.0f}s: "
          f"{x_trainval.shape[1]} per candidate")

    rows = [one_seed(x_trainval, y_trainval, x_test, y_test, trainval, seed, args, print)
            for seed in args.seeds]
    wall = time.perf_counter() - started
    rows = [r for r in rows if r["oracle_review_reduction"] is not None]
    if not rows:
        print("no seed reached a point inside the budget", file=sys.stderr)
        return 1

    # The network, re-scored on the identical array so the two numbers share a
    # basis. Its own entries carry the same figure; recomputing it here is what
    # stops this table drifting from them.
    network = None
    if args.predictions.exists():
        stored = np.load(args.predictions)
        probabilities, labels = stored["probabilities"], stored["labels"]
        network = best_at_escape_budget(
            sweep(probabilities[:, false_call_index], labels, false_call_index), args.budget)

    reviews = spread([r["oracle_review_reduction"] for r in rows])
    record = {"commit": commit(), "seeds": args.seeds, "budget": args.budget,
              "features": NAMES, "diff_threshold": DIFF_THRESHOLD,
              "wall_seconds": round(wall, 1), "rows": rows,
              "review_removed": reviews,
              "network_review_removed": None if network is None else network.review_reduction}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")

    counted = {}
    for row in rows:
        for name, value in row["top_features"]:
            counted[name] = counted.get(name, 0.0) + value / len(rows)
    ranked = sorted(counted.items(), key=lambda kv: -kv[1])[:6]

    lines = [
        f"## {date.today().isoformat()} · commit {commit()}",
        "",
        "### The floor with shape in it — hand features and a tree, still no network",
        "",
        f"The 2026-08-31 model-free entry ends by naming the baseline it is not: "
        f"connected-component geometry and a gradient-boosted tree. That is the "
        f"floor a reviewer actually asks for, because the scalar one re-ranks "
        f"candidates by the very quantity the AOI stage already selected them "
        f"for. This is that baseline. {len(NAMES)} features per candidate from the "
        f"same 64 px patch -- blob area, aspect, extent, solidity, circularity, "
        f"elongation, distance to the nearest template edge, and the foreground "
        f"the test image has that the template does not -- into a "
        f"`HistGradientBoostingClassifier`. Same {len(test.labels):,} test "
        f"candidates, same by-image split per seed, same sweep, same budget. "
        f"{wall:.0f} s, CPU. `scripts/feature_baseline.py`.",
        "",
        "**The tree gets the easier problem on purpose**: binary false-call vs "
        "defect, where the network solves seven classes and is read on one of "
        "them. A floor should be given its best shot; handicapping it would make "
        "the gap above it meaningless. Its one hyper-parameter, the number of "
        "boosting iterations, is chosen exactly the way the network's epoch is: "
        "on the validation half of a by-image split, by review removed at the "
        "budget. sklearn's own early stopping splits by row, which puts patches "
        "from one board on both sides.",
        "",
        "| seed | iterations (val-chosen) | review removed @ budget | achieved escape | accuracy |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['iterations']} | "
                     f"**{row['oracle_review_reduction']:.2%}** | "
                     f"{row['oracle_escape_rate']:.3%} | {row['test_accuracy']:.1%} |")
    lines.append(f"| **median** | — | **{reviews['median']:.2%}** | — | — |")
    if network is not None:
        lines.append(f"| ResNet-18, same array, same oracle | — | "
                     f"**{network.review_reduction:.2%}** | "
                     f"{network.escape_rate:.3%} | — |")
    lines += [
        "",
        "Features carrying the most signal, by permutation importance on each "
        "seed's validation split (mean over seeds, AUC drop): "
        + ", ".join(f"`{name}` {value:+.3f}" for name, value in ranked) + ".",
        "",
        "**What this does not establish.** One feature set, chosen by hand from "
        "what the difference of two binarised images makes available, and one "
        "tree with default-ish settings and no hyper-parameter search -- a "
        "stronger feature set or a tuned tree would move this floor up, and "
        "nothing here bounds how far. The patch is 64 px, so every feature is "
        "local: a defect's relation to the wider board is not in it. And this is "
        "DeepPCB, where the images are binarised and the residuals are crisp; on "
        "photographs the shape features would be measuring a different object, "
        "the same way the differencing stage was.",
        "",
    ]
    body = "\n".join(lines)
    if args.dry_run:
        print("\n" + body)
    else:
        with args.benchmarks.open("a") as handle:
            handle.write("\n" + body)
        print(f"\nappended to {args.benchmarks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
