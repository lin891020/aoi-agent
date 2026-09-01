"""Does calibrating the probabilities let a threshold survive a retrain?

Three places in this project's documents say the same thing and none of them
measures it: the re-verifier's softmax is not calibrated, the swept threshold
absorbs that, and the stated cost is that **0.961 means "this model's 0.961"**,
so a new checkpoint invalidates it and `retraining-the-reverifier` makes a
re-sweep mandatory. That is a claim about transfer, and the 2026-08-31 seed
entry supplies the instrument for testing it: several models trained by the
same recipe, differing only in seed.

So the measurement is a cross-application. Each seed's threshold is chosen on
its own by-image validation half at the escape budget -- never on test -- and
then applied to *every other seed's* test scores. If the probabilities carry no
shared meaning, a threshold that buys 0.5% escape on the model that chose it
will buy something else entirely on a sibling. Temperature scaling is then
fitted on the same validation half and the whole cross-application repeated.

Stated before running: **if the off-diagonal escape rates are no tighter after
calibration, temperature scaling does not buy threshold transfer here**, and
the re-sweep stays mandatory rather than becoming a convenience.

One subtlety worth naming, because it decides whether this can move the curve
at all: with a multiclass softmax a single temperature is *not* a monotone
transform of `P(false_call)` across candidates -- the other six logits move too
-- so calibration can change the operating-point curve itself, not merely
relabel its thresholds. `log(p)` is used in place of the logits, which is exact:
softmax is invariant to the additive constant that separates them.

    uv run python scripts/calibration_report.py           # ~6 min, 3 seeds
    uv run python scripts/calibration_report.py --seeds 0 1

Writes `models/calibration.json`; appends to docs/benchmarks.md unless
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
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aoi_agent.vision.dataset import CandidateDataset  # noqa: E402
from aoi_agent.vision.model import select_device  # noqa: E402
from aoi_agent.vision.operating_point import best_at_escape_budget, sweep  # noqa: E402
from aoi_agent.vision.patches import PatchSet  # noqa: E402
from train import DEFAULTS, fit, predict, split_by_image  # noqa: E402

EPSILON = 1e-12


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def temper(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Re-soften a softmax by a scalar temperature.

    `log(p)` stands in for the logits: they differ by one additive constant per
    row and softmax does not see it.
    """
    scaled = np.log(probabilities + EPSILON) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """The temperature minimising negative log-likelihood on the validation half."""
    def loss(log_temperature: float) -> float:
        tempered = temper(probabilities, float(np.exp(log_temperature)))
        return float(-np.log(tempered[np.arange(len(labels)), labels] + EPSILON).mean())

    result = minimize_scalar(loss, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(result.x))


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray,
                               bins: int = 15) -> float:
    """Standard ECE over the predicted class's confidence."""
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            error += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return float(error)


def escape_at(scores: np.ndarray, labels: np.ndarray, index: int, threshold: float):
    """What one threshold buys on one model's test scores."""
    return sweep(scores, labels, index, thresholds=np.array([threshold]))[0]


def one_seed(trainval: PatchSet, test: PatchSet, seed: int, args, device, log) -> dict:
    label_names = trainval.label_names
    index = label_names.index("false_call")

    log(f"\n=== seed {seed} ===")
    train_idx, val_idx = split_by_image(trainval, DEFAULTS["val_fraction"], seed)
    model, _history, _best = fit(
        Subset(CandidateDataset(trainval, augment=True), train_idx),
        Subset(CandidateDataset(trainval, augment=False), val_idx),
        trainval, label_names,
        epochs=args.epochs, batch_size=DEFAULTS["batch_size"], lr=DEFAULTS["lr"],
        seed=seed, device=device, escape_budget=args.budget,
        checkpoint=None, log=lambda line: log("  " + line),
    )
    val_loader = DataLoader(Subset(CandidateDataset(trainval, augment=False), val_idx),
                            batch_size=256)
    test_loader = DataLoader(CandidateDataset(test, augment=False), batch_size=256)
    val_probabilities, val_labels = predict(model, val_loader, device)
    test_probabilities, test_labels = predict(model, test_loader, device)
    del model

    temperature = fit_temperature(val_probabilities, val_labels)
    warm_val = temper(val_probabilities, temperature)
    warm_test = temper(test_probabilities, temperature)

    # The threshold each arm would deploy, chosen on the validation half only.
    def choose(probabilities, labels):
        point = best_at_escape_budget(
            sweep(probabilities[:, index], labels, index), args.budget)
        return None if point is None else point.threshold

    row = {
        "seed": seed,
        "temperature": temperature,
        "ece_val_raw": expected_calibration_error(val_probabilities, val_labels),
        "ece_val_calibrated": expected_calibration_error(warm_val, val_labels),
        "ece_test_raw": expected_calibration_error(test_probabilities, test_labels),
        "ece_test_calibrated": expected_calibration_error(warm_test, test_labels),
        "threshold_raw": choose(val_probabilities, val_labels),
        "threshold_calibrated": choose(warm_val, val_labels),
        "_scores_raw": test_probabilities[:, index],
        "_scores_calibrated": warm_test[:, index],
        "_test_labels": test_labels,
        "_index": index,
    }
    log(f"seed {seed}: T={temperature:.3f}  ECE(test) {row['ece_test_raw']:.4f} -> "
        f"{row['ece_test_calibrated']:.4f}  threshold {row['threshold_raw']:.4f} -> "
        f"{row['threshold_calibrated']:.4f}")
    return row


def cross_apply(rows: list[dict], arm: str) -> dict:
    """Every seed's threshold on every seed's model, in one arm."""
    cells = []
    for chooser in rows:
        threshold = chooser[f"threshold_{arm}"]
        if threshold is None:
            continue
        for owner in rows:
            point = escape_at(owner[f"_scores_{arm}"], owner["_test_labels"],
                              owner["_index"], threshold)
            cells.append({"chooser": chooser["seed"], "owner": owner["seed"],
                          "escape_rate": point.escape_rate,
                          "review_reduction": point.review_reduction,
                          "own": chooser["seed"] == owner["seed"]})
    off = [c["escape_rate"] for c in cells if not c["own"]]
    diagonal = [c["escape_rate"] for c in cells if c["own"]]
    return {"cells": cells,
            "off_diagonal": {"median": statistics.median(off), "min": min(off),
                             "max": max(off), "spread": max(off) - min(off)},
            "diagonal": {"median": statistics.median(diagonal), "min": min(diagonal),
                         "max": max(diagonal)}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=Path, default=ROOT / "data" / "patches")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "calibration.json")
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "docs" / "benchmarks.md")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--budget", type=float, default=DEFAULTS["escape_budget"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    trainval = PatchSet.load(args.patches / "trainval.npz")
    test = PatchSet.load(args.patches / "test.npz")
    device = select_device(args.device)
    started = time.perf_counter()
    rows = [one_seed(trainval, test, seed, args, device, print) for seed in args.seeds]
    wall = time.perf_counter() - started

    raw, calibrated = cross_apply(rows, "raw"), cross_apply(rows, "calibrated")
    public = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    record = {"commit": commit(), "seeds": args.seeds, "budget": args.budget,
              "epochs": args.epochs, "wall_seconds": round(wall, 1),
              "rows": public, "raw": raw, "calibrated": calibrated}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")

    tighter = calibrated["off_diagonal"]["spread"] < raw["off_diagonal"]["spread"]
    lines = [
        f"## {date.today().isoformat()} · commit {commit()}",
        "",
        "### Does calibration let a threshold survive a retrain?",
        "",
        f"Three documents in this project say the softmax is uncalibrated, that "
        f"the swept threshold absorbs it, and that the cost is a threshold "
        f"meaning nothing on a different checkpoint. None of them measured it. "
        f"{len(rows)} models by the same recipe differing only in seed; each "
        f"chooses its threshold on its own by-image validation half at the "
        f"budget, never on test; each threshold is then applied to every "
        f"model's test scores. Temperature scaling is fitted on the same "
        f"validation half and everything repeats. {wall / 60:.0f} min. "
        f"`scripts/calibration_report.py`.",
        "",
        "| seed | temperature | ECE test raw → calibrated | val-chosen threshold raw → calibrated |",
        "|---|---|---|---|",
    ]
    for row in public:
        lines.append(
            f"| {row['seed']} | {row['temperature']:.3f} | "
            f"{row['ece_test_raw']:.4f} → {row['ece_test_calibrated']:.4f} | "
            f"{row['threshold_raw']:.4f} → {row['threshold_calibrated']:.4f} |")
    lines += [
        "",
        "**A threshold on the model that chose it, and on a sibling.** The "
        "budget is 0.50%; the diagonal is what each threshold buys at home, the "
        "off-diagonal what it buys on a model it never saw.",
        "",
        "| | own model (diagonal) | a sibling (off-diagonal) | off-diagonal spread |",
        "|---|---|---|---|",
        f"| uncalibrated | {raw['diagonal']['median']:.3%} "
        f"({raw['diagonal']['min']:.3%}–{raw['diagonal']['max']:.3%}) | "
        f"**{raw['off_diagonal']['median']:.3%}** "
        f"({raw['off_diagonal']['min']:.3%}–{raw['off_diagonal']['max']:.3%}) | "
        f"{raw['off_diagonal']['spread']:.3%} |",
        f"| temperature-scaled | {calibrated['diagonal']['median']:.3%} "
        f"({calibrated['diagonal']['min']:.3%}–{calibrated['diagonal']['max']:.3%}) | "
        f"**{calibrated['off_diagonal']['median']:.3%}** "
        f"({calibrated['off_diagonal']['min']:.3%}–{calibrated['off_diagonal']['max']:.3%}) | "
        f"{calibrated['off_diagonal']['spread']:.3%} |",
        "",
        ("**The stated failure condition was met: calibration does not buy threshold "
         "transfer here.**" if not tighter else
         "**Calibration tightens the off-diagonal**, which is the outcome the "
         "failure condition was written against."),
        "",
        "**What this does not establish.** Temperature scaling is the cheapest "
        "calibration there is -- one scalar over all seven classes -- and a "
        "per-class or vector scaling, or isotonic regression on `P(false_call)` "
        "alone, would be a different experiment. The siblings differ only by "
        "seed, so this says nothing about a threshold surviving a change of "
        "dataset, architecture or recipe, which is the transfer a line would "
        "actually ask about. And ECE is measured over the predicted class's "
        "confidence, which is not the quantity the threshold reads.",
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
