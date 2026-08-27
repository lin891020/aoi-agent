"""The crop re-verifier against the detector's own ordering, on the same boxes.

Two orderings of one candidate set. The detector's ``1 - confidence`` removed
1.2% of the queue at the ≤0.5% escape budget; this reads the re-verifier's
``P(false_call)`` over the identical test candidates and puts the two side by
side at every budget, with the defects the detector never boxed stated first,
because neither ordering can reach a defect that was never a candidate.

    uv run python scripts/crop_reverifier_report.py --dry-run
    uv run python scripts/crop_reverifier_report.py            # appends to docs/benchmarks.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.stats import wilson  # noqa: E402
from aoi_agent.vision.operating_point import best_at_escape_budget, sweep  # noqa: E402

BUDGETS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)
FALSE_CALL = "false_call"


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def table(scores: np.ndarray, labels: np.ndarray, fc: int, n_defects: int) -> list[tuple]:
    """(budget, point or None) at every budget, over the flagged candidates."""
    points = sweep(scores, labels, fc)
    rows = []
    for budget in BUDGETS:
        rows.append((budget, best_at_escape_budget(points, budget)))
    return rows


def render(predictions: Path, sidecar: Path, history: Path) -> list[str]:
    out: list[str] = []

    def emit(text: str = "") -> None:
        out.append(text)

    pred = np.load(predictions, allow_pickle=False)
    side = np.load(sidecar, allow_pickle=False)
    label_names = [str(n) for n in pred["label_names"]]
    fc = label_names.index(FALSE_CALL)
    labels = pred["labels"]
    reverifier = pred["probabilities"][:, fc]
    detector = side["detector_false_call_probability"]
    assert len(detector) == len(labels), "sidecar and predictions describe different candidates"

    n_images = int(len(side["candidates_per_image"]))
    n_defects = int(side["defects_by_class"].sum())
    n_unflagged = int(side["unflagged_by_class"].sum())
    flagged_defects = int((labels != fc).sum())
    false_calls = int((labels == fc).sum())
    fragments = int(side["fragments"]) if "fragments" in side.files else None
    prevalence = flagged_defects / (flagged_defects + false_calls)
    hist = json.loads(history.read_text()) if history.exists() else []
    epochs = len(hist)
    seconds = sum(h.get("seconds", 0) for h in hist)

    emit("### Crop re-verifier — the ResNet-18 over the detector's boxes, against the detector's own ordering")
    emit()
    emit(
        f"**Basis: {n_images} test images, {n_defects} annotated defects, of which "
        f"{n_unflagged} were never boxed by the detector and are outside every figure "
        f"below.** {flagged_defects + false_calls} candidates at the detector's floor "
        f"({flagged_defects} covering a defect, {false_calls} false calls). "
        f"One training run, {epochs} epochs in {seconds:.0f} s on the M5 Air, seed 0, "
        f"RGB crops with no template channel; the training candidates come from a "
        f"detector that had seen the training images (see "
        f"`scripts/build_detector_patches.py`). {date.today().isoformat()}, commit `{commit()}`."
    )
    emit()
    emit(
        f"The queue is **{prevalence:.1%} genuine defects**, so review removed cannot exceed "
        f"{1 - prevalence:.1%} on any ordering; read every figure below against that ceiling. "
        + (
            f"The detector's row differs from its own entry above because this basis holds out "
            f"the {fragments} boxes that split a defect (neither class, so not trainable); "
            f"both orderings are read over the identical {flagged_defects + false_calls} candidates."
            if fragments is not None else ""
        )
    )
    emit()
    emit("| escape budget | ordering | achieved escape | review removed | 95% on escape |")
    emit("|---|---|---|---|---|")
    for name, scores in (("detector 1 − confidence", detector), ("crop re-verifier P(false call)", reverifier)):
        for budget, point in table(scores, labels, fc, flagged_defects):
            if point is None:
                emit(f"| ≤{budget:.2%} | {name} | — | — | no threshold meets the budget |")
                continue
            escapes = int(round(point.escape_rate * flagged_defects))
            low, high = wilson(escapes, flagged_defects)
            emit(
                f"| ≤{budget:.2%} | {name} | {point.escape_rate:.2%} | "
                f"**{point.review_reduction:.1%}** | {low:.2%}–{high:.2%} |"
            )
    emit()
    at_budget = {
        name: best_at_escape_budget(sweep(s, labels, fc), 0.005)
        for name, s in (("detector", detector), ("reverifier", reverifier))
    }
    d = at_budget["detector"].review_reduction if at_budget["detector"] else 0.0
    r = at_budget["reverifier"].review_reduction if at_budget["reverifier"] else 0.0
    emit(
        f"At the ≤0.5% budget the detector's confidence removes **{d:.1%}** of the same "
        f"queue and the crop re-verifier removes **{r:.1%}**. "
        + (
            "The re-verifier orders what the detector could only locate."
            if r > d + 0.05 else
            f"That is {r / (1 - prevalence):.0%} of the false calls the queue holds, and the "
            f"gap between the two is inside what sixty images can resolve: a re-verifier over "
            f"RGB crops with no template channel is not yet an ordering either. What both "
            f"front ends now agree on is that on this data the false calls are not separable "
            f"from the defects on appearance alone at this budget -- which is the finding, "
            f"and the reason the differencing front end's template channel was never a "
            f"convenience."
        )
    )
    emit()
    emit(
        "What this does not establish: one seed, sixty test images and wide intervals; "
        "a re-verifier trained on in-sample detector boxes; and no template channel, so "
        "nothing here transfers to DeepPCB or says anything about the differencing front "
        "end. `scripts/build_detector_patches.py`, `scripts/train.py --patches "
        "data/patches_pcbaoi` and this script rebuild every number."
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("models/pcbaoi_reverifier/test_predictions.npz"))
    parser.add_argument("--sidecar", type=Path, default=Path("data/patches_pcbaoi/test_detector.npz"))
    parser.add_argument("--history", type=Path, default=Path("models/pcbaoi_reverifier/history.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lines = render(args.predictions, args.sidecar, args.history)
    text = "\n".join(lines) + "\n"
    print(text)
    if not args.dry_run:
        with args.out.open("a") as f:
            f.write("\n" + text)
        print(f"appended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
