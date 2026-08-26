"""The detector front end, read the only way this project reads a model.

Sixty test images, touched once. Every box the detector emits at the floor
is a candidate; a candidate covering an annotated defect carries that
defect's class and every other one is a false call; ``P(false_call)`` is
``1 - confidence``; and the operating point is swept over that with the same
`vision.operating_point.sweep` the re-verifier is scored with. A defect no
box covers is *unflagged* -- the detector's S0 -- and is reported before any
figure downstream of it, because the sweep cannot see a defect that was never
a candidate.

The headline is **manual review removed at an escape budget**, with a Wilson
interval, on a basis the first line of the entry names. mAP is printed once,
for reference, the way accuracy is for the re-verifier.

Run:
    uv run python scripts/detector_report.py --dry-run
    uv run python scripts/detector_report.py            # appends to docs/benchmarks.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.aoi.matching import match  # noqa: E402
from aoi_agent.data import pcbaoi  # noqa: E402
from aoi_agent.stats import wilson  # noqa: E402
from aoi_agent.vision.detector import CONF_FLOOR, DEFAULT_CHECKPOINT, Detector, false_call_probabilities  # noqa: E402
from aoi_agent.vision.operating_point import best_at_escape_budget, sweep  # noqa: E402

BUDGETS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)
FALSE_CALL = "false_call"
HISTORY = Path("models/detector_history.json")


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def score(detector: Detector, items: list[pcbaoi.Item]) -> dict:
    """Every candidate on the test set with its label, plus what was never flagged."""
    scores, labels, per_image = [], [], []
    classes = list(pcbaoi.CLASS_NAMES) + [FALSE_CALL]
    defects_total = Counter()
    unflagged = Counter()
    for item in items:
        image = np.array(Image.open(item.image_path).convert("RGB"))
        boxes = item.load_boxes()
        cands = detector.detect(image, floor=CONF_FLOOR)
        result = match(cands, boxes)
        per_image.append(len(cands))
        for b in boxes:
            defects_total[b.class_name] += 1
        for b in result.missed_annotations:
            unflagged[b.class_name] += 1
        for labelled, cand in zip(result.labelled, cands, strict=True):
            target = labelled.matched_annotation
            labels.append(classes.index(target.class_name if target is not None else FALSE_CALL))
            scores.append(cand.false_call_probability)
    return {
        "classes": classes,
        "scores": np.array(scores), "labels": np.array(labels),
        "candidates_per_image": per_image,
        "defects_total": defects_total, "unflagged": unflagged,
    }


def render(scored: dict, detector: Detector, images: int, mean_ap: float | None) -> list[str]:
    out: list[str] = []

    def emit(text: str = "") -> None:
        out.append(text)

    classes = scored["classes"]
    fc = classes.index(FALSE_CALL)
    labels, scores = scored["labels"], scored["scores"]
    n_defects = sum(scored["defects_total"].values())
    n_unflagged = sum(scored["unflagged"].values())
    flagged_defects = int((labels != fc).sum())
    false_calls = int((labels == fc).sum())
    history = json.loads(HISTORY.read_text()) if HISTORY.exists() else {}

    emit("### Detector front end — YOLO26n on PCB-AoI, read at the escape budget")
    emit()
    emit(
        f"**Basis: {images} test images, {n_defects} annotated defects "
        f"({', '.join(f'{k} {v}' for k, v in sorted(scored['defects_total'].items()))}).** "
        f"Sixty images is a small test set and every interval below says so. "
        f"The detector emitted {len(labels)} candidates at a confidence floor of {CONF_FLOOR} "
        f"({np.mean(scored['candidates_per_image']):.1f} an image): {flagged_defects} covering a "
        f"defect and {false_calls} false calls, a prevalence of "
        f"{flagged_defects / len(labels) if len(labels) else 0:.1%}. "
        f"Trained {history.get('epochs', '?')} epochs on {history.get('train_images', '?')} images "
        f"with {history.get('val_boards', '?')} boards held out, {history.get('wall_seconds', 0) / 60:.0f} min on "
        f"{history.get('device', '?')}"
        + (" -- **beside a busy GPU**" if history.get("contended") else "")
        + f". `scripts/detector_report.py`, checkpoint `{detector.checkpoint.name}`, commit `{commit()}`."
    )
    emit()
    emit("**S0 first.** " + (
        f"{n_unflagged} of {n_defects} defects ({n_unflagged / n_defects:.1%}) were covered by no box at the floor "
        f"-- " + ", ".join(f"{k} {scored['unflagged'][k]}/{scored['defects_total'][k]}" for k in sorted(scored['defects_total']))
        + ". Those are escapes no threshold below can recover, and the table below is conditional on the rest."
        if n_defects else "no defects annotated."
    ))
    emit()
    emit("| escape budget | achieved | manual review removed | escapes | 95% interval on the escape rate | false calls dismissed |")
    emit("|---|---|---|---|---|---|")
    points = sweep(scores, labels, fc)
    for budget in BUDGETS:
        best = best_at_escape_budget(points, budget)
        if best is None:
            emit(f"| ≤{budget:.2%} | — | *not reachable* | — | — | — |")
            continue
        lo, hi = wilson(best.escapes, best.defects_total)
        emit(
            f"| ≤{budget:.2%} | {best.escape_rate:.2%} | **{best.review_reduction:.1%}** | "
            f"{best.escapes}/{best.defects_total} | {lo:.2%}–{hi:.2%} | "
            f"{best.false_calls_dismissed}/{best.false_calls_total} |"
        )
    emit()
    emit(
        "The escape rate in this table divides by the defects the detector *flagged*, "
        "the way the re-verifier's table divides by the candidates it was handed. "
        "Add the unflagged row above to read it as a line rate."
    )
    emit()
    if mean_ap is not None:
        emit(f"mAP50-95 on the same images: {mean_ap:.3f} (reported for reference only — it weighs every box the same and answers no question about a budget).")
        emit()
    emit(
        "**What this does not establish.** One run, one seed, sixty images; no CPU timing "
        "(nothing about inference speed is claimed until it is measured the way the re-verifier's was); "
        "and the two classes are the dataset's, with no work instruction behind either. The comparison "
        "with DeepPCB is a comparison of *readings*, not of numbers: the populations differ and so does "
        "the prevalence, which is why both are printed in the first line."
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument("--no-map", action="store_true", help="skip the ultralytics val pass")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    detector = Detector(args.checkpoint, device=args.device)
    items = pcbaoi.load("test_data")
    scored = score(detector, items)

    mean_ap = None
    if not args.no_map:
        yaml = pcbaoi.DEFAULT_EXPORT / "data.yaml"
        if yaml.exists():
            metrics = detector.model.val(data=str(yaml), split="test", verbose=False, plots=False, device=args.device)
            mean_ap = float(metrics.box.map)

    lines = render(scored, detector, len(items), mean_ap)
    header = f"## {date.today().isoformat()} · commit {commit()}"
    body = "\n".join([header, "", *lines])
    print(body)
    if args.dry_run:
        print("\n(dry run -- nothing appended)", file=sys.stderr)
        return 0
    existing = args.out.read_text() if args.out.exists() else "# Benchmarks\n"
    args.out.write_text(existing.rstrip() + "\n\n" + body + "\n")
    print(f"\nappended to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
