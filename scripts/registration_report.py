"""How much registration error this pipeline survives, and where it stops.

**This project has no registration stage.** The single `warpAffine` in it is in
`simulator.apply_perturbation`, which *introduces* misalignment to test robustness. Nothing
aligns an unaligned pair, because DeepPCB never handed it one: the dataset's own
README says "the image registration and thresholding techniques are common
process for high-accuracy PCB defect localization", and its authors did both
before shipping.

So the headline -- 56.2% of manual review removed -- is measured on a pipeline
that begins *after* the hardest stage of real AOI. That is a larger gap than the
prevalence one and until now neither `CLAUDE.md` nor the README named it, though
`Perturbation`'s own docstring has been saying the right thing all along:
"DeepPCB ships pre-aligned, binarised images, so these disturbances have already
been engineered away. Adding them back is restoring the operating conditions,
not corrupting the data."

Adding them back is what this does. It sweeps a per-image translation over the
official test split and reports what the AOI stage does with it -- candidates
emitted, and recall of the annotations that must survive to be re-verified at
all. Two quantities, and they fail in opposite directions:

- **candidates** is the queue the re-verifier has to absorb. It grows, because
  every trace edge becomes a sliver of difference.
- **recall** is the ceiling on everything downstream. A defect the AOI never
  emits is not in the escape budget, it is *before* it -- no threshold reaches
  it and no re-verifier recovers it.

What this cannot do is tell you which shift a real line runs at. That is a
property of a stage, a conveyor and a camera, none of which are here. What it
gives is the curve, so a line that knows its own repeatability can read its own
number off it.

Usage::

    uv run python scripts/registration_report.py --limit 100
    uv run python scripts/registration_report.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_check import evaluate  # noqa: E402

from aoi_agent.aoi.simulator import DetectorConfig, Perturbation  # noqa: E402
from aoi_agent.data.deeppcb import load_split  # noqa: E402

#: Translations to sweep, in pixels. DeepPCB scans at ~48 px/mm, so 4 px is
#: about 83 microns -- well inside what a production stage repeats to, which is
#: the point: these are not extreme values.
SHIFTS = (0, 1, 2, 3, 4)


def commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover
        return "uncommitted"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="use only the first N test pairs")
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pairs = load_split("test")
    if args.limit:
        pairs = pairs[: args.limit]

    config = DetectorConfig()
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    started = time.perf_counter()
    rows = []
    for shift in SHIFTS:
        perturbation = None if shift == 0 else Perturbation(max_shift_px=shift)
        summary = evaluate(pairs, config, perturbation)
        rows.append((shift, summary))
        print(
            f"  shift {shift}px: recall {summary['recall']:.1%}, "
            f"{summary['mean_candidates']:.1f} candidates/board",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    # Seconds under a minute. `{:.0f}` on 75 seconds printed "0 min", which
    # reads as though nothing ran -- and a report whose own timing says it did
    # not run is not a report anybody should believe the rest of.
    took = f"{elapsed / 60:.0f} min" if elapsed >= 60 else f"{elapsed:.0f} s"
    baseline = rows[0][1]

    emit()
    emit("### Registration — the stage this pipeline does not have")
    emit()
    emit(
        f"The only `warpAffine` in this project is in `simulator.apply_perturbation`, and it "
        f"*introduces* misalignment. **Nothing here aligns an unaligned pair**, "
        f"because DeepPCB never handed it one — its README says the registration "
        f"and thresholding were done before shipping. The headline figure is "
        f"therefore measured on a pipeline that begins after the hardest stage of "
        f"real AOI, and that had not been written down anywhere. Swept over "
        f"{len(pairs)} test pairs in {took}, "
        f"`scripts/registration_report.py`, commit `{commit()}`."
    )
    emit()
    emit(
        "DeepPCB scans at about 48 px/mm, so the largest shift below is roughly "
        "83 microns. These are not extreme values for a stage and a conveyor."
    )
    emit()
    emit("| shift | recall of annotations | candidates / board | vs 0 px | "
         "boards with no false call |")
    emit("|---|---|---|---|---|")
    for shift, summary in rows:
        ratio = summary["mean_candidates"] / baseline["mean_candidates"]
        emit(
            f"| {shift} px | {summary['recall']:.1%} | "
            f"{summary['mean_candidates']:.1f} | {ratio:.1f}× | "
            f"{summary['images_without_false_calls']}/{len(pairs)} |"
        )

    worst = rows[-1][1]
    emit()
    lost = baseline["recall"] - worst["recall"]
    emit(
        f"**The queue grows, and that is the half the design already answers.** "
        f"Candidates per board go {baseline['mean_candidates']:.1f} → "
        f"{worst['mean_candidates']:.1f}, "
        f"{worst['mean_candidates'] / baseline['mean_candidates']:.1f}×. It lands "
        f"on the re-verifier, which is the layer that exists to absorb it, and at "
        f"2.5 ms a candidate on CPU the arithmetic still works. What it does move "
        f"is every published *review reduction* figure, whose denominator grows "
        f"with it — the operating point was swept at 0 px and nothing has been "
        f"re-swept here."
    )
    emit()
    emit(
        f"**Recall is the half nothing answers, and it is the serious one.** "
        f"{baseline['recall']:.1%} at perfect alignment down to "
        f"{worst['recall']:.1%} at {SHIFTS[-1]} px — **{lost:.1%} of annotations "
        f"the AOI stops emitting at all.** A defect that never becomes a "
        f"candidate is not inside the escape budget, it is *before* it: no "
        f"threshold reaches it, no re-verifier recovers it, and it does not "
        f"appear in any escape figure this project publishes, because those are "
        f"computed over candidates. Against a budget of 0.5%, losing "
        f"{lost:.1%} upstream is not a rounding error — it is an order of "
        f"magnitude more defect than the entire re-verification stage is allowed "
        f"to miss."
    )
    emit()
    emit(
        "The 3×3 opening is the mechanism on both sides: it erases the one- and "
        "two-pixel slivers misalignment leaves along every trace edge, which is "
        "what keeps the queue from exploding further, and it erases small real "
        "defects along with them. `scripts/opening_kernel_sweep.py` swept that "
        "trade at 0 px. **This table is the same trade at a shift the sweep "
        "never considered**, and it moves against the defects."
    )
    emit()
    emit(
        "**What this does not establish.** It sweeps a pure translation on "
        "images that are already binarised. A real misregistration is a "
        "translation, a rotation and a scale, on grey images under a lamp that "
        "ages — and binarisation is the other thing DeepPCB removed. So this is "
        "a lower bound on the disturbance and an upper bound on the recall, and "
        "it says nothing about which shift a line actually runs at: that is a "
        "property of a stage and a camera, neither of which is in this "
        "repository. What it gives a line that knows its own repeatability is a "
        "curve to read its own number off."
    )
    emit()

    report = "\n".join(lines)
    if not args.dry_run:
        with args.out.open("a") as handle:
            handle.write(report + "\n")
        print(f"\nappended to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
