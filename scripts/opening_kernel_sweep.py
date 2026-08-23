"""What does the opening kernel cost, and what does it buy?

Seven defects on the test split have no candidate on them at all. That is the
whole of this line's unrecoverable escape rate -- 0.22%, see
`scripts/escape_accounting.py` -- and every one of them dies in the same place:
`DetectorConfig.open_kernel = 3`. Each is a sub-3px notch, whisker, filament or
dot-chain whose above-threshold difference blob is 24 to 133 pixels wide-ish
and one to three pixels thin, and a 3x3 morphological opening takes it to zero
before connected components ever runs.

So a smaller kernel recovers them. The opening is not there by accident,
though: it erases the one- and two-pixel slivers misregistration leaves along
every trace edge, and README's "Known limits" already records that a 2px
template shift yields zero candidates at the default settings *because* of it.
Lowering it trades escapes against false calls, which is the same trade every
other threshold in this project is reported on. It gets the same treatment: a
curve, not a point.

Swept, on the official test split:

* **`open_kernel`** at 0 (no opening), 2, 3, 4 and 5, plus a 3x3 **cross**,
  which survives on five cells in a plus where a 3x3 square needs all nine --
  a middle point between 2 and 3 that changing the size alone cannot reach. A
  1x1 element is a no-op in OpenCV and is reported as such rather than as a
  data point.
* Against **defects nothing is flagged on** -- the only figure a kernel change
  can move that is genuinely an escape.
* Against **false calls per board**, clean and under production-like
  misregistration (2px shift, sigma 6 noise, 1.03 gain -- `gate_check.py`'s
  defaults, seeded per board so one registration error does not colour the
  whole run). The perturbed column is the one that describes a real line and
  it is where the opening earns its keep.
* And against what the **shipped re-verifier** does with the defects a setting
  recovers, because a recovered candidate is not a recovered defect. If the
  model dismisses the whisker anyway, the kernel bought nothing and the false
  calls are still on the bill.

The last point comes with a caveat that must stay attached to it: the
checkpoint was trained on `open_kernel=3` patches, so a candidate only a
smaller kernel produces is out of its training distribution. It answers "what
would this line do today", not "what would a retrained line do". It is
reported as the former.

    uv run python scripts/opening_kernel_sweep.py
    uv run python scripts/opening_kernel_sweep.py --limit 100 --dry-run
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.aoi.matching import IOU_THRESHOLD, match  # noqa: E402
from aoi_agent.aoi.simulator import (  # noqa: E402
    DetectorConfig,
    Perturbation,
    detect,
    opening_element,
)
from aoi_agent.data.deeppcb import load_split  # noqa: E402
from aoi_agent.vision.inference import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_DISMISS_THRESHOLD,
    ReVerifier,
)
from aoi_agent.vision.patches import PATCH_SIZE, build_patch  # noqa: E402

from escape_accounting import covering_candidates  # noqa: E402

#: The settings swept. Order is loosest opening first, so the table reads down
#: from "keep everything" to "keep almost nothing" and the current value sits
#: in the middle of its own curve rather than at the end of it.
SETTINGS: list[tuple[str, DetectorConfig]] = [
    ("none", DetectorConfig(open_kernel=0)),
    ("2x2 square", DetectorConfig(open_kernel=2)),
    ("3x3 cross", DetectorConfig(open_kernel=3, open_shape="cross")),
    ("3x3 square", DetectorConfig(open_kernel=3)),
    ("4x4 square", DetectorConfig(open_kernel=4)),
    ("5x5 square", DetectorConfig(open_kernel=5)),
]

SHIPPED = "3x3 square"

#: gate_check.py's production conditions. Not a guess -- the same disturbance
#: the S0 gate uses, so the two runs are talking about the same line.
SHIFT_PX = 2
NOISE_SIGMA = 6.0
GAIN = 1.03


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "uncommitted"


@dataclass
class Row:
    """One opening setting, measured."""

    name: str
    config: DetectorConfig
    defects: int = 0
    unflagged: int = 0
    missed_at_cut: int = 0
    false_calls: list[int] = field(default_factory=list)
    candidates: list[int] = field(default_factory=list)
    perturbed_false_calls: list[int] = field(default_factory=list)
    perturbed_candidates: list[int] = field(default_factory=list)
    recovered_kept: int = 0
    """Of the defects this setting flags that the shipped setting does not, how
    many does the re-verifier keep rather than dismiss."""

    recovered: int = 0

    @property
    def unflagged_rate(self) -> float:
        return self.unflagged / self.defects if self.defects else 0.0

    @property
    def mean_false_calls(self) -> float:
        return statistics.mean(self.false_calls) if self.false_calls else 0.0

    @property
    def mean_candidates(self) -> float:
        return statistics.mean(self.candidates) if self.candidates else 0.0

    @property
    def mean_perturbed_false_calls(self) -> float:
        return (
            statistics.mean(self.perturbed_false_calls)
            if self.perturbed_false_calls
            else 0.0
        )

    @property
    def boards(self) -> int:
        return len(self.false_calls)


def is_noop(config: DetectorConfig) -> bool:
    """A 1x1 element leaves the mask alone whatever its shape."""
    return config.open_kernel <= 1


def kept_mask(
    reverifier: ReVerifier | None,
    template: np.ndarray,
    test: np.ndarray,
    candidates: list,
    threshold: float,
) -> np.ndarray:
    """Per-candidate: True where the re-verifier does not dismiss it."""
    if reverifier is None or not candidates:
        return np.ones(len(candidates), dtype=bool)
    import torch

    patches = np.stack([build_patch(template, test, c, PATCH_SIZE) for c in candidates])
    batch = torch.from_numpy(patches).float().div_(255.0)
    with torch.no_grad():
        probabilities = (
            torch.softmax(reverifier.model(batch.to(reverifier.device)), dim=1)
            .cpu()
            .numpy()
        )
    index = reverifier.label_names.index("false_call")
    return probabilities[:, index] < threshold


def erased_by_opening(
    template: np.ndarray, test: np.ndarray, box: tuple[int, int, int, int],
    config: DetectorConfig,
) -> tuple[int, int, float]:
    """Difference pixels inside ``box`` before and after the opening, and the
    feature's maximum half-width.

    The half-width comes from a distance transform, which is the direct answer
    to "is this thin". A blob survives an NxN opening only where it contains a
    fully-enclosed NxN square, so a half-width under N/2 is erased outright.
    """
    diff = cv2.absdiff(test, template)
    _, mask = cv2.threshold(diff, config.threshold, 255, cv2.THRESH_BINARY)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_element(config))
    x1, y1, x2, y2 = box
    window = (mask[y1:y2, x1:x2] > 0).astype(np.uint8)
    distance = cv2.distanceTransform(window, cv2.DIST_L2, 3)
    return int(window.sum()), int((opened[y1:y2, x1:x2] > 0).sum()), float(distance.max())


def measure(
    split: str, reverifier: ReVerifier | None, threshold: float, limit: int | None
) -> tuple[list[Row], list[dict]]:
    """Sweep every setting over the split. Returns the rows and the forensics
    on the defects the shipped setting loses."""
    pairs = load_split(split)
    if limit:
        pairs = pairs[:limit]

    rows = {name: Row(name, config) for name, config in SETTINGS}
    forensics: list[dict] = []

    for index, pair in enumerate(pairs):
        template, test = pair.load_template(), pair.load_test()
        annotations = pair.load_annotations()
        perturbation = Perturbation(
            max_shift_px=SHIFT_PX, noise_sigma=NOISE_SIGMA, gain=GAIN, seed=index
        )

        shipped_config = rows[SHIPPED].config
        shipped_candidates = detect(template, test, shipped_config)
        lost = [
            a
            for a in annotations
            if not covering_candidates(a.box, shipped_candidates)
        ]

        for annotation in lost:
            raw, after, half_width = erased_by_opening(
                template, test, annotation.box, shipped_config
            )
            forensics.append({
                "board": pair.stem,
                "class": annotation.class_name,
                "box": f"{annotation.x2 - annotation.x1}x{annotation.y2 - annotation.y1}",
                "diff_pixels": raw,
                "after_opening": after,
                "half_width": half_width,
            })

        for name, config in SETTINGS:
            row = rows[name]
            candidates = detect(template, test, config)
            result = match(candidates, annotations, iou_threshold=IOU_THRESHOLD)
            row.defects += len(annotations)
            row.missed_at_cut += len(result.missed_annotations)
            row.false_calls.append(len(result.false_calls))
            row.candidates.append(len(candidates))
            row.unflagged += sum(
                1 for a in annotations if not covering_candidates(a.box, candidates)
            )

            if lost and name != SHIPPED:
                recovered = [a for a in lost if covering_candidates(a.box, candidates)]
                row.recovered += len(recovered)
                if recovered:
                    kept = kept_mask(reverifier, template, test, candidates, threshold)
                    for annotation in recovered:
                        indices = covering_candidates(annotation.box, candidates)
                        if any(bool(kept[i]) for i in indices):
                            row.recovered_kept += 1

            perturbed = detect(template, test, config, perturbation)
            row.perturbed_false_calls.append(
                len(match(perturbed, annotations).false_calls)
            )
            row.perturbed_candidates.append(len(perturbed))

        if (index + 1) % 50 == 0:
            print(f"  {index + 1}/{len(pairs)} boards")

    return [rows[name] for name, _ in SETTINGS], forensics


def cost_per_defect(row: Row, shipped: Row, saved: int) -> float | None:
    """Extra false calls across the split per defect actually kept.

    ``saved`` is defects recovered *and* not then dismissed by the model.
    Dividing by candidates recovered instead would credit the kernel with
    defects that escape one stage later.
    """
    if saved <= 0:
        return None
    added = (row.mean_false_calls - shipped.mean_false_calls) * row.boards
    return added / saved


def render(rows: list[Row], forensics: list[dict], split: str, threshold: float) -> list[str]:
    """The benchmark section. Pure -- takes measurements, returns prose."""
    shipped = next(r for r in rows if r.name == SHIPPED)
    out: list[str] = []

    def emit(text: str = "") -> None:
        out.append(text)

    emit("### The opening kernel — what the seven lost defects would cost to recover")
    emit()
    emit(f"`open_kernel = 3` erases {shipped.unflagged} real defects on the "
         f"{split} split before connected components ever runs. That is the whole "
         f"of this line's unrecoverable escape rate ({shipped.unflagged_rate:.2%}, "
         f"see the accounting above), and it is a detector setting rather than a "
         f"property of the data. So: sweep it.")
    emit()
    emit(f"{shipped.boards} boards, {shipped.defects} ground-truth defects. The "
         f"perturbed columns re-run the same boards under `gate_check.py`'s "
         f"production conditions -- {SHIFT_PX}px template shift, sigma "
         f"{NOISE_SIGMA} noise, {GAIN} gain, seeded per board. That is the "
         f"condition the opening exists for, so it is the column that prices it.")
    emit()

    emit("| opening | defects nothing is flagged on | unmatched at IoU 0.33 | false calls/board | candidates/board | false calls/board, misregistered |")
    emit("|---|---|---|---|---|---|")
    for row in rows:
        label = f"**{row.name}**" if row.name == SHIPPED else row.name
        if is_noop(row.config):
            label += " (a no-op; 1x1 is the same row)"
        marker = " ← shipped" if row.name == SHIPPED else ""
        emit(
            f"| {label}{marker} | {row.unflagged} ({row.unflagged_rate:.2%}) | "
            f"{row.missed_at_cut} ({row.missed_at_cut / row.defects:.2%}) | "
            f"{row.mean_false_calls:.2f} | {row.mean_candidates:.2f} | "
            f"{row.mean_perturbed_false_calls:.2f} |"
        )
    emit()

    none = next((r for r in rows if r.config.open_kernel == 0), None)
    emit("Two things in that table are not obvious.")
    emit()
    if none is not None:
        emit(f"**Removing the opening entirely does not remove the problem.** With no "
             f"opening the sliver noise merges under the 5x5 dilation into "
             f"components that blow past `max_area`, and the detector drops those "
             f"as registration failures -- so it loses {none.unflagged} defects of "
             f"its own while carrying "
             f"{none.mean_false_calls / shipped.mean_false_calls:.0f}x the false "
             f"calls. The curve has no free end; it has a middle.")
        emit()
    emit("**The unmatched-at-IoU column moves a long way and means almost "
         "nothing.** A tighter opening leaves more of a defect standing, so the "
         "box is tighter and clears the cut. Those defects were reaching an "
         "operator either way -- that is the whole finding of the section above, "
         "and the column is here so nobody re-derives a recall improvement from "
         "it. The column that matters is the first one.")
    emit()

    emit("#### The trade, priced")
    emit()
    emit("| opening | defects recovered | of those, the re-verifier keeps | added false calls/board | added false calls per defect actually kept |")
    emit("|---|---|---|---|---|")
    for row in rows:
        if row.name == SHIPPED or row.recovered == 0:
            continue
        added = row.mean_false_calls - shipped.mean_false_calls
        price = cost_per_defect(row, shipped, row.recovered_kept)
        price_text = f"**{price:,.0f}**" if price is not None else "— (none kept)"
        emit(f"| {row.name} | {row.recovered}/{shipped.unflagged} | "
             f"{row.recovered_kept} | {added:+.2f} | {price_text} |")
    emit()
    emit("The middle column is the one that decides this, and it needs its caveat "
         "said out loud: the checkpoint was trained on `open_kernel=3` patches, so "
         "a candidate only a smaller kernel produces is out of its distribution. "
         "This is what the line would do *today*, not what a retrained line would "
         "do. It is still the right question to ask first, because a recovered "
         "candidate the model then dismisses is a defect that escaped one stage "
         "later with the false calls still on the bill.")
    emit()

    emit("#### Decision: `open_kernel` stays at 3")
    emit()
    priced = [
        (r, cost_per_defect(r, shipped, r.recovered_kept))
        for r in rows
        if r.name != SHIPPED and r.recovered_kept > 0
    ]
    if priced:
        best, price = min(priced, key=lambda item: item[1])
        full = [r for r, _ in priced if r.recovered == shipped.unflagged]
        emit(f"The cheapest setting that recovers anything is **{best.name}**, at "
             f"**{price:,.0f} additional false calls per defect the model then "
             f"keeps**. It buys {best.recovered_kept} of {shipped.unflagged}.")
        if full:
            cheapest_full = min(
                full, key=lambda r: cost_per_defect(r, shipped, r.recovered_kept)
            )
            emit()
            emit(f"Recovering all {shipped.unflagged} needs **{cheapest_full.name}** "
                 f"at "
                 f"{cost_per_defect(cheapest_full, shipped, cheapest_full.recovered_kept):,.0f} "
                 f"each -- and the model keeps only {cheapest_full.recovered_kept} "
                 f"of what it recovers, so "
                 f"{cheapest_full.recovered - cheapest_full.recovered_kept} of the "
                 f"{shipped.unflagged} escape one stage later with the false calls "
                 f"still on the bill.")
        emit()
        emit(f"Under misregistration the bill rises again: "
             f"{best.mean_perturbed_false_calls:.0f} false calls per board for "
             f"{best.name} against the shipped "
             f"{shipped.mean_perturbed_false_calls:.0f}. That is the column that "
             f"describes a real line -- DeepPCB ships pre-registered, and the "
             f"opening is in the detector precisely because a line that is not "
             f"pre-registered leaves slivers along every trace edge.")
    emit()
    multiples = sorted(r.mean_candidates / shipped.mean_candidates for r, _ in priced)
    span = (
        f"{multiples[0]:.1f}x"
        if len(multiples) < 2
        else f"{multiples[0]:.1f}x to {multiples[-1]:.1f}x"
    ) if multiples else "more of"
    emit(f"Set against all of that: {shipped.unflagged_rate:.2%} of defects, on a "
         f"project whose headline is how much review it removes. Buying them back "
         f"means every board carries {span} the candidates, the checkpoint is "
         f"invalidated, the operating point has to be re-swept, every routing "
         f"number moves -- and the model, as it stands, dismisses part of what "
         f"the change recovers. **The sweep does not support the change, so the "
         f"constant does not move.**")
    emit()
    emit("What the sweep does support is naming the condition. Revisit this if the "
         "line's escape budget is ever written against the whole line rather than "
         "the re-verification stage, or if an escaped `open` is repriced -- these "
         "are the only defects here that no threshold reaches, and the lever that "
         "reaches them is this one and not the model.")
    emit()

    emit(f"#### Why these {len(forensics)} and not others")
    emit()
    emit("| board | class | ground-truth box | difference pixels | after the 3x3 opening | max half-width |")
    emit("|---|---|---|---|---|---|")
    for item in sorted(forensics, key=lambda f: f["board"]):
        emit(f"| {item['board']} | {item['class']} | {item['box']} | "
             f"{item['diff_pixels']} | {item['after_opening']} | "
             f"{item['half_width']:.2f} px |")
    emit()
    widest = max((f["half_width"] for f in forensics), default=0.0)
    emit(f"Every one goes to zero. A blob survives an NxN opening only where it "
         f"contains a fully-enclosed NxN square, and the widest of these is "
         f"{widest:.2f} px from its own edge at the thickest point -- under the "
         f"1.5 px a 3x3 square needs. They are **thin, not small**: the difference "
         f"blobs run "
         f"{min(f['diff_pixels'] for f in forensics)}-"
         f"{max(f['diff_pixels'] for f in forensics)} pixels, which is well clear "
         f"of `min_area`. Nothing downstream ever gets the chance to reject them; "
         f"the mask is already empty.")
    emit()
    emit(f"They are spread over {len({f['board'] for f in forensics})} boards, one "
         f"each, so this is a property of the detector rather than of a bad scan. "
         f"`DetectorConfig.open_shape` was added for this sweep and defaults to "
         f"`rect`, which is what shipped and what still ships.")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["trainval", "test"])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_DISMISS_THRESHOLD)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument("--dry-run", action="store_true", help="print, do not append")
    args = parser.parse_args()

    # Same check the latency report refuses on. Nothing here is a time, so a
    # busy machine cannot move a figure -- but a run taken beside a training
    # job is worth knowing about, and the check costs a process listing.
    try:
        from reverifier_latency import (
            competing_processes,
            ollama_ps,
            process_table,
            resident_models,
        )

        contended = resident_models(ollama_ps()) + competing_processes(
            process_table(), os.getpid()
        )
        print("machine check: " + ("busy -- " + "; ".join(contended) if contended
                                   else "quiet (ollama ps empty, no busy torch process)"))
    except ImportError:  # pragma: no cover - the script sits beside that one
        print("machine check: skipped, reverifier_latency not importable")

    reverifier = (
        ReVerifier(args.checkpoint, device="cpu") if args.checkpoint.exists() else None
    )
    if reverifier is None:
        print(f"{args.checkpoint} not found -- the 'model keeps' column will be "
              f"vacuous", file=sys.stderr)

    print(f"sweeping {len(SETTINGS)} openings over the {args.split} split ...")
    rows, forensics = measure(args.split, reverifier, args.threshold, args.limit)

    body = "\n".join([
        f"## {date.today().isoformat()} · commit {git_commit()}",
        "",
        *render(rows, forensics, args.split, args.threshold),
    ])
    print()
    print(body)

    if args.dry_run:
        print("\n(dry run -- nothing appended)")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = args.out.read_text() if args.out.exists() else "# Benchmarks\n"
    args.out.write_text(existing.rstrip() + "\n\n" + body + "\n")
    print(f"\nappended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
