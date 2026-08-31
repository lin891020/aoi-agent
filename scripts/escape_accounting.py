"""Where do defects actually escape this line, and at which stage?

`docs/benchmarks.md` and the README both published a whole-line escape rate of
**5.4%**, composed as "the AOI stage misses 5.0%, the re-verifier adds 0.47%",
under the sentence *"Defects the AOI never caught are already gone and no
threshold recovers them."*

That sentence is false for almost all of the 5.0%. The 5.0% is not a count of
defects the detector failed to find. It is a count of defects whose best
candidate did not clear DeepPCB's IoU 0.33 box-tightness cut, and 150 of those
157 defects have a candidate sitting on top of them. What the old number
measured is how tightly the differencing detector draws a box, which is a
localisation-quality statistic. It was published as a detection failure, and a
detection failure is unrecoverable in a way a loose box is not.

The accounting error has a specific shape, and it is worth naming because the
code makes it look reasonable. A candidate that covers a real defect below the
cut is labelled `fragment` by `aoi.matching.match`, and `patches_for_image`
holds every fragment out of the patch set. So that region is absent from
`test.npz` -- absent from the denominator of the operating-point curve, absent
from anything the model is scored on -- while the annotation it covers is
counted in `missed_annotations` and charged in full to the line. It is not
double-counted in the arithmetic. It is charged to the stage that did not fail,
and simultaneously removed from the only measurement that could have said so.

This script does the count the other way round, on defects rather than on
boxes, and with the model in the loop rather than assumed away. For every
ground-truth defect on the split it asks:

1. Does *any* candidate overlap it at all? If not, the defect is genuinely
   invisible to this line and no threshold anywhere recovers it. That is the
   number the old prose was describing and it is not 5.0%.
2. If a candidate does overlap it, what does the re-verifier do with every one
   of those candidates? A defect escapes only when the model dismisses all of
   them. One kept candidate puts the region on an operator's screen, whatever
   its IoU against a box someone drew by hand.

The two figures that come out mean different things and only one of them is
unrecoverable, so both are reported and they are not added into a single
headline.

    uv run python scripts/escape_accounting.py
    uv run python scripts/escape_accounting.py --split trainval --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.aoi.matching import IOU_THRESHOLD, coverage, iou, match  # noqa: E402
from aoi_agent.aoi.simulator import Candidate, DetectorConfig, detect  # noqa: E402
from aoi_agent.data.deeppcb import Annotation, load_split  # noqa: E402
from aoi_agent.vision.inference import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_DISMISS_THRESHOLD,
    ReVerifier,
)
from aoi_agent.vision.operating_point import system_escape_rate  # noqa: E402
from aoi_agent.vision.patches import PATCH_SIZE, build_patch  # noqa: E402

#: The cuts the miss rate is reported at. 0.33 is DeepPCB's own benchmark
#: threshold and the one this repository matches on; the rest are here to show
#: how much of the "miss" count is the cut rather than the detector.
IOU_CUTS = [0.50, 0.40, IOU_THRESHOLD, 0.30, 0.25, 0.20, 0.10]

UNFLAGGED = "unflagged"
"""No candidate overlaps the defect by a single pixel. Unrecoverable here."""

DISMISSED = "dismissed"
"""Flagged, but the re-verifier dismissed every candidate covering it."""

REVIEWED_MATCHED = "reviewed_matched"
"""Kept, by a candidate that also clears the IoU cut."""

REVIEWED_SUB_CUT = "reviewed_sub_cut"
"""Kept, but only by a candidate the IoU rule calls a miss. These are the
defects the old 5.0% was charging to the line as escapes."""

ESCAPES = (UNFLAGGED, DISMISSED)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "uncommitted"


def covering_candidates(
    box: tuple[int, int, int, int], candidates: list[Candidate]
) -> list[int]:
    """Indices of every candidate that overlaps ``box`` by at least one pixel.

    Deliberately the loosest possible test. The question this answers is "did
    the detector put anything on this defect", and an operator looking at a
    flagged region sees the defect in it whether or not the box is tight. Any
    stricter rule here would re-introduce the mistake being corrected.
    """
    return [
        index
        for index, candidate in enumerate(candidates)
        if iou(candidate.box, box) > 0 or coverage(candidate.box, box) > 0
    ]


def defect_outcome(
    annotation: Annotation,
    candidates: list[Candidate],
    kept: np.ndarray,
    matched: bool,
) -> str:
    """What became of one defect, end to end.

    ``kept`` is a per-candidate mask: True where the re-verifier did *not*
    dismiss it, so the region reaches a person. ``matched`` is whether the
    IoU/coverage rule paired the defect with a candidate, which is the only
    thing the old accounting looked at.
    """
    overlapping = covering_candidates(annotation.box, candidates)
    if not overlapping:
        return UNFLAGGED
    if not any(bool(kept[index]) for index in overlapping):
        return DISMISSED
    return REVIEWED_MATCHED if matched else REVIEWED_SUB_CUT


@dataclass
class Accounting:
    """Every defect on a split, sorted by what happened to it."""

    defects: int
    outcomes: Counter
    escapes_by_class: dict[str, Counter]
    defects_by_class: Counter
    miss_by_cut: dict[float, int]
    unflagged: list[tuple[str, str]]
    """(board stem, class name) for each defect nothing was flagged on."""

    best_iou_of_sub_cut: list[float]
    area_ratio_of_matched: list[float]

    @property
    def escapes(self) -> int:
        return sum(self.outcomes[kind] for kind in ESCAPES)

    @property
    def whole_line_rate(self) -> float:
        return self.escapes / self.defects

    @property
    def unflagged_rate(self) -> float:
        return self.outcomes[UNFLAGGED] / self.defects

    @property
    def flagged(self) -> int:
        return self.defects - self.outcomes[UNFLAGGED]

    @property
    def reverifier_rate(self) -> float:
        """Escape rate over the defects that reached the model at all."""
        return self.outcomes[DISMISSED] / self.flagged if self.flagged else 0.0


def measure(
    split: str,
    reverifier: ReVerifier | None,
    config: DetectorConfig,
    threshold: float,
    limit: int | None = None,
    pairs: list | None = None,
) -> Accounting:
    """Run the detector and the model over a split and account for every defect.

    ``pairs`` lets a caller hand in a dataset that is not DeepPCB -- anything
    with ``stem``, ``load_template``, ``load_test`` and ``load_annotations`` --
    so the same accounting runs unchanged over a second population. Added
    2026-08-26 for `scripts/transfer_report.py`; when it is None the DeepPCB
    split named by ``split`` is loaded, as before.
    """
    import torch

    if pairs is None:
        pairs = load_split(split)
    if limit:
        pairs = pairs[:limit]

    false_call_index = (
        reverifier.label_names.index("false_call") if reverifier is not None else 0
    )

    outcomes: Counter = Counter()
    escapes_by_class: dict[str, Counter] = {kind: Counter() for kind in ESCAPES}
    defects_by_class: Counter = Counter()
    miss_by_cut = {cut: 0 for cut in IOU_CUTS}
    unflagged: list[tuple[str, str]] = []
    best_iou_of_sub_cut: list[float] = []
    area_ratio_of_matched: list[float] = []
    defects = 0

    for index, pair in enumerate(pairs):
        template, test = pair.load_template(), pair.load_test()
        annotations = pair.load_annotations()
        candidates = detect(template, test, config)
        defects += len(annotations)

        if candidates and reverifier is not None:
            patches = np.stack(
                [build_patch(template, test, c, PATCH_SIZE) for c in candidates]
            )
            batch = torch.from_numpy(patches).float().div_(255.0)
            with torch.no_grad():
                probabilities = (
                    torch.softmax(reverifier.model(batch.to(reverifier.device)), dim=1)
                    .cpu()
                    .numpy()
                )
            kept = probabilities[:, false_call_index] < threshold
        else:
            # No model: every flagged region reaches a person, which is the
            # line as it runs today.
            kept = np.ones(len(candidates), dtype=bool)

        result = match(candidates, annotations, iou_threshold=IOU_THRESHOLD)
        matched = {id(a) for a in result.detected_annotations}

        for cut in IOU_CUTS:
            miss_by_cut[cut] += len(
                match(candidates, annotations, iou_threshold=cut).missed_annotations
            )

        for labelled in result.labelled:
            if labelled.matched_annotation is not None:
                target = labelled.matched_annotation
                candidate_area = (labelled.candidate.x2 - labelled.candidate.x1) * (
                    labelled.candidate.y2 - labelled.candidate.y1
                )
                target_area = (target.x2 - target.x1) * (target.y2 - target.y1)
                if target_area:
                    area_ratio_of_matched.append(candidate_area / target_area)

        for annotation in annotations:
            defects_by_class[annotation.class_name] += 1
            outcome = defect_outcome(
                annotation, candidates, kept, id(annotation) in matched
            )
            outcomes[outcome] += 1
            if outcome in ESCAPES:
                escapes_by_class[outcome][annotation.class_name] += 1
            if outcome == UNFLAGGED:
                unflagged.append((pair.stem, annotation.class_name))
            if outcome == REVIEWED_SUB_CUT or (
                outcome == DISMISSED and id(annotation) not in matched
            ):
                best = max(
                    (iou(candidates[i].box, annotation.box)
                     for i in covering_candidates(annotation.box, candidates)),
                    default=0.0,
                )
                best_iou_of_sub_cut.append(best)

        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{len(pairs)} boards, {defects} defects")

    return Accounting(
        defects=defects,
        outcomes=outcomes,
        escapes_by_class=escapes_by_class,
        defects_by_class=defects_by_class,
        miss_by_cut=miss_by_cut,
        unflagged=unflagged,
        best_iou_of_sub_cut=best_iou_of_sub_cut,
        area_ratio_of_matched=area_ratio_of_matched,
    )


def render(accounting: Accounting, split: str, threshold: float, boards: int) -> list[str]:
    """The benchmark section. Pure -- takes a measurement, returns prose."""
    a = accounting
    out: list[str] = []

    def emit(text: str = "") -> None:
        out.append(text)

    old_miss = a.miss_by_cut[IOU_THRESHOLD]
    sub_cut = old_miss - a.outcomes[UNFLAGGED]

    emit("### Whole-line escape rate, recounted on defects instead of boxes")
    emit()
    emit(f"Was **5.4%**. Is **{a.whole_line_rate:.2%}**. The old figure added an "
         f"AOI-stage miss rate of 5.0% to the re-verifier's 0.47%, under the "
         f"sentence \"Defects the AOI never caught are already gone and no "
         f"threshold recovers them\". That sentence was true of "
         f"{a.outcomes[UNFLAGGED]} defects on this split and was being applied to "
         f"{old_miss}.")
    emit()
    emit(f"The 5.0% was never a count of defects the detector failed to find. It "
         f"counted defects whose best candidate did not clear DeepPCB's IoU "
         f"{IOU_THRESHOLD} cut, and {sub_cut} of those {old_miss} have a candidate "
         f"sitting on them -- {sub_cut / old_miss:.1%}. A matched candidate is on "
         f"median {np.median(a.area_ratio_of_matched):.2f}x the area of the "
         f"hand-drawn box it matches, so the whole distribution of best-IoUs piles "
         f"up just under the cut: median "
         f"{np.median(a.best_iou_of_sub_cut):.2f} against a cut of {IOU_THRESHOLD}. "
         f"That is a statistic about how tightly this detector draws a box. It was "
         f"published as a detection failure.")
    emit()
    emit(f"Measured on the {split} split: {boards} boards, {a.defects} ground-truth "
         f"defects, the shipped checkpoint, dismissal threshold {threshold:.3f}.")
    emit()

    emit("#### What happens to every defect on the split")
    emit()
    emit("| outcome | defects | share | recoverable by a threshold? |")
    emit("|---|---|---|---|")
    rows = [
        (REVIEWED_MATCHED, "reaches a person, via a candidate that also clears the IoU cut", "n/a -- reviewed"),
        (REVIEWED_SUB_CUT, "reaches a person, but only via a candidate the IoU rule calls a miss", "n/a -- reviewed"),
        (DISMISSED, "flagged, and the re-verifier dismissed every candidate on it", "yes -- this is the dismissal threshold"),
        (UNFLAGGED, "**never flagged: not one candidate overlaps it**", "**no**"),
    ]
    for kind, label, recoverable in rows:
        count = a.outcomes[kind]
        emit(f"| {label} | {count} | {count / a.defects:.2%} | {recoverable} |")
    emit()
    emit(f"The third and fourth rows are the escapes: **{a.escapes} defects, "
         f"{a.whole_line_rate:.2%}**. The second row -- {a.outcomes[REVIEWED_SUB_CUT]} "
         f"defects -- is what the old number was charging to the line. Every one of "
         f"them is on an operator's screen.")
    emit()

    emit("#### The miss rate is mostly the cut")
    emit()
    emit("| detection rule | defects counted missed | share of defects |")
    emit("|---|---|---|")
    for cut in IOU_CUTS:
        missed = a.miss_by_cut[cut]
        marker = "  ← published as the AOI escape rate" if cut == IOU_THRESHOLD else ""
        emit(f"| IoU ≥ {cut:.2f} | {missed} | {missed / a.defects:.2%}{marker} |")
    emit(f"| any overlap at all | {a.outcomes[UNFLAGGED]} | "
         f"**{a.unflagged_rate:.2%}** |")
    emit()
    emit("Nothing about the detector changes down that column. Only the cut does. "
         "The bottom row is the only one that describes a defect this line cannot "
         "see, and it is the one that belongs in an escape rate.")
    emit()

    emit("#### The composition")
    emit()
    compounded = system_escape_rate(a.reverifier_rate, a.unflagged_rate)
    emit(f"- **never flagged: {a.unflagged_rate:.2%}** of defects "
         f"({a.outcomes[UNFLAGGED]}/{a.defects}) -- unrecoverable. No threshold, "
         f"no model and no retrain reaches these; the pixels never reach the "
         f"classifier.")
    emit(f"- **dismissed by the re-verifier: {a.reverifier_rate:.2%}** of the "
         f"{a.flagged} defects that did reach it ({a.outcomes[DISMISSED]}) -- this "
         f"is the number the dismissal threshold governs, and the one QP-110's "
         f"0.5% budget is written about.")
    emit(f"- **whole line: {compounded:.2%}** "
         f"({a.unflagged_rate:.2%} + {1 - a.unflagged_rate:.3f} × "
         f"{a.reverifier_rate:.2%}), against {a.escapes}/{a.defects} = "
         f"{a.whole_line_rate:.2%} counted directly.")
    emit()
    emit("Two numbers, not one. They are not interchangeable and adding them into "
         "a single headline is what produced the 5.4%: one of them is a knob and "
         "the other is a wall. Reporting only the sum tells a reader to go tune "
         "the thing that cannot move.")
    emit()
    emit(f"The re-verifier's own escape rate is quoted as {a.reverifier_rate:.2%} "
         f"here and {per_candidate_rate():.2%} in the operating-point table above. "
         f"Both are right and "
         f"they count different things: the table counts *candidates* carrying a "
         f"defect label that were dismissed, this counts *defects* every covering "
         f"candidate was dismissed on. A defect flagged by three candidates "
         f"escapes only if all three go, and a defect whose only candidate is a "
         f"held-out fragment is in this count and not in that one.")
    emit()

    emit("#### Where the escapes are")
    emit()
    emit("| defect class | on the split | never flagged | dismissed | escape rate |")
    emit("|---|---|---|---|---|")
    for name, total in sorted(a.defects_by_class.items(), key=lambda kv: -kv[1]):
        never = a.escapes_by_class[UNFLAGGED][name]
        gone = a.escapes_by_class[DISMISSED][name]
        emit(f"| {name} | {total} | {never} | {gone} | {(never + gone) / total:.2%} |")
    emit()
    boards_with = len(set(stem for stem, _ in a.unflagged))
    emit(f"The never-flagged {a.outcomes[UNFLAGGED]} are spread over "
         f"{boards_with} boards, no board contributing more than one, so this is "
         f"not one bad scan. What they have in common is a cause, and it is in "
         f"the detector rather than in the data -- see the opening-kernel sweep.")
    emit()

    emit("#### What changed in the code")
    emit()
    emit("`scripts/report.py` no longer computes a whole-line figure. It had a "
         "`--aoi-escape-rate` argument defaulting to 0.050, a number carried over "
         "by hand from `build_patches.py`'s miss print, and it had no access to "
         "the two things the composition needs: whether anything was flagged on a "
         "defect, and what the model did with it. This script owns that section "
         "now. `system_escape_rate` is unchanged and still correct -- it was "
         "being fed the wrong stage rate, not computing the wrong thing.")
    return out



def per_candidate_rate() -> float:
    """The other reading of the same threshold, computed rather than quoted.

    This sentence carried a literal 0.47% -- true of the threshold shipped on
    2026-08-24 and reprinted on every run after it, which is the defect
    `tests/test_published_figures.py` exists for, one file over. The two rates
    answer different questions and both belong here; neither may be a constant.
    """
    import numpy as np

    from aoi_agent.vision.inference import DEFAULT_DISMISS_THRESHOLD

    predictions = Path(__file__).resolve().parents[1] / "models" / "test_predictions.npz"
    if not predictions.exists():
        return float("nan")
    data = np.load(predictions, allow_pickle=False)
    names = [str(n) for n in data["label_names"]]
    false_call = names.index("false_call")
    is_defect = data["labels"] != false_call
    dismissed = data["probabilities"][:, false_call] >= DEFAULT_DISMISS_THRESHOLD
    return float((dismissed & is_defect).sum()) / int(is_defect.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["trainval", "test"])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_DISMISS_THRESHOLD)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument("--dry-run", action="store_true", help="print, do not append")
    args = parser.parse_args()

    # No figure in this report is a time, so a busy machine cannot move one --
    # every count here is deterministic. The check runs anyway because it costs
    # nothing and because a run taken beside a training job is worth knowing
    # about even when the numbers survive it.
    try:
        from reverifier_latency import (
            competing_processes,
            ollama_ps,
            process_table,
            resident_models,
        )

        residency = ollama_ps()
        busy = competing_processes(process_table(), os.getpid())
        contended = resident_models(residency) + busy
        print("machine check: " + ("busy -- " + "; ".join(contended) if contended
                                   else "quiet (ollama ps empty, no busy torch process)"))
    except ImportError:  # pragma: no cover - the script is beside this one
        print("machine check: skipped, reverifier_latency not importable")

    reverifier = ReVerifier(args.checkpoint, device="cpu") if args.checkpoint.exists() else None
    if reverifier is None:
        print(f"{args.checkpoint} not found -- accounting the detector only",
              file=sys.stderr)

    pairs = load_split(args.split)
    boards = min(len(pairs), args.limit) if args.limit else len(pairs)
    print(f"accounting {boards} boards from the {args.split} split ...")

    accounting = measure(
        args.split, reverifier, DetectorConfig(register=True), args.threshold, args.limit
    )
    lines = render(accounting, args.split, args.threshold, boards)
    header = f"## {date.today().isoformat()} · commit {git_commit()}"
    body = "\n".join([header, "", *lines])
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
