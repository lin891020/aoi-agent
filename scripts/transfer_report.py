"""Does the shipped operating point survive a dataset it was not swept on?

The project's own notes name what would make the escape evidence stronger: "a
second dataset with a different prevalence and its own registration problem".
HRIPCB is that dataset, and this script runs the pipeline over it **unchanged**
-- the shipped differencing threshold, the shipped opening kernel, the shipped
registration stage, the shipped checkpoint and the shipped dismissal threshold
-- so that whatever comes out is the operating point applied to a population it
has never seen, which is the question a line asks.

Three things it measures, in the order a defect meets them:

1. **S0, the differencing stage.** How many defects are flagged at all. On
   DeepPCB this stage was gated at recall >= 95% with >= 2 false calls per
   image; whether it clears that bar on photographs is reported before any
   figure downstream of it, because a re-verifier is only asked about what the
   detector hands it.
2. **Registration.** On the ``rotated`` subset -- the same images turned by
   -10..+10 degrees against an un-turned template -- how often
   `aoi/registration.py` acts, and how often it refuses, and why. This is the
   first test of "it does not recover rotation" on a disturbance nobody here
   synthesised.
3. **The re-verifier at the shipped threshold.** Escapes, review reduction and
   prevalence over the queue the detector actually produced.

Every figure is per subset and the two subsets are not averaged. A second
grey-level threshold is reported beside the shipped one, labelled as what it
is -- the best-recall setting on this data, chosen after looking, and not a
number anything ships with.

Run:
    uv run python scripts/transfer_report.py            # both subsets, ~10 min
    uv run python scripts/transfer_report.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aoi_agent.aoi.registration import align  # noqa: E402
from aoi_agent.aoi.simulator import DetectorConfig  # noqa: E402
from aoi_agent.data import hripcb  # noqa: E402
from aoi_agent.vision.inference import DEFAULT_CHECKPOINT, DEFAULT_DISMISS_THRESHOLD, ReVerifier  # noqa: E402
import numpy as np  # noqa: E402

from aoi_agent.vision.patches import PATCH_SIZE, build_patch  # noqa: E402
from escape_accounting import (  # noqa: E402
    ESCAPES,
    REVIEWED_SUB_CUT,
    UNFLAGGED,
    Accounting,
    measure,
)

#: The differencing threshold the project ships. Everything here runs at it
#: first; the second value is whatever the S0 gate found for recall on this
#: data, read out of the gate's own result file rather than typed here -- a
#: literal would be true of the run it was typed after and reprinted on every
#: run since, which is the shape this repository spent 2026-08-26 removing.
SHIPPED_GREY_THRESHOLD = DetectorConfig().threshold
GATE_RESULT = Path("eval/results/gate_check_hripcb_aligned.json")


def best_recall_threshold(path: Path = GATE_RESULT) -> int:
    """The grey threshold the S0 gate found best for recall on HRIPCB."""
    import json

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run the gate first:\n"
            "  uv run python scripts/gate_check.py --dataset hripcb --split aligned "
            "--limit 693 --thresholds 10 15 20 30 45 60 --out " + str(path)
        )
    runs = json.loads(path.read_text())
    runs = runs.get("runs", runs) if isinstance(runs, dict) else runs
    return int(max(runs, key=lambda r: r["recall"])["threshold"])

RECALL_TARGET = 0.95
FALSE_CALLS_TARGET = 2.0


def commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass
class Registration:
    """What the registration stage did over one subset."""

    acted: int = 0
    refused: Counter = None  # type: ignore[assignment]
    angles_acted: list = None  # type: ignore[assignment]
    angles_refused: list = None  # type: ignore[assignment]

    def __post_init__(self):
        self.refused = Counter()
        self.angles_acted = []
        self.angles_refused = []


def registration_over(pairs) -> Registration:
    """Run the stage alone and count its decisions.

    `measure()` runs it too, inside `detect(register=True)`, and throws the
    decision away; this second pass is a phase correlation per pair and costs
    seconds. It is kept separate so `measure()` stays the function every other
    report uses.
    """
    out = Registration()
    for pair in pairs:
        _corrected, alignment = align(pair.load_template(), pair.load_test())
        if alignment.refused is None:
            out.acted += 1
            out.angles_acted.append(abs(pair.angle))
        else:
            out.refused[alignment.refused] += 1
            out.angles_refused.append(abs(pair.angle))
    return out


def prevalence_of(a: Accounting, candidates: int) -> float:
    """Share of the queue that is a real defect.

    Every flagged defect is one candidate covering it; the rest of the queue is
    false calls. HRIPCB ships no false calls of its own, so every one of these
    was manufactured by the differencing stage on this photograph.
    """
    flagged = a.defects - a.outcomes[UNFLAGGED]
    return flagged / candidates if candidates else 0.0


def queue_of(pairs, config: DetectorConfig, reverifier, threshold: float) -> tuple[int, int]:
    """(candidates, dismissed) over the subset, counted per candidate.

    `measure()` accounts per *defect* -- that is its job -- so review reduction,
    which is a statement about the queue, has to be counted here per candidate
    the same way `operating_point.sweep` counts it: dismissed / total.
    """
    import torch

    from aoi_agent.aoi.simulator import detect

    candidates = dismissed = 0
    if reverifier is not None:
        false_call = reverifier.label_names.index("false_call")
    for pair in pairs:
        template, test = pair.load_template(), pair.load_test()
        found = detect(template, test, config)
        candidates += len(found)
        if not found or reverifier is None:
            continue
        batch = torch.from_numpy(
            np.stack([build_patch(template, test, c, PATCH_SIZE) for c in found])
        ).float().div_(255.0)
        with torch.no_grad():
            probabilities = torch.softmax(reverifier.model(batch), dim=1).numpy()
        dismissed += int((probabilities[:, false_call] >= threshold).sum())
    return candidates, dismissed


def subset_section(
    name: str,
    pairs,
    reverifier: ReVerifier | None,
    threshold: float,
    registration: Registration | None,
) -> list[str]:
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)

    emit(f"#### `{name}` — {len(pairs)} boards, {sum(len(p.load_annotations()) for p in pairs)} defects")
    emit()

    if registration is not None:
        total = registration.acted + sum(registration.refused.values())
        emit(
            f"**Registration acted on {registration.acted} of {total} pairs and "
            f"refused {sum(registration.refused.values())}**: "
            + ", ".join(f"`{why}` {n}" for why, n in registration.refused.most_common())
            + "."
        )
        if registration.angles_acted:
            acted = registration.angles_acted
            emit(
                f"Where it acted the rotation was {min(acted):.0f}–{max(acted):.0f}°, "
                f"median {sorted(acted)[len(acted) // 2]:.0f}°; where it refused, "
                f"{min(registration.angles_refused):.0f}–{max(registration.angles_refused):.0f}°, "
                f"median {sorted(registration.angles_refused)[len(registration.angles_refused) // 2]:.0f}°. "
                "Phase correlation estimates a translation, so on a pure rotation "
                "it either finds a small spurious shift and applies it, or finds no "
                "peak and declines. Neither is a correction, and the candidate "
                "counts below are what the differencing stage sees either way."
            )
        emit()

    emit("| grey threshold | candidates / board | defects flagged | of which sub-cut IoU | never flagged | prevalence | escapes | escape rate | review removed |")
    emit("|---|---|---|---|---|---|---|---|---|")
    for grey, label in ((SHIPPED_GREY_THRESHOLD, "shipped"), (best_recall_threshold(), "best recall on this data, per the gate")):
        config = DetectorConfig(threshold=grey, register=True)
        a = measure(name, reverifier, config, threshold, pairs=pairs)
        candidates, dismissed = queue_of(pairs, config, reverifier, threshold)
        flagged = a.defects - a.outcomes[UNFLAGGED]
        escapes = sum(a.outcomes[k] for k in ESCAPES)
        emit(
            f"| {grey} ({label}) | {candidates / len(pairs):.1f} | "
            f"{flagged}/{a.defects} = {flagged / a.defects:.1%} | {a.outcomes[REVIEWED_SUB_CUT]} | "
            f"{a.outcomes[UNFLAGGED]} | {prevalence_of(a, candidates):.1%} | "
            f"{escapes} | {a.whole_line_rate:.2%} | "
            f"{(dismissed / candidates if candidates else 0):.1%} |"
        )
    emit()
    return lines


def render(sections: list[str], threshold: float, checkpoint: Path) -> list[str]:
    out: list[str] = []

    def emit(text: str = "") -> None:
        out.append(text)
    emit("### Transfer — the shipped pipeline on HRIPCB, a dataset it was never swept on")
    emit()
    emit(
        f"Ten photographed bare boards, one template each, 693 images with defects "
        f"drawn onto the template, and the same 693 rotated by -10..+10° in the "
        f"dataset's own `rotation/` set. Images are downscaled by {hripcb.SCALE} so "
        f"the median defect is 39 px long, which is what it is on DeepPCB and what "
        f"the 64 px patch was sized against. Nothing else changes: differencing "
        f"threshold, opening kernel, registration stage, checkpoint "
        f"`{checkpoint.name}` and dismissal threshold {threshold} are the shipped "
        f"values. `scripts/transfer_report.py`, commit `{commit()}`."
    )
    emit()
    emit(
        "**Read S0 before anything under it.** The re-verifier is asked only about "
        "candidates the differencing stage produces. On DeepPCB that stage was "
        f"gated at recall ≥ {RECALL_TARGET:.0%} with ≥ {FALSE_CALLS_TARGET:.0f} false "
        "calls per image (`scripts/gate_check.py`); the same gate was run on this "
        "data and its result is in the section that precedes this one. Whatever it "
        "found, every escape figure below is conditional on the queue that stage "
        "handed over, and a defect it never flagged is not an escape the model "
        "could have prevented."
    )
    emit()
    emit(
        "**What `prevalence` means here.** HRIPCB contains no false calls of its "
        "own -- every annotated box is a real defect -- so every false call in the "
        "queue was manufactured by differencing this photograph against its "
        "template. The figure is a property of the detector on this imagery, not "
        "of the dataset, and it is the other prevalence the project's notes asked "
        "to see the operating point under."
    )
    emit()
    out.extend(sections)
    emit(
        "**What this does not establish.** One checkpoint, trained on binarised "
        "640 px pairs, read on colour photographs at half resolution; the grey "
        "threshold labelled *best recall here* was chosen after looking at this "
        "data and ships nowhere. `missing_hole` is a class the model has never "
        "seen and DeepPCB does not have, so its rows measure only whether the "
        "model will dismiss an unfamiliar defect. And the rotated subset's "
        "candidates are dominated by misregistration, which is the condition, "
        "not a confound: the queue a rotation produces is the queue a line "
        "would have to review."
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_DISMISS_THRESHOLD)
    parser.add_argument("--subsets", nargs="+", default=list(hripcb.SETS), choices=hripcb.SETS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    reverifier = ReVerifier(args.checkpoint, device="cpu") if args.checkpoint.exists() else None
    if reverifier is None:
        print(f"{args.checkpoint} not found -- detector only", file=sys.stderr)

    sections: list[str] = []
    for subset in args.subsets:
        pairs = hripcb.load(subset)
        if args.limit:
            pairs = pairs[: args.limit]
        print(f"{subset}: {len(pairs)} pairs ...", file=sys.stderr)
        registration = registration_over(pairs) if subset == "rotated" else None
        sections.extend(subset_section(subset, pairs, reverifier, args.threshold, registration))

    lines = render(sections, args.threshold, args.checkpoint)
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
