"""The stage this pipeline does not have.

The only `warpAffine` in the project *introduces* misalignment. Nothing aligns
an unaligned pair, because DeepPCB never handed it one -- its authors did the
registration and the binarisation before shipping. So every published figure
here is measured on a pipeline that begins after the hardest stage of real AOI,
and what these tests hold is that the fact stays stated and the curve stays
measured.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "aoi_agent"


def test_the_pipeline_still_has_no_registration_stage():
    """Not a wish -- a fact about the source, asserted so it cannot stop being
    true quietly.

    If somebody adds registration, this fails, and the right response is to
    delete this test and rewrite the benchmarks section it guards. What must not
    happen is a registration stage arriving while the report still says there is
    none, or the report being deleted while there still is none.
    """
    # `--include=*.py`: a compiled `.pyc` carries the same call and matching it
    # made this depend on whether anything had been imported yet.
    aligners = subprocess.run(
        ["grep", "-rn", "-E", "--include=*.py",
         r"warpAffine|findHomography|estimateAffine|"
         r"matchTemplate|phaseCorrelate", str(SOURCE)],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()

    assert len(aligners) == 1, (
        "the project gained or lost an image-alignment call:\n  "
        + "\n  ".join(aligners)
    )
    assert "simulator.py" in aligners[0]
    # And it is the one that *adds* misalignment, not one that removes it.
    source = (SOURCE / "aoi" / "simulator.py").read_text()
    assert "def apply_perturbation" in source
    assert source.index("def apply_perturbation") < source.index("cv2.warpAffine")


@pytest.mark.dataset
def test_recall_falls_with_misregistration_and_the_queue_grows():
    """Both halves of the curve, at the two ends the report publishes.

    Recall is the serious one: a defect the AOI never emits is not inside the
    escape budget but before it, where no threshold reaches it.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from gate_check import evaluate

    from aoi_agent.aoi.simulator import DetectorConfig, Perturbation
    from aoi_agent.data.deeppcb import load_split

    pairs = load_split("test")[:40]
    config = DetectorConfig()

    clean = evaluate(pairs, config, None)
    shifted = evaluate(pairs, config, Perturbation(max_shift_px=4))

    assert shifted["recall"] < clean["recall"], (
        "misregistration has to cost recall, or the sweep is not doing anything"
    )
    assert shifted["mean_candidates"] > clean["mean_candidates"] * 2, (
        "and it has to cost queue, which is the half the re-verifier absorbs"
    )


def test_the_benchmarks_section_states_the_missing_stage():
    """The claim is the point of the section, so it is held rather than left to
    survive an edit."""
    benchmarks = (ROOT / "docs" / "benchmarks.md").read_text()

    assert "### Registration — the stage this pipeline does not have" in benchmarks
    assert "Nothing here aligns an unaligned pair" in benchmarks
