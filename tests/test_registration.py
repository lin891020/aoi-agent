"""What misalignment costs, measured before there was a stage to answer it.

`test_the_pipeline_still_has_no_registration_stage` lived here and asserted the
absence -- exactly one image-alignment call, in the function that *adds*
misalignment. It failed the day `aoi/registration.py` was written, which is what
it was for: the failure is the prompt to rewrite the section it guarded, not a
regression. Retired rather than relaxed.

What stays is the cost curve, which is still true and is what justified building
the stage. What the stage recovers is `tests/test_registration_stage.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


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
