"""The shot list fits the band it is shown in, in both languages.

Until 2026-09-05 the demo burned one paragraph per scene across the page.
The rule now is one sentence per cue, two subtitle lines at most, each under
the length Netflix's Chinese (Traditional) and English guides allow, and
never up for less than a subtitle's minimum. The two languages must be the
same shot list -- same scenes, same number of cues, framing the same
elements -- or a take in one language shows something the other never
mentions.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def script():
    path = ROOT / "scripts" / "demo_script.py"
    spec = importlib.util.spec_from_file_location("demo_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["demo_script"] = module  # dataclasses resolves annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def test_every_cue_wraps_to_two_lines_under_the_guide(script):
    for key, i, cue in script.CUES:
        for lang, limit in (("zh-TW", script.ZH_LINE), ("en", script.EN_LINE)):
            wrapped = script.lines(cue, lang)
            assert len(wrapped) <= 2, f"{script.cue_id(key, i)} ({lang}) needs {len(wrapped)} lines"
            for line in wrapped:
                assert len(line) <= limit, f"{script.cue_id(key, i)} ({lang}): {len(line)} > {limit}: {line!r}"
            assert "".join(wrapped).replace(" ", "") == script.text(cue, lang).replace(" ", ""), "wrap lost text"


def test_every_cue_can_be_read_in_its_hold(script):
    """The hold is what the silent cut gives a viewer; the guide's reading
    speed is the most a subtitle may ask of them."""
    for key, i, cue in script.CUES:
        zh = script.hold(cue, "zh-TW")
        assert zh >= script.MIN_HOLD
        assert len(cue.zh) / zh <= script.ZH_CPS, f"{script.cue_id(key, i)} zh reads at {len(cue.zh)/zh:.1f} cps"
        en = script.hold(cue, "en")
        assert en >= script.MIN_HOLD
        assert len(cue.en) / en <= script.EN_CPS, f"{script.cue_id(key, i)} en reads at {len(cue.en)/en:.1f} cps"


def test_both_languages_are_the_same_shot_list(script):
    for key, cues in script.SCENES:
        for cue in cues:
            assert cue.zh.strip() and cue.en.strip(), f"{key}: an empty cue"
    ids = [script.cue_id(k, i) for k, i, _ in script.CUES]
    assert len(ids) == len(set(ids))


def test_the_audience_is_told_what_the_system_is_before_any_page(script):
    """Somebody who has never seen an AOI queue watches this. The first
    scene is a card, not a page, and it says what an AOI is for."""
    first_key, first_cues = script.SCENES[0]
    assert first_key == "intro"
    assert first_cues[0].spot is None
    assert "AOI" in first_cues[0].zh and "AOI" in first_cues[0].en


def test_the_shot_list_document_is_the_table(script):
    """docs/demo-script.md is generated; a hand edit there is a second
    source that the recorder does not read."""
    doc = (ROOT / "docs" / "demo-script.md").read_text()
    assert doc == script.shot_list(), "run: uv run python scripts/demo_script.py"


def test_every_spot_names_one_element(script):
    for key, i, cue in script.CUES:
        if cue.spot is None:
            continue
        spec = cue.spot.removeprefix("first:")
        sel = spec.split("@@")[0]
        assert sel.strip() and "," not in sel, f"{script.cue_id(key, i)}: {cue.spot!r} is a list, not one element"


def test_every_cue_has_a_known_phase(script):
    for key, i, cue in script.CUES:
        assert cue.phase in script.PHASES, f"{script.cue_id(key, i)}: phase {cue.phase!r}"
    # The scenes that wait on a model say something while they wait.
    for key in ("ask", "control", "switch"):
        assert script.cues_in(key, "wait"), f"{key} has no cue for the wait"
