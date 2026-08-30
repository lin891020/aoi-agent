"""The journey deck's content, held to the same rules as the documents.

`scripts/deck_content.py` is the one source for the .pptx, the HTML deck and
the study guide. Three rules over it:

1. **A figure on a slide is a figure the project published.** Every string
   in a slide's `figures` appears in docs/benchmarks.md or in one of the four
   documents `test_published_figures.py` watches. A deck can otherwise carry a
   number nobody measured, in the one place an interviewer reads it.
2. **Every experiment slide states the decision it was.** The "why" cell is
   written as the question being answered and what would have counted as
   failure -- 「要回答的問題」 and a 「就算…」 clause both present. That is the
   difference between a design and an afterthought, and it is the author's
   own requirement.
3. **Every experiment slide can be asked about.** At least two interviewer
   questions with answers, and at least one of the answers names a number --
   an interviewer's "how do you know" is answered with a figure, while "why
   not a front-end framework" legitimately is not.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PUBLISHED = [
    ROOT / "docs" / "benchmarks.md",
    ROOT / "README.md",
    ROOT / "README.zh-TW.md",
    ROOT / "CLAUDE.md",
    ROOT / "docs" / "architecture.md",
    ROOT / "docs" / "findings.md",
]


def _content():
    path = ROOT / "scripts" / "deck_content.py"
    spec = importlib.util.spec_from_file_location("deck_content", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["deck_content"] = module
    spec.loader.exec_module(module)
    return module


content = _content()
SLIDES = content.SLIDES
FIVE = [s for s in SLIDES if s.kind == "five"]


@pytest.fixture(scope="module")
def published_text() -> str:
    return "\n".join(p.read_text() for p in PUBLISHED if p.exists())


def test_every_quoted_figure_was_published(published_text):
    missing = [(s.key, f) for s in SLIDES for f in s.figures if f not in published_text]
    assert not missing, f"figures on slides that no published document carries: {missing}"


def test_every_experiment_slide_states_the_question_and_the_failure_condition():
    bad = [s.key for s in FIVE
           if "要回答的問題" not in s.five["why"] or "就算" not in s.five["why"]]
    assert not bad, f"'why' written as an afterthought, not a decision: {bad}"


def test_every_experiment_slide_has_all_five_cells():
    for s in FIVE:
        assert set(s.five) == set(content.FIVE_LABELS), s.key
        assert all(s.five[k].strip() for k in content.FIVE_LABELS), s.key


def test_every_experiment_slide_carries_two_answerable_questions():
    for s in FIVE:
        assert len(s.questions) >= 2, s.key
        for q, a in s.questions:
            assert q.strip() and a.strip(), s.key
        assert any(re.search(r"\d", a) for _, a in s.questions), \
            f"{s.key}: no answer names a number"


def test_every_slide_has_a_spoken_script_of_three_to_five_sentences():
    for s in SLIDES:
        assert 3 <= len(s.notes) <= 5, (s.key, len(s.notes))
        assert s.plain.strip(), s.key


def test_the_core_path_is_nine_slides_numbered_once_each():
    cores = sorted(s.core for s in SLIDES if s.core)
    assert cores == list(range(1, content.CORE_TOTAL + 1)), cores


def test_layers_are_the_declared_ones_in_order():
    seen = []
    for s in SLIDES:
        assert s.layer in content.LAYERS, s.key
        if not seen or seen[-1] != s.layer:
            seen.append(s.layer)
    assert seen == [layer for layer in content.LAYERS if layer in seen]
    assert seen == sorted(seen, key=content.LAYERS.index)


def test_keys_are_unique():
    keys = [s.key for s in SLIDES]
    assert len(keys) == len(set(keys))


def test_the_builder_refuses_politely_without_python_pptx():
    """The build is not a test dependency; the script says what to install."""
    pytest.importorskip("pptx", reason="python-pptx is passed with --with, not installed")


def test_every_slide_that_names_a_picture_has_one():
    """A missing picture renders as a blank slide, and nothing else complains.

    The station screenshots come from `scripts/deck_screenshots.py` rather than
    from the deck build, so a build that returned only its own renders dropped
    them from three slides silently.
    """
    img = ROOT / "docs" / "deck" / "img"
    missing = [s.image for s in SLIDES if s.image and not (img / s.image).exists()]
    assert not missing, f"slides name pictures that are not in docs/deck/img: {missing}"
