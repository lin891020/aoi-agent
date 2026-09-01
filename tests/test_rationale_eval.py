"""What the rationale scorer counts, and what it refuses to publish.

The run itself needs the model and the store, so what is held here is the
classifier of findings -- the part that decides what a number in the table
means. Three things it must not get wrong:

1. **A checked kind is exact.** `foreign_document` compares the documents the
   rationale cites against the documents retrieval returned; if that comparison
   drifts, the count stops being reproducible from the stored run and becomes
   an opinion.
2. **A missing rationale is one finding, not six.** A run that produced no
   prose cannot also be missing a class name or citing a foreign document, and
   counting it under every kind would make the table sum to nonsense.
3. **One language may not be published.** A rationale is written once, in the
   line's language; a count over one of them reads as a claim about both. This
   is `synthesis_eval.py`'s rule, inherited.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scorer():
    pytest.importorskip("langgraph", reason="the scorer drives the disposition graph")
    return _script("rationale_eval")


def _state(rationale, documents=("WI-201",), flags=(), status="explained"):
    return {
        "agent_rationale": rationale,
        "rationale_flags": list(flags),
        "standards": [{"document": d, "heading": "h", "text": "t"} for d in documents],
        "explanation_status": status,
    }


def test_a_document_the_retrieval_never_returned_is_a_finding(scorer):
    found = scorer.findings(
        _state("Per WI-206 a break inside a pad is rejected.", documents=("WI-201",)),
        "open")
    assert found["foreign_document"] == ["WI-206"]


def test_a_document_that_was_retrieved_is_not(scorer):
    found = scorer.findings(
        _state("WI-201 says any confirmed open is critical.", documents=("WI-201",)),
        "open")
    assert "foreign_document" not in found


def test_a_run_with_no_rationale_raises_exactly_one_kind(scorer):
    found = scorer.findings(_state("", documents=(), status="timed_out"), "open")
    assert found == {"no_explanation": ["timed_out"]}


def test_the_class_the_classifier_decided_has_to_appear(scorer):
    # i18n.LANGUAGE_NOTE tells both languages to spell identifiers as the data
    # spells them, so this also checks whether that instruction is obeyed.
    assert "class_not_named" in scorer.findings(_state("此區域為開路。"), "open")
    assert "class_not_named" not in scorer.findings(_state("此區域為 open。"), "open")


def test_naming_another_class_is_raised_for_a_person_not_counted_as_wrong(scorer):
    found = scorer.findings(_state("An open, not a short."), "open")
    assert found["other_class_named"] == ["short"]
    assert "other_class_named" in scorer.FLAGGED
    assert "other_class_named" not in scorer.CHECKED


def test_an_acceptance_limit_on_a_class_that_admits_none_is_flagged(scorer):
    found = scorer.findings(
        _state("An open is acceptable up to 0.3 mm.", documents=("WI-201",)), "open")
    assert found["limit_for_a_zero_tolerance_class"]
    # mousebite's instruction is conditional on a measurement, so the same
    # sentence there is not a finding.
    assert "limit_for_a_zero_tolerance_class" not in scorer.findings(
        _state("A mousebite is acceptable above 80% of nominal width."), "mousebite")


def test_a_single_language_run_is_refused_rather_than_published(scorer):
    source = (ROOT / "scripts" / "rationale_eval.py").read_text()
    assert "len(args.languages) < 2" in source
    assert "return 2" in source


def test_the_contention_check_does_not_report_the_measurement_itself(scorer):
    import os

    seen = scorer.contention()["busy_processes"]
    assert not any(str(os.getpid()) == line.split()[0] for line in seen)
