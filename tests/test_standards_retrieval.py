"""The criteria retrieval may not answer about one class out of another's rules.

This is a regression suite for a shipped defect, so it is written against the
real documents and the real embedding index rather than a fixture. Unscoped,
the top-ranked passage for the disposition path's own `open` query was
WI-206's "Within limits and outside pads: release. Inside a pad: reject." --
a pin-hole limit, ranked above WI-201's own classification section, which says
any confirmed open is critical with no size below which it is acceptable. The
model read both, fused them, and told five operators to establish whether the
open was inside a pad "(critical) or outside". That sentence points at
releasing a critical defect and no document contains it.

Three things are checked, and the third is the one that matters:

* every document declares which class it governs, and every class the
  classifier can emit has exactly one document -- so a seventh class arrives
  with a scope or not at all;
* the scope of a class is that class's document plus the class-agnostic policy
  documents, computed from the documents rather than from a list kept beside
  them;
* no phrasing the system actually issues returns a passage from another
  class's document.

The last one runs real queries against a real index, built into a temporary
directory so a test run cannot disturb the station's. It costs a few seconds
and it is the only check that would have caught the defect, because every layer
above it did exactly what it was told.
"""

from __future__ import annotations

import pytest

from aoi_agent.data.deeppcb import CLASS_NAMES, FALSE_CALL
from aoi_agent.store import standards

from retrieval_report import class_documents, phrasings

#: What the classifier can emit, from the dataset's own class list. Adding a
#: class there and not here is the drift this file exists to prevent, so it is
#: read, not restated.
CLASSES = sorted(CLASS_NAMES.values())


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    """The real documents, indexed somewhere disposable."""
    directory = tmp_path_factory.mktemp("chroma")
    original = standards.CHROMA_DIR
    standards.CHROMA_DIR = directory
    try:
        count = standards.build_index()
        assert count > 0, "the documents produced no passages to search"
        yield
    finally:
        standards.CHROMA_DIR = original


def test_every_document_declares_the_class_it_governs():
    documents = sorted(
        path for path in standards.STANDARDS_DIR.glob("*.md") if path.name != "README.md"
    )
    assert documents, "no standards documents found"
    for path in documents:
        # Raises UndeclaredDocument if it does not, which is the point: the
        # message names the file and the accepted values.
        standards.declared_class(path.read_text(), path.name)


@pytest.mark.parametrize("defect_class", CLASSES)
def test_every_class_has_exactly_one_work_instruction(defect_class):
    """A class with no document of its own would silently fall back to the
    policy documents, and a class with two would make "the wrong document" an
    ambiguous idea."""
    owned = [
        path.name
        for path in sorted(standards.STANDARDS_DIR.glob("*.md"))
        if path.name != "README.md"
        and standards.declared_class(path.read_text(), path.name) == defect_class
    ]
    assert len(owned) == 1, f"{defect_class} is governed by {owned}"


@pytest.mark.parametrize("defect_class", CLASSES)
def test_a_class_scopes_to_its_own_document_and_the_policy_documents(defect_class):
    assert sorted(standards.scope_of(defect_class)) == sorted(
        {defect_class, standards.ANY}
    )


def test_false_call_scopes_to_policy_only():
    """No acceptance limit applies to a region that has no defect. WI-300 still
    says what to do with it."""
    assert standards.scope_of(FALSE_CALL) == [standards.ANY]


def test_a_class_nobody_classifies_is_refused_rather_than_ignored():
    with pytest.raises(standards.UnknownDefectClass):
        standards.scope_of("pinhole")  # WI-206's class is spelled `pin-hole`


def test_an_undeclared_document_stops_the_build_instead_of_defaulting(tmp_path):
    (tmp_path / "flux-residue.md").write_text("# WI-207 Flux residue\n\n## Limits\nx\n")
    with pytest.raises(standards.UndeclaredDocument) as error:
        standards.read_passages(tmp_path)
    assert "flux-residue.md" in str(error.value)


def test_a_document_declaring_an_unknown_class_stops_the_build(tmp_path):
    (tmp_path / "flux-residue.md").write_text(
        "---\ndefect_class: flux\n---\n# WI-207 Flux residue\n\n## Limits\nx\n"
    )
    with pytest.raises(standards.UndeclaredDocument):
        standards.read_passages(tmp_path)


def test_front_matter_is_not_indexed_as_a_passage():
    """It is metadata, not a rule, and a passage reading `defect_class: open`
    quoted at an operator is noise in a quality record."""
    _, documents, _ = standards.read_passages()
    assert not any("defect_class:" in text for text in documents)


@pytest.mark.parametrize("defect_class", CLASSES)
def test_no_phrasing_returns_another_classs_criteria(defect_class, index):
    """The regression itself: over every phrasing the system really issues."""
    own = class_documents()[defect_class]
    offenders = []
    for query in phrasings(defect_class):
        for passage in standards.search(query, top_k=2, defect_class=defect_class):
            if passage.defect_class == standards.ANY:
                continue  # QP-110 and WI-300 govern every class by declaration
            if passage.document != own:
                offenders.append(f"{query!r} -> {passage.document}/{passage.heading}")
    assert not offenders, (
        f"{defect_class} was answered out of another class's document: "
        + "; ".join(offenders)
    )


def test_the_measurement_still_detects_the_fault_it_was_written_for(index):
    """Teeth check.

    A regression test that passes because the retrieval got quietly weaker --
    an empty index, a filter that drops everything -- is not a test. Unscoped,
    the same query set must still show the contamination, and the specific
    passage the fabricated rule came from must still be what it pulls for
    `open`.
    """
    unscoped = standards.search(
        "acceptance criteria and disposition for open", top_k=2
    )
    assert any(p.document == "pin-hole" for p in unscoped), (
        "unscoped retrieval no longer reproduces the fault, so passing the "
        "scoped case proves nothing"
    )

    scoped = standards.search(
        "acceptance criteria and disposition for open", top_k=2, defect_class="open"
    )
    assert scoped, "scoping returned nothing at all"
    assert all(p.document == "open-circuit" for p in scoped)


def test_an_unscoped_search_still_reaches_every_document(index):
    """`/ask` has questions that belong to no class -- what stops the line, what
    the escape budget is -- and scoping is the caller's to ask for, not a filter
    welded on. So the open-ended path must still see everything."""
    passages = standards.search("when do we stop the line", top_k=5)
    assert passages
    assert {p.document for p in passages} - {"open-circuit"}
