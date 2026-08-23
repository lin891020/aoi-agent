"""Retrieval over the acceptance-criteria documents.

Chroma with its default local embedding model. Documents are split on markdown
headings rather than a fixed window: each work instruction is already written
in short titled sections, and those sections are the unit an operator would
actually be quoted.

Retrieval here is scoped, not merely ranked. Every document declares in its
front matter which defect class it governs, that declaration is carried onto
every passage cut from it, and a caller that names a class gets passages from
that class's work instruction and from the class-agnostic policy documents --
never from another class's.

That is a boundary, not a preference, because similarity has no notion of
jurisdiction. Measured before this scoping existed, the top-ranked passage for
the graph's own `open` query was ``pin-hole.md`` -- "Within limits and outside
pads: release. Inside a pad: reject." -- ahead of WI-201's own classification
section. It reads as an acceptance limit for opens and there is none: any
confirmed open is critical. The model read both passages, fused them, and every
escalation it wrote told an operator to check whether the open was inside a pad
"(critical) or outside". No document says that, and it points at releasing a
critical defect. 27.8% of passages retrieved across the six classes came from
the wrong class's document.

The defence is not a prompt telling the model to ignore what it was handed.
Evidence about `open` that is not about `open` must not reach the prompt at
all.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Chroma's default embedding function is ONNX MiniLM, so importing chromadb is
# how onnxruntime enters this process -- this module is the only door to either.
# onnxruntime has Microsoft's 1DS "OneCollector" telemetry client statically
# linked into its wheel -- the macOS one this is developed on and the manylinux
# one CI installs, both -- and constructs it at import: two threads, one of them
# an HTTP uploader posting to mobile.events.data.microsoft.com. Nothing joins
# that uploader at exit. If a response lands while the C++ static destructors are
# running, its `DebugEventSource` locks a recursive_mutex that has already been
# destroyed, throws `system_error` on a thread with no handler, and the process
# aborts -- after every test has passed. It cost this suite a 60% abort rate
# (6 of 10 full runs) and exit 134 on a green CI job.
#
# This is the only lever that removes the thread rather than quieting it. The
# Python API, `onnxruntime.disable_telemetry_events()`, runs after import and so
# after the uploader already exists; measured, it took the abort rate to 20%,
# not to zero. The variable is read while onnxruntime builds its Env, which is
# why it is set here, above the import that triggers it, and not in a fixture --
# the station is a long-running process and has the same race.
#
# `setdefault`, so an operator who has already made this choice keeps theirs.
# It sits beside Chroma's own `anonymized_telemetry=False` below: this project
# does not phone home, and now neither does the library underneath it.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import chromadb  # noqa: E402  -- must follow ORT_DISABLE_TELEMETRY above
from chromadb.config import Settings  # noqa: E402

from aoi_agent.data.deeppcb import CLASS_NAMES, FALSE_CALL

STANDARDS_DIR = Path("data/standards")
CHROMA_DIR = Path("data/chroma")
COLLECTION = "acceptance_criteria"

#: The declaration a document carries when it governs no single class: QP-110's
#: escape budget and WI-300's operating procedure apply whatever the class is.
#: They are the only documents a scoped search adds to the class's own.
ANY = "any"

#: The six the vision model can emit, read from the dataset's own class list so
#: that a class cannot exist for the classifier and not for the criteria.
DEFECT_CLASSES = frozenset(CLASS_NAMES.values())

#: What ``defect_class`` will accept. ``false_call`` is a class the classifier
#: emits and no work instruction governs, so it scopes to the policy documents
#: alone -- WI-300 says what to do with a region nobody has to disposition,
#: and no acceptance limit applies to a region that has no defect.
SCOPES = DEFECT_CLASSES | {FALSE_CALL}


@dataclass(frozen=True)
class Passage:
    document: str
    heading: str
    text: str
    distance: float
    """Chroma's squared L2 distance -- lower is closer. Reported raw rather
    than converted to a similarity, because the conversion depends on the
    embedding space and a made-up 0-1 score would imply a calibration that
    does not exist."""
    defect_class: str = ANY
    """Which class this passage's document governs, or ``any``. Carried out to
    the caller so that "which document was this" and "was it allowed to be
    here" are answerable from the passage itself."""


class UndeclaredDocument(ValueError):
    """A standards document with no usable ``defect_class`` declaration.

    Raised at index build, not swallowed. The alternative default -- treat an
    undeclared document as applying to everything -- is exactly the bug this
    module exists to prevent, and it would return silently the moment someone
    adds a seventh class.
    """


def parse_front_matter(markdown: str) -> tuple[dict[str, str], str]:
    """Split ``---`` front matter off the top of a document.

    Deliberately not YAML: the front matter holds one flat ``key: value`` per
    line and a parser that accepts more than the documents contain is a parser
    nobody has read the failure modes of.
    """
    match = re.match(r"\A---\n(.*?)\n---\n?", markdown, flags=re.DOTALL)
    if not match:
        return {}, markdown

    meta = {}
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(":")
        if separator:
            meta[key.strip()] = value.strip()
    return meta, markdown[match.end():]


def declared_class(markdown: str, name: str) -> str:
    """The class a document governs, or raise."""
    meta, _ = parse_front_matter(markdown)
    value = meta.get("defect_class", "")
    if not value:
        raise UndeclaredDocument(
            f"{name} declares no defect_class. Add front matter naming the "
            f"class it governs, or {ANY!r} if it governs all of them: "
            f"one of {sorted(DEFECT_CLASSES | {ANY})}"
        )
    if value not in DEFECT_CLASSES and value != ANY:
        raise UndeclaredDocument(
            f"{name} declares defect_class {value!r}, which is not a class this "
            f"system classifies: expected one of {sorted(DEFECT_CLASSES | {ANY})}"
        )
    return value


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split a work instruction into ``(heading, body)`` sections."""
    _, markdown = parse_front_matter(markdown)
    parts = re.split(r"^(#{1,3} .+)$", markdown, flags=re.MULTILINE)
    sections: list[tuple[str, str]] = []

    preamble = parts[0].strip()
    heading = ""
    for index in range(1, len(parts), 2):
        heading = parts[index].lstrip("# ").strip()
        body = parts[index + 1].strip() if index + 1 < len(parts) else ""
        if body:
            sections.append((heading, body))

    if not sections and preamble:
        sections.append((heading or "document", preamble))
    return sections


class UnknownDefectClass(ValueError):
    """A scope nobody classifies. Its own type because the caller has to tell
    it apart from "the index is not built yet", which is an operational
    problem with a different remedy and, in some Chroma versions, the same
    exception type."""


def scope_of(defect_class: str) -> list[str]:
    """The ``defect_class`` values a search for ``defect_class`` may return."""
    if defect_class not in SCOPES:
        raise UnknownDefectClass(
            f"unknown defect_class {defect_class!r}; expected one of "
            f"{sorted(SCOPES)}, or omit it for a question that is not about "
            f"one class"
        )
    if defect_class == FALSE_CALL:
        return [ANY]
    return [defect_class, ANY]


@lru_cache(maxsize=4)
def _client_at(path: str) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=path, settings=Settings(anonymized_telemetry=False)
    )


def _client() -> chromadb.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return _client_at(str(CHROMA_DIR))


@lru_cache(maxsize=4)
def _collection_at(path: str):
    """The collection, held open.

    Opening it per query rebuilds the embedding model's session each time: it
    is the same cost as the query and it is paid on the disposition path, where
    the whole point of the vision model is that most regions are settled in
    milliseconds. Keyed by path so that pointing ``CHROMA_DIR`` somewhere else
    -- a test, a second index -- gets its own, and cleared on rebuild because a
    deleted collection's handle answers nothing useful.
    """
    return _client_at(path).get_collection(COLLECTION)


def read_passages(
    standards_dir: Path = STANDARDS_DIR,
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """Every passage on disk, as Chroma's three parallel lists.

    Separate from the indexing so that an undeclared document raises before
    anything is deleted. Validating inside the rebuild would drop the working
    index on the way to discovering that the new document cannot be indexed,
    and leave the station with no criteria at all.
    """
    ids, documents, metadatas = [], [], []
    for path in sorted(standards_dir.glob("*.md")):
        if path.name == "README.md":
            continue
        text = path.read_text()
        governs = declared_class(text, path.name)
        for index, (heading, body) in enumerate(split_sections(text)):
            ids.append(f"{path.stem}::{index}")
            documents.append(f"{heading}\n\n{body}")
            metadatas.append(
                {
                    "document": path.stem,
                    "heading": heading,
                    "defect_class": governs,
                }
            )
    return ids, documents, metadatas


def build_index(standards_dir: Path = STANDARDS_DIR) -> int:
    """(Re)build the collection from the documents on disk."""
    ids, documents, metadatas = read_passages(standards_dir)

    client = _client()
    _collection_at.cache_clear()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)

    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def search(
    query: str, top_k: int = 3, defect_class: str | None = None
) -> list[Passage]:
    """Return the passages most relevant to ``query``.

    ``defect_class`` scopes the search to the work instruction governing that
    class plus the class-agnostic policy documents. Leave it unset for a
    question that is not about one class -- "what stops the line" is a real
    question and answering it out of one class's document would be its own kind
    of wrong answer.

    The scope is the caller's to state because only the caller knows whether it
    has a class in hand. The disposition path always does: it is reasoning
    about one classified region, and it passes it. The analysis path usually
    does not.
    """
    where = None
    if defect_class is not None:
        where = {"defect_class": {"$in": scope_of(defect_class)}}

    _client()  # so a missing directory is created before the collection is read
    try:
        result = _collection_at(str(CHROMA_DIR)).query(
            query_texts=[query], n_results=top_k, where=where
        )
    except Exception:
        # The held handle points at a collection somebody rebuilt underneath
        # it -- the index is rebuilt by a script, and the station is a
        # long-running process that would otherwise answer nothing until it
        # was restarted. Re-open once; a second failure is a real one and is
        # raised, because the tool turns it into "build the index first".
        _collection_at.cache_clear()
        result = _collection_at(str(CHROMA_DIR)).query(
            query_texts=[query], n_results=top_k, where=where
        )

    passages = []
    for document, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
        strict=True,
    ):
        passages.append(
            Passage(
                document=str(metadata["document"]),
                heading=str(metadata["heading"]),
                text=document,
                distance=float(distance),
                defect_class=str(metadata.get("defect_class", ANY)),
            )
        )
    return passages
