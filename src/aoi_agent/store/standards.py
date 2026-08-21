"""Retrieval over the acceptance-criteria documents.

Chroma with its default local embedding model. Documents are split on markdown
headings rather than a fixed window: each work instruction is already written
in short titled sections, and those sections are the unit an operator would
actually be quoted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

STANDARDS_DIR = Path("data/standards")
CHROMA_DIR = Path("data/chroma")
COLLECTION = "acceptance_criteria"


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


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split a work instruction into ``(heading, body)`` sections."""
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


def _client() -> chromadb.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
    )


def build_index(standards_dir: Path = STANDARDS_DIR) -> int:
    """(Re)build the collection from the documents on disk."""
    client = _client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)

    ids, documents, metadatas = [], [], []
    for path in sorted(standards_dir.glob("*.md")):
        if path.name == "README.md":
            continue
        for index, (heading, body) in enumerate(split_sections(path.read_text())):
            ids.append(f"{path.stem}::{index}")
            documents.append(f"{heading}\n\n{body}")
            metadatas.append({"document": path.stem, "heading": heading})

    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def search(query: str, top_k: int = 3) -> list[Passage]:
    """Return the passages most relevant to ``query``."""
    collection = _client().get_collection(COLLECTION)
    result = collection.query(query_texts=[query], n_results=top_k)

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
            )
        )
    return passages
