"""Does a passage about one defect class come back as evidence about another?

The criteria retrieval ranks by embedding similarity, and similarity has no
notion of jurisdiction: WI-206's "inside a pad: reject" is written about pin
holes and reads, out of context, like an acceptance limit for anything. This
measures how often that happens, per class, over the phrasings the system
really issues -- the graph's own query, the planner's, a bare class name, and
the Chinese question forms a supervisor types into `/ask`.

Both modes are measured in one run, so the fix is reported against its own
baseline rather than against a number from an older index: `scoped` passes the
class to `search_standards`, `unscoped` is what the tool did before it could be
told one.

Needs the Chroma index and no GPU. Rebuild the index first if the documents
have changed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.store import standards  # noqa: E402

#: Which document governs each class. Read off the documents themselves rather
#: than written down twice -- a seventh class is added by writing its work
#: instruction, and this map has to follow it there.
def class_documents(standards_dir: Path = standards.STANDARDS_DIR) -> dict[str, str]:
    documents: dict[str, str] = {}
    for path in sorted(standards_dir.glob("*.md")):
        if path.name == "README.md":
            continue
        governs = standards.declared_class(path.read_text(), path.name)
        if governs != standards.ANY:
            documents[governs] = path.stem
    return documents


#: A class's long form, for the phrasings that spell it out. Absent classes are
#: already spelled the way a person writes them.
LONG_FORM = {
    "open": "open circuit",
    "short": "short circuit",
    "copper": "spurious copper",
    "pin-hole": "pin hole",
}


def phrasings(defect_class: str) -> list[str]:
    """The queries this class is really asked about, and where each comes from."""
    long_form = LONG_FORM.get(defect_class, defect_class)
    return [
        # graph/flow.py, the disposition path's own query
        f"acceptance criteria and disposition for {defect_class}",
        # analysis/prompts.py, the planner's few-shot phrasing
        f"{long_form} acceptance and disposition",
        # the bare class, which is what a short plan tends to produce
        defect_class,
        # /ask, passed through in the supervisor's own words
        f"{defect_class} 的驗收標準是什麼？",
        f"{defect_class} 到什麼程度算 reject？WI 裡面怎麼寫的？",
        # the question the operator actually has in front of the images
        f"how do I confirm a {long_form}",
    ]


def measure(top_k: int = 2, scoped: bool = True) -> dict[str, tuple[int, int]]:
    """Cross-class passages per class, as ``{class: (wrong, returned)}``.

    A passage from a policy document -- QP-110, WI-300 -- is not cross-class:
    those govern every class by declaration, and quoting the escape budget at
    an open is quoting the document that covers it.
    """
    documents = class_documents()
    result = {}
    for defect_class, own in documents.items():
        wrong = returned = 0
        for query in phrasings(defect_class):
            passages = standards.search(
                query,
                top_k=top_k,
                defect_class=defect_class if scoped else None,
            )
            for passage in passages:
                returned += 1
                if passage.defect_class != standards.ANY and passage.document != own:
                    wrong += 1
        result[defect_class] = (wrong, returned)
    return result


def rate(counts: dict[str, tuple[int, int]]) -> tuple[int, int]:
    wrong = sum(w for w, _ in counts.values())
    returned = sum(r for _, r in counts.values())
    return wrong, returned


def render(before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]],
           top_k: int) -> str:
    lines = [
        "### Cross-class contamination in the criteria retrieval",
        "",
        f"Six classes x {len(phrasings('open'))} phrasings, top_k={top_k}: the "
        "graph's own query, the planner's, a bare class name, two Chinese "
        "question forms from `/ask`, and the question an operator has in front "
        "of the images. A passage is counted wrong when it comes from another "
        "class's work instruction; QP-110 and WI-300 govern every class by "
        "declaration and are not counted against anything.",
        "",
        "| class | unscoped | scoped |",
        "|---|---|---|",
    ]
    for defect_class in sorted(before):
        wrong_before, total_before = before[defect_class]
        wrong_after, total_after = after[defect_class]
        lines.append(
            f"| `{defect_class}` | {wrong_before}/{total_before} "
            f"({wrong_before / total_before:.0%}) | {wrong_after}/{total_after} "
            f"({wrong_after / total_after:.0%}) |"
        )
    wrong_before, total_before = rate(before)
    wrong_after, total_after = rate(after)
    lines += [
        f"| **all** | **{wrong_before}/{total_before} "
        f"({wrong_before / total_before:.1%})** | **{wrong_after}/{total_after} "
        f"({wrong_after / total_after:.1%})** |",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=2,
                        help="what the disposition path asks for")
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        before = measure(args.top_k, scoped=False)
        after = measure(args.top_k, scoped=True)
    except Exception as error:
        print(f"{error}\nBuild the index first: uv run python -c "
              "'from aoi_agent.store.standards import build_index; build_index()'",
              file=sys.stderr)
        return 1

    report = render(before, after, args.top_k)
    print(report)
    if not args.dry_run:
        existing = args.out.read_text() if args.out.exists() else "# Benchmarks\n"
        args.out.write_text(existing.rstrip() + "\n\n" + report + "\n")
        print(f"appended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
