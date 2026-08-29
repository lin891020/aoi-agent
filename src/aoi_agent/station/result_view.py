"""What a tool returned, in a form a person can check the prose against.

The synthesis model can describe correct data incorrectly, and the defence
against that is not a better prompt: it is putting the numbers next to the
sentence so a reader can see the sentence is wrong. Only two of the five
plannable tools have chart builders, so for the other three this view is the
*only* thing on the page that came from the store rather than from a model.

The station never shows ``ground_truth``. That invariant is what shapes this
module -- and, three fix rounds in, so does *how* the invariant is enforced.
Every leak found here has had the same shape: one traversal branch doing what a
sibling branch did not. A key filter that ran on dicts but not on lists; a
whitelist that voided a record for a shape the top level accepted; a composed
label checked in one branch of two. So this module has **one walk**, not four
near-parallel ones. ``_walk`` is the only thing that looks at a value, and it
asks four questions in exactly one place each:

* **Is this shape renderable?** ``_CONTAINERS`` -- the whitelist, as a table.
  It says which containers a *slot* may hold and which slot their contents get,
  and nothing else in the module decides it. A shape absent from the table is
  dropped unread, wherever it sits. That matters because ``str()`` on a
  container prints the field names inside it, which walks a payload straight
  past a name check: a list holding a record dumps that record verbatim, and so
  does a tuple used as a key. Nothing is ever ``str()``-ed unless it is a
  scalar -- values and keys alike.
* **Is this key hidden?** ``_is_hidden`` takes the whole *path* of key names
  that reached the value, not the last key, so a name split across levels
  cannot slip between them: ``{"ground": {"truth": ...}}`` and
  ``{"ground": [{"truth": ...}]}`` compose the same name and are both filtered.
  Names are normalised first, so ``Ground_Truth``, ``groundTruth``,
  ``groundtruth``, ``" ground truth "`` and ``ground-truth`` are one key.
* **Is this value clipped?** ``_Rendered.__post_init__`` -- every row in the
  module is built by constructing one of those, so the length cap applies to a
  paragraph-long passage and a 200-item list identically, at whatever depth
  either of them sat and whichever way it reached the page.
* **Is this omission counted?** Each walk returns its own dropped count and its
  caller adds it in, so a hidden key inside a record, a shape nobody could
  read, a list cut at ``MAX_ITEMS`` and rows cut at ``MAX_ROWS`` all reach the
  reader as one visible "n things not shown" line, rather than as a payload
  that quietly lost a field.

Enforced here, at the dict boundary, in the same place and for the same reason
as ``store.boards.resolve_candidate`` -- not by grepping the HTML afterwards.
No tool returns ``ground_truth`` today; the guard exists because this function
renders whatever a tool hands back, and the sixth tool is not written yet.

``readable_rows`` is not the only route from a tool's data to the page, and a
boundary with a second door is not a boundary. ``build_synthesis_messages``
serialises the raw payload into the model's context and the sentence that comes
back is rendered verbatim, so a field this walk would have dropped can reach a
reader in prose instead. ``strip_hidden`` closes that route, using the same
``_is_hidden`` this walk uses -- one rule about what the operator must not be
shown, applied on both paths, rather than two that can drift.

It is also deliberately not complete. ``search_standards`` returns passages that
are paragraphs long and ``query_machine_stats`` returns a row per machine. A
block that floods the page is one nobody reads, and a reader who needs the whole
payload has the CLI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: Compared against the *normalised* key, never the raw one.
HIDDEN_KEYS = {"ground_truth"}

#: The label on the one row that is not a field of the payload. Named rather
#: than written twice, so the count beside the block and the row it must not
#: count cannot disagree.
OVERFLOW_LABEL = "…"

MAX_ROWS = 14
MAX_ITEMS = 6
MAX_CHARS = 160
#: Shorter, because a label is a field name and a real one is under 30
#: characters. A tool returning forty 500-character keys is a 17.6KB block
#: otherwise, which defeats the cap on values by going round it.
MAX_LABEL = 60

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEPARATORS = re.compile(r"[^a-z0-9]+")
_PATH = re.compile(r"[.\[\]]+")

#: The same names with every separator gone, so ``groundtruth`` -- a spelling no
#: amount of separator normalisation reaches -- is still the same key.
_SQUASHED = {name.replace("_", "") for name in HIDDEN_KEYS}

#: How long a composed name can be before it cannot be a hidden one any more.
#: Read off the names rather than written down, so adding a longer one to
#: HIDDEN_KEYS does not quietly shorten the search that has to find it.
_LONGEST = max(len(name) for name in HIDDEN_KEYS)
_LONGEST_SQUASHED = max(len(name) for name in _SQUASHED)

# ---------------------------------------------------------------------------
# The whitelist, as one table.
#
# A *slot* is a position a value can occupy. The table says which container
# types that slot may hold, and which slot the container's contents occupy in
# turn; a type that is not listed for a slot is not renderable there. Reading
# the four entries top to bottom is the whole grammar of what this page shows:
#
#     value  := scalar | record | list of (scalar | record)
#     record := dict of (key -> scalar | list of scalars)
#
# which is two levels of names and no more, so nothing can hide under a third.
# ---------------------------------------------------------------------------

VALUE = "value"  #: a value of the returned payload
CELL = "cell"  #: one item of a list that sits at VALUE
FIELD = "field"  #: a value inside a record
ITEM = "item"  #: one item of a list that sits inside a record

_CONTAINERS: dict[str, dict[type, str]] = {
    VALUE: {dict: FIELD, list: CELL},
    CELL: {dict: FIELD},
    FIELD: {list: ITEM},
    ITEM: {},
}


def normalise_key(key: str) -> str:
    """``" Ground-Truth "``, ``groundTruth`` and ``ground_truth`` are one key."""
    return _SEPARATORS.sub("_", _CAMEL.sub("_", key).casefold()).strip("_")


def _is_hidden(path: tuple[str, ...]) -> bool:
    """True when the path of keys that reached this value names a hidden field.

    The check is on the path rather than on the last key because the same name
    can be split across container levels in more than one way --
    ``{"ground": {"truth": ...}}``, ``{"ground": [{"truth": ...}]}`` and
    ``{"a": {"b.ground_truth": ...}}`` all compose ``ground_truth`` -- and a
    per-level check sees only halves of it. So every contiguous run of the path
    is still tested, and a key that is itself a path is split before testing.

    It grows those runs one segment at a time and abandons a run as soon as it
    is longer than the longest name it could be, which is what keeps the cost
    linear in the length of the key. Testing every run outright was quadratic
    in the number of segments and cubic with the normalising, and the path is
    whatever a tool returned: ``{"a." * 2000: 1}`` took 101 seconds to render
    one row, which is a synchronous hang of ``GET /ask/{id}`` bought with a
    single dotted key. Nothing is given up for it -- a run longer than the name
    cannot be the name -- so the shapes the leak table probes are unaffected.
    """
    parts = [
        normalised
        for key in path
        for segment in _PATH.split(key)
        if (normalised := normalise_key(segment))
    ]
    for start in range(len(parts)):
        joined = ""
        squashed = ""
        # By index, not `parts[start:]`: that slice copies the tail of the list
        # on every start, which is quadratic again for a long enough key.
        for index in range(start, len(parts)):
            part = parts[index]
            joined = f"{joined}_{part}" if joined else part
            squashed += part.replace("_", "")
            if joined in HIDDEN_KEYS or squashed in _SQUASHED:
                return True
            if len(joined) > _LONGEST and len(squashed) > _LONGEST_SQUASHED:
                break  # every longer run is longer still
    return False


def _is_scalar(value: Any) -> bool:
    """A value that carries no field names inside it."""
    return value is None or isinstance(value, (str, int, float, bool))


def clip(text: str, limit: int = MAX_CHARS) -> str:
    """One string, capped. Public because `station.app` caps a streamed tool
    error with it, and a module reaching across for a private name is a
    boundary that was never drawn."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _scalar(value: Any) -> str:
    """One value, short enough to read at a glance.

    Reached only with a scalar by construction. The container branch is there
    so that a future caller that gets it wrong renders an ellipsis rather than
    a Python repr of somebody's row.
    """
    if not _is_scalar(value):
        return "…"
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, int):
        return str(value)
    return clip(" ".join(str(value).split()))


@dataclass
class _Rendered:
    """What one value became: rows to print, and how much was left out.

    Labels are *relative* to the value they describe -- ``""`` for a bare
    scalar, the field name inside a record, ``[i]`` inside a list -- and the
    caller composes the full label. That is what lets one walk serve a value
    that becomes its own rows and a value that becomes part of its parent's.
    """

    rows: list[tuple[str, str]] = field(default_factory=list)
    dropped: int = 0

    def __post_init__(self) -> None:
        """Clip here, because every row in the module is born here.

        Not in the callers that compose a text: there are four of them, and
        clipping in three is how a 200-item list under a record reached the
        page as one 10.4KB row while the same list at the top was cut. Labels
        go through it too -- a key is a value the tool chose, and a cap that
        holds only for the right-hand column is one a payload walks around.
        """
        self.rows = [
            (clip(label, MAX_LABEL), clip(text)) for label, text in self.rows
        ]

    @property
    def inline(self) -> str:
        """This value on one line, for a row it shares with its siblings."""
        if not self.rows:
            return "（空）"
        if len(self.rows) == 1 and not self.rows[0][0]:
            return self.rows[0][1]
        return ", ".join(f"{label}={text}" for label, text in self.rows)


def _contents_slot(value: Any, slot: str) -> str | None:
    """Which slot this container's contents occupy, or None if it cannot render."""
    for kind, contents in _CONTAINERS[slot].items():
        if isinstance(value, kind):
            return contents
    return None


def _walk(value: Any, slot: str, path: tuple[str, ...]) -> _Rendered | None:
    """One value, wherever it sits. None means a shape to drop unread.

    The only entry point to the traversal: a dict, a list and a scalar all
    arrive here, so the whitelist is consulted once per value and cannot be
    consulted differently by two branches that mean the same thing.
    """
    if _is_scalar(value):
        return _Rendered([("", _scalar(value))])

    contents = _contents_slot(value, slot)
    if contents is None:
        return None
    if isinstance(value, dict):
        return _walk_mapping(value, contents, path)
    return _walk_sequence(value, contents, path)


def _walk_mapping(
    mapping: dict, contents: str, path: tuple[str, ...], *, lift: bool = False
) -> _Rendered | None:
    """A dict, as one row per entry.

    ``lift`` is the payload's own dict and nothing else. It differs in exactly
    two ways, and both are properties of being the top: its entries become
    rows of their own rather than one folded line, and a value it cannot read
    costs that entry rather than the whole payload. Inside, one value nobody
    understands voids its record -- partially rendering a shape nobody
    understands is how a field ends up on screen that nobody decided to put
    there.
    """
    rows: list[tuple[str, str]] = []
    dropped = 0
    for key, value in mapping.items():
        if not _is_scalar(key):
            # Not even a label: `str()` on a tuple or an object prints whatever
            # is inside it, which is the leak this module exists to prevent.
            dropped += 1
            continue
        label = str(key)
        here = (*path, label)
        if _is_hidden(here):
            dropped += 1
            continue
        made = _walk(value, contents, here)
        if made is None:
            if not lift:
                return None
            dropped += 1
            continue
        dropped += made.dropped
        if lift:
            rows.extend((_join(label, inner), text) for inner, text in made.rows)
        else:
            rows.append((label, made.inline))
    if not rows and not lift:
        return _Rendered([("", "（空）")], dropped)
    return _Rendered(rows, dropped)


def _walk_sequence(
    items: list, contents: str, path: tuple[str, ...]
) -> _Rendered | None:
    """A list, as one row per item -- or as one bracketed row, if it is scalars.

    The path does not grow here: an index is a position, not a field name, so
    ``{"ground": [{"truth": ...}]}`` composes the same name as
    ``{"ground": {"truth": ...}}`` and is filtered by the same check.
    """
    if not items:
        return _Rendered([("", "（空）")])

    rows: list[tuple[str, str]] = []
    dropped = max(0, len(items) - MAX_ITEMS)
    scalars = True
    for index, item in enumerate(items[:MAX_ITEMS]):
        made = _walk(item, contents, path)
        if made is None:
            return None
        dropped += made.dropped
        scalars = scalars and _is_scalar(item)
        rows.append((f"[{index}]", made.inline))
    if scalars:
        # A list of scalars is one value, not n fields: it carries no names
        # inside it. Bracketed, because it can sit in a `k=v, k=v` line where
        # an unbracketed join would read as three more fields. `list_candidates`
        # returns a box this way, and it is the whole content of that tool.
        return _Rendered([("", "[" + ", ".join(t for _, t in rows) + "]")], dropped)
    return _Rendered(rows, dropped)


def _join(parent: str, child: str) -> str:
    """``meta`` + ``keep`` is ``meta.keep``; ``rows`` + ``[0]`` is ``rows[0]``."""
    if not child:
        return parent
    if not parent:
        return child
    return f"{parent}{child}" if child.startswith("[") else f"{parent}.{child}"


def readable_rows(data: dict | None) -> list[tuple[str, str]]:
    """A tool's return value as label/value pairs, safe to render."""
    if not isinstance(data, dict):
        return []

    made = _walk_mapping(data, VALUE, (), lift=True)
    assert made is not None  # lift=True never voids the payload
    rows, dropped = made.rows, made.dropped

    if len(rows) > MAX_ROWS:
        dropped += len(rows) - MAX_ROWS
        rows = rows[:MAX_ROWS]
    if dropped:
        # Stated rather than silent, and counted once for the whole payload:
        # a reader who is shown less than there was should be told so.
        rows = [*rows, (OVERFLOW_LABEL, f"另外 {dropped} 項未顯示")]
    return rows


def shown_count(rows: list[tuple[str, str]]) -> int:
    """How many of these rows are data the tool returned.

    The "n things not shown" line is a note about the payload, not a field of
    it. The page's summary counted it as one, so a payload cut from 20 fields
    to 14 announced "15 items" -- a number that is neither what the tool
    returned nor what the reader is looking at.
    """
    return sum(1 for label, _ in rows if label != OVERFLOW_LABEL)


def strip_hidden(value: Any, path: tuple[str, ...] = ()) -> Any:
    """The same payload with every hidden key removed, structure intact.

    ``readable_rows`` is the boundary for the *table*. It is not the only route
    from a tool's data to the page: `prompts.build_synthesis_messages` serialises
    the raw payload into the model's context, and the sentence it writes back is
    rendered verbatim. A model told "describe what the results show" will
    happily reproduce a field the table would have dropped, so the invariant
    needs enforcing on both routes or it is enforced on neither.

    Two things it deliberately does not do. It does not clip, and it does not
    whitelist shapes: the model is given the whole payload because it has to
    describe it, and the length and repr concerns that shape the table are
    display concerns. The key rule is the one that is about the operator's
    judgement rather than about the page, so the key rule is the one that
    travels. ``_is_hidden`` is shared with the table's walk, so the two cannot
    disagree about what ``ground_truth`` is spelled like.
    """
    if isinstance(value, dict):
        return {
            key: strip_hidden(inner, (*path, str(key)))
            for key, inner in value.items()
            if not (_is_scalar(key) and _is_hidden((*path, str(key))))
        }
    if isinstance(value, (list, tuple)):
        # The path does not grow: an index is a position, not a field name,
        # the same reasoning as `_walk_sequence`.
        return [strip_hidden(item, path) for item in value]
    return value


def error_text(data: Any) -> str | None:
    """The message a tool reported instead of an answer, if it reported one.

    The tools signal trouble with ``{"error": "..."}`` rather than by raising,
    and the page prints that line as written. It comes through here rather than
    out of the template because Jinja would happily ``str()`` a dict-valued
    error, which is the same repr-dump hole the walk guards against -- latent
    today, since every error path in the repo returns a string, and one edit
    away from not being.
    """
    if not isinstance(data, dict) or "error" not in data:
        return None
    value = data["error"]
    if isinstance(value, str):
        # Clipped like every other text on the page: this one is printed
        # outside the rows block, so nothing else would cap it.
        return clip(" ".join(value.split()))
    if _is_scalar(value):
        return _scalar(value)
    return "工具回報了一個無法顯示的錯誤"


def sql_table(data: dict | None) -> tuple[list[str], list[list[str]], int] | None:
    """A `run_sql` payload's columns and rows, clipped for the page.

    The rows are positional, not keyed, so the hidden-key walk has nothing to
    read in them -- and nothing to hide, because the column was never on the
    database they came from (`analysis/sql_guard.py`). Cells still go through
    `clip`, and at most `MAX_ROWS` rows are shown with the rest counted.
    """
    if not isinstance(data, dict) or not isinstance(data.get("columns"), list):
        return None
    rows = data.get("rows") or []
    if not isinstance(rows, list):
        return None
    shown = [[clip(_scalar(cell)) for cell in row] for row in rows[:MAX_ROWS]]
    return [clip(str(c), MAX_LABEL) for c in data["columns"]], shown, max(len(rows) - MAX_ROWS, 0)
