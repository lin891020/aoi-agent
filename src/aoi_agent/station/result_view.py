"""What a tool returned, in a form a person can check the prose against.

The synthesis model can describe correct data incorrectly, and the defence
against that is not a better prompt: it is putting the numbers next to the
sentence so a reader can see the sentence is wrong. Only two of the five
plannable tools have chart builders, so for the other three this view is the
*only* thing on the page that came from the store rather than from a model.

The station never shows ``ground_truth``. That invariant is what shapes this
module, and it is why the rule here is a whitelist rather than a list of banned
names:

* **Nothing is ever ``str()``-ed unless it is a scalar** -- values and keys
  alike. ``str()`` on a container prints the field names inside it, which walks
  a payload straight past a name check: a list holding a record dumps that
  record verbatim, and so does a tuple used as a key. So a value is rendered
  only when its shape is one this module can positively account for: a scalar,
  a flat record (a dict whose values are scalars or lists of scalars), a list of
  scalars, or a list of flat records. Every other shape is dropped unread, with
  a count, because a blacklist has to predict the next tool's return shape and
  a whitelist does not.
* **Keys are normalised before they are compared**, so ``Ground_Truth``,
  ``groundTruth``, ``groundtruth``, ``" ground truth "`` and ``ground-truth``
  are one key. A composed label is checked segment by segment as well, so a key
  that is itself a path (``{"a": {"b.ground_truth": ...}}``) cannot smuggle one
  through.
* **Every omission is counted**, wherever it happens -- a hidden key nested
  inside a record included -- so the reader is told something was left out
  rather than shown a payload that quietly lost a field.

Enforced here, at the dict boundary, in the same place and for the same reason
as ``store.boards.resolve_candidate`` -- not by grepping the HTML afterwards.
No tool returns ``ground_truth`` today; the guard exists because this function
renders whatever a tool hands back, and the sixth tool is not written yet.

It is also deliberately not complete. ``search_standards`` returns passages that
are paragraphs long and ``query_machine_stats`` returns a row per machine. A
block that floods the page is one nobody reads, and a reader who needs the whole
payload has the CLI.
"""

from __future__ import annotations

import re
from typing import Any

#: Compared against the *normalised* key, never the raw one.
HIDDEN_KEYS = {"ground_truth"}

MAX_ROWS = 14
MAX_ITEMS = 6
MAX_CHARS = 160

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEPARATORS = re.compile(r"[^a-z0-9]+")
_PATH = re.compile(r"[.\[\]]+")

#: The same names with every separator gone, so ``groundtruth`` -- a spelling no
#: amount of separator normalisation reaches -- is still the same key.
_SQUASHED = {name.replace("_", "") for name in HIDDEN_KEYS}


def normalise_key(key: str) -> str:
    """``" Ground-Truth "``, ``groundTruth`` and ``ground_truth`` are one key."""
    return _SEPARATORS.sub("_", _CAMEL.sub("_", key).casefold()).strip("_")


def _is_hidden(label: str) -> bool:
    """True for the key itself, or for any segment of a composed label.

    Segments matter because a key can arrive already containing a path
    separator, and the label built from it would then read as the banned path
    while neither level matched on its own.
    """
    for segment in [label, *_PATH.split(label)]:
        normalised = normalise_key(segment)
        if normalised in HIDDEN_KEYS or normalised.replace("_", "") in _SQUASHED:
            return True
    return False


def _is_scalar(value: Any) -> bool:
    """A value that carries no field names inside it."""
    return value is None or isinstance(value, (str, int, float, bool))


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
    return _clip(" ".join(str(value).split()))


def _clip(text: str) -> str:
    return text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 1] + "…"


def _scalar_list(value: Any) -> str | None:
    """A list of scalars, bracketed. None if it holds anything else.

    Bracketed because it is rendered inside a record's ``k=v, k=v`` line, where
    an unbracketed join would read as three more fields. `list_candidates`
    returns a box this way, and it is the whole content of that tool.
    """
    if not isinstance(value, list) or not all(_is_scalar(item) for item in value):
        return None
    return "[" + ", ".join(_scalar(item) for item in value) + "]"


def _record_pairs(item: Any) -> tuple[list[tuple[str, str]], int] | None:
    """A flat record as label/value pairs, plus how many entries were left out.

    Returns None -- disqualifying the whole record rather than the one entry --
    when a value is a shape this module cannot account for. Partially rendering
    a shape nobody understands is how a field ends up on screen that nobody
    decided to put there. A list of scalars is not such a shape: it has no field
    names inside it, so it renders.
    """
    if not isinstance(item, dict):
        return None

    pairs: list[tuple[str, str]] = []
    dropped = 0
    for key, value in item.items():
        if not _is_scalar(key):
            dropped += 1
            continue
        label = str(key)
        if _is_hidden(label):
            dropped += 1
            continue
        if _is_scalar(value):
            pairs.append((label, _scalar(value)))
            continue
        listed = _scalar_list(value)
        if listed is None:
            return None
        pairs.append((label, listed))
    return pairs, dropped


def _rows_for(label: str, value: Any) -> tuple[list[tuple[str, str]], int] | None:
    """Rows for one renderable shape, or None for a shape to drop unread."""
    if _is_scalar(value):
        return [(label, _scalar(value))], 0

    if isinstance(value, dict):
        made = _record_pairs(value)
        if made is None:
            return None
        pairs, dropped = made
        rows = [
            (f"{label}.{key}", text)
            for key, text in pairs
            if not _is_hidden(f"{label}.{key}")
        ]
        dropped += len(pairs) - len(rows)
        return rows or [(label, "（空）")], dropped

    if isinstance(value, list):
        if not value:
            return [(label, "（空）")], 0

        listed = _scalar_list(value)
        if listed is not None:
            return [(label, _clip(listed))], 0

        records = [_record_pairs(item) for item in value]
        if any(made is None for made in records):
            return None

        rows = []
        dropped = 0
        for index, (pairs, hidden) in enumerate(records[:MAX_ITEMS]):
            rows.append(
                (
                    f"{label}[{index}]",
                    _clip(", ".join(f"{key}={text}" for key, text in pairs)),
                )
            )
            dropped += hidden
        if len(value) > MAX_ITEMS:
            rows.append((label, f"… 另外 {len(value) - MAX_ITEMS} 筆未顯示"))
        return rows, dropped

    return None


def readable_rows(data: dict | None) -> list[tuple[str, str]]:
    """A tool's return value as label/value pairs, safe to render."""
    if not isinstance(data, dict):
        return []

    rows: list[tuple[str, str]] = []
    dropped = 0
    for key, value in data.items():
        if not _is_scalar(key):
            # Not even a label: `str()` on a tuple or an object prints whatever
            # is inside it, which is the leak this module exists to prevent.
            dropped += 1
            continue
        label = str(key)
        if _is_hidden(label):
            dropped += 1
            continue
        made = _rows_for(label, value)
        if made is None:
            # A shape this module cannot account for. Counted, not printed:
            # the reader learns something was left out without being shown it.
            dropped += 1
            continue
        made_rows, inner = made
        rows.extend(made_rows)
        dropped += inner

    if len(rows) > MAX_ROWS:
        dropped += len(rows) - MAX_ROWS
        rows = rows[:MAX_ROWS]
    if dropped:
        rows.append(("…", f"另外 {dropped} 個欄位未顯示"))
    return rows


def error_text(data: Any) -> str | None:
    """The message a tool reported instead of an answer, if it reported one.

    The tools signal trouble with ``{"error": "..."}`` rather than by raising,
    and the page prints that line as written. It comes through here rather than
    out of the template because Jinja would happily ``str()`` a dict-valued
    error, which is the same repr-dump hole the rows guard against -- latent
    today, since every error path in the repo returns a string, and one edit
    away from not being.
    """
    if not isinstance(data, dict) or "error" not in data:
        return None
    value = data["error"]
    if isinstance(value, str):
        return " ".join(value.split())
    if _is_scalar(value):
        return _scalar(value)
    return "工具回報了一個無法顯示的錯誤"
