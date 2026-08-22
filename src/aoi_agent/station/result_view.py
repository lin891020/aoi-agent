"""What a tool returned, in a form a person can check the prose against.

The synthesis model can describe correct data incorrectly, and the defence
against that is not a better prompt: it is putting the numbers next to the
sentence so a reader can see the sentence is wrong. Only two of the five
plannable tools have chart builders, so for the other three this view is the
*only* thing on the page that came from the store rather than from a model.

The station never shows ``ground_truth``. That invariant is what shapes this
module, and it is why the rule here is a whitelist rather than a list of banned
names:

* **Nothing is ever ``str()``-ed unless it is a scalar.** ``str()`` on a
  container prints the keys inside it, which walks a payload straight past a
  name check -- a list holding a list holding a record dumps that record's field
  names verbatim. So a value is rendered only when its shape is one this module
  can positively account for: a scalar, a flat dict of scalars, a list of
  scalars, or a list of flat dicts of scalars. Every other shape is dropped
  unread, with a count of what went, because a blacklist has to predict the
  next tool's return shape and a whitelist does not.
* **Keys are normalised before they are compared**, so ``Ground_Truth``,
  ``" ground truth "`` and ``ground-truth`` are the same key as far as the
  filter is concerned.

Enforced here, at the dict boundary, in the same place and for the same reason
as ``store.boards.resolve_candidate`` -- not by grepping the HTML afterwards.
No tool returns ``ground_truth`` today; the guard exists because this function
renders whatever a tool hands back, and the sixth tool is not written yet.

It is also deliberately not complete. ``search_standards`` returns passages that
are paragraphs long and ``query_machine_stats`` returns a row per machine. A
block that floods the page is one nobody reads, and a reader who needs the whole
payload has the CLI. Long values are truncated and long lists cut off with a
count of what was left out, so every omission is visible rather than silent.
"""

from __future__ import annotations

import re
from typing import Any

#: Compared against the *normalised* key, never the raw one.
HIDDEN_KEYS = {"ground_truth"}

MAX_ROWS = 14
MAX_ITEMS = 6
MAX_CHARS = 160

_SEPARATORS = re.compile(r"[^a-z0-9]+")


def normalise_key(key: Any) -> str:
    """``" Ground-Truth "`` and ``ground_truth`` are one key, not three."""
    return _SEPARATORS.sub("_", str(key).casefold()).strip("_")


def _is_hidden(key: Any) -> bool:
    return normalise_key(key) in HIDDEN_KEYS


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
    text = " ".join(str(value).split())
    return text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 1] + "…"


def _clip(text: str) -> str:
    return text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 1] + "…"


def _flat_pairs(item: Any) -> list[tuple[str, str]] | None:
    """A dict of scalars as label/value pairs, or None if it is anything else.

    One non-scalar value disqualifies the whole dict rather than just that
    entry: partially rendering a shape this module does not understand is how a
    field ends up on screen that nobody decided to put there.
    """
    if not isinstance(item, dict):
        return None
    pairs = []
    for key, value in item.items():
        if _is_hidden(key):
            continue
        if not _is_scalar(value):
            return None
        pairs.append((str(key), _scalar(value)))
    return pairs


def _rows_for(label: str, value: Any) -> list[tuple[str, str]] | None:
    """Rows for one renderable shape, or None for a shape to drop unread."""
    if _is_scalar(value):
        return [(label, _scalar(value))]

    if isinstance(value, dict):
        pairs = _flat_pairs(value)
        if pairs is None:
            return None
        return [(f"{label}.{key}", text) for key, text in pairs] or [(label, "（空）")]

    if isinstance(value, list):
        if not value:
            return [(label, "（空）")]

        if all(_is_scalar(item) for item in value):
            return [(label, _clip(", ".join(_scalar(item) for item in value)))]

        records = [_flat_pairs(item) for item in value]
        if any(pairs is None for pairs in records):
            return None

        rows = [
            (f"{label}[{index}]", _clip(", ".join(f"{k}={v}" for k, v in pairs)))
            for index, pairs in enumerate(records[:MAX_ITEMS])
        ]
        if len(value) > MAX_ITEMS:
            rows.append((label, f"… 另外 {len(value) - MAX_ITEMS} 筆未顯示"))
        return rows

    return None


def readable_rows(data: dict | None) -> list[tuple[str, str]]:
    """A tool's return value as label/value pairs, safe to render."""
    if not isinstance(data, dict):
        return []

    rows: list[tuple[str, str]] = []
    dropped = 0
    for key, value in data.items():
        if _is_hidden(key):
            dropped += 1
            continue
        made = _rows_for(str(key), value)
        if made is None:
            # A shape this module cannot account for. Counted, not printed:
            # the reader learns something was left out without being shown it.
            dropped += 1
            continue
        rows.extend(made)

    if len(rows) > MAX_ROWS:
        dropped += len(rows) - MAX_ROWS
        rows = rows[:MAX_ROWS]
    if dropped:
        rows.append(("…", f"另外 {dropped} 個欄位未顯示"))
    return rows
