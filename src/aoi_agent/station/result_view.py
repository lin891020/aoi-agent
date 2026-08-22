"""What a tool returned, in a form a person can check the prose against.

The synthesis model can describe correct data incorrectly, and the defence
against that is not a better prompt: it is putting the numbers next to the
sentence so a reader can see the sentence is wrong. Only two of the five
plannable tools have chart builders, so for the other three this view is the
*only* thing on the page that came from the store rather than from a model.

Two things it deliberately is not:

* Not a JSON dump. ``ground_truth`` is skipped by name at every level, and the
  guard is not decoration: none of today's five tools return it, but this
  function renders whatever a tool hands back, so the day a sixth tool selects
  a whole ``CandidateRecord`` row a dump would put the answer key on the
  operator's screen. Enforced here, at the dict boundary, in the same place and
  for the same reason as ``store.boards.resolve_candidate`` -- not by grepping
  the HTML afterwards.
* Not complete. ``search_standards`` returns passages that are paragraphs long
  and ``query_machine_stats`` returns a row per machine. A block that floods
  the page is one nobody reads, and a reader who needs the whole payload has
  the CLI. Long values are truncated and long lists are cut off with a count of
  what was left out, so the omission is visible rather than silent.
"""

from __future__ import annotations

from typing import Any

#: Never rendered, at any depth. See the module docstring.
HIDDEN_KEYS = {"ground_truth"}

MAX_ROWS = 14
MAX_ITEMS = 6
MAX_CHARS = 160


def _scalar(value: Any) -> str:
    """One value, short enough to read at a glance."""
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, int):
        return str(value)
    text = " ".join(str(value).split())
    return text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 1] + "…"


def _inline(item: dict) -> str:
    """One row of a list of records, as ``k=v`` pairs.

    Nested containers inside a record are dropped rather than flattened
    further: two levels is the depth at which a key/value view stops being
    easier to read than the JSON it replaced.
    """
    return ", ".join(
        f"{key}={_scalar(value)}"
        for key, value in item.items()
        if key not in HIDDEN_KEYS and not isinstance(value, (dict, list))
    )


def readable_rows(data: dict | None) -> list[tuple[str, str]]:
    """A tool's return value as label/value pairs, safe to render."""
    if not isinstance(data, dict):
        return []

    rows: list[tuple[str, str]] = []
    for key, value in data.items():
        if key in HIDDEN_KEYS:
            continue

        if isinstance(value, dict):
            rows.extend(
                (f"{key}.{sub}", _scalar(inner))
                for sub, inner in value.items()
                if sub not in HIDDEN_KEYS and not isinstance(inner, (dict, list))
            )
        elif isinstance(value, list):
            for index, item in enumerate(value[:MAX_ITEMS]):
                rows.append(
                    (
                        f"{key}[{index}]",
                        _inline(item) if isinstance(item, dict) else _scalar(item),
                    )
                )
            if len(value) > MAX_ITEMS:
                rows.append((key, f"… 另外 {len(value) - MAX_ITEMS} 筆未顯示"))
        else:
            rows.append((key, _scalar(value)))

    if len(rows) > MAX_ROWS:
        remaining = len(rows) - MAX_ROWS
        rows = rows[:MAX_ROWS] + [("…", f"另外 {remaining} 個欄位未顯示")]
    return rows
