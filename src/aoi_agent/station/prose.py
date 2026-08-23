"""Model-authored prose, turned into blocks a template can render.

The synthesised answer arrives as Markdown -- the model writes tables, bold
labels and bullet lists because that is how a supervisor's report reads -- and
the page printed it into a single ``<p>``. Asterisks and pipe rows came out on
screen as asterisks and pipe rows.

**This function returns structure, never markup.** That is the whole design. A
renderer that produced an HTML string would need ``|safe`` at the template, and
``|safe`` on model-authored text is the hole that `result_view.py` took five
review rounds to close one door of: the answer is written by a model, from tool
payloads, and neither is a trusted author. What comes back here is a list of
blocks of spans; Jinja escapes every one of them on the way out, and there is no
switch to turn that off.

So the safety property is not "the escaping is careful". It is that **no code
path here can emit a tag at all**. `<script>` in an answer is a text span
containing the characters `<script>`, the same as any other word, because a
text span is the only thing text can become.

What is understood is what the model actually writes: paragraphs, ``**bold**``,
``` `code` ```, ``-``/``*``/``1.`` lists, ``|`` tables, and ``#`` headings.
Everything else -- including any HTML in the source -- is text. Unrecognised
Markdown is left as the characters the model typed rather than guessed at: a
half-understood table is worse than a visible pipe row, because the pipe row
tells the reader to go and check the figures panel and the mangled one does not.
"""

from __future__ import annotations

import re

#: ``**bold**`` and `` `code` ``. Ordered: the code pattern wins inside a run of
#: backticks so `**` inside code stays literal, which is what a reader typing an
#: expression expects.
_INLINE = re.compile(r"`([^`]+)`|\*\*(.+?)\*\*")

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^[-*]\s+(.*)$")
_ORDERED = re.compile(r"^(\d+)[.)]\s+(.*)$")
#: A table's rule row: ``|---|:--:|``. Its alignment colons are read and
#: discarded -- the station's tables align by column type, and honouring a
#: model's alignment hint would let prose restyle the page.
_RULE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")


def _spans(text: str) -> list[dict]:
    """One line of text, split into plain, strong and code runs."""
    spans: list[dict] = []
    at = 0
    for match in _INLINE.finditer(text):
        if match.start() > at:
            spans.append({"kind": "text", "text": text[at:match.start()]})
        code, strong = match.group(1), match.group(2)
        spans.append(
            {"kind": "code", "text": code} if code is not None
            else {"kind": "strong", "text": strong}
        )
        at = match.end()
    if at < len(text):
        spans.append({"kind": "text", "text": text[at:]})
    return spans or [{"kind": "text", "text": ""}]


def _cells(line: str) -> list[str]:
    """A table row's cells.

    Splits on the pipes that separate cells, then drops the empty edges a
    leading and trailing pipe produce. A cell containing a pipe cannot be
    written in this syntax and is not rescued here -- it would split, and the
    row would be one cell wider than its header, which `_table` refuses.
    """
    parts = line.strip().split("|")
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return [part.strip() for part in parts]


def _table(lines: list[str], start: int) -> tuple[dict | None, int]:
    """A table beginning at ``start``, or nothing.

    A header, a rule, and at least one row. Every row has to have the header's
    width: a ragged table is a misread, and rendering one silently puts figures
    under the wrong headings -- which on this page is the failure the whole
    boundary exists to prevent, arriving by a different door.
    """
    if start + 2 >= len(lines) + 1 or "|" not in lines[start]:
        return None, start
    if start + 1 >= len(lines) or not _RULE.match(lines[start + 1]):
        return None, start

    head = _cells(lines[start])
    if not head:
        return None, start

    rows = []
    at = start + 2
    while at < len(lines) and "|" in lines[at] and lines[at].strip():
        cells = _cells(lines[at])
        if len(cells) != len(head):
            break
        rows.append([_spans(cell) for cell in cells])
        at += 1

    if not rows:
        return None, start
    return {"kind": "table", "head": [_spans(cell) for cell in head], "rows": rows}, at


def blocks(answer: str) -> list[dict]:
    """The blocks of one answer, in order.

    Every returned block is one of ``heading``, ``paragraph``, ``list`` or
    ``table``, and every piece of text in it sits in a span whose ``text`` is
    the model's characters unchanged. Nothing here builds markup.
    """
    lines = (answer or "").replace("\r\n", "\n").split("\n")
    out: list[dict] = []
    paragraph: list[str] = []
    items: list[str] | None = None
    ordered = False

    def flush_paragraph() -> None:
        if paragraph:
            out.append({"kind": "paragraph", "spans": _spans(" ".join(paragraph))})
            paragraph.clear()

    def flush_list() -> None:
        nonlocal items
        if items:
            out.append({
                "kind": "list",
                "ordered": ordered,
                "items": [_spans(item) for item in items],
            })
        items = None

    at = 0
    while at < len(lines):
        line = lines[at]

        table, after = _table(lines, at)
        if table is not None:
            flush_paragraph()
            flush_list()
            out.append(table)
            at = after
            continue

        if not line.strip():
            flush_paragraph()
            flush_list()
            at += 1
            continue

        heading = _HEADING.match(line)
        if heading:
            flush_paragraph()
            flush_list()
            out.append({
                "kind": "heading",
                # Clamped so a model writing `####` cannot outrank the page's
                # own headings in a screen reader's outline.
                "level": min(len(heading.group(1)) + 2, 6),
                "spans": _spans(heading.group(2).strip()),
            })
            at += 1
            continue

        bullet = _BULLET.match(line.strip())
        numbered = _ORDERED.match(line.strip())
        if bullet or numbered:
            flush_paragraph()
            wants_ordered = numbered is not None
            if items is None or wants_ordered != ordered:
                flush_list()
                items = []
                ordered = wants_ordered
            items.append((numbered.group(2) if numbered else bullet.group(1)).strip())
            at += 1
            continue

        flush_list()
        paragraph.append(line.strip())
        at += 1

    flush_paragraph()
    flush_list()
    return out
