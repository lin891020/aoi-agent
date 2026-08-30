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
_INLINE = re.compile(r"`([^`]+)`|\*\*(.+?)\*\*|\*([^*\s](?:[^*]*?[^*\s])?)\*")

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^[-*]\s+(.*)$")
_ORDERED = re.compile(r"^(\d+)[.)]\s+(.*)$")
#: A table's rule row: ``|---|:--:|``. Its alignment colons are read and
#: discarded -- the station's tables align by column type, and honouring a
#: model's alignment hint would let prose restyle the page.
_RULE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")


def _spans(text: str) -> list[dict]:
    """One line of text, split into plain, strong, emphasis and code runs.

    Single-asterisk emphasis joined 2026-08-30: the Chinese rationales write
    `*false_call*` for a class name, and a queue row that prints the asterisks
    reads as a typo beside the bold it did render.
    """
    spans: list[dict] = []
    at = 0
    for match in _INLINE.finditer(text):
        if match.start() > at:
            spans.append({"kind": "text", "text": text[at:match.start()]})
        code, strong, em = match.group(1), match.group(2), match.group(3)
        if code is not None:
            spans.append({"kind": "code", "text": code})
        elif strong is not None:
            spans.append({"kind": "strong", "text": strong})
        else:
            spans.append({"kind": "em", "text": em})
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

        if items is not None and not paragraph:
            # A plain line directly under a list item continues that item --
            # Markdown's lazy continuation. The standards documents are
            # hard-wrapped at 72 columns, so without this every bullet longer
            # than a line was cut into a bullet and a stray paragraph on the
            # region page ("...per the product's" / "repair class. Class 3...").
            items[-1] = f"{items[-1]} {line.strip()}"
            at += 1
            continue

        flush_list()
        paragraph.append(line.strip())
        at += 1

    flush_paragraph()
    flush_list()
    return out


#: Where a sentence ends. A CJK full stop ends one wherever it stands; an
#: ASCII one only when followed by a space or the end of the text, because
#: ``0.61`` is not two sentences. ``！？`` and ``!?`` are read the same way.
_SENTENCE_END = re.compile(r"[。！？]|[.!?](?=\s|$)")


def inline(text: str) -> list[dict]:
    """One line's spans, for text that is a record and may not be restructured.

    The planner's interpretation and assumptions are shown as written; the
    only thing this changes is that `per_board` reads as code rather than
    as two backticks. No headings, lists or tables are recognised here.
    """
    return _spans(text or "")


def plain_text(text: str) -> str:
    """The same text with Markdown structure flattened to sentences.

    For the queue's two-sentence lead, which is read as a scan of prose and
    must not open with "以下為對此區域的評估與說明： 1. **視覺模型輸出**".
    Headings become sentences, list items become sentences, emphasis marks
    are dropped; the characters of the words are unchanged.
    """
    pieces: list[str] = []
    for block in blocks(text):
        if block["kind"] == "table":
            continue
        runs = [block["spans"]] if block["kind"] in ("heading", "paragraph") else block["items"]
        for spans in runs:
            piece = "".join(span["text"] for span in spans).strip()
            if piece:
                pieces.append(piece)
    return " ".join(pieces)


def assumption_items(items: object) -> list[str]:
    """The planner's assumptions, one per line, however the model packed them.

    The schema asks for a list. `gpt-oss:20b` sometimes returns one string
    holding the whole list with the characters backslash-n between items --
    not a newline, the two characters -- and the page then showed
    "；\\n2. 前後窗口…" as one bullet. Split on both, drop a leading "1."
    the model wrote itself, keep the text otherwise unchanged: these are
    records and are never rewritten.
    """
    out: list[str] = []
    for item in items if isinstance(items, list) else [items]:
        if not isinstance(item, str):
            continue
        for line in re.split(r"\\n|\n", item):
            line = re.sub(r"^\s*\d+[.、)]\s*", "", line).strip()
            if line:
                out.append(line)
    return out


def lead_and_rest(text: str, sentences: int = 2) -> tuple[str, str]:
    """The first few sentences of a rationale, and everything after them.

    For the queue, where a paragraph per row made the one column an operator
    scans the one they could not. Nothing is dropped: ``lead + rest`` is the
    text, so the disclosure that shows ``rest`` shows the rationale whole.
    """
    text = text or ""
    ends = [m.end() for m in _SENTENCE_END.finditer(text)]
    if len(ends) <= sentences:
        return text, ""
    cut = ends[sentences - 1]
    return text[:cut], text[cut:]
