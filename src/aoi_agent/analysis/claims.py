"""Is the sentence true of the numbers it was written from?

`scripts/analysis_eval.py` scores the plan. Nothing scored what happens after
the tools return, and that is where the failure this module exists for actually
happened: the criteria retrieval was handing back another class's rules, the
synthesis model repeated them faithfully, and eight stored explanations cited a
rule no document contains. The model was not wrong about its input. It was
given the wrong input and wrote what it said.

So the question here is narrow on purpose. **Not** "is the answer true" -- this
module cannot know that. "Is the prose true *of the results it was written
from*." The results are deterministic and stored beside the prose, so for a
large class of claims that question is arithmetic rather than judgement: a
figure in a sentence either appears in the payload or it does not, and the
entity it is attached to either matches or it does not.

Five kinds, because one accuracy number over them would repeat the mistake this
project's headline invariant exists to prevent. They fail differently and they
cost a supervisor differently:

``fabricated_figure``
    A number in the prose that appears nowhere in the results and cannot be
    derived from them by the renderings below. The reader has no way to catch
    this: it looks exactly like the numbers beside it.

``misattributed_figure``
    A number that *is* in the results, attached in the prose to an entity it
    does not belong to -- the right rate against the wrong machine, the wrong
    line, the wrong class. Worse than a fabrication in one way: every figure on
    the page checks out, so a reader auditing the numbers finds nothing.

``unsupported_claim``
    A statement the results cannot carry whatever their values. A cause, or a
    movement over time. No plannable tool returns a time series, so a trend
    claim has no support that could exist -- it is unsupported by the shape of
    the tool surface, not by this run's numbers.

``unhedged_gap``
    A tool failed, returned an error payload or returned nothing, and the prose
    does not say so. The supervisor reads a complete-looking answer built on
    part of the data.

``misquoted_criterion``
    A normative rule -- reject, accept, critical, scrap -- asserted about a
    class no retrieved passage governs or mentions, or attributed to a work
    instruction that was never retrieved. This is the incident above, made
    checkable.

**Where this stops.** Grounding a figure is arithmetic. Deciding that a
sentence *asserts* a cause is not, and the last three kinds are pattern
matches that raise a candidate for a person to adjudicate -- they are reported
as flags with the sentence attached, never as verdicts. And every kind is
fidelity to the results, not truth about the line: a tool that labels a 9-day
window ``"days": 14`` is repeated faithfully by the model and passes every
check here, correctly. That defect belongs to the tool.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

#: The classes the store's own tools speak, plus the classifier's extra one.
DEFECT_CLASSES = (
    "open", "short", "mousebite", "spur", "copper", "pin-hole", "false_call",
)

KINDS = (
    "fabricated_figure",
    "misattributed_figure",
    "unsupported_claim",
    "unhedged_gap",
    "misquoted_criterion",
)

#: Kinds decided by comparing the prose against the stored payload, with no
#: judgement anywhere in the decision. The rest are candidate flags.
CHECKED_KINDS = ("fabricated_figure", "misattributed_figure")

#: Keys whose string value is free text a supervisor's entity names would
#: appear in for reasons unrelated to what the payload is *about*. `why` is the
#: plan's own justification and names machines; a passage's `text` names every
#: class it contrasts with. Reading entities out of those would scope a payload
#: to whatever the prose around it happened to mention, which is the opposite
#: of a scope.
_FREE_TEXT_KEYS = frozenset(
    {"why", "text", "heading", "query", "interpretation", "error", "question"}
)

#: Child dicts whose entity names describe the parent rather than themselves.
_HOISTED = frozenset({"filters", "args"})

#: Words that place a claim in time. No plannable tool returns a time-bucketed
#: series -- `query_defect_history` and `query_machine_stats` both aggregate a
#: single window into one row per class or machine -- so a claim that something
#: rose or fell has no support the tool surface could have provided. The
#: exception is a plan that asked the same tool for two different windows,
#: which `check` suppresses on.
TREND_WORDS = (
    "increas", "decreas", "rising", "risen", "rose", "falling", "fell",
    "declin", "trend", "worsen", "improv", "spike", "climb", "drop",
    "uptick", "higher than last", "lower than last", "compared with last",
    "上升", "下降", "增加", "減少", "變多", "變少", "惡化", "改善", "趨勢",
)

#: Words that assert a cause. The synthesis prompt forbids one outright; the
#: tools carry association only.
CAUSE_WORDS = (
    "caused by", "cause of", "causes the", "because of", "because the",
    "due to", "led to", "leads to", "resulted in", "results from",
    "responsible for", "driven by", "attributable to", "the culprit",
    "造成", "導致", "原因是", "肇因",
)

#: Words that mark a sentence as *disclaiming* rather than asserting cause.
#: The prompt asks for exactly this sentence when a question asked why, so a
#: cause word inside one is compliance, not a finding.
CAUSE_DISCLAIMERS = (
    "not a cause", "not cause", "association", "associated", "correlat",
    "cannot establish", "does not establish", "not causal", "no causal",
    "would be needed", "not evidence of cause", "關聯", "非因果", "不能證明",
)

#: What a sentence has to contain before a class name in it counts as a
#: normative statement about that class.
NORMATIVE_WORDS = (
    "critical", "reject", "accept", "scrap", "rework", "disposition",
    "threshold", "limit", "criteri", "must be", "shall", "allowable",
    "判退", "允收", "報廢", "重工",
)

#: Sentences saying the criteria are *silent* on a class. The model is
#: supposed to write these -- "no specific threshold is listed for mousebite"
#: names a class no passage governs and is the honest answer, not a
#: fabricated rule. Without this the check would penalise the behaviour it
#: exists to encourage.
ABSENCE_WORDS = (
    "no specific", "not listed", "not covered", "no rule", "no threshold",
    "silent on", "does not", "do not", "no acceptance", "none of the",
    "were not retrieved", "no criteria", "沒有", "未提及", "未列",
)

#: What counts as the prose owning up to a gap.
GAP_MARKERS = (
    "fail", "error", "unavailable", "no data", "not available", "missing",
    "could not", "returned nothing", "returned no", "empty", "none were",
    "no results", "no passage", "no record", "沒有", "無法", "查無", "缺",
)

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20,
}

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD_RE = re.compile(r"\b(" + "|".join(_WORD_NUMBERS) + r")\b", re.IGNORECASE)
_WI_RE = re.compile(r"\bWI[-\s]?(\d{3})\b", re.IGNORECASE)

_MENTION_RE = re.compile(
    r"\bL\d-M\d+\b|\bM\d{2,}\b|\bL\d\b|"
    r"\b(?:open|short|mousebite|spur|copper|pin[-\s]?hole|false[_\s]call)\b",
    re.IGNORECASE,
)

#: How far past an entity name a figure is still read as attached to it, when
#: no other entity intervenes. Long enough for "line L1 inspected 137 boards
#: with 966 defects (7.05 per board)", short enough not to swallow a paragraph.
ATTRIBUTION_SPAN = 140


@dataclass(frozen=True)
class Finding:
    """One thing the prose says that the results do not.

    ``checked`` is the load-bearing field. True means the verdict came from
    comparing a value against the stored payload and a reader can reproduce it;
    False means a pattern matched and somebody has to look. Publishing them
    under one count would be the same error as publishing one accuracy over
    five failure modes.
    """

    kind: str
    claim: str
    evidence: str
    sentence: str

    @property
    def checked(self) -> bool:
        return self.kind in CHECKED_KINDS


def normalise(text: str) -> str:
    """NFKC, and the model's typographic hyphens back to ASCII.

    `gpt-oss:20b` writes `2026‑08‑09` with U+2011 and `7‑day` likewise, so a
    checker reading raw output sees no dates and no numbers where a reader sees
    both. Getting this wrong makes the whole module silently lenient.
    """
    text = unicodedata.normalize("NFKC", text)
    for dash in "‐‑‒–—−":
        text = text.replace(dash, "-")
    return text.replace(" ", " ")


def numbers_in(
    text: str, words: bool = True
) -> list[tuple[Decimal, int, int, int]]:
    """Every figure in a piece of text: value, decimal places, start, end.

    Decimal places travel with the value because they are the tolerance. A
    prose `0.38` is a correct rendering of a stored `0.375` and a prose `0.375`
    is not a rendering of `0.38`; comparing them at a fixed epsilon would get
    one of those two wrong.

    ``words=False`` drops the spelled-out forms, and the checks that score
    prose pass it. Measured on real output, a spelled-out number in this
    model's answers is a count of things -- "the two lines", "six defect
    classes", "nine candidates" -- and never a measurement. Reading one as a
    figure produced only false positives, including a swap finding against
    "the worse of the **two** lines". Digits are what a supervisor acts on and
    digits are what is scored. Grounding still reads word forms out of the
    payload's own strings, where they cost nothing.
    """
    found: list[tuple[Decimal, int, int, int]] = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group().replace(",", "")
        if raw.endswith("."):
            raw = raw[:-1]
        try:
            value = Decimal(raw)
        except InvalidOperation:  # pragma: no cover - the regex forbids it
            continue
        places = len(raw.split(".")[1]) if "." in raw else 0
        found.append((value, places, match.start(), match.end()))
    if words:
        for match in _WORD_RE.finditer(text):
            found.append(
                (Decimal(_WORD_NUMBERS[match.group(1).lower()]), 0,
                 match.start(), match.end())
            )
    return sorted(found, key=lambda item: item[2])


def _renderings(value: Decimal) -> set[Decimal]:
    """The forms a stored value legitimately takes in a sentence.

    A share of 0.375 is written `0.375` or `37.5%`. Nothing else is added --
    no pairwise differences, no ratios between rows -- because every extra
    rendering is a number the checker will accept without it being in the
    results, and the perturbation figure in the report is what that latitude
    costs.
    """
    forms = {value}
    if 0 <= value <= 1:
        forms.add(value * 100)
    return forms


def _tokens_from(value: str) -> set[str]:
    """Entity names in one string value, in the store's own vocabulary."""
    text = value.strip()
    tokens: set[str] = set()
    machine_line = re.fullmatch(r"(L\d)-(M\d+)", text, re.IGNORECASE)
    if machine_line:
        return {machine_line.group(1).upper(), machine_line.group(2).upper()}
    if re.fullmatch(r"M\d{2,}", text, re.IGNORECASE):
        tokens.add(text.upper())
    elif re.fullmatch(r"L\d", text, re.IGNORECASE):
        tokens.add(text.upper())
    elif text.lower() in DEFECT_CLASSES:
        tokens.add(text.lower())
    elif re.fullmatch(r"\d{6,}", text):
        tokens.add(text)
    return tokens


def _family(token: str) -> str:
    if re.fullmatch(r"M\d+", token):
        return "machine"
    if re.fullmatch(r"L\d", token):
        return "line"
    if re.fullmatch(r"\d+", token):
        return "board"
    return "class"


class Grounding:
    """Every figure the results hold, globally and per entity.

    Two questions are asked of it. `holds` answers "does this number exist in
    the results at all", which is the fabrication check. `holds_for` answers
    "does it exist under this entity", which is the attribution one -- and the
    difference between the two answers is the whole of the second failure kind.
    """

    def __init__(self, results: Iterable[dict]) -> None:
        self.everything: set[Decimal] = set()
        self.scopes: dict[str, set[Decimal]] = {}
        #: Figures that came out of a passage's own prose rather than out of a
        #: field. A sentence quoting "Class 3 product" or "two or more
        #: confirmed criticals" is repeating a document, not attributing a
        #: figure to whichever machine the sentence also names, so the
        #: attribution check has nothing to say about it.
        self.quoted: set[Decimal] = set()
        self.passages: list[dict] = []
        for result in results:
            payload = {
                "tool": result.get("tool"),
                "args": result.get("args") or {},
                "data": result.get("data"),
            }
            self._walk(payload, frozenset())
            data = result.get("data")
            if isinstance(data, dict) and isinstance(data.get("passages"), list):
                self.passages.extend(
                    p for p in data["passages"] if isinstance(p, dict)
                )

    # -- construction --------------------------------------------------
    def _add(self, value: Decimal, tokens: frozenset[str]) -> None:
        for form in _renderings(value):
            self.everything.add(form)
            for token in tokens:
                self.scopes.setdefault(token, set()).add(form)

    def _walk(self, node: Any, tokens: frozenset[str]) -> None:
        if isinstance(node, dict):
            here = set(tokens)
            for key, value in node.items():
                if key in _FREE_TEXT_KEYS:
                    if isinstance(value, str):
                        for figure, _p, _s, _e in numbers_in(normalise(value)):
                            self.quoted.add(figure)
                    continue
                if isinstance(value, str):
                    here |= _tokens_from(value)
                elif isinstance(value, dict) and key in _HOISTED:
                    for inner in value.values():
                        if isinstance(inner, str):
                            here |= _tokens_from(inner)
            frozen = frozenset(here)
            for key, value in node.items():
                if isinstance(value, dict) and _looks_like_by_class(value):
                    # `by_class` and its shape-alikes: the key names the class
                    # the value belongs to, so the value is scoped to it and
                    # not to the payload's other five classes.
                    for name, count in value.items():
                        self._walk(count, frozen | _tokens_from(str(name)))
                    self._numeric(sum_of(value), frozen)
                else:
                    self._walk(value, frozen)
            self._numeric(Decimal(len(node)), frozen)
        elif isinstance(node, list):
            self._numeric(Decimal(len(node)), tokens)
            for item in node:
                self._walk(item, tokens)
        else:
            self._leaf(node, tokens)

    def _leaf(self, node: Any, tokens: frozenset[str]) -> None:
        if isinstance(node, bool) or node is None:
            return
        if isinstance(node, (int, float)):
            self._numeric(Decimal(str(node)), tokens)
        elif isinstance(node, str):
            # Identifiers and timestamps carry figures a sentence quotes back:
            # `LOT-2608000`, `20085294#3`, `2026-08-09T13:36:00`, and the
            # `0.5%` written into a standards passage.
            for value, _places, _start, _end in numbers_in(normalise(node)):
                self._numeric(value, tokens)

    def _numeric(self, value: Decimal | None, tokens: frozenset[str]) -> None:
        if value is not None:
            self._add(value, tokens)

    # -- queries -------------------------------------------------------
    @staticmethod
    def _near(value: Decimal, places: int, pool: set[Decimal]) -> bool:
        tolerance = _tolerance(places)
        return any(abs(value - stored) <= tolerance for stored in pool)

    def holds(self, value: Decimal, places: int) -> bool:
        return self._near(value, places, self.everything)

    def holds_for(self, value: Decimal, places: int, tokens: Iterable[str]) -> bool:
        pools = [self.scopes.get(token) for token in tokens]
        known = [pool for pool in pools if pool]
        if not known:
            return True          # nothing known about this entity; do not judge
        shared = set.intersection(*known) if len(known) > 1 else known[0]
        return self._near(value, places, shared)

    def elsewhere(self, value: Decimal, places: int, tokens: set[str]) -> str:
        """An entity of the same kind that *does* hold this figure, if any.

        A number missing from the entity it was attached to is only a
        misattribution when some sibling entity holds it. Without that it is
        just a number the payload keeps somewhere else -- a fleet average
        quoted beside a machine, most often -- and calling it a swap would be
        the checker inventing a finding.
        """
        families = {_family(token) for token in tokens}
        for other, pool in sorted(self.scopes.items()):
            if other in tokens or _family(other) not in families:
                continue
            if self._near(value, places, pool):
                return other
        return ""


def _looks_like_by_class(node: dict) -> bool:
    return bool(node) and all(
        str(key).lower() in DEFECT_CLASSES and isinstance(value, (int, float))
        and not isinstance(value, bool)
        for key, value in node.items()
    )


def sum_of(node: dict) -> Decimal | None:
    try:
        return sum((Decimal(str(v)) for v in node.values()), Decimal(0))
    except (InvalidOperation, TypeError):  # pragma: no cover
        return None


def sentences(prose: str) -> list[str]:
    """Prose split into units an entity's figures can be attached within.

    Newlines split as hard as full stops. The model writes bullet lists, and a
    bullet about M22 followed by a bullet about M11 is two claims however the
    first one is punctuated.
    """
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", normalise(prose))
    return [part.strip() for part in parts if part.strip()]


def _mentions(sentence: str) -> list[tuple[set[str], int, int]]:
    """Entity names in one sentence, in the store's spelling.

    The prose spells a class however it likes -- `pin hole`, `pin-hole`,
    `false call` -- and the store spells it one way. Folding the prose's
    spelling into the store's here is what keeps the scope lookup from missing
    silently, which would show up as a checker that flags nothing.
    """
    found = []
    for match in _MENTION_RE.finditer(sentence):
        raw = match.group()
        if re.fullmatch(r"[LM]\d+(-M\d+)?", raw, re.IGNORECASE):
            text = raw.upper()
        else:
            text = re.sub(r"[\s_-]+", "-", raw.strip().lower())
            if text == "false-call":
                text = "false_call"
        tokens = _tokens_from(text)
        if tokens:
            found.append((tokens, match.start(), match.end()))
    return found


def plan_figures(plan: dict | None) -> set[Decimal]:
    """Figures the plan itself stated, which the prose is *told* to repeat.

    `SYNTHESIS_PROMPT` asks the model to restate the plan's assumptions in its
    own words, so "a 24-hour window" comes back in the answer having never been
    in a payload. Counting that as a fabrication would score the synthesis node
    for the planner's work -- `scripts/analysis_eval.py` is where a plan is
    judged. They are counted apart instead, so the latitude is visible.
    """
    plan = plan or {}
    text = " ".join(
        [plan.get("interpretation", ""), *(plan.get("assumptions") or [])]
        + [str(value) for call in plan.get("calls") or []
           for value in (call.get("args") or {}).values()]
    )
    return {value for value, _p, _s, _e in numbers_in(normalise(text))}


def figure_findings(
    prose: str, grounding: Grounding, restated: set[Decimal] | None = None
) -> tuple[list[Finding], int]:
    """The two kinds decided by arithmetic: fabricated, and misattributed.

    Returns the findings and how many figures were waved through as restated
    from the plan, because a latitude nobody counts is a latitude nobody can
    argue with.
    """
    restated = restated or set()
    findings: list[Finding] = []
    waved = 0
    for sentence in sentences(prose):
        mentions = _mentions(sentence)
        for value, places, start, _end in numbers_in(sentence, words=False):
            if not grounding.holds(value, places):
                if any(abs(value - stated) <= _tolerance(places)
                       for stated in restated):
                    waved += 1
                    continue
                findings.append(Finding(
                    "fabricated_figure", str(value),
                    "no value in the results renders as this", sentence,
                ))
                continue
            if value in grounding.quoted:
                continue
            tokens = _attached_to(mentions, start)
            if not tokens or grounding.holds_for(value, places, tokens):
                continue
            other = grounding.elsewhere(value, places, tokens)
            if other:
                findings.append(Finding(
                    "misattributed_figure",
                    f"{value} attributed to {'/'.join(sorted(tokens))}",
                    f"that figure is {other}'s in the results", sentence,
                ))
    return findings, waved


def _tolerance(places: int) -> Decimal:
    return Decimal(5) * Decimal(10) ** -(places + 1)


def _attached_to(
    mentions: list[tuple[set[str], int, int]], position: int
) -> set[str]:
    """Which entities a figure at this offset is being said about.

    The nearest preceding name, plus any name of a *different* kind earlier in
    the same sentence. `copper 138` inside a sentence about L1 is a claim about
    both, and checking it against the class alone would pass L2's copper count
    printed under L1's heading -- which is precisely the swap this kind is for.
    """
    attached: set[str] = set()
    nearest = None
    for tokens, start, end in mentions:
        if end <= position:
            nearest = (tokens, end)
        else:
            break
    if nearest is None or position - nearest[1] > ATTRIBUTION_SPAN:
        return attached
    attached |= nearest[0]
    families = {_family(token) for token in attached}
    for tokens, _start, end in mentions:
        if end > position:
            break
        for token in tokens:
            if _family(token) not in families:
                attached.add(token)
    return attached


def gaps(results: Iterable[dict]) -> list[str]:
    """Everything the prose owes the reader a sentence about."""
    missing = []
    for result in results:
        tool = result.get("tool", "a tool")
        if not result.get("ok", True):
            missing.append(f"{tool} failed: {result.get('error')}")
            continue
        data = result.get("data")
        if not isinstance(data, dict):
            missing.append(f"{tool} returned nothing")
            continue
        if "error" in data:
            missing.append(f"{tool} returned an error payload: {data['error']}")
            continue
        for key in ("passages", "candidates", "machines", "by_class"):
            if key in data and not data[key]:
                missing.append(f"{tool} returned an empty {key}")
        if data.get("boards_inspected") == 0:
            missing.append(f"{tool} matched no boards")
    return missing


def claim_findings(prose: str, plan: dict, results: list[dict]) -> list[Finding]:
    """The three kinds a pattern raises and a person settles."""
    findings: list[Finding] = []
    windows = {
        (call.get("tool"), (call.get("args") or {}).get("days"))
        for call in (plan or {}).get("calls") or []
    }
    tools_asked_twice = len({tool for tool, _ in windows}) < len(windows)
    normalised = normalise(prose)
    lowered = normalised.lower()

    for sentence in sentences(normalised):
        low = sentence.lower()
        if not tools_asked_twice and any(word in low for word in TREND_WORDS):
            findings.append(Finding(
                "unsupported_claim", "a movement over time",
                "no plannable tool returns a time series, and this plan asked "
                "for one window", sentence,
            ))
        if any(word in low for word in CAUSE_WORDS) and not any(
            word in low for word in CAUSE_DISCLAIMERS
        ):
            findings.append(Finding(
                "unsupported_claim", "a cause",
                "the tools carry association only", sentence,
            ))

    findings += _criterion_findings(normalised, results)

    outstanding = [gap for gap in gaps(results)]
    if outstanding and not any(marker in lowered for marker in GAP_MARKERS):
        for gap in outstanding:
            findings.append(Finding(
                "unhedged_gap", "the answer reads as complete", gap, "",
            ))
    return findings


def _criterion_findings(prose: str, results: list[dict]) -> list[Finding]:
    """Rules asserted about classes and documents nothing retrieved."""
    grounding = Grounding(results)
    passages = grounding.passages
    corpus = " ".join(
        f"{p.get('document', '')} {p.get('governs', '')} {p.get('text', '')}"
        for p in passages
    ).lower()
    findings: list[Finding] = []

    for match in _WI_RE.finditer(prose):
        if f"wi-{match.group(1)}" not in corpus.replace(" ", "-"):
            findings.append(Finding(
                "misquoted_criterion", f"WI-{match.group(1)}",
                "no retrieved passage comes from that instruction", "",
            ))

    for sentence in sentences(prose):
        low = sentence.lower()
        if not any(word in low for word in NORMATIVE_WORDS):
            continue
        if any(word in low for word in ABSENCE_WORDS):
            continue
        for klass in DEFECT_CLASSES:
            if not re.search(rf"\b{re.escape(klass)}\b", low):
                continue
            if not passages:
                findings.append(Finding(
                    "misquoted_criterion", f"a rule about {klass}",
                    "no criteria were retrieved on this run at all", sentence,
                ))
            elif klass not in corpus:
                findings.append(Finding(
                    "misquoted_criterion", f"a rule about {klass}",
                    "no retrieved passage governs or mentions that class",
                    sentence,
                ))
    return findings


def check(
    prose: str, plan: dict | None, results: list[dict]
) -> tuple[list[Finding], int]:
    """Every finding against one answer, checked kinds first.

    Second element is the count of figures accepted because the plan had
    already stated them.
    """
    grounding = Grounding(results)
    figures, waved = figure_findings(prose, grounding, plan_figures(plan))
    return figures + claim_findings(prose, plan or {}, results), waved


def perturbations(
    prose: str, grounding: Grounding
) -> tuple[int, int, int, int]:
    """How much a wrong number would have to be wrong before this notices.

    Every figure that grounded is re-asked at 1.3x and 0.7x. A checker whose
    perturbed figures ground too is not checking anything, and this is the
    number that says so -- the "so lenient nothing can fail it" test, run on
    the real distribution rather than argued about.

    Reported twice, because the two halves say different things. Over every
    figure it is dominated by small integers -- a box coordinate moved 30%
    lands on another box coordinate, and the results are full of them. Over
    figures written with a decimal point, which is where a rate or a share
    lives, it is the rate at which a wrong *rate* would go unnoticed.

    Returns (accepted, tried, accepted_decimal, tried_decimal).
    """
    accepted = tried = accepted_decimal = tried_decimal = 0
    for sentence in sentences(prose):
        for value, places, _start, _end in numbers_in(sentence, words=False):
            if not grounding.holds(value, places) or value == 0:
                continue
            for factor in (Decimal("1.3"), Decimal("0.7")):
                tried += 1
                tried_decimal += 1 if places else 0
                moved = (value * factor).quantize(
                    Decimal(1).scaleb(-places) if places else Decimal(1)
                )
                if grounding.holds(moved, places):
                    accepted += 1
                    accepted_decimal += 1 if places else 0
    return accepted, tried, accepted_decimal, tried_decimal
