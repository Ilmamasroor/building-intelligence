"""Story 5.1: the rule-based classifier for the `structured` mode.

Pure functions: no I/O, no database, no model, no network. It turns a question (and an optional
chapter) into `StructuredSignals`. Only the validated signals ever reach a database predicate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from building_with_rag.contracts import StructuredSignals

Outcome = Literal["exact", "needs_clarification", "recommendation"]

CHAPTER_RE = re.compile(r"^[IVXL]{1,6}$")
SECTION_RE = re.compile(r"^[1-9]\d{0,3}$")

RECOMMEND = (
    "The structured mode only does exact section lookups such as 'BNS section 103'. "
    "For other questions try the hybrid or semantic mode."
)

_ACT_PATTERNS = (
    ("BNS", re.compile(r"\bBNS\b|\bBharatiya\s+Nyaya\s+Sanhita\b", re.I)),
    ("IPC", re.compile(r"\bIPC\b|\bIndian\s+Penal\s+Code\b", re.I)),
)
_ACT_WORD = r"(?:BNS|IPC|Bharatiya\s+Nyaya\s+Sanhita|Indian\s+Penal\s+Code)"
# The captured token is deliberately wide (digits, optional letter, optional range/list tail);
# validation below accepts only a plain integer.
_TOKEN = r"(\d+[A-Za-z]?(?:\s*(?:-|–|to|and|or|&|,)\s*\d+[A-Za-z]?)*)"
_SECTION_PATTERNS = (
    ("section keyword", re.compile(rf"(?<!\w)(sections?|secs?|s)\b\.?\s*{_TOKEN}", re.I)),
    ("number of act", re.compile(rf"(?<![\w.]){_TOKEN}\s+of\s+(?:the\s+)?{_ACT_WORD}\b", re.I)),
    ("compact act number", re.compile(rf"\b{_ACT_WORD}[\s-]+{_TOKEN}", re.I)),
)
_AGGREGATION_RE = re.compile(
    r"\bhow\s+many\b|\bcount\b|\btotal\s+number\b|\bnumber\s+of\s+sections\b", re.I
)
_FILTER_RES = (
    re.compile(r"\b(?:list|all|every)\b[^.?!]*\bsections?\b", re.I),
    re.compile(r"\bsections?\b[^.?!]*\b(?:in|of|under)\s+chapter\b", re.I),
    re.compile(r"\b(?:repealed|in[\s-]force)\s+sections?\b", re.I),
)


@dataclass
class Classification:
    signals: StructuredSignals | None
    outcome: Outcome
    reason: str
    rule: str
    matched: str | None = None


def _acts(question: str) -> list[str]:
    return [name for name, pattern in _ACT_PATTERNS if pattern.search(question)]


def _clarify(rule: str, reason: str, signals: StructuredSignals | None = None) -> Classification:
    return Classification(signals=signals, outcome="needs_clarification", reason=reason, rule=rule)


def classify(question: str, chapter: str | None) -> Classification:
    chapter_value = (chapter or "").strip().upper() or None
    if chapter_value is not None and not CHAPTER_RE.match(chapter_value):
        return _clarify(
            "chapter invalid",
            "The chapter must be an uppercase Roman numeral such as VI (letters I, V, X, L).",
        )

    acts = _acts(question)
    refs: list[tuple[str, bool]] = []  # (token, plural keyword)
    for _name, pattern in _SECTION_PATTERNS:
        for m in pattern.finditer(question):
            token = re.sub(r"\s+", "", m.group(m.lastindex))
            plural = m.lastindex == 2 and m.group(1).lower() in ("sections", "secs")
            refs.append((token, plural))

    if not refs:
        act = acts[0] if len(acts) == 1 else None
        if _AGGREGATION_RE.search(question):
            signals = StructuredSignals(intent="aggregation", act=act)
            return Classification(signals, "recommendation", RECOMMEND, "aggregation phrase")
        if any(p.search(question) for p in _FILTER_RES):
            signals = StructuredSignals(intent="filter", act=act)
            return Classification(signals, "recommendation", RECOMMEND, "filter phrase")
        if not acts and not re.search(r"\d", question):
            return Classification(None, "recommendation", RECOMMEND, "no exact reference")

    # An exact-looking request: it either becomes a validated lookup or a clarification.
    if len(acts) != 1:
        return _clarify("act missing or ambiguous", "Say which act: BNS or IPC.")
    act = acts[0]
    numbers = {t for t, _ in refs}
    if not refs:
        return _clarify("section missing", "Say which section number, for example 'section 103'.")
    if any(p for _, p in refs) or len(numbers) > 1 or not all(SECTION_RE.match(t) for t in numbers):
        return _clarify(
            "section ambiguous",
            "Ask for one plain section number at a time (digits only, no letters or ranges).",
        )
    number = numbers.pop()
    signals = StructuredSignals(
        intent="exact_lookup", act=act, section_number=number, chapter=chapter_value
    )
    return Classification(
        signals, "exact", "Exact section lookup.", "exact_lookup", f"{act} {number}"
    )
