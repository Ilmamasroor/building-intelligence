"""Story 6.1: bounded agentic RAG, a fixed read-only controller (no model, no planner).

It makes a fixed choice between three existing retrieval adapters, checks what each returns and
stops. At most two adapter calls (at most one HyDE call) and a deadline checked before and after
every call. It never builds, edits or re-ranks passage text and never keeps the question or any
adapter trace or HyDE text in its own result.

Diagnostic command:
    uv run python -m building_with_rag.retrieval.agentic "BNS section 103" \
        [--limit N] [--act BNS_2023]
"""

from __future__ import annotations

import argparse
import re
import sys
import time

from building_with_rag.contracts import (
    QueryRequest,
    QueryResult,
    RetrievedChunk,
    SemanticFilters,
)
from building_with_rag.generation.answer import SUPPORT_MIN_SHARE, meaningful_words
from building_with_rag.retrieval import hyde, semantic, structured
from building_with_rag.retrieval.decomposition import key_terms
from building_with_rag.retrieval.structured_classifier import classify
from building_with_rag.settings import get_settings

PATTERN = "agentic"
MAX_CALLS = 2  # not configurable
MAX_HYDE_CALLS = 1  # not configurable
DEFAULT_DEADLINE_SECONDS = 30
DEADLINE_RANGE = (1, 120)

CLARIFY_HINT = (
    "Say which act and section, for example 'BNS section 103', or ask a full question."
)

# Fixed allowlist: name -> (adapter module, attribute, adapter pattern). Never chosen by the
# question or by any model output; resolved by attribute at call time so tests can replace them.
_ADAPTERS = {
    "structuredLookup": (structured, "structured_retrieve", "structured"),
    "semanticRetrieve": (semantic, "semantic_retrieve", "semantic"),
    "hydeRetrieve": (hyde, "hyde_retrieve", "hyde"),
}

# Everyday wording the Acts word differently. It only decides whether HyDE is *allowed*; it is
# never used to rewrite, expand or search the query.
LAY_PHRASES = (
    re.compile(r"\bhand(?:s|ed|ing)?\s+over\b", re.I),
    re.compile(r"\bthreat(?:en\w*|s)?\b", re.I),
    re.compile(r"\bforc(?:e|es|ed|ing)\s+(?:me|him|her|them|us)\b", re.I),
    re.compile(r"\bpretend(?:s|ed|ing)?\s+to\s+be\b", re.I),
    re.compile(r"\brun(?:s|ning)?\s+away\s+with\b", re.I),
    re.compile(r"\bscar(?:e|es|ed|ing)\b", re.I),
)


def _clock() -> float:
    return time.monotonic()


class _Stop(Exception):
    def __init__(self, reason: str):
        self.reason = reason


class _Unavailable(Exception):
    def __init__(self, adapter: str, category: str):
        self.adapter, self.category = adapter, category


def deadline_seconds() -> tuple[int, bool]:
    """(seconds, valid). Anything outside 1-120 or not a whole number uses the default."""
    raw = get_settings().agentic_deadline_seconds
    try:
        value = int(raw) if not isinstance(raw, bool) else None
    except (TypeError, ValueError):
        value = None
    if value is None or not DEADLINE_RANGE[0] <= value <= DEADLINE_RANGE[1]:
        return DEFAULT_DEADLINE_SECONDS, False
    return value, True


def is_usable(chunk: RetrievedChunk, filters: SemanticFilters | None) -> bool:
    if not (chunk.text.strip() and chunk.chunk_id.strip()
            and chunk.section_id.strip() and chunk.act.strip()):
        return False
    if filters and filters.act is not None and chunk.act not in filters.act:
        return False
    if filters and filters.status is not None:
        if (chunk.source or {}).get("status") not in filters.status:
            return False
    return True


def _share(question_terms: set[str], passages: list[RetrievedChunk]) -> float:
    if not question_terms:
        return 0.0
    words: set[str] = set()
    for p in passages:
        words |= meaningful_words(p.text)
    return len(question_terms & words) / len(question_terms)


def _adequate_share(question: str, passages: list[RetrievedChunk]) -> bool:
    return bool(passages) and _share(key_terms(question), passages) >= SUPPORT_MIN_SHARE


class _Run:
    """One controller run: owns the call budget, the deadline and the trace."""

    def __init__(self, request: QueryRequest):
        self.request = request
        self.start = _clock()
        self.deadline, self.deadline_ok = deadline_seconds()
        self.calls = 0
        self.hyde_calls = 0
        self.call_lines: list[str] = []
        self.adequacy_lines: list[str] = []

    def expired(self) -> bool:
        return _clock() - self.start >= self.deadline

    def call(self, name: str) -> tuple[QueryResult, list[RetrievedChunk]]:
        """One allowlisted adapter call; returns (result, usable passages)."""
        if name not in _ADAPTERS:
            raise _Stop("call_budget_exhausted")
        second_hyde = name == "hydeRetrieve" and self.hyde_calls >= MAX_HYDE_CALLS
        if self.calls >= MAX_CALLS or second_hyde:
            raise _Stop("call_budget_exhausted")
        if self.expired():
            raise _Stop("deadline_exceeded")
        module, attr, pattern = _ADAPTERS[name]
        self.calls += 1
        if name == "hydeRetrieve":
            self.hyde_calls += 1
        sub = self.request.model_copy(update={"pattern": pattern, "generate_answer": False})
        try:
            result = getattr(module, attr)(sub)
        except Exception as exc:  # never keep the raw error text
            self.call_lines.append(
                f"call {self.calls}: {name} status=error passages=0 usable=0"
            )
            raise _Unavailable(name, f"error ({type(exc).__name__})") from None
        passages = list(result.results)
        usable = [p.model_copy(deep=True) for p in passages if is_usable(p, self.request.filters)]
        late = self.expired()
        self.call_lines.append(
            f"call {self.calls}: {name} status={result.status} passages={len(passages)} "
            f"usable={0 if late else len(usable)}"
        )
        # hyde_* fields and the adapter's trace are dropped here with the adapter result.
        if late:
            raise _Stop("deadline_exceeded")
        return result, usable

    def adequacy(self, rule: str, ok: bool) -> bool:
        self.adequacy_lines.append(
            f"adequacy: {rule} -> {'adequate' if ok else 'not adequate'}"
        )
        return ok

    def finish(
        self, status: str, message: str, stop: str, *, validation: str, path: str,
        results: list[RetrievedChunk] | None = None,
    ) -> QueryResult:
        elapsed = _clock() - self.start
        trace = [
            f"mode: {PATTERN}",
            f"validation: {validation}",
            f"path: {path}",
            *self.call_lines,
            *self.adequacy_lines,
            f"calls: {self.calls} of {MAX_CALLS}",
        ]
        if not self.deadline_ok:
            trace.append(f"deadline setting invalid; using {DEFAULT_DEADLINE_SECONDS}")
        trace += [f"deadline: {self.deadline} s, elapsed {elapsed:.1f} s", f"stop: {stop}"]
        return QueryResult(
            pattern=PATTERN, status=status, message=message, trace=trace,
            results=results or [],
        )


def _ready(n: int) -> str:
    return f"Retrieved {n} passage(s) for synthesis."


def _partial(n: int, stop: str) -> str:
    return f"Limited evidence: {n} passage(s) kept; the bounded search stopped ({stop})."


def _is_requested_section(chunk: RetrievedChunk, act: str, number: str) -> bool:
    return (
        chunk.act == structured.ACT_NAMES[act]
        and chunk.section_id.strip().lower() == f"{act.lower()}:{number}"
    )


def _merge(first: list[RetrievedChunk], second: list[RetrievedChunk], limit: int):
    out: list[RetrievedChunk] = []
    seen: set[str] = set()
    for chunk in (*first, *second):
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            out.append(chunk)
    return out[:limit]


def agentic_retrieve(request: QueryRequest) -> QueryResult:
    run = _Run(request)

    # Rule 1: validate first (no adapter call).
    c = classify(request.question, request.chapter)
    category: str | None = None
    if c.outcome == "needs_clarification":
        category = c.rule
    elif c.signals is not None and c.signals.intent in ("filter", "aggregation"):
        category = f"unsupported request ({c.signals.intent})"
    elif not key_terms(request.question):
        category = "no usable words"
    if category:
        return run.finish(
            "clarify", CLARIFY_HINT, "validation_rejected",
            validation=f"rejected ({category})", path="none",
        )

    accepted = {"validation": "accepted"}
    retained: list[RetrievedChunk] = []
    path = "none"
    try:
        if c.outcome == "exact":
            # Rule 2: act-qualified section request.
            path = "structured"
            signals = c.signals
            result, usable = run.call("structuredLookup")
            if result.status == "needs_clarification":
                return run.finish(
                    "clarify", "More than one stored record matched; say which one is meant.",
                    "needs_clarification", path=path, **accepted,
                )
            if result.status == "not_found":
                usable = []
            elif result.status != "ok":
                raise _Unavailable("structuredLookup", _category(result.status))
            matching = [
                p for p in usable if _is_requested_section(p, signals.act, signals.section_number)
            ]
            run.adequacy("act-qualified lookup", bool(matching))
            if not matching:
                return run.finish(
                    "clarify",
                    "The record was not found in the indexed documents; check the act and number.",
                    "record_not_found", path=path, **accepted,
                )
            return run.finish(
                "ready_for_synthesis", _ready(len(matching)), "evidence_adequate",
                path=path, results=matching, **accepted,
            )

        # Rules 3 and 4 start with one semantic call.
        path = "semantic"
        result, direct = run.call("semanticRetrieve")
        if result.status not in ("ok", "no_results"):
            raise _Unavailable("semanticRetrieve", _category(result.status))
        retained = direct
        lay = any(p.search(request.question) for p in LAY_PHRASES)

        if not lay:  # Rule 4: ordinary valid question
            if run.adequacy("ordinary question", bool(direct)):
                return run.finish(
                    "ready_for_synthesis", _ready(len(direct)), "evidence_adequate",
                    path=path, results=direct, **accepted,
                )
            return run.finish(
                "partial_result", _partial(0, "no usable evidence"), "no_usable_evidence",
                path=path, **accepted,
            )

        # Rule 3: prepared vocabulary-mismatch question.
        if run.adequacy("vocabulary mismatch (direct)", _adequate_share(request.question, direct)):
            return run.finish(
                "ready_for_synthesis", _ready(len(direct)), "evidence_adequate",
                path=path, results=direct, **accepted,
            )
        path = "semantic+hyde"
        result, via_hyde = run.call("hydeRetrieve")
        if result.status not in ("ok", "no_results"):
            raise _Unavailable("hydeRetrieve", _category(result.status))
        combined = _merge(via_hyde, direct, request.limit)
        retained = combined
        enough = _adequate_share(request.question, combined)
        if run.adequacy("vocabulary mismatch (combined)", enough):
            return run.finish(
                "ready_for_synthesis", _ready(len(combined)), "evidence_adequate",
                path=path, results=combined, **accepted,
            )
        return run.finish(
            "partial_result", _partial(len(combined), "no adequate evidence"),
            "no_usable_evidence", path=path, results=combined, **accepted,
        )
    except _Stop as stop:
        return run.finish(
            "partial_result", _partial(len(retained), stop.reason), stop.reason,
            path=path, results=retained, **accepted,
        )
    except _Unavailable as exc:
        if retained:
            return run.finish(
                "partial_result", _partial(len(retained), "tool unavailable"), "tool_unavailable",
                path=path, results=retained, **accepted,
            )
        return run.finish(
            "unavailable", f"Retrieval unavailable: {exc.adapter} ({exc.category}).",
            "tool_unavailable", path=path, **accepted,
        )


def _category(status: str) -> str:
    return "adapter reported unavailable" if "unavailable" in status else "unexpected status"


def main() -> int:
    ap = argparse.ArgumentParser(description="Compact bounded agentic RAG diagnostic.")
    ap.add_argument("question")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--act", action="append", help="repeatable, e.g. BNS_2023")
    args = ap.parse_args()

    from building_with_rag.pipeline import run_pattern

    request = QueryRequest(
        question=args.question, pattern=PATTERN, limit=args.limit,
        filters=SemanticFilters(act=args.act) if args.act else None,
    )
    result = run_pattern(request)
    print(f"status: {result.status}\nmessage: {result.message}")
    for line in result.trace:
        print(f"trace: {line}")
    for r in result.results:
        print(
            f"  chunk_id={r.chunk_id} section_id={r.section_id} heading={r.heading!r} "
            f"text_len={len(r.text)}"
        )
    return 1 if result.status == "unavailable" else 0


if __name__ == "__main__":
    sys.exit(main())
