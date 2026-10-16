"""Story 5.1: structured exact retrieval (read-only lookup of one section by act and number).

Only the classifier's validated signals reach the database; the question text never does.

Diagnostic command:
    uv run python -m building_with_rag.retrieval.structured "BNS section 103" \
        [--limit N] [--chapter I] [--status in_force]
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from pymongo.errors import PyMongoError

from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.ingestion.mongodb_schema import CHUNKS, SECTIONS, SOURCES
from building_with_rag.retrieval import semantic
from building_with_rag.retrieval.semantic import RetrievalUnavailable
from building_with_rag.retrieval.structured_classifier import classify

PATTERN = "structured"
ACT_NAMES = {"BNS": "BNS_2023", "IPC": "IPC_1860"}


def _check_ready(db) -> None:
    for name in (SECTIONS, SOURCES):
        if db[name].estimated_document_count() == 0:
            raise RetrievalUnavailable(f"corpus not ready: collection '{name}' is empty")


def _predicate(signals, access_levels: list[str]) -> dict[str, Any]:
    """Built only from validated values: ACT_NAMES act, int section number, access, chapter."""
    predicate: dict[str, Any] = {
        "act": ACT_NAMES[signals.act],
        "section_number": int(signals.section_number),
        "access_level": {"$in": list(access_levels)},
    }
    if signals.chapter:
        predicate["chapter"] = signals.chapter
    return predicate


def _recorded(source: dict | None) -> str:
    status = (source or {}).get("status") or "unknown"
    version = (source or {}).get("source_status_version")
    return f"recorded status: {status}" + (f" (source: {version})" if version else "")


def structured_retrieve(request: QueryRequest) -> QueryResult:
    c = classify(request.question, request.chapter)
    trace: list[Any] = [f"mode: {PATTERN}", f"rule: {c.rule}"]
    if c.signals:
        s = c.signals
        trace.append(
            f"signals: intent={s.intent}, act={s.act}, section_number={s.section_number}, "
            f"chapter={s.chapter}"
        )
    if c.outcome != "exact":
        trace.append("database lookup skipped")
        return QueryResult(pattern=PATTERN, status=c.outcome, message=c.reason, trace=trace)

    signals = c.signals
    label = f"{signals.act} {signals.section_number}"
    filters = semantic.effective_filters(request)
    predicate = _predicate(signals, filters["access_level"])
    trace.append(f"matched: {c.matched}")
    trace.append(
        f"predicate fields: {', '.join(predicate)}; access levels: {filters['access_level']}"
    )
    not_found = QueryResult(
        pattern=PATTERN, status="not_found",
        message=f"No section {label} was found in the indexed documents.", trace=trace,
    )

    try:
        db = semantic._database()
        _check_ready(db)
        if "act" in filters and ACT_NAMES[signals.act] not in filters["act"]:
            trace.append("request act filter excludes this act")
            return not_found
        sections = list(db[SECTIONS].find(predicate))
        sources = {
            s["_id"]: s
            for s in db[SOURCES].find(
                {"_id": {"$in": list({x["source_id"] for x in sections if x.get("source_id")})}}
            )
        }
        if "status" in filters:
            wanted = set(filters["status"])
            sections = [
                x for x in sections
                if (sources.get(x.get("source_id")) or {}).get("status") in wanted
            ]
            trace.append(f"status filter applied through sources: {sorted(wanted)}")
        if not sections:
            return not_found
        if len(sections) > 1:
            ids = sorted(x["section_id"] for x in sections)
            return QueryResult(
                pattern=PATTERN, status="needs_clarification", trace=trace,
                message=(
                    f"More than one stored section matches {label}: {', '.join(ids)}. "
                    "Say which one is meant."
                ),
            )
        section = sections[0]
        chunk_filter = {
            "section_id": section["section_id"], "access_level": predicate["access_level"],
        }
        chunk_ids = [
            ch["chunk_id"]
            for ch in db[CHUNKS].find(chunk_filter, {"chunk_id": 1}).sort("chunk_index", 1)
        ]
        if len(chunk_ids) > request.limit:
            trace.append(f"limit {request.limit} cut {len(chunk_ids) - request.limit} chunk(s)")
            chunk_ids = chunk_ids[: request.limit]
        results, dropped = semantic.resolve_hits(db, [{"chunk_id": cid} for cid in chunk_ids])
    except RetrievalUnavailable as exc:
        return QueryResult(
            pattern=PATTERN, status="unavailable", message=f"Retrieval unavailable: {exc}.",
            trace=trace,
        )
    except PyMongoError as exc:
        return QueryResult(
            pattern=PATTERN, status="unavailable",
            message=f"Retrieval unavailable: MongoDB error ({type(exc).__name__}).", trace=trace,
        )

    trace.append(
        f"section: {section['section_id']}; chunks returned: {len(results)}, dropped: {dropped}"
    )
    if not results:
        return not_found
    trace.append(_recorded(sources.get(section.get("source_id"))))
    return QueryResult(
        pattern=PATTERN, status="ok", trace=trace, results=results,
        message=(
            f"Retrieved {len(results)} chunk(s) of section {section['section_id']}; "
            f"{_recorded(sources.get(section.get('source_id')))}."
        ),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Compact structured exact-lookup diagnostic.")
    ap.add_argument("question")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--chapter")
    ap.add_argument("--status", action="append")
    args = ap.parse_args()

    from building_with_rag.contracts import SemanticFilters
    from building_with_rag.pipeline import run_pattern

    request = QueryRequest(
        question=args.question, pattern=PATTERN, limit=args.limit, chapter=args.chapter,
        filters=SemanticFilters(status=args.status) if args.status else None,
    )
    result = run_pattern(request)
    c = classify(request.question, request.chapter)
    s = c.signals
    print(f"status: {result.status}\nmessage: {result.message}")
    print(
        f"signals: intent={s.intent if s else None} act={s.act if s else None} "
        f"section_number={s.section_number if s else None} chapter={s.chapter if s else None}"
    )
    print(f"rule: {c.rule}")
    for r in result.results:
        print(
            f"  chunk_id={r.chunk_id} section_id={r.section_id} act={r.act} "
            f"heading={r.heading!r} status={r.source.get('status')} "
            f"source_status_version={r.source.get('source_status_version')} text_len={len(r.text)}"
        )
    return 0 if result.status != "unavailable" else 1


if __name__ == "__main__":
    sys.exit(main())
