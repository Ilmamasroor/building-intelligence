"""Story 4.1: hybrid retrieval (vector search + Atlas Search keyword search, fused by RRF).

Only retrieval differs from `semantic`: the fused passages use the same `RetrievedChunk` shape,
context assembly, grounded answer, citations and streaming.

Diagnostic command:
    uv run python -m building_with_rag.retrieval.hybrid "What is culpable homicide?" \
        --limit 5 --act BNS_2023
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any

from pymongo.errors import PyMongoError

from building_with_rag.contracts import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    QueryRequest,
    QueryResult,
    RetrievedChunk,
)
from building_with_rag.ingestion.mongodb_schema import (
    CHUNKS,
    KEYWORD_FILTER_FIELDS,
    KEYWORD_INDEX_NAME,
    KEYWORD_TEXT_FIELD,
)
from building_with_rag.retrieval import semantic
from building_with_rag.retrieval.semantic import RetrievalUnavailable

PATTERN = "hybrid"
MIN_ROUTE_DEPTH = 20
RRF_K = 60
FUSION_RULE = (
    f"Reciprocal Rank Fusion: fused = sum over routes of 1/({RRF_K}+rank), 1-based ranks, "
    "equal weights, raw scores not compared"
)


def route_depth(limit: int) -> int:
    return max(limit, MIN_ROUTE_DEPTH)


@dataclass
class Fused:
    chunk_id: str
    fused_score: float
    semantic_score: float | None = None
    semantic_rank: int | None = None
    keyword_score: float | None = None
    keyword_rank: int | None = None

    @property
    def route(self) -> str:
        if self.semantic_rank is not None and self.keyword_rank is not None:
            return "both"
        return "semantic only" if self.semantic_rank is not None else "keyword only"

    @property
    def best_rank(self) -> int:
        return min(r for r in (self.semantic_rank, self.keyword_rank) if r is not None)


def fuse(semantic_hits: list[dict], keyword_hits: list[dict]) -> list[Fused]:
    """Reciprocal Rank Fusion over two ranked `{chunk_id, score}` lists (best first).

    Ranks only: the vector similarity and the keyword score are not comparable and are carried
    through for inspection, never added or normalised. Ties: better best-route rank, then chunk_id.
    """
    by_id: dict[str, Fused] = {}
    for route, hits in (("semantic", semantic_hits), ("keyword", keyword_hits)):
        for rank, hit in enumerate(hits, start=1):
            f = by_id.setdefault(hit["chunk_id"], Fused(hit["chunk_id"], 0.0))
            if getattr(f, f"{route}_rank") is not None:  # one candidate per chunk per route
                continue
            setattr(f, f"{route}_rank", rank)
            setattr(f, f"{route}_score", float(hit["score"]))
            f.fused_score += 1.0 / (RRF_K + rank)
    return sorted(by_id.values(), key=lambda f: (-f.fused_score, f.best_rank, f.chunk_id))


def keyword_stages(
    question: str, filters: dict[str, list[str]], depth: int
) -> list[dict[str, Any]]:
    """Atlas Search stages. The question is a `query` value only; scope filters live in $search."""
    scope = [
        {"in": {"path": field, "value": values}}
        for field, values in filters.items()
        if field in KEYWORD_FILTER_FIELDS
    ]
    compound: dict[str, Any] = {
        "must": [{"text": {"query": question, "path": KEYWORD_TEXT_FIELD}}]
    }
    if scope:
        compound["filter"] = scope
    return [
        {"$search": {"index": KEYWORD_INDEX_NAME, "compound": compound}},
        {"$limit": depth},
        {"$project": {"_id": 0, "chunk_id": 1, "score": {"$meta": "searchScore"}}},
    ]


def check_keyword_ready(db) -> None:
    found = list(db[CHUNKS].list_search_indexes(KEYWORD_INDEX_NAME))
    if not found:
        raise RetrievalUnavailable(
            f"keyword index not ready: '{KEYWORD_INDEX_NAME}' does not exist "
            "(run the setup_db command)"
        )
    if not (found[0].get("queryable") or found[0].get("status") == "READY"):
        raise RetrievalUnavailable(f"keyword index not ready: '{KEYWORD_INDEX_NAME}' not queryable")


def semantic_route(db, vector, filters, depth) -> list[dict]:
    return semantic.vector_search(db, vector, filters, depth, semantic.num_candidates(depth))


def keyword_route(db, question, filters, depth) -> list[dict]:
    return list(db[CHUNKS].aggregate(keyword_stages(question, filters, depth)))


def _passage_line(r: RetrievedChunk, route: str) -> str:
    sem = f"semantic #{r.semantic_rank} ({r.semantic_score:.4f})" if r.semantic_rank else None
    kw = f"keyword #{r.keyword_rank} ({r.keyword_score:.2f})" if r.keyword_rank else None
    parts = " ".join(p for p in (sem, kw) if p)
    return f"#{r.fused_rank} {r.chunk_id} fused={r.fused_score:.4f} {parts} {route}"


def base_trace(request: QueryRequest, pattern: str, depth: int, filters: dict) -> list[Any]:
    trace: list[Any] = [
        f"mode: {pattern}",
        f"query: {request.question}",
        f"embedding: {EMBEDDING_MODEL} ({EMBEDDING_DIMENSIONS} dimensions)",
        f"effective filters: {filters}",
        f"limit: {request.limit}, route depth: {depth}, "
        f"numCandidates: {semantic.num_candidates(depth)}",
        f"fusion: {FUSION_RULE}",
        f"keyword index: {KEYWORD_INDEX_NAME} on {CHUNKS}.{KEYWORD_TEXT_FIELD}",
    ]
    ignored = [n for n in ("required_acts", "chapter") if getattr(request, n)]
    if ignored:
        trace.append(f"not applied in {pattern} mode: {', '.join(ignored)}")
    return trace


def gather_candidates(request, depth, filters, trace):
    """Readiness, single question embedding, both routes and fusion.

    Returns `(db, sem_hits, kw_hits, fused)` with `fused` the full fused list. Raises
    RetrievalUnavailable / PyMongoError; the callers turn those into `unavailable`.
    """
    db = semantic._database()
    semantic.check_ready(db)
    check_keyword_ready(db)
    if any(not values for values in filters.values()):
        sem_hits: list[dict] = []
        kw_hits: list[dict] = []
        trace.append("a filter list is empty after caller scoping; search skipped")
    else:
        vector = semantic.embed_query(request.question)  # once per request
        try:
            sem_hits = semantic_route(db, vector, filters, depth)
        except PyMongoError as exc:
            raise RetrievalUnavailable(f"semantic route failed ({type(exc).__name__})") from None
        try:
            kw_hits = keyword_route(db, request.question, filters, depth)
        except PyMongoError as exc:
            raise RetrievalUnavailable(f"keyword route failed ({type(exc).__name__})") from None
    return db, sem_hits, kw_hits, fuse(sem_hits, kw_hits)


def route_counts(sem_hits: list[dict], kw_hits: list[dict]) -> str:
    sem_ids = {h["chunk_id"] for h in sem_hits}
    kw_ids = {h["chunk_id"] for h in kw_hits}
    return (
        f"semantic returned {len(sem_ids)}, keyword returned {len(kw_ids)}, "
        f"in both {len(sem_ids & kw_ids)}, semantic only {len(sem_ids - kw_ids)}, "
        f"keyword only {len(kw_ids - sem_ids)}"
    )


def apply_fused(r: RetrievedChunk, f: Fused, rank: int) -> None:
    r.score = r.fused_score = f.fused_score
    r.fused_rank = rank
    r.semantic_score, r.semantic_rank = f.semantic_score, f.semantic_rank
    r.keyword_score, r.keyword_rank = f.keyword_score, f.keyword_rank


def hybrid_retrieve(request: QueryRequest) -> QueryResult:
    limit = request.limit
    depth = route_depth(limit)
    filters = semantic.effective_filters(request)
    trace = base_trace(request, PATTERN, depth, filters)

    try:
        db, sem_hits, kw_hits, fused = gather_candidates(request, depth, filters, trace)
        by_id = {f.chunk_id: f for f in fused}
        results, dropped = (
            semantic.resolve_hits(
                db, [{"chunk_id": f.chunk_id, "score": f.fused_score} for f in fused]
            )
            if fused
            else ([], 0)
        )
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

    for rank, r in enumerate(results, start=1):
        apply_fused(r, by_id[r.chunk_id], rank)
    trace.append(f"counts: {route_counts(sem_hits, kw_hits)}, dropped (unresolved) {dropped}")
    results = results[:limit]
    trace.extend(_passage_line(r, by_id[r.chunk_id].route) for r in results)
    if not results:
        return QueryResult(
            pattern=PATTERN, status="no_results",
            message="The search ran successfully but no passages matched.", trace=trace,
        )
    return QueryResult(
        pattern=PATTERN, status="ok", message=f"Retrieved {len(results)} passage(s).",
        trace=trace, results=results,
    )


def _route_label(r: RetrievedChunk) -> str:
    if r.semantic_rank and r.keyword_rank:
        return "both"
    return "semantic only" if r.semantic_rank else "keyword only"


def main() -> int:
    ap = argparse.ArgumentParser(description="Compact hybrid retrieval diagnostic.")
    ap.add_argument("question")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--act", action="append", help="repeatable, e.g. BNS_2023")
    ap.add_argument("--status", action="append")
    args = ap.parse_args()

    from building_with_rag.contracts import SemanticFilters
    from building_with_rag.pipeline import run_pattern

    filters = (
        SemanticFilters(act=args.act, status=args.status) if args.act or args.status else None
    )
    request = QueryRequest(
        question=args.question, pattern=PATTERN, limit=args.limit, filters=filters
    )
    result = run_pattern(request)
    print(f"status: {result.status}\nmode: {result.pattern}\nquery: {request.question}")
    print(f"message: {result.message}")
    print(f"fusion: {FUSION_RULE}")
    print("\n".join(t for t in result.trace if str(t).startswith("counts:")))
    for r in result.results:
        sem = f"#{r.semantic_rank}/{r.semantic_score:.4f}" if r.semantic_rank else "-"
        kw = f"#{r.keyword_rank}/{r.keyword_score:.2f}" if r.keyword_rank else "-"
        print(
            f"  #{r.fused_rank} fused={r.fused_score:.4f} chunk_id={r.chunk_id} "
            f"section_id={r.section_id} heading={r.heading!r} act={r.act} semantic={sem} "
            f"keyword={kw} route={_route_label(r)} text_len={len(r.text)}"
        )
    return 0 if result.status != "unavailable" else 1


if __name__ == "__main__":
    sys.exit(main())
