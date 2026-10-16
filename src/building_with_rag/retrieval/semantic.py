"""Story 2.3: semantic retrieval (embed query -> $vectorSearch -> resolve chunks/sections/sources).

Diagnostic command:
    uv run python -m building_with_rag.retrieval.semantic "What is the punishment for murder?" \
        --limit 3 --act BNS_2023
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from building_with_rag.contracts import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_VERSION,
    QueryRequest,
    QueryResult,
    RetrievedChunk,
)
from building_with_rag.ingestion.db import (
    PrerequisiteError,
    require_mongodb_uri,
    require_voyage_key,
)
from building_with_rag.ingestion.mongodb_schema import (
    ACCESS_LEVEL_PUBLIC,
    CHUNKS,
    EMBEDDINGS,
    SECTIONS,
    SOURCES,
    VECTOR_FIELD,
    VECTOR_FILTER_FIELDS,
    VECTOR_INDEX_NAME,
)
from building_with_rag.settings import get_settings

PATTERN = "semantic"
MIN_NUM_CANDIDATES = 50
CANDIDATE_MULTIPLE = 10
MAX_NUM_CANDIDATES = 1000  # Atlas maximum
VOYAGE_TIMEOUT_SECONDS = 30
MONGO_TIMEOUT_MS = 10_000
# The one fixed local demo caller may only see public passages (no users, roles or authorization).
CALLER_ACCESS_LEVELS = [ACCESS_LEVEL_PUBLIC]

_clients: dict[str, MongoClient] = {}
_embeddings_verified: set[str] = set()


class RetrievalUnavailable(Exception):
    """A dependency, index or setting is missing or failing. Message is safe to return."""


def num_candidates(limit: int) -> int:
    return min(MAX_NUM_CANDIDATES, max(MIN_NUM_CANDIDATES, limit * CANDIDATE_MULTIPLE))


def _database():
    try:
        uri = require_mongodb_uri()
    except PrerequisiteError as exc:
        raise RetrievalUnavailable(str(exc).split(".")[0]) from None
    client = _clients.get(uri)
    if client is None:
        client = _clients[uri] = MongoClient(uri, serverSelectionTimeoutMS=MONGO_TIMEOUT_MS)
    return client[get_settings().mongodb_db_name]


def check_ready(db) -> None:
    """Counts and one sample only on each request; the full model check runs once per database."""
    for name in (SECTIONS, CHUNKS, EMBEDDINGS):
        if db[name].estimated_document_count() == 0:
            raise RetrievalUnavailable(f"corpus not ready: collection '{name}' is empty")
    found = list(db[EMBEDDINGS].list_search_indexes(VECTOR_INDEX_NAME))
    if not found:
        raise RetrievalUnavailable(f"vector index not ready: '{VECTOR_INDEX_NAME}' does not exist")
    if not (found[0].get("queryable") or found[0].get("status") == "READY"):
        raise RetrievalUnavailable(f"vector index not ready: '{VECTOR_INDEX_NAME}' not queryable")
    approved = {
        "model": EMBEDDING_MODEL,
        "model_version": EMBEDDING_MODEL_VERSION,
        "dimensions": EMBEDDING_DIMENSIONS,
    }
    if db.name in _embeddings_verified:
        sample = db[EMBEDDINGS].find_one({}, {k: 1 for k in approved})
        if sample and any(sample.get(k) != v for k, v in approved.items()):
            raise RetrievalUnavailable("embeddings do not carry the approved model settings")
        return
    mismatch = {"$or": [{k: {"$ne": v}} for k, v in approved.items()]}
    if db[EMBEDDINGS].count_documents(mismatch, limit=1):
        raise RetrievalUnavailable("embeddings do not carry the approved model settings")
    _embeddings_verified.add(db.name)


def embed_query(question: str) -> list[float]:
    try:
        import voyageai

        key = require_voyage_key()
        vector = voyageai.Client(
            api_key=key, max_retries=0, timeout=VOYAGE_TIMEOUT_SECONDS
        ).embed(
            [question],
            model=EMBEDDING_MODEL,
            input_type="query",
            output_dimension=EMBEDDING_DIMENSIONS,
        ).embeddings[0]
    except PrerequisiteError as exc:
        raise RetrievalUnavailable(str(exc).split(".")[0]) from None
    except Exception:
        raise RetrievalUnavailable("embedding service unavailable") from None
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise RetrievalUnavailable("embedding service returned an unexpected vector length")
    return vector


def effective_filters(request: QueryRequest) -> dict[str, list[str]]:
    """Requested lists narrow scope; access_level is always intersected with the caller's."""
    requested = request.filters
    out: dict[str, list[str]] = {}
    for field in ("act", "status"):
        values = getattr(requested, field, None) if requested else None
        if values is not None:
            out[field] = list(values)
    wanted = requested.access_level if requested else None
    out["access_level"] = (
        list(CALLER_ACCESS_LEVELS)
        if wanted is None
        else [v for v in CALLER_ACCESS_LEVELS if v in wanted]
    )
    return out


def vector_filter(filters: dict[str, list[str]]) -> dict[str, Any] | None:
    clauses = [{f: {"$in": v}} for f, v in filters.items() if f in VECTOR_FILTER_FIELDS]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _source_details(chunk: dict, section: dict, source: dict | None) -> dict[str, Any]:
    details = {
        "source_pdf": section.get("source_pdf"),
        "source_sha256": section.get("source_sha256"),
        "page_start": section.get("page_start"),
        "page_end": section.get("page_end"),
        "chapter": section.get("chapter") or chunk.get("chapter"),
        "status": chunk.get("status") or (source or {}).get("status"),
        "source_status_version": (source or {}).get("source_status_version"),
    }
    return {k: v for k, v in details.items() if v is not None}


def vector_search(
    db, vector: list[float], filters: dict[str, list[str]], limit: int, candidates: int
) -> list[dict]:
    """The $vectorSearch step: `(chunk_id, score)` hits in score order, filter inside the stage."""
    stage: dict[str, Any] = {
        "index": VECTOR_INDEX_NAME,
        "path": VECTOR_FIELD,
        "queryVector": vector,
        "numCandidates": candidates,
        "limit": limit,
    }
    flt = vector_filter(filters)
    if flt:
        stage["filter"] = flt
    return list(
        db[EMBEDDINGS].aggregate(
            [
                {"$vectorSearch": stage},
                {"$project": {"_id": 0, "chunk_id": 1, "score": {"$meta": "vectorSearchScore"}}},
            ]
        )
    )


def resolve_hits(db, hits: list[dict]) -> tuple[list[RetrievedChunk], int]:
    """Batched chunk/section/source lookups. Keeps hit order; `score` is the hit's score.

    Unresolvable hits are dropped and counted. No route-specific fields are set here.
    """
    chunk_ids = [h["chunk_id"] for h in hits]
    chunks = {c["chunk_id"]: c for c in db[CHUNKS].find({"chunk_id": {"$in": chunk_ids}})}
    section_ids = list({c["section_id"] for c in chunks.values()})
    sections = {s["section_id"]: s for s in db[SECTIONS].find({"section_id": {"$in": section_ids}})}
    source_ids = list({s["source_id"] for s in sections.values() if s.get("source_id")})
    sources = {s["_id"]: s for s in db[SOURCES].find({"_id": {"$in": source_ids}})}
    results: list[RetrievedChunk] = []
    dropped = 0
    for hit in hits:
        chunk = chunks.get(hit["chunk_id"])
        section = sections.get(chunk["section_id"]) if chunk else None
        if not chunk or not section:
            dropped += 1
            continue
        score = float(hit["score"]) if hit.get("score") is not None else None
        results.append(
            RetrievedChunk(
                chunk_id=chunk["chunk_id"],
                section_id=section["section_id"],
                act=chunk["act"],
                text=chunk["text"],
                heading=section.get("heading"),
                score=score,
                source=_source_details(chunk, section, sources.get(section.get("source_id"))),
            )
        )
    return results, dropped


def resolve_ranked(db, hits: list[dict]) -> tuple[list[RetrievedChunk], int]:
    results, dropped = resolve_hits(db, hits)
    for r in results:
        r.semantic_score = r.score
    results.sort(key=lambda r: r.score or 0.0, reverse=True)
    for rank, r in enumerate(results, start=1):
        r.semantic_rank = rank
    return results, dropped


def semantic_retrieve(request: QueryRequest) -> QueryResult:
    limit = request.limit
    candidates = num_candidates(limit)
    filters = effective_filters(request)
    trace: list[Any] = [
        f"mode: {PATTERN}",
        f"query: {request.question}",
        f"embedding: {EMBEDDING_MODEL} ({EMBEDDING_DIMENSIONS} dimensions)",
        f"effective filters: {filters}",
        f"limit: {limit}, numCandidates: {candidates}",
    ]
    ignored = [n for n in ("required_acts", "chapter") if getattr(request, n)]
    if ignored:
        trace.append(f"not applied in semantic mode: {', '.join(ignored)}")

    try:
        db = _database()
        check_ready(db)
        if any(not values for values in filters.values()):
            hits: list[dict] = []
            trace.append("a filter list is empty after caller scoping; search skipped")
        else:
            vector = embed_query(request.question)
            hits = vector_search(db, vector, filters, limit, candidates)
        results, dropped = resolve_ranked(db, hits) if hits else ([], 0)
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

    trace.append(f"hits returned: {len(results)}, dropped (unresolved): {dropped}")
    results = results[:limit]
    if not results:
        return QueryResult(
            pattern=PATTERN, status="no_results",
            message="The search ran successfully but no passages matched.", trace=trace,
        )
    return QueryResult(
        pattern=PATTERN, status="ok", message=f"Retrieved {len(results)} passage(s).",
        trace=trace, results=results,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Compact semantic retrieval diagnostic.")
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
    for r in result.results:
        print(
            f"  #{r.semantic_rank} score={r.score:.4f} chunk_id={r.chunk_id} "
            f"section_id={r.section_id} act={r.act} heading={r.heading!r} text_len={len(r.text)}"
        )
    return 0 if result.status != "unavailable" else 1


if __name__ == "__main__":
    sys.exit(main())
