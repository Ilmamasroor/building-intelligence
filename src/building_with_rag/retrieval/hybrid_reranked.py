"""Story 4.2: hybrid-reranked search.

Hybrid candidates (Story 4.1) -> candidate / send / return limits -> one re-rank provider call.
Only retrieval differs; the evidence shape, answer, citations and streaming are shared.

Diagnostic command:
    uv run python -m building_with_rag.retrieval.hybrid_reranked \
        "What is the difference between culpable homicide and murder?" --limit 5 --act IPC_1860
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from pymongo.errors import PyMongoError

from building_with_rag.contracts import OmittedCandidate, QueryRequest, QueryResult
from building_with_rag.retrieval import hybrid, semantic
from building_with_rag.retrieval.rerank import RerankUnavailable, check_limits, rerank
from building_with_rag.retrieval.semantic import RetrievalUnavailable
from building_with_rag.settings import get_settings

PATTERN = "hybrid-reranked"


def route_depth(candidate_limit: int) -> int:
    """Deep enough that the candidate limit can actually be filled."""
    return max(candidate_limit, hybrid.MIN_ROUTE_DEPTH)


def _unavailable(message: str, trace: list[Any]) -> QueryResult:
    return QueryResult(
        pattern=PATTERN, status="unavailable", message=f"Retrieval unavailable: {message}.",
        trace=trace,
    )


def hybrid_reranked_retrieve(request: QueryRequest) -> QueryResult:
    settings = get_settings()
    cand_limit = settings.rerank_candidate_limit
    send_limit = settings.rerank_send_limit
    ret_limit = settings.rerank_return_limit
    depth = route_depth(cand_limit)
    filters = semantic.effective_filters(request)
    trace = hybrid.base_trace(request, PATTERN, depth, filters)
    try:
        check_limits(settings)
    except RerankUnavailable as exc:
        return _unavailable(f"re-ranking unavailable ({exc})", trace)
    k = min(request.limit, ret_limit)
    trace.append(
        f"re-ranking: model {settings.rerank_model_name}; RERANK_CANDIDATE_LIMIT {cand_limit}, "
        f"RERANK_SEND_LIMIT {send_limit}, RERANK_RETURN_LIMIT {ret_limit}; returning {k}"
        + (f" (lowered by request limit {request.limit})" if k < ret_limit else "")
    )

    try:
        db, sem_hits, kw_hits, fused = hybrid.gather_candidates(request, depth, filters, trace)
        total = len(fused)
        kept = fused[:cand_limit]
        to_send = kept[:send_limit]
        sent_chunks, unresolved = (
            semantic.resolve_hits(
                db, [{"chunk_id": f.chunk_id, "score": f.fused_score} for f in to_send]
            )
            if to_send
            else ([], 0)
        )
    except RetrievalUnavailable as exc:
        return _unavailable(str(exc), trace)
    except PyMongoError as exc:
        return _unavailable(f"MongoDB error ({type(exc).__name__})", trace)

    fused_rank = {f.chunk_id: i for i, f in enumerate(fused, start=1)}
    by_id = {f.chunk_id: f for f in fused}
    resolved_ids = {c.chunk_id for c in sent_chunks}
    omitted: list[OmittedCandidate] = []
    for i, f in enumerate(fused, start=1):
        if i > cand_limit:
            reason = f"beyond RERANK_CANDIDATE_LIMIT (fused rank {i} of {total})"
        elif i > send_limit:
            reason = f"not sent to re-ranker: beyond RERANK_SEND_LIMIT (fused rank {i})"
        elif f.chunk_id not in resolved_ids:
            reason = f"could not be resolved from the corpus (fused rank {i})"
        else:
            continue
        omitted.append(OmittedCandidate(chunk_id=f.chunk_id, omitted_reason=reason))
    cut_before = total - len(to_send)
    trace.append(
        f"counts: {hybrid.route_counts(sem_hits, kw_hits)}; fused candidates {total}, "
        f"kept {len(kept)}, sent {len(sent_chunks)}, cut before re-ranking {cut_before}, "
        f"unresolved {unresolved}"
    )
    for r in sent_chunks:
        hybrid.apply_fused(r, by_id[r.chunk_id], fused_rank[r.chunk_id])
    trace.append("before re-ranking (fused order, NOT re-ranked):")
    trace.extend(
        f"  #{r.fused_rank} {r.chunk_id} section={r.section_id} fused={r.fused_score:.4f} "
        f"{by_id[r.chunk_id].route}" for r in sent_chunks
    )

    if not sent_chunks:
        return QueryResult(
            pattern=PATTERN, status="no_results",
            message="The search ran successfully but no passages matched.",
            trace=trace, omitted_candidates=omitted,
        )

    try:
        pairs = rerank(request.question, [r.text for r in sent_chunks])
    except RerankUnavailable as exc:
        trace.append("re-ranking failed; the order above was not re-ranked and is not returned")
        return _unavailable(f"re-ranking unavailable ({exc})", trace)

    # score descending; ties keep fused order (sent_chunks is in fused order)
    ordered = sorted(pairs, key=lambda p: (-p[1], p[0]))
    results = []
    for rank, (index, score) in enumerate(ordered, start=1):
        r = sent_chunks[index]
        if rank > k:
            omitted.append(OmittedCandidate(
                chunk_id=r.chunk_id,
                omitted_reason=f"cut after re-ranking: beyond RERANK_RETURN_LIMIT "
                f"(rerank rank {rank}, score {score:.4f})"
                if k == ret_limit else
                f"cut after re-ranking: beyond the returned limit {k} "
                f"(rerank rank {rank}, score {score:.4f})",
            ))
            continue
        r.rerank_score, r.rerank_rank = score, rank
        r.score = score
        results.append(r)
    trace.append(
        f"counts: returned {len(results)}, cut after re-ranking {len(sent_chunks) - len(results)}, "
        f"omitted total {len(omitted)}"
    )
    trace.append("after re-ranking:")
    trace.extend(
        f"  #{r.rerank_rank} {r.chunk_id} section={r.section_id} rerank={r.rerank_score:.4f} "
        f"(was fused #{r.fused_rank})" for r in results
    )
    return QueryResult(
        pattern=PATTERN, status="ok", message=f"Retrieved {len(results)} passage(s).",
        trace=trace, results=results, omitted_candidates=omitted,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Compact hybrid-reranked retrieval diagnostic.")
    ap.add_argument("question")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--act", action="append", help="repeatable, e.g. IPC_1860")
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
    for line in result.trace:
        s = str(line)
        if s.startswith(("re-ranking", "counts:", "before", "after", "  #")):
            print(s)
    print("final results:")
    for r in result.results:
        print(
            f"  rerank #{r.rerank_rank} score={r.rerank_score:.4f} chunk_id={r.chunk_id} "
            f"section_id={r.section_id} heading={r.heading!r} fused_rank={r.fused_rank} "
            f"text_len={len(r.text)}"
        )
    print("omitted:")
    for o in result.omitted_candidates:
        print(f"  {o.chunk_id}: {o.omitted_reason}")
    return 0 if result.status != "unavailable" else 1


if __name__ == "__main__":
    sys.exit(main())
