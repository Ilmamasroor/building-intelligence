"""The single run_pattern path shared by /v1/query and /v1/chat/completions.

`run_pattern_events` performs the whole operation once and yields small internal events, ending
with `Final(result)`. `run_pattern` drains it; chat renders the same events. One retrieval, one
generation operation, one QueryResult.
"""

from collections.abc import Iterator

from .contracts import QueryRequest, QueryResult
from .generation.answer import stream_grounded_answer
from .generation.events import Event, Final, GenerationDone
from .retrieval.agentic import agentic_retrieve
from .retrieval.decomposition import decomposition_retrieve, generation_question
from .retrieval.hybrid import hybrid_retrieve
from .retrieval.hybrid_reranked import hybrid_reranked_retrieve
from .retrieval.hyde import hyde_retrieve
from .retrieval.semantic import semantic_retrieve
from .retrieval.structured import structured_retrieve

# Patterns whose retrieval is implemented; they share the one answer path.
IMPLEMENTED_PATTERNS = (
    "semantic", "hybrid", "hybrid-reranked", "structured", "decomposition", "hyde", "agentic",
)
# Statuses that go on to answer generation; every other status makes no model call.
_GENERATE_ON = {
    "structured": ("ok",),
    "decomposition": ("ok", "partial_answer"),
    "agentic": ("ready_for_synthesis", "partial_result"),
}
_NOTE_STATUSES = ("ok", "partial_answer", "ready_for_synthesis", "partial_result")
_DEFAULT_GENERATE_ON = ("ok", "no_results")

_GENERATION_NOTE = {
    "answered": "Answer generated.",
    "insufficient_evidence": "The evidence was insufficient to answer.",
    "unavailable": "Answer generation unavailable.",
    "malformed": "Answer generation returned an invalid reply.",
}
_LOW_CONFIDENCE_NOTE = "Answer failed evidence checks (low confidence)."


def _retriever(pattern: str):
    """Looked up by module name at call time, so each mode's retrieval can be replaced in tests."""
    if pattern == "hybrid-reranked":
        return hybrid_reranked_retrieve
    if pattern == "structured":
        return structured_retrieve
    if pattern == "decomposition":
        return decomposition_retrieve
    if pattern == "hyde":
        return hyde_retrieve
    if pattern == "agentic":
        return agentic_retrieve
    return hybrid_retrieve if pattern == "hybrid" else semantic_retrieve


def run_pattern_events(request: QueryRequest) -> Iterator[Event]:
    if request.pattern not in IMPLEMENTED_PATTERNS:
        yield Final(
            QueryResult(
                pattern=request.pattern,
                status="not_implemented",
                message=f"The '{request.pattern}' pattern is not implemented yet.",
            )
        )
        return
    retrieve = _retriever(request.pattern)
    result = retrieve(request)  # once, non-streamed, before any text event
    # Retrieval `unavailable` / `hyde_unavailable` stay as is (no model call, generation null).
    generate_on = _GENERATE_ON.get(request.pattern, _DEFAULT_GENERATE_ON)
    if request.generate_answer and result.status in generate_on:
        question = request.question
        if request.pattern == "decomposition":
            question = generation_question(question, result)  # only evidenced subquestions
        for event in stream_grounded_answer(question, result):
            if isinstance(event, GenerationDone):
                gen = event.generation
                result.generation = gen
                if result.status in _NOTE_STATUSES:
                    note = _GENERATION_NOTE[gen.outcome]
                    if gen.outcome == "malformed" and gen.confidence is not None:
                        note = _LOW_CONFIDENCE_NOTE
                    result.message += f" {note}"
            else:
                yield event
    yield Final(result)


def run_pattern(request: QueryRequest) -> QueryResult:
    final: QueryResult | None = None
    for event in run_pattern_events(request):
        if isinstance(event, Final):
            final = event.result
    assert final is not None
    return final
