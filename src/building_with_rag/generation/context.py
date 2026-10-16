"""Story 3.1: bounded, labelled evidence context built from retrieved passages."""

from __future__ import annotations

from dataclasses import dataclass, field

from building_with_rag.contracts import RetrievedChunk

MAX_PASSAGES = 6
MAX_CONTEXT_CHARS = 12_000


@dataclass
class ContextPassage:
    label: str
    chunk: RetrievedChunk


@dataclass
class AssembledContext:
    outcome: str  # assembled | no_results | empty_after_budget
    passages: list[ContextPassage] = field(default_factory=list)
    not_included: list[str] = field(default_factory=list)  # chunk_ids
    trace: list[str] = field(default_factory=list)

    @property
    def by_label(self) -> dict[str, RetrievedChunk]:
        return {p.label: p.chunk for p in self.passages}

    @property
    def chunks(self) -> list[RetrievedChunk]:
        return [p.chunk for p in self.passages]


def assemble_context(results: list[RetrievedChunk]) -> AssembledContext:
    """Whole passages in rank order; stop at the first one that would break a budget."""
    if not results:
        return AssembledContext("no_results", trace=["context: no retrieved passages"])
    passages: list[ContextPassage] = []
    total = 0
    cut = len(results)
    for i, chunk in enumerate(results):
        if len(passages) >= MAX_PASSAGES or total + len(chunk.text) > MAX_CONTEXT_CHARS:
            cut = i
            break
        total += len(chunk.text)
        passages.append(ContextPassage(f"E{len(passages) + 1}", chunk))
    not_included = [c.chunk_id for c in results[cut:]]
    outcome = "assembled" if passages else "empty_after_budget"
    trace = [
        f"context: {outcome}; included {len(passages)}, not included {len(not_included)}, "
        f"chars {total}/{MAX_CONTEXT_CHARS}, max passages {MAX_PASSAGES}",
        "included: " + ", ".join(f"{p.label}={p.chunk.chunk_id}" for p in passages),
        "not included: " + ", ".join(not_included),
    ]
    return AssembledContext(outcome, passages, not_included, trace)
