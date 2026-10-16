"""Story 3.2: small internal events for one generation run. Not a public API."""

from __future__ import annotations

from dataclasses import dataclass, field

from building_with_rag.contracts import GenerationResult, QueryResult


@dataclass
class TextPiece:
    text: str


@dataclass
class AttemptStarted:
    attempt: int
    max_attempts: int


@dataclass
class AttemptChecked:
    attempt: int
    max_attempts: int
    passed: bool
    issues: list[str] = field(default_factory=list)  # plain-language details of this attempt
    will_retry: bool = False


@dataclass
class GenerationDone:
    """Internal: ends a generation run; the pipeline folds it into the QueryResult."""

    generation: GenerationResult


@dataclass
class Final:
    result: QueryResult


Event = TextPiece | AttemptStarted | AttemptChecked | GenerationDone | Final
