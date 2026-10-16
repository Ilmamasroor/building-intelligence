"""Shared contracts. Typed models only, no behavior. Extend additively; never replace."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Pattern = Literal[
    "semantic", "hybrid", "hybrid-reranked", "structured", "decomposition", "hyde", "agentic"
]

# Fixed embedding choice (see docs/architecture.md).
EMBEDDING_MODEL = "voyage-3.5"
EMBEDDING_MODEL_VERSION = "voyage-3.5"
EMBEDDING_DIMENSIONS = 1024


# ---- Retrieval / generation contracts ----


class RetrievedChunk(BaseModel):
    chunk_id: str
    section_id: str  # act-qualified, e.g. "BNS-103"
    act: str
    text: str
    heading: str | None = None
    score: float | None = None
    source: dict[str, Any] = Field(default_factory=dict)  # source origin fields
    # Added by later stories only, as named in their handouts.
    semantic_score: float | None = None
    semantic_rank: int | None = None
    keyword_score: float | None = None
    keyword_rank: int | None = None
    fused_score: float | None = None
    fused_rank: int | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None


class OmittedCandidate(BaseModel):
    chunk_id: str
    omitted_reason: str


class SubquestionEvidence(BaseModel):
    subquestion: str
    status: Literal["evidenced", "no_evidence"]
    results: list[RetrievedChunk] = Field(default_factory=list)
    reason: str | None = None


class StructuredSignals(BaseModel):
    intent: Literal["exact_lookup", "filter", "aggregation"]
    act: str | None = None
    section_number: str | None = None
    chapter: str | None = None


class GenerationResult(BaseModel):
    outcome: Literal["answered", "insufficient_evidence", "unavailable", "malformed"]
    answer: str | None = None
    claims: list[Any] = Field(default_factory=list)
    citations: list[Any] = Field(default_factory=list)
    supporting_passages: list[RetrievedChunk] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    trace: list[Any] = Field(default_factory=list)
    context_outcome: str | None = None
    confidence: float | None = None
    draft_answer: str | None = None
    issues: list[str] = Field(default_factory=list)
    attempts: int | None = None
    low_confidence_reason: str | None = None


# ---- /v1/query contracts ----


class SemanticFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown fields are a 422, never ignored

    act: list[str] | None = None
    status: list[str] | None = None
    access_level: list[str] | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    pattern: Pattern
    caller_id: str | None = None
    filters: SemanticFilters | None = None
    limit: int = Field(default=5, ge=1, le=20)
    generate_answer: bool = False
    required_acts: list[str] | None = None
    chapter: str | None = None

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class QueryResult(BaseModel):
    pattern: Pattern
    status: str
    message: str
    trace: list[Any] = Field(default_factory=list)
    results: list[RetrievedChunk] = Field(default_factory=list)
    generation: GenerationResult | None = None
    omitted_candidates: list[OmittedCandidate] = Field(default_factory=list)
    subquestions: list[SubquestionEvidence] = Field(default_factory=list)
    hyde_direct_candidates: list[RetrievedChunk] = Field(default_factory=list)
    hyde_query_candidates: list[RetrievedChunk] = Field(default_factory=list)
    hyde_hypothetical_text_debug: str | None = None


# ---- OpenAI contracts ----


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str


class RagOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: Pattern | None = None
    filters: SemanticFilters | None = None  # nested form sent by the Open WebUI Pipe
    act: list[str] | None = None
    status: list[str] | None = None
    access_level: list[str] | None = None
    limit: int | None = Field(default=None, ge=1, le=20)
    required_acts: list[str] | None = None
    chapter: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    n: int = 1
    rag_options: RagOptions | None = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "building-with-rag"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class OpenAIErrorBody(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class OpenAIErrorEnvelope(BaseModel):
    error: OpenAIErrorBody


# ---- MongoDB schema contract ----


class PassageDocument(BaseModel):
    """Shape of one document in the passages collection (Story 2.x fills it)."""

    chunk_id: str
    section_id: str
    act: str
    text: str
    heading: str | None = None
    chapter: str | None = None
    status: str | None = None
    access_level: str | None = None
    source: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    embedding_model: str = EMBEDDING_MODEL
    embedding_model_version: str = EMBEDDING_MODEL_VERSION
    embedding_dimensions: int = EMBEDDING_DIMENSIONS
