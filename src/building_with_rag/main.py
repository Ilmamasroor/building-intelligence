import hmac
import json
import time
import uuid
from collections.abc import Iterator

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .chat_format import UNEXPECTED_LINE_NO_TEXT, render_final, stream_chat_text
from .contracts import (
    ChatCompletionRequest,
    ModelCard,
    ModelList,
    OpenAIErrorBody,
    OpenAIErrorEnvelope,
    QueryRequest,
    QueryResult,
    SemanticFilters,
)
from .pipeline import run_pattern, run_pattern_events
from .registry import MODES
from .settings import get_settings

app = FastAPI(title="Building with RAG")


class ChatError(Exception):
    def __init__(self, status: int, message: str, type_: str, code: str | None = None):
        self.status, self.message, self.type_, self.code = status, message, type_, code


def _error(status: int, message: str, type_: str, code: str | None = None) -> JSONResponse:
    body = OpenAIErrorEnvelope(error=OpenAIErrorBody(message=message, type=type_, code=code))
    return JSONResponse(status_code=status, content=body.model_dump())


@app.exception_handler(ChatError)
async def _chat_error(_: Request, exc: ChatError) -> JSONResponse:
    return _error(exc.status, exc.message, exc.type_, exc.code)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/v1/chat/completions"):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return _error(
            400, f"Invalid request: {loc}: {first.get('msg', 'invalid')}", "invalid_request_error"
        )
    detail = jsonable_encoder(exc.errors(), exclude={"input", "ctx"})
    return JSONResponse(status_code=422, content={"detail": detail})


def _authorized(request: Request) -> bool:
    secret = get_settings().capstone_api_key
    expected = secret.get_secret_value().strip() if secret else ""
    if not expected:
        return True  # no key configured: behave as before
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(
        token.strip().encode(), expected.encode()
    )


def require_api_key(request: Request) -> None:
    if not _authorized(request):
        raise HTTPException(
            401, "Invalid or missing API key.", headers={"WWW-Authenticate": "Bearer"}
        )


def require_api_key_openai(request: Request) -> None:
    if not _authorized(request):
        raise ChatError(
            401, "Invalid or missing API key.", "authentication_error", "invalid_api_key"
        )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/query", response_model=QueryResult, dependencies=[Depends(require_api_key)])
def query(request: QueryRequest, response: Response) -> QueryResult:
    result = run_pattern(request)
    if result.status in ("unavailable", "hyde_unavailable"):
        response.status_code = 503
    return result


@app.get(
    "/v1/models", response_model=ModelList, dependencies=[Depends(require_api_key_openai)]
)
def models() -> ModelList:
    return ModelList(data=[ModelCard(id=model_id) for model_id in MODES])


def _to_query_request(body: ChatCompletionRequest) -> QueryRequest:
    if body.model not in MODES:
        raise ChatError(
            404,
            f"The model '{body.model}' does not exist.",
            "invalid_request_error",
            "model_not_found",
        )
    if body.n != 1:
        raise ChatError(400, "Only n=1 is supported.", "invalid_request_error")
    user_messages = [m.content for m in body.messages if m.role == "user"]
    question = user_messages[-1].strip() if user_messages else ""
    if not 1 <= len(question) <= 4000:
        raise ChatError(
            400, "The latest user message must be 1-4,000 characters.", "invalid_request_error"
        )
    opts = body.rag_options
    filters = None
    if opts:
        nested = opts.filters
        act = opts.act or (nested.act if nested else None)
        status = opts.status or (nested.status if nested else None)
        access = opts.access_level or (nested.access_level if nested else None)
        if any((act, status, access)):
            filters = SemanticFilters(act=act, status=status, access_level=access)
    return QueryRequest(
        question=question,
        pattern=(opts.pattern if opts and opts.pattern else MODES[body.model]),
        caller_id=get_settings().webui_demo_caller_id,
        filters=filters,
        limit=opts.limit if opts and opts.limit else 5,
        generate_answer=True,  # server-side only; the Pipe never controls it
        required_acts=opts.required_acts if opts else None,
        chapter=opts.chapter if opts else None,
    )


def _sse(chunk_id: str, created: int, model: str, delta: dict, finish: str | None) -> str:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key_openai)])
def chat_completions(body: ChatCompletionRequest):
    request = _to_query_request(body)
    chunk_id, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())

    if body.stream:

        def frames() -> Iterator[str]:
            yield _sse(chunk_id, created, body.model, {"role": "assistant"}, None)
            # Retrieval and context assembly run here, after the role chunk.
            for text in stream_chat_text(run_pattern_events(request)):
                yield _sse(chunk_id, created, body.model, {"content": text}, None)
            yield _sse(chunk_id, created, body.model, {}, "stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(frames(), media_type="text/event-stream")

    try:
        text = render_final(run_pattern(request))
    except Exception:  # never leak details
        text = UNEXPECTED_LINE_NO_TEXT
    return {
        "id": chunk_id,
        "object": "chat.completion",
        "created": created,
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
