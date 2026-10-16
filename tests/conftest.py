import pytest

from building_with_rag import pipeline
from building_with_rag.contracts import QueryResult, RetrievedChunk
from building_with_rag.generation import answer as answer_mod
from building_with_rag.settings import get_settings

SECRET = "sk-test-secret-123"
CAPSTONE = "capstone-test-key-456"
BASE = "https://llm.example.invalid/v1"


def make_chunk(n: int, text: str = "Whoever commits murder shall be punished.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"bns:{n}:c0", section_id=f"bns:{n}", act="BNS_2023", text=text,
        heading=f"Heading {n}", score=0.9,
        source={"source_pdf": "BNS.pdf", "page_start": n, "page_end": n},
    )


def pieces(text: str, size: int = 7) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setenv("GENERATION_API_BASE_URL", BASE)
    monkeypatch.setenv("GENERATION_API_KEY", SECRET)
    monkeypatch.setenv("CAPSTONE_API_KEY", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def stub(monkeypatch):
    """Stub retrieval and the streamed provider; records calls.

    `replies` has one entry per attempt (the last repeats): a reply string, an Exception, or a
    (partial_text, Exception) tuple for a failure after some text.
    """
    calls = {"retrieval": 0, "provider": 0, "messages": [], "replies": []}
    chunks = [make_chunk(103), make_chunk(101)]

    def retrieve(request):
        calls["retrieval"] += 1
        return QueryResult(pattern="semantic", status="ok", message="m", results=list(chunks))

    def stream(messages):
        i = calls["provider"]
        calls["provider"] += 1
        calls["messages"].append(messages)
        reply = calls["replies"][min(i, len(calls["replies"]) - 1)]
        if isinstance(reply, tuple):
            yield from pieces(reply[0])
            raise reply[1]
        if isinstance(reply, Exception):
            raise reply
        yield from pieces(reply)

    monkeypatch.setattr(pipeline, "semantic_retrieve", retrieve)
    monkeypatch.setattr(answer_mod, "stream_chat_completion", stream)
    calls["chunks"] = chunks
    return calls
