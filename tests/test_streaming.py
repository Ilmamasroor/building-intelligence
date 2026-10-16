"""Story 3.2: one event path, streamed chat text, attempts/validation, auth. Offline only."""

import json

import pytest
from fastapi.testclient import TestClient

from building_with_rag import pipeline
from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.generation.extractor import AnswerExtractor
from building_with_rag.main import app
from building_with_rag.settings import get_settings
from conftest import BASE, CAPSTONE, SECRET

client = TestClient(app)
CHAT = {"model": "rag-semantic", "messages": [{"role": "user", "content": "murder?"}]}
QUERY = {"question": "murder?", "pattern": "semantic", "generate_answer": True}
GOOD = json.dumps({
    "answer": "Murder is punished [E1].", "outcome": "answered",
    "claims": [{"text": "Murder is punished.", "evidence": ["E1"]}],
})
BAD_LABEL = json.dumps({
    "answer": "Murder is punished [E7].", "outcome": "answered",
    "claims": [{"text": "Murder is punished.", "evidence": ["E7"]}],
})
UNSUPPORTED = json.dumps({
    "answer": "Quantum treaties apply.", "outcome": "answered",
    "claims": [{"text": "Quantum computing exceeds treaty obligations.", "evidence": ["E1"]}],
})


def sse(text: str) -> tuple[list[dict], str]:
    frames = [ln[6:] for ln in text.split("\n\n") if ln.startswith("data: ")]
    assert frames[-1] == "[DONE]"
    chunks = [json.loads(f) for f in frames[:-1]]  # every other frame is valid JSON
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    return chunks, content


def chat_text(**extra) -> str:
    stub_calls = extra.pop("stub", None)
    if stub_calls is not None:
        stub_calls["provider"] = 0  # replies are per attempt of ONE request: start over
    return sse(client.post("/v1/chat/completions", json={**CHAT, "stream": True, **extra}).text)[1]


def test_one_retrieval_one_generation_and_stream_matches_result(stub):
    stub["replies"] = [GOOD]
    gen = client.post("/v1/query", json=QUERY).json()["generation"]
    assert stub["retrieval"] == 1 and stub["provider"] == 1
    text = chat_text(stub=stub)
    assert stub["retrieval"] == 2 and stub["provider"] == 1  # exactly one each per request
    streamed = text.split("DRAFT — checking evidence\n\n", 1)[1].split("\n\n---", 1)[0]
    assert streamed == gen["answer"]


def test_query_and_chat_agree(stub):
    stub["replies"] = [BAD_LABEL, GOOD]
    gen = client.post("/v1/query", json=QUERY).json()["generation"]
    stub["provider"] = 0
    result = pipeline.run_pattern(QueryRequest(question="q", pattern="semantic",
                                               generate_answer=True))
    text = chat_text(stub=stub)
    again = result.generation
    assert (gen["outcome"], gen["confidence"], gen["attempts"]) == ("answered", 0.9, 2)
    assert (again.outcome, again.confidence, again.attempts) == ("answered", 0.9, 2)
    assert [c["label"] for c in gen["citations"]] == [c["label"] for c in again.citations]
    assert "attempt 2 of 2" in text


def test_sse_shape(stub):
    stub["replies"] = [GOOD]
    chunks, _ = sse(client.post("/v1/chat/completions", json={**CHAT, "stream": True}).text)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert all("content" in c["choices"][0]["delta"] for c in chunks[1:-1]) and len(chunks) > 3
    assert chunks[-1]["choices"][0]["delta"] == {}
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)


def test_first_attempt_valid(stub):
    stub["replies"] = [GOOD]
    gen = client.post("/v1/query", json=QUERY).json()["generation"]
    assert gen["outcome"] == "answered" and gen["confidence"] == 0.9 and gen["attempts"] == 1
    assert gen["issues"] == [] and gen["draft_answer"] is None
    assert gen["low_confidence_reason"] is None
    text = chat_text()
    assert text.startswith("DRAFT — checking evidence\n\nMurder is punished [E1].")
    assert "Evidence check passed — confidence: high (attempt 1 of 2)" in text
    assert "Sources:\n- E1 · bns:103 · Heading 103 · BNS.pdf · p. 103" in text
    assert "{" not in text  # no claims JSON


@pytest.mark.parametrize("bad", [BAD_LABEL, UNSUPPORTED])
def test_retry_then_valid(stub, bad):
    stub["replies"] = [bad, GOOD]
    gen = client.post("/v1/query", json=QUERY).json()["generation"]
    assert gen["outcome"] == "answered" and gen["attempts"] == 2 and gen["confidence"] == 0.9
    assert len(gen["issues"]) == 1 and gen["issues"][0].startswith("attempt 1:")
    assert stub["provider"] == 2
    retry = stub["messages"][1]  # original messages + previous reply + failed checks
    assert retry[2] == {"role": "assistant", "content": bad}
    assert "failed these checks" in retry[3]["content"]
    assert "Whoever commits" not in retry[3]["content"]
    text = chat_text(stub=stub)
    assert text.count("DRAFT — checking evidence") == 2
    assert "Check failed (attempt 1 of 2)" in text and "Retrying…" in text


def test_both_attempts_invalid(stub):
    stub["replies"] = [BAD_LABEL]
    r = client.post("/v1/query", json=QUERY)
    gen = r.json()["generation"]
    assert gen["outcome"] == "malformed" and gen["answer"] is None and gen["citations"] == []
    assert gen["draft_answer"] == "Murder is punished [E7]."  # label left in place
    assert gen["confidence"] == 0.2 and gen["attempts"] == 2 and gen["low_confidence_reason"]
    assert len(gen["issues"]) == 2 and all("E7" in i for i in gen["issues"])
    assert "low confidence" in r.json()["message"]
    text = chat_text()
    assert "[E7]" in text
    assert "DRAFT — low confidence, not the final answer" in text
    assert "Sources:" not in text
    # non-streaming: same outcome, rendered with the DRAFT labels
    body = client.post("/v1/chat/completions", json=CHAT).json()
    content = body["choices"][0]["message"]["content"]
    assert "DRAFT — low confidence, not the final answer" in content


@pytest.mark.parametrize("reply", ["Murder is punished with death.", json.dumps({"outcome": "x"})])
def test_no_answer_text_is_not_low_confidence(stub, reply):
    stub["replies"] = [reply]
    gen = client.post("/v1/query", json=QUERY).json()["generation"]
    assert gen["outcome"] == "malformed" and gen["confidence"] is None
    assert gen["draft_answer"] is None and gen["issues"] and gen["attempts"] == 2
    text = chat_text()
    assert "DRAFT" not in text and "No validated answer" in text


def test_provider_failure_after_text(stub):
    stub["replies"] = [('{"answer": "Murder is pun', RuntimeError(f"{SECRET} {BASE}"))]
    gen = client.post("/v1/query", json=QUERY).json()["generation"]
    assert gen["outcome"] == "unavailable" and gen["answer"] is None and gen["confidence"] is None
    assert gen["draft_answer"] == "Murder is pun" and stub["provider"] == 1
    text = chat_text()
    assert "Murder is pun" in text
    assert "UNAVAILABLE — the answer service stopped while writing" in text
    assert SECRET not in text and BASE not in text


def test_provider_failure_before_text(stub):
    from building_with_rag.generation.provider import ProviderUnavailable

    stub["replies"] = [ProviderUnavailable("timeout")]
    gen = client.post("/v1/query", json=QUERY).json()["generation"]
    assert gen["outcome"] == "unavailable" and gen["draft_answer"] is None
    raw = client.post("/v1/chat/completions", json={**CHAT, "stream": True})
    text = sse(raw.text)[1]
    assert raw.status_code == 200 and "UNAVAILABLE" in text and "DRAFT" not in text


def test_insufficient_and_no_results_have_no_draft(stub, monkeypatch):
    stub["replies"] = [json.dumps({"outcome": "insufficient_evidence", "reason": "not covered"})]
    gen = client.post("/v1/query", json=QUERY).json()["generation"]
    assert gen["outcome"] == "insufficient_evidence" and gen["confidence"] is None
    assert gen["draft_answer"] is None
    text = chat_text()
    assert "DRAFT" not in text and "not covered" in text and "confidence" not in text

    calls = stub["provider"]
    monkeypatch.setattr(pipeline, "semantic_retrieve",
                        lambda r: QueryResult(pattern="semantic", status="no_results", message="m"))
    gen = client.post("/v1/query", json=QUERY).json()["generation"]
    assert gen["outcome"] == "insufficient_evidence" and stub["provider"] == calls
    assert "DRAFT" not in chat_text()


def test_retrieval_unavailable_is_a_labelled_line(monkeypatch):
    monkeypatch.setattr(pipeline, "semantic_retrieve", lambda r: QueryResult(
        pattern="semantic", status="unavailable", message="Retrieval unavailable: db down."))
    raw = client.post("/v1/chat/completions", json={**CHAT, "stream": True})
    assert raw.status_code == 200
    assert sse(raw.text)[1].strip() == "UNAVAILABLE — retrieval: db down."


def test_unexpected_error_still_closes(monkeypatch):
    def boom(request):
        raise RuntimeError(f"{SECRET}")

    monkeypatch.setattr(pipeline, "semantic_retrieve", boom)
    raw = client.post("/v1/chat/completions", json={**CHAT, "stream": True})
    chunks, text = sse(raw.text)
    assert "UNAVAILABLE — an unexpected error" in text and SECRET not in raw.text
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_extractor_split_pieces_and_escapes():
    reply = '```json\n{"answer": "Line\\nTwo \\"q\\" \\u00e9 \\ud83d\\ude00 end", "outcome": ' \
            '"answered", "claims": [{"text": "answer", "evidence": ["E1"]}]}\n```'
    for size in (1, 2, 3, 5, 50):
        ex = AnswerExtractor()
        out = "".join(ex.feed(reply[i : i + size]) for i in range(0, len(reply), size))
        assert out == 'Line\nTwo "q" é 😀 end'
    ex = AnswerExtractor()
    assert ex.feed('{"claims": [{"text": "answer"}], "reason": "answer", "outcome": "x"}') == ""
    assert AnswerExtractor().feed("Murder is punished with death.") == ""
    assert AnswerExtractor().feed('{"outcome": "insufficient_evidence", "reason": "r"}') == ""


def test_auth(stub, monkeypatch):
    stub["replies"] = [GOOD]
    monkeypatch.setenv("CAPSTONE_API_KEY", CAPSTONE)
    get_settings.cache_clear()
    good = {"Authorization": f"Bearer {CAPSTONE}"}
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": CAPSTONE}):
        r = client.post("/v1/query", json=QUERY, headers=headers)
        assert r.status_code == 401 and r.json()["detail"] and CAPSTONE not in r.text
        for r in (client.get("/v1/models", headers=headers),
                  client.post("/v1/chat/completions", json=CHAT, headers=headers)):
            assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
    assert client.post("/v1/query", json=QUERY, headers=good).status_code == 200
    assert client.get("/v1/models", headers=good).status_code == 200
    assert client.post("/v1/chat/completions", json=CHAT, headers=good).status_code == 200
    assert client.get("/healthz").status_code == 200
    # key unset: unchanged
    monkeypatch.setenv("CAPSTONE_API_KEY", "")
    get_settings.cache_clear()
    assert client.get("/v1/models").status_code == 200


def test_no_secret_in_any_output(stub, monkeypatch):
    monkeypatch.setenv("CAPSTONE_API_KEY", CAPSTONE)
    get_settings.cache_clear()
    h = {"Authorization": f"Bearer {CAPSTONE}"}
    for replies in ([GOOD], [BAD_LABEL], [("{\"answer\": \"x", RuntimeError(SECRET))]):
        stub["replies"] = replies
        outputs = [
            client.post("/v1/query", json=QUERY, headers=h).text,
            client.post("/v1/chat/completions", json={**CHAT, "stream": True}, headers=h).text,
            client.post("/v1/chat/completions", json=CHAT, headers=h).text,
        ]
        for out in outputs:
            assert SECRET not in out and CAPSTONE not in out and BASE not in out
