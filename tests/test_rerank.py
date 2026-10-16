"""Story 4.2: bounded re-ranking step, no silent fallback, shared answer path. Offline only."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from building_with_rag import pipeline
from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.generation import answer as answer_mod
from building_with_rag.main import app
from building_with_rag.retrieval import hybrid, hybrid_reranked, rerank, semantic
from building_with_rag.settings import get_settings
from conftest import make_chunk, pieces

client = TestClient(app)
KEY = "rr-secret-key-789"
TEXT = "Whoever commits murder shall be punished. "
REQ = QueryRequest(question="culpable homicide", pattern="hybrid-reranked", limit=5)


@pytest.fixture(autouse=True)
def rr_env(monkeypatch):
    monkeypatch.setenv("RERANK_API_KEY", KEY)
    monkeypatch.setenv("RERANK_CANDIDATE_LIMIT", "6")
    monkeypatch.setenv("RERANK_SEND_LIMIT", "4")
    monkeypatch.setenv("RERANK_RETURN_LIMIT", "3")
    get_settings.cache_clear()


@pytest.fixture
def world(monkeypatch):
    """8 semantic hits -> 8 fused candidates (bns:1..8); stub provider records calls."""
    state = {"sem": [{"chunk_id": f"bns:{n}:c0", "score": 1 - n / 100} for n in range(1, 9)],
             "kw": [], "sent": [], "response": None}
    monkeypatch.setattr(semantic, "_database", lambda: object())
    monkeypatch.setattr(semantic, "check_ready", lambda db: None)
    monkeypatch.setattr(hybrid, "check_keyword_ready", lambda db: None)
    monkeypatch.setattr(semantic, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(hybrid, "semantic_route", lambda *a: state["sem"])
    monkeypatch.setattr(hybrid, "keyword_route", lambda *a: state["kw"])
    monkeypatch.setattr(
        semantic, "resolve_hits",
        lambda db, hits: ([make_chunk(int(h["chunk_id"].split(":")[1]), text=TEXT + h["chunk_id"])
                           for h in hits], 0),
    )

    def fake_rerank(query, docs):
        state["sent"].append(list(docs))
        if state["response"] is not None:
            return state["response"](docs)
        return [(i, 1.0 - i / 10) for i in range(len(docs))]  # keeps order

    monkeypatch.setattr(hybrid_reranked, "rerank", fake_rerank)
    return state


def test_bounded_step_and_omitted_reasons(world):
    r = hybrid_reranked.hybrid_reranked_retrieve(REQ)
    assert r.status == "ok"
    assert world["sent"] == [[f"{TEXT}bns:{n}:c0" for n in range(1, 5)]]  # top SEND, fused order
    assert [c.chunk_id for c in r.results] == ["bns:1:c0", "bns:2:c0", "bns:3:c0"]
    om = {o.chunk_id: o.omitted_reason for o in r.omitted_candidates}
    assert len(om) == 5 and not set(om) & {c.chunk_id for c in r.results}
    assert om["bns:4:c0"].startswith("cut after re-ranking") and "rerank rank 4" in om["bns:4:c0"]
    assert om["bns:5:c0"].startswith("not sent to re-ranker: beyond RERANK_SEND_LIMIT")
    assert om["bns:6:c0"].startswith("not sent")
    assert om["bns:7:c0"] == "beyond RERANK_CANDIDATE_LIMIT (fused rank 7 of 8)"
    assert om["bns:8:c0"].startswith("beyond RERANK_CANDIDATE_LIMIT")
    text = "\n".join(map(str, r.trace))
    assert "before re-ranking" in text and "after re-ranking" in text and "(was fused #" in text
    assert "Whoever" not in text and KEY not in text


def test_reordering_fields(world):
    world["response"] = lambda docs: [(i, 0.1 * (i + 1)) for i in range(len(docs))]  # reversed
    r = hybrid_reranked_retrieve_ok()
    assert [c.chunk_id for c in r.results] == ["bns:4:c0", "bns:3:c0", "bns:2:c0"]
    assert [c.rerank_rank for c in r.results] == [1, 2, 3]
    for c in r.results:
        assert c.score == c.rerank_score
    top = r.results[0]
    assert top.fused_rank == 4 and top.fused_score is not None and top.semantic_rank == 4
    assert top.keyword_rank is None


def hybrid_reranked_retrieve_ok():
    r = hybrid_reranked.hybrid_reranked_retrieve(REQ)
    assert r.status == "ok"
    return r


def test_request_limit_lowers_k_and_invalid_limits(world, monkeypatch):
    r = hybrid_reranked.hybrid_reranked_retrieve(REQ.model_copy(update={"limit": 2}))
    assert len(r.results) == 2 and "lowered by request limit 2" in "\n".join(map(str, r.trace))
    monkeypatch.setenv("RERANK_RETURN_LIMIT", "5")  # > SEND (4)
    get_settings.cache_clear()
    r = hybrid_reranked.hybrid_reranked_retrieve(REQ)
    assert r.status == "unavailable" and "RERANK_RETURN_LIMIT" in r.message and not r.results
    assert len(world["sent"]) == 1  # invalid limits: no further provider call


@pytest.mark.parametrize("bad", [
    [(0, 0.5), (0, 0.4), (1, 0.3), (2, 0.2)],           # duplicate
    [(0, 0.5), (1, 0.4), (2, 0.3), (9, 0.2)],           # out of range
    [(0, 0.5), (1, 0.4), (2, 0.3), (3, "high")],        # non-numeric
    [(0, 0.5), (1, 0.4)],                               # missing
])
def test_malformed_via_parse_response(bad):
    payload = {"data": [{"index": i, "relevance_score": s} for i, s in bad]}
    resp = httpx.Response(200, json=payload)
    with pytest.raises(rerank.RerankUnavailable, match="malformed re-rank response"):
        rerank.parse_response(resp, 4)


def run_http(monkeypatch, behaviour):
    monkeypatch.setattr(httpx, "post", behaviour)
    return hybrid_reranked.hybrid_reranked_retrieve(REQ)


def real_rerank(monkeypatch, world):
    monkeypatch.setattr(hybrid_reranked, "rerank", rerank.rerank)


def _timeout(*a, **k):
    raise httpx.ReadTimeout("slow https://x.invalid")


@pytest.mark.parametrize("behaviour,expect", [
    (_timeout, "timeout"),
    (lambda *a, **k: httpx.Response(500, text=KEY), "http status 500"),
    (lambda *a, **k: httpx.Response(200, json={"data": [{"index": 5, "relevance_score": 1}] * 4}),
     "malformed re-rank response"),
])
def test_no_silent_fallback(monkeypatch, world, behaviour, expect):
    real_rerank(monkeypatch, world)
    r = run_http(monkeypatch, behaviour)
    assert r.status == "unavailable" and expect in r.message and not r.results
    assert KEY not in r.message and "invalid" not in r.message.split("(")[0]
    assert KEY not in json.dumps(r.model_dump(), default=str)


def test_missing_key_unavailable_and_no_generation(monkeypatch, world):
    real_rerank(monkeypatch, world)
    monkeypatch.setenv("RERANK_API_KEY", "")
    get_settings.cache_clear()
    calls = []
    monkeypatch.setattr(answer_mod, "stream_chat_completion", lambda m: calls.append(m) or iter(()))
    r = client.post("/v1/query", json={"question": "q", "pattern": "hybrid-reranked",
                                       "generate_answer": True})
    body = r.json()
    assert r.status_code == 503 and body["status"] == "unavailable"
    assert "RERANK_API_KEY" in body["message"] and body["results"] == []
    assert body["generation"] is None and calls == []


def test_zero_candidates_no_provider_call(world):
    world["sem"] = []
    r = hybrid_reranked.hybrid_reranked_retrieve(REQ)
    assert r.status == "no_results" and world["sent"] == []


GOOD = json.dumps({"answer": "Murder is punished [E1].", "outcome": "answered",
                   "claims": [{"text": "Murder is punished.", "evidence": ["E1"]}]})


def test_shared_answer_path_and_other_modes(monkeypatch, world):
    world["response"] = lambda docs: [(i, 0.1 * (i + 1)) for i in range(len(docs))]
    provider = []

    def stream(messages):
        provider.append(messages)
        yield from pieces(GOOD)

    monkeypatch.setattr(answer_mod, "stream_chat_completion", stream)
    body = client.post("/v1/query", json={"question": "murder?", "pattern": "hybrid-reranked",
                                          "generate_answer": True}).json()
    assert len(world["sent"]) == 1 and len(provider) == 1
    assert body["generation"]["outcome"] == "answered"
    assert [c["section_id"] for c in body["generation"]["citations"]] == ["bns:4"]
    prompt = json.dumps(provider[0])
    assert "bns:4:c0" in prompt
    assert "bns:5:c0" not in prompt  # only the returned passages
    h = client.post("/v1/query", json={"question": "murder?", "pattern": "hybrid"}).json()
    assert h["results"] and all(x["rerank_score"] is None and x["rerank_rank"] is None
                                for x in h["results"])
    assert len(world["sent"]) == 1  # hybrid made no re-rank call


def test_models_and_chat_use_run_pattern_events(monkeypatch):
    assert len(client.get("/v1/models").json()["data"]) == 7
    seen = []
    real = pipeline.run_pattern_events
    monkeypatch.setattr("building_with_rag.main.run_pattern_events",
                        lambda r: seen.append(r.pattern) or real(r))
    monkeypatch.setattr(pipeline, "hybrid_reranked_retrieve", lambda r: QueryResult(
        pattern="hybrid-reranked", status="no_results", message="none"))
    r = client.post("/v1/chat/completions", json={
        "model": "rag-hybrid-reranked", "messages": [{"role": "user", "content": "x"}],
        "stream": True})
    assert r.status_code == 200 and seen == ["hybrid-reranked"]
