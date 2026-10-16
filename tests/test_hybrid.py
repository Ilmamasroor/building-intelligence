"""Story 4.1: fusion, keyword stage, hybrid_retrieve and the shared answer path. Offline only."""

import json

import pytest
from fastapi.testclient import TestClient
from pymongo.errors import OperationFailure

from building_with_rag import pipeline
from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.generation import answer as answer_mod
from building_with_rag.main import app
from building_with_rag.registry import MODES
from building_with_rag.retrieval import hybrid, semantic
from conftest import make_chunk, pieces

client = TestClient(app)


def hit(cid, score=1.0):
    return {"chunk_id": cid, "score": score}


def test_fuse_both_routes_outranks_single_route():
    out = hybrid.fuse([hit("a"), hit("b")], [hit("c"), hit("b")])
    by = {f.chunk_id: f for f in out}
    assert by["b"].fused_score == pytest.approx(1 / 62 + 1 / 62)
    assert by["b"].fused_score > by["a"].fused_score
    assert out[0].chunk_id == "b"


def test_fuse_single_route_fields_none_and_ranks():
    by = {f.chunk_id: f for f in hybrid.fuse([hit("a", 0.8)], [hit("k", 7.4)])}
    assert by["a"].keyword_rank is None and by["a"].keyword_score is None
    assert by["a"].semantic_rank == 1 and by["a"].semantic_score == 0.8
    assert by["k"].semantic_rank is None and by["k"].semantic_score is None
    assert by["k"].route == "keyword only" and by["a"].route == "semantic only"


def test_fuse_ties_are_deterministic():
    # equal fused score and equal best rank: chunk_id decides, whatever the input order
    a = hybrid.fuse([hit("z")], [hit("m")])
    b = hybrid.fuse([hit("z")], [hit("m")])
    assert [f.chunk_id for f in a] == ["m", "z"] == [f.chunk_id for f in b]
    # equal fused score, different best rank: the better rank wins
    out = hybrid.fuse([hit("p"), hit("q")], [hit("r"), hit("s")])
    assert [f.chunk_id for f in out] == ["p", "r", "q", "s"]


def test_keyword_stages_scope_inside_search_and_question_only_as_value():
    q = '{"$where": "1"} culpable'
    filters = {"act": ["BNS_2023"], "access_level": ["public"]}
    stages = hybrid.keyword_stages(q, filters, 20)
    s = stages[0]["$search"]
    assert s["index"] == "chunks_keyword_index"
    assert s["compound"]["must"] == [{"text": {"query": q, "path": "text"}}]
    assert s["compound"]["filter"] == [
        {"in": {"path": "act", "value": ["BNS_2023"]}},
        {"in": {"path": "access_level", "value": ["public"]}},
    ]
    assert stages[1] == {"$limit": 20}
    assert json.dumps(stages).count(q.replace('"', '\\"')) == 1


@pytest.fixture
def routes(monkeypatch):
    state = {"sem": [hit("bns:103:c0", 0.81), hit("bns:101:c0", 0.7)],
             "kw": [hit("bns:101:c0", 7.4), hit("bns:105:c0", 3.0)], "embeds": 0}
    monkeypatch.setattr(semantic, "_database", lambda: object())
    monkeypatch.setattr(semantic, "check_ready", lambda db: None)
    monkeypatch.setattr(hybrid, "check_keyword_ready", lambda db: None)

    def embed(q):
        state["embeds"] += 1
        return [0.0]

    monkeypatch.setattr(semantic, "embed_query", embed)

    def route(name):
        def fn(*a):
            if isinstance(state[name], Exception):
                raise state[name]
            return state[name]
        return fn

    monkeypatch.setattr(hybrid, "semantic_route", route("sem"))
    monkeypatch.setattr(hybrid, "keyword_route", route("kw"))
    monkeypatch.setattr(
        semantic, "resolve_hits",
        lambda db, hits: ([make_chunk(int(h["chunk_id"].split(":")[1])) for h in hits
                           if h["chunk_id"] != "bns:999:c0"], 0),
    )
    return state


REQ = QueryRequest(question="culpable homicide", pattern="hybrid", limit=5)


def test_hybrid_ok_fills_fields_and_counts(routes):
    r = hybrid.hybrid_retrieve(REQ)
    assert r.status == "ok" and routes["embeds"] == 1
    top = r.results[0]
    assert top.chunk_id == "bns:101:c0" and top.fused_rank == 1
    assert top.score == top.fused_score
    assert (top.semantic_rank, top.keyword_rank) == (2, 1)
    assert top.rerank_score is None and top.rerank_rank is None
    assert [x.fused_rank for x in r.results] == [1, 2, 3]
    only_kw = next(x for x in r.results if x.chunk_id == "bns:105:c0")
    assert only_kw.semantic_rank is None and only_kw.keyword_rank == 2
    text = "\n".join(map(str, r.trace))
    assert "in both 1, semantic only 1, keyword only 1" in text and "both" in text
    assert "Reciprocal Rank Fusion" in text


def test_hybrid_one_route_empty_is_ok_both_empty_is_no_results(routes):
    routes["kw"] = []
    r = hybrid.hybrid_retrieve(REQ)
    assert r.status == "ok" and all(x.keyword_rank is None for x in r.results)
    routes["sem"] = []
    assert hybrid.hybrid_retrieve(REQ).status == "no_results"


def test_hybrid_unavailable_cases(routes, monkeypatch):
    routes["kw"] = OperationFailure("boom secret-uri")
    r = hybrid.hybrid_retrieve(REQ)
    assert r.status == "unavailable" and "keyword route" in r.message and "secret" not in r.message
    routes["kw"] = []
    routes["sem"] = OperationFailure("boom")
    assert hybrid.hybrid_retrieve(REQ).status == "unavailable"

    def missing(db):
        raise semantic.RetrievalUnavailable("keyword index not ready: 'x' does not exist")

    monkeypatch.setattr(hybrid, "check_keyword_ready", missing)
    r = hybrid.hybrid_retrieve(REQ)
    assert r.status == "unavailable" and "keyword index" in r.message


def test_check_keyword_ready_states():
    class Coll:
        def __init__(self, found):
            self.found = found

        def list_search_indexes(self, name):
            return self.found

    class Db(dict):
        pass

    with pytest.raises(semantic.RetrievalUnavailable, match="does not exist"):
        hybrid.check_keyword_ready(Db(chunks=Coll([])))
    with pytest.raises(semantic.RetrievalUnavailable, match="not queryable"):
        hybrid.check_keyword_ready(Db(chunks=Coll([{"queryable": False, "status": "PENDING"}])))
    hybrid.check_keyword_ready(Db(chunks=Coll([{"queryable": True}])))


GOOD = json.dumps({
    "answer": "Murder is punished [E1].", "outcome": "answered",
    "claims": [{"text": "Murder is punished.", "evidence": ["E1"]}],
})


def test_query_hybrid_uses_shared_answer_path(monkeypatch):
    calls = {"retrieval": 0, "provider": 0}
    chunks = [make_chunk(103)]
    for i, c in enumerate(chunks, 1):
        c.fused_rank, c.fused_score = i, 0.0328

    def retrieve(request):
        calls["retrieval"] += 1
        return QueryResult(pattern="hybrid", status="ok", message="m", results=list(chunks))

    def stream(messages):
        calls["provider"] += 1
        yield from pieces(GOOD)

    monkeypatch.setattr(pipeline, "hybrid_retrieve", retrieve)
    monkeypatch.setattr(answer_mod, "stream_chat_completion", stream)
    r = client.post("/v1/query", json={"question": "murder?", "pattern": "hybrid",
                                       "generate_answer": True})
    body = r.json()
    assert r.status_code == 200 and calls == {"retrieval": 1, "provider": 1}
    assert body["generation"]["outcome"] == "answered"
    assert [c["section_id"] for c in body["generation"]["citations"]] == ["bns:103"]


def test_semantic_shape_unchanged(stub):
    stub["replies"] = ["x"]
    body = client.post("/v1/query", json={"question": "q", "pattern": "semantic"}).json()
    res = body["results"][0]
    assert res["score"] == 0.9
    for f in ("keyword_score", "keyword_rank", "fused_score", "fused_rank"):
        assert res[f] is None


def test_models_seven_and_chat_hybrid_uses_run_pattern_events(monkeypatch):
    assert len(client.get("/v1/models").json()["data"]) == 7 == len(MODES)
    seen = []
    real = pipeline.run_pattern_events

    def spy(request):
        seen.append(request.pattern)
        return real(request)

    monkeypatch.setattr("building_with_rag.main.run_pattern_events", spy)
    monkeypatch.setattr(pipeline, "hybrid_retrieve", lambda r: QueryResult(
        pattern="hybrid", status="no_results", message="none"))
    r = client.post("/v1/chat/completions", json={
        "model": "rag-hybrid", "messages": [{"role": "user", "content": "x"}], "stream": True})
    assert r.status_code == 200 and seen == ["hybrid"]
