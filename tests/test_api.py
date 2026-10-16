from fastapi.testclient import TestClient

from building_with_rag import pipeline
from building_with_rag.main import app
from building_with_rag.registry import MODES

client = TestClient(app)


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_query_other_pattern_not_implemented(monkeypatch):
    monkeypatch.setattr(pipeline, "IMPLEMENTED_PATTERNS", ())
    r = client.post("/v1/query", json={"question": "What is murder?", "pattern": "decomposition"})
    assert r.status_code == 200
    assert r.json()["status"] == "not_implemented"
    assert r.json()["results"] == []


def test_query_rejects_empty_question():
    assert client.post("/v1/query", json={"question": "", "pattern": "semantic"}).status_code == 422


def test_models_lists_seven():
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert sorted(ids) == sorted(MODES) and len(ids) == 7


def test_chat_other_pattern_placeholder(monkeypatch):
    monkeypatch.setattr(pipeline, "IMPLEMENTED_PATTERNS", ())
    req = {"model": "rag-decomposition", "messages": [{"role": "user", "content": "hi"}]}
    r = client.post("/v1/chat/completions", json=req)
    assert r.json()["choices"][0]["message"]["content"] == (
        "The 'decomposition' pattern is not implemented yet."
    )
    s = client.post("/v1/chat/completions", json={**req, "stream": True})
    assert "not implemented yet" in s.text and s.text.rstrip().endswith("data: [DONE]")


def test_chat_error_envelope():
    r = client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
    r = client.post("/v1/chat/completions", json={"model": "rag-semantic"})
    assert r.status_code == 400 and "error" in r.json()


def test_chat_accepts_nested_rag_options_filters(monkeypatch):
    seen = {}

    def run(request):
        seen["filters"] = request.filters
        from building_with_rag.contracts import QueryResult

        return QueryResult(pattern="semantic", status="no_results", message="m")

    monkeypatch.setattr("building_with_rag.main.run_pattern", run)
    opts = {"pattern": "semantic", "limit": 5, "filters": {"act": ["BNS_2023"], "status": None}}
    r = client.post(
        "/v1/chat/completions",
        json={"model": "rag-semantic", "messages": [{"role": "user", "content": "x"}],
              "rag_options": opts},
    )
    assert r.status_code == 200 and seen["filters"].act == ["BNS_2023"]
