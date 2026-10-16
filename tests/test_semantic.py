import pytest
from fastapi.testclient import TestClient

from building_with_rag.contracts import QueryRequest, SemanticFilters
from building_with_rag.main import app
from building_with_rag.retrieval import semantic

client = TestClient(app)


def _req(**kw):
    return QueryRequest(question="q", pattern="semantic", **kw)


def test_num_candidates_bounds():
    assert semantic.num_candidates(1) == 50
    assert semantic.num_candidates(5) == 50
    assert semantic.num_candidates(20) == 200


def test_access_level_is_intersected_with_caller():
    assert semantic.effective_filters(_req())["access_level"] == ["public"]
    f = SemanticFilters(access_level=["restricted"])
    assert semantic.effective_filters(_req(filters=f))["access_level"] == []
    f = SemanticFilters(access_level=["public", "restricted"])
    assert semantic.effective_filters(_req(filters=f))["access_level"] == ["public"]


def test_vector_filter_uses_in_and_and():
    flt = semantic.vector_filter({"act": ["BNS_2023"], "access_level": ["public"]})
    assert flt == {"$and": [{"act": {"$in": ["BNS_2023"]}}, {"access_level": {"$in": ["public"]}}]}
    assert semantic.vector_filter({}) is None


@pytest.mark.parametrize(
    "filters",
    [{"nope": ["x"]}, {"act": {"$ne": "BNS_2023"}}, {"act": "BNS_2023"}, {"act": [{"$in": []}]}],
)
def test_bad_filters_are_422(filters):
    body = {"question": "q", "pattern": "semantic", "filters": filters}
    assert client.post("/v1/query", json=body).status_code == 422


def test_blank_question_and_limit_are_422():
    body = {"question": "   ", "pattern": "semantic"}
    assert client.post("/v1/query", json=body).status_code == 422
    body = {"question": "q", "pattern": "semantic", "limit": 21}
    assert client.post("/v1/query", json=body).status_code == 422


def test_unavailable_is_503_without_secret(monkeypatch):
    def boom():
        raise semantic.RetrievalUnavailable("MONGODB_URI is empty")

    monkeypatch.setattr(semantic, "_database", boom)
    r = client.post("/v1/query", json={"question": "q", "pattern": "semantic"})
    assert r.status_code == 503 and r.json()["status"] == "unavailable"
    assert r.json()["results"] == []
