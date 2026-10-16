"""Story 5.1: classifier, exact lookup, answer boundary. Offline only."""

import pytest
from fastapi.testclient import TestClient

from building_with_rag import pipeline
from building_with_rag.contracts import QueryRequest
from building_with_rag.generation import answer as answer_mod
from building_with_rag.main import app
from building_with_rag.registry import MODES
from building_with_rag.retrieval import semantic, structured
from building_with_rag.retrieval.semantic import RetrievalUnavailable
from building_with_rag.retrieval.structured_classifier import classify

client = TestClient(app)


@pytest.mark.parametrize(
    "question,act,number",
    [
        ("BNS section 103", "BNS", "103"),
        ("What does IPC s. 302 say?", "IPC", "302"),
        ("section 103 of the BNS", "BNS", "103"),
        ("IPC-302", "IPC", "302"),
        ("sec. 7 Indian Penal Code", "IPC", "7"),
    ],
)
def test_classifier_exact(question, act, number):
    c = classify(question, None)
    assert c.outcome == "exact"
    assert (c.signals.intent, c.signals.act, c.signals.section_number) == (
        "exact_lookup", act, number,
    )


@pytest.mark.parametrize(
    "question,chapter",
    [
        ("section 103", None),
        ("BNS and IPC section 103", None),
        ("IPC 153A", None),
        ("IPC sections 153A", None),
        ("sections 100 and 101", None),
        ("BNS sections 100 and 101", None),
        ("IPC 302-304", None),
        ("BNS section 007", None),
        ("BNS section 103 and section 104", None),
        ("BNS", None),
        ("BNS section 103", "vi!"),
        ('103"}, {"$where": "..."}', None),
    ],
)
def test_classifier_clarifies(question, chapter):
    c = classify(question, chapter)
    assert c.outcome == "needs_clarification"
    assert c.signals is None or c.signals.section_number is None
    assert c.reason


@pytest.mark.parametrize(
    "question,intent",
    [
        ("list all sections in chapter VI of the BNS", "filter"),
        ("list repealed sections", "filter"),
        ("how many sections in the IPC", "aggregation"),
        ("what is murder", None),
    ],
)
def test_classifier_recommends(question, intent):
    c = classify(question, None)
    assert c.outcome == "recommendation"
    assert (c.signals.intent if c.signals else None) == intent


def test_classifier_chapter():
    assert classify("BNS section 103", " vi ").signals.chapter == "VI"
    assert classify("BNS section 103", None).signals.chapter is None


# ---- retrieval with an in-memory database ----


class Cursor(list):
    def sort(self, *_):
        return self


class Coll:
    def __init__(self, docs):
        self.docs = docs
        self.queries = []

    def estimated_document_count(self):
        return len(self.docs)

    def find(self, query=None, projection=None):
        self.queries.append(query)
        return Cursor(d for d in self.docs if self._match(d, query or {}))

    @staticmethod
    def _match(doc, query):
        for k, v in query.items():
            if isinstance(v, dict) and "$in" in v:
                if doc.get(k) not in v["$in"]:
                    return False
            elif doc.get(k) != v:
                return False
        return True


class FakeDB(dict):
    pass


def section(sid, number, act="BNS_2023"):
    return {"section_id": sid, "source_id": "src1", "act": act, "chapter": "VI",
            "section_number": number, "heading": f"H {sid}", "access_level": "public",
            "source_pdf": "x.pdf"}


def chunk(sid, i):
    return {"chunk_id": f"{sid}:c{i}", "section_id": sid, "chunk_index": i, "act": "BNS_2023",
            "text": f"text {sid} {i}", "status": "in_force", "access_level": "public"}


@pytest.fixture
def db(monkeypatch):
    d = FakeDB(
        sections=Coll([section("bns:103", 103), section("ipc:153", 153, "IPC_1860"),
                       section("ipc:153A", 153, "IPC_1860")]),
        chunks=Coll([chunk("bns:103", 1), chunk("bns:103", 0), chunk("ipc:153", 0)]),
        sources=Coll([{"_id": "src1", "status": "in_force", "source_status_version": "v1"}]),
    )
    # chunks are stored out of order; the fake cursor does not sort, so pre-sort here
    d["chunks"].docs.sort(key=lambda c: c["chunk_index"])
    d.calls = 0
    d.name = "t"

    def database():
        d.calls += 1
        return d

    monkeypatch.setattr(semantic, "_database", database)
    return d


def req(q, **kw):
    return QueryRequest(question=q, pattern="structured", **kw)


def test_one_match_returns_ordered_chunks_without_score(db):
    r = structured.structured_retrieve(req("BNS section 103"))
    assert r.status == "ok"
    assert [x.chunk_id for x in r.results] == ["bns:103:c0", "bns:103:c1"]
    assert all(x.score is None and x.semantic_score is None for x in r.results)
    assert r.results[0].source["status"] == "in_force"
    assert r.results[0].source["source_status_version"] == "v1"
    assert "recorded status: in_force" in r.message
    assert "BNS section 103" not in "\n".join(map(str, r.trace))


def test_predicate_has_only_validated_keys(db):
    structured.structured_retrieve(req("BNS section 103", chapter="vi"))
    assert db["sections"].queries[0] == {
        "act": "BNS_2023", "section_number": 103,
        "access_level": {"$in": ["public"]}, "chapter": "VI",
    }


def test_limit_cuts_chunks_and_is_traced(db):
    r = structured.structured_retrieve(req("BNS section 103", limit=1))
    assert len(r.results) == 1 and "cut 1 chunk" in "\n".join(r.trace)


def test_not_found(db):
    r = structured.structured_retrieve(req("BNS section 999"))
    assert r.status == "not_found" and r.results == []
    assert "indexed documents" in r.message


def test_status_filter_through_sources(db):
    from building_with_rag.contracts import SemanticFilters

    r = structured.structured_retrieve(
        req("BNS section 103", filters=SemanticFilters(status=["repealed"]))
    )
    assert r.status == "not_found"


def test_act_filter_excluding_act(db):
    from building_with_rag.contracts import SemanticFilters

    r = structured.structured_retrieve(
        req("BNS section 103", filters=SemanticFilters(act=["IPC_1860"]))
    )
    assert r.status == "not_found"


def test_lettered_siblings_need_clarification(db):
    r = structured.structured_retrieve(req("IPC section 153"))
    assert r.status == "needs_clarification" and r.results == []
    assert "ipc:153, ipc:153A" in r.message


@pytest.mark.parametrize("q", ["section 103", "how many sections in the IPC", "what is murder"])
def test_non_exact_never_touches_database(db, q):
    r = structured.structured_retrieve(req(q))
    assert r.status in ("needs_clarification", "recommendation")
    assert db.calls == 0 and r.results == []
    assert "database lookup skipped" in r.trace


def test_missing_uri_is_unavailable(monkeypatch):
    def boom():
        raise RetrievalUnavailable("MONGODB_URI is not set")

    monkeypatch.setattr(semantic, "_database", boom)
    r = structured.structured_retrieve(req("BNS section 103"))
    assert r.status == "unavailable" and "MONGODB_URI" in r.message


def test_empty_collection_is_unavailable(db):
    db["sources"].docs.clear()
    assert structured.structured_retrieve(req("BNS section 103")).status == "unavailable"


# ---- pipeline / API ----


@pytest.fixture
def gen(monkeypatch, db):
    calls = {"n": 0}

    def stream(messages):
        calls["n"] += 1
        yield "[]"

    monkeypatch.setattr(answer_mod, "stream_chat_completion", stream)
    return calls


@pytest.mark.parametrize("q", ["BNS section 999", "section 103", "how many sections in the IPC"])
def test_no_model_call_unless_ok(gen, q):
    r = pipeline.run_pattern(req(q, generate_answer=True))
    assert r.status in ("not_found", "needs_clarification", "recommendation")
    assert gen["n"] == 0 and r.generation is None


def test_ok_runs_generation_once(gen):
    r = pipeline.run_pattern(req("BNS section 103", generate_answer=True))
    assert r.status == "ok" and r.generation is not None and gen["n"] >= 1


def test_api_status_codes(db):
    ok = client.post("/v1/query", json={"question": "BNS section 103", "pattern": "structured"})
    assert ok.status_code == 200 and ok.json()["status"] == "ok"
    amb = client.post("/v1/query", json={"question": "section 103", "pattern": "structured"})
    assert amb.status_code == 200 and amb.json()["status"] == "needs_clarification"
    db["sources"].docs.clear()
    down = client.post("/v1/query", json={"question": "BNS section 103", "pattern": "structured"})
    assert down.status_code == 503


def test_models_still_seven():
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert sorted(ids) == sorted(MODES) and len(ids) == 7
