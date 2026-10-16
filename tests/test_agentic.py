"""Story 6.1: bounded agentic RAG controller. Offline only (stubbed adapters and model)."""

import json

import pytest
from fastapi.testclient import TestClient

from building_with_rag.contracts import QueryRequest, QueryResult, SemanticFilters
from building_with_rag.generation import answer as answer_mod
from building_with_rag.main import app
from building_with_rag.registry import MODES
from building_with_rag.retrieval import agentic, hyde, semantic, structured
from building_with_rag.settings import get_settings
from conftest import make_chunk, pieces

client = TestClient(app)

MURDER = "Whoever commits murder shall be punished with death or imprisonment for life."
LAY_Q = "What happens if someone makes me hand over my property by threatening to hurt me?"
LAY_TEXT = "Extortion: putting a person in fear of injury to hand over property, threatening hurt."
ORDINARY = "What is culpable homicide?"
MARKER = "HYPOTHETICAL-MARKER-XYZ"


@pytest.fixture
def adapters(monkeypatch):
    """Stub the three allowlisted adapters; `calls` records (name, request)."""
    state = {"calls": [], "structured": None, "semantic": None, "hyde": None}

    def make(name, key, status_default):
        def adapter(request):
            state["calls"].append((name, request))
            reply = state[key]
            if isinstance(reply, Exception):
                raise reply
            if isinstance(reply, QueryResult):
                return reply
            chunks = list(reply or [])
            return QueryResult(
                pattern=name, status="ok" if chunks else status_default, message="m",
                results=chunks, trace=[f"query: {request.question}"],
            )
        return adapter

    for module, attr, key, empty in (
        (structured, "structured_retrieve", "structured", "not_found"),
        (semantic, "semantic_retrieve", "semantic", "no_results"),
        (hyde, "hyde_retrieve", "hyde", "no_results"),
    ):
        monkeypatch.setattr(module, attr, make(key, key, empty))
    return state


def names(state):
    return [n for n, _ in state["calls"]]


def req(question, **kw):
    return QueryRequest(question=question, pattern="agentic", **kw)


def text_of(result):
    return "\n".join([result.message, *map(str, result.trace)])


# ---- the four rule paths ----


def test_act_qualified_lookup(adapters):
    adapters["structured"] = [make_chunk(103, MURDER)]
    r = agentic.agentic_retrieve(req("BNS section 103"))
    assert names(adapters) == ["structured"] and r.status == "ready_for_synthesis"
    assert [c.section_id for c in r.results] == ["bns:103"]
    assert r.trace[:3] == ["mode: agentic", "validation: accepted", "path: structured"]
    assert "call 1: structuredLookup status=ok passages=1 usable=1" in r.trace
    assert "calls: 1 of 2" in r.trace and r.trace[-1] == "stop: evidence_adequate"


def test_not_found_is_clarify_without_fall_through(adapters):
    adapters["structured"] = []
    r = agentic.agentic_retrieve(req("BNS section 103"))
    assert names(adapters) == ["structured"] and r.status == "clarify" and r.results == []
    assert r.trace[-1] == "stop: record_not_found"


def test_wrong_section_from_adapter_is_not_evidence(adapters):
    adapters["structured"] = [make_chunk(104, MURDER)]
    r = agentic.agentic_retrieve(req("BNS section 103"))
    assert r.status == "clarify" and r.results == []


def test_ordinary_question_one_semantic_call(adapters):
    adapters["semantic"] = [make_chunk(100, MURDER)]
    r = agentic.agentic_retrieve(req(ORDINARY))
    assert names(adapters) == ["semantic"] and r.status == "ready_for_synthesis"
    assert "path: semantic" in r.trace and "calls: 1 of 2" in r.trace


def test_ordinary_without_evidence_is_partial_and_stops(adapters):
    adapters["semantic"] = []
    r = agentic.agentic_retrieve(req(ORDINARY))
    assert names(adapters) == ["semantic"] and r.status == "partial_result" and r.results == []
    assert r.trace[-1] == "stop: no_usable_evidence"


def test_mismatch_adequate_direct_needs_no_hyde(adapters):
    adapters["semantic"] = [make_chunk(308, LAY_TEXT)]
    r = agentic.agentic_retrieve(req(LAY_Q))
    assert names(adapters) == ["semantic"] and r.status == "ready_for_synthesis"
    assert "adequacy: vocabulary mismatch (direct) -> adequate" in r.trace


def test_mismatch_inadequate_direct_calls_hyde_once(adapters):
    adapters["semantic"] = [make_chunk(1, "Unrelated provision about weights and measures.")]
    adapters["hyde"] = [make_chunk(308, LAY_TEXT)]
    r = agentic.agentic_retrieve(req(LAY_Q))
    assert names(adapters) == ["semantic", "hyde"] and r.status == "ready_for_synthesis"
    assert "path: semantic+hyde" in r.trace and "calls: 2 of 2" in r.trace
    assert "adequacy: vocabulary mismatch (direct) -> not adequate" in r.trace
    assert [c.chunk_id for c in r.results] == ["bns:308:c0", "bns:1:c0"]


def test_mismatch_still_inadequate_is_partial_with_passages(adapters):
    adapters["semantic"] = [make_chunk(1, "Unrelated provision about weights and measures.")]
    adapters["hyde"] = []
    r = agentic.agentic_retrieve(req(LAY_Q))
    assert r.status == "partial_result" and [c.section_id for c in r.results] == ["bns:1"]


@pytest.mark.parametrize(
    "question", ["section 103", "BNS sections 103 and 104", "BNS 153A", "how many sections in BNS",
                 "BNS", "?!"]
)
def test_ambiguous_input_is_clarify_with_zero_calls(adapters, question):
    r = agentic.agentic_retrieve(req(question))
    assert r.status == "clarify" and adapters["calls"] == [] and "calls: 0 of 2" in r.trace
    trace = "\n".join(r.trace)
    assert "validation: rejected (" in trace and r.trace[-1] == "stop: validation_rejected"
    assert question not in trace and r.message == agentic.CLARIFY_HINT


def test_invalid_chapter_is_rejected(adapters):
    r = agentic.agentic_retrieve(req(ORDINARY, chapter="six"))
    assert r.status == "clarify" and adapters["calls"] == []


# ---- scope, bounds, deadline, failures ----


def test_filters_and_caller_pass_unchanged_to_every_call(adapters):
    adapters["semantic"] = [make_chunk(1, "Unrelated provision about weights and measures.")]
    adapters["hyde"] = [make_chunk(308, LAY_TEXT)]
    f = SemanticFilters(act=["BNS_2023"], access_level=["public"])
    request = req(LAY_Q, filters=f, caller_id="c1", limit=7, chapter="XVII", required_acts=["BNS"])
    agentic.agentic_retrieve(request)
    assert len(adapters["calls"]) == 2
    for _, sub in adapters["calls"]:
        assert sub.filters == f and sub.caller_id == "c1" and sub.limit == 7
        assert sub.chapter == "XVII" and sub.required_acts == ["BNS"]
        assert sub.generate_answer is False
    assert [s.pattern for _, s in adapters["calls"]] == ["semantic", "hyde"]


def test_passages_outside_requested_act_are_dropped(adapters):
    adapters["semantic"] = [make_chunk(100, MURDER)]  # act BNS_2023
    r = agentic.agentic_retrieve(req(ORDINARY, filters=SemanticFilters(act=["IPC_1860"])))
    assert r.status == "partial_result" and r.results == []
    assert "call 1: semanticRetrieve status=ok passages=1 usable=0" in r.trace


def test_guard_refuses_third_and_second_hyde_call(adapters):
    run = agentic._Run(req(LAY_Q))
    run.call("semanticRetrieve")
    run.call("hydeRetrieve")
    with pytest.raises(agentic._Stop) as third:
        run.call("semanticRetrieve")
    assert third.value.reason == "call_budget_exhausted"
    run2 = agentic._Run(req(LAY_Q))
    run2.call("hydeRetrieve")
    with pytest.raises(agentic._Stop):
        run2.call("hydeRetrieve")
    with pytest.raises(agentic._Stop):
        run2.call("rawDatabase")
    assert len(adapters["calls"]) == 3


def test_deadline_before_hyde_keeps_direct_passages(adapters, monkeypatch):
    adapters["semantic"] = [make_chunk(1, "Unrelated provision about weights and measures.")]
    ticks = iter([0, 0, 0, 31, 31, 31, 31])
    monkeypatch.setattr(agentic, "_clock", lambda: next(ticks))
    r = agentic.agentic_retrieve(req(LAY_Q))
    assert names(adapters) == ["semantic"] and r.status == "partial_result"
    assert r.trace[-1] == "stop: deadline_exceeded" and len(r.results) == 1


def test_call_returning_after_deadline_is_discarded(adapters, monkeypatch):
    adapters["semantic"] = [make_chunk(100, MURDER)]
    ticks = iter([0, 0, 99, 99, 99])
    monkeypatch.setattr(agentic, "_clock", lambda: next(ticks))
    r = agentic.agentic_retrieve(req(ORDINARY))
    assert r.status == "partial_result" and r.results == []
    assert "usable=0" in "\n".join(r.trace) and r.trace[-1] == "stop: deadline_exceeded"


@pytest.mark.parametrize(
    "raw,expected",
    [("5", (5, True)), ("0", (30, False)), ("999", (30, False)), ("abc", (30, False)),
     ("", (30, False))],
)
def test_deadline_setting(monkeypatch, raw, expected):
    monkeypatch.setenv("AGENTIC_DEADLINE_SECONDS", raw)
    get_settings.cache_clear()
    assert agentic.deadline_seconds() == expected


def test_unavailable_without_evidence(adapters):
    adapters["semantic"] = QueryResult(pattern="semantic", status="unavailable",
                                       message="Retrieval unavailable: https://secret.example")
    r = agentic.agentic_retrieve(req(ORDINARY))
    assert r.status == "unavailable" and r.trace[-1] == "stop: tool_unavailable"
    assert "semanticRetrieve" in r.message and "secret.example" not in text_of(r)


def test_exception_is_unavailable_without_raw_error(adapters):
    adapters["structured"] = RuntimeError("password=hunter2")
    r = agentic.agentic_retrieve(req("BNS section 103"))
    assert r.status == "unavailable" and "hunter2" not in text_of(r)


def test_hyde_unavailable_keeps_direct_passage_as_partial(adapters):
    adapters["semantic"] = [make_chunk(1, "Unrelated provision about weights and measures.")]
    adapters["hyde"] = QueryResult(pattern="hyde", status="hyde_unavailable", message="x")
    r = agentic.agentic_retrieve(req(LAY_Q))
    assert r.status == "partial_result" and len(r.results) == 1
    assert r.trace[-1] == "stop: tool_unavailable"


def test_unexpected_status_is_unavailable(adapters):
    adapters["structured"] = QueryResult(pattern="structured", status="weird", message="x")
    assert agentic.agentic_retrieve(req("BNS section 103")).status == "unavailable"


# ---- HyDE text and question never leak ----


def test_hyde_text_and_question_never_leak(adapters, monkeypatch):
    adapters["semantic"] = [make_chunk(1, "Unrelated provision about weights and measures.")]
    leaky = QueryResult(
        pattern="hyde", status="ok", message=MARKER, trace=[f"hypothetical {MARKER}"],
        results=[make_chunk(308, LAY_TEXT)], hyde_direct_candidates=[make_chunk(5, MARKER)],
        hyde_query_candidates=[make_chunk(6, MARKER)], hyde_hypothetical_text_debug=MARKER,
    )
    adapters["hyde"] = leaky
    seen = {"messages": []}

    def stream(messages):
        seen["messages"].append(messages)
        yield from pieces(json.dumps({
            "answer": "Extortion [E1].", "outcome": "answered",
            "claims": [{"text": "Extortion hand over property.", "evidence": ["E1"]}],
        }))

    monkeypatch.setattr(answer_mod, "stream_chat_completion", stream)
    r = client.post("/v1/query", json={"question": LAY_Q, "pattern": "agentic",
                                       "generate_answer": True}).json()
    assert r["status"] == "ready_for_synthesis"
    assert MARKER not in json.dumps(r) and MARKER not in json.dumps(seen["messages"])
    assert r["hyde_direct_candidates"] == [] and r["hyde_query_candidates"] == []
    assert r["hyde_hypothetical_text_debug"] is None
    trace = "\n".join(r["trace"])
    assert "hand over" not in trace and "threatening" not in trace and "query:" not in trace
    assert r["generation"]["outcome"] == "answered"
    cited = {c["chunk_id"] for c in r["generation"]["citations"]}
    assert cited <= {x["chunk_id"] for x in r["results"]}
    chat = client.post("/v1/chat/completions", json={
        "model": "rag-agentic", "messages": [{"role": "user", "content": LAY_Q}]})
    assert MARKER not in chat.text


# ---- API / pipeline ----


def test_models_lists_seven_agentic_included():
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert len(ids) == 7 == len(MODES) and "rag-agentic" in ids


def test_clarify_and_unavailable_never_generate(adapters, monkeypatch):
    called = []
    monkeypatch.setattr(
        answer_mod, "stream_chat_completion", lambda m: called.append(1) or iter(())
    )
    r = client.post("/v1/query", json={"question": "section 103", "pattern": "agentic",
                                       "generate_answer": True})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "clarify" and body["generation"] is None
    adapters["semantic"] = QueryResult(pattern="semantic", status="unavailable", message="x")
    r = client.post("/v1/query", json={"question": ORDINARY, "pattern": "agentic",
                                       "generate_answer": True})
    assert r.status_code == 503 and r.json()["status"] == "unavailable" and called == []


def test_chat_stream_agentic_ends_with_done(adapters, monkeypatch):
    adapters["semantic"] = [make_chunk(100, MURDER)]
    reply = json.dumps({"answer": "Murder is punished [E1].", "outcome": "answered",
                        "claims": [{"text": "Murder is punished.", "evidence": ["E1"]}]})
    monkeypatch.setattr(answer_mod, "stream_chat_completion", lambda m: iter(pieces(reply)))
    r = client.post("/v1/chat/completions", json={
        "model": "rag-agentic", "stream": True,
        "messages": [{"role": "user", "content": ORDINARY}]})
    assert r.text.rstrip().endswith("data: [DONE]")
    frames = [f[6:] for f in r.text.split("\n\n") if f.startswith("data: {")]
    text = "".join(json.loads(f)["choices"][0]["delta"].get("content", "") for f in frames)
    assert "DRAFT — checking evidence" in text and "Sources:" in text


def test_chat_partial_result_line(adapters, monkeypatch):
    adapters["semantic"] = [make_chunk(1, "Unrelated provision about weights and measures.")]
    adapters["hyde"] = []
    reply = json.dumps({"answer": "Weights [E1].", "outcome": "answered",
                        "claims": [{"text": "Weights and measures.", "evidence": ["E1"]}]})
    monkeypatch.setattr(answer_mod, "stream_chat_completion", lambda m: iter(pieces(reply)))
    r = client.post("/v1/chat/completions", json={
        "model": "rag-agentic", "messages": [{"role": "user", "content": LAY_Q}]})
    content = r.json()["choices"][0]["message"]["content"]
    assert "Partial result — evidence for this question was limited" in content
