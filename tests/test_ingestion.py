import json
from pathlib import Path

from building_with_rag.ingestion.chunking import split_text
from building_with_rag.ingestion.load_and_embed import build_documents
from building_with_rag.ingestion.mongodb_schema import (
    CHUNKING_VERSION,
    VECTOR_FILTER_FIELDS,
    chunk_id_for,
    embedding_id_for,
    vector_index_definition,
)
from building_with_rag.ingestion.setup_db import definition_differences

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def test_short_text_is_one_chunk():
    assert split_text("Short section.") == ["Short section."]


def test_long_text_chunks_are_bounded_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(400))
    chunks = split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 2048 for c in chunks)
    assert chunks[1][:40] in chunks[0]  # overlap carries context across the cut


def test_ids_are_deterministic_and_change_with_text():
    cid = chunk_id_for("bns:103", 0)
    assert cid == "bns:103:c0"
    assert embedding_id_for(cid, "a") == embedding_id_for(cid, "a")
    assert embedding_id_for(cid, "a") != embedding_id_for(cid, "b")


def test_vector_index_definition_and_diff():
    d = vector_index_definition()
    vec = d["fields"][0]
    assert (vec["type"], vec["path"], vec["numDimensions"], vec["similarity"]) == (
        "vector", "vector", 1024, "cosine")
    assert [f["path"] for f in d["fields"][1:]] == VECTOR_FILTER_FIELDS
    assert definition_differences(d) == []
    assert definition_differences({"fields": d["fields"][:1]})


def test_corpus_builds_unique_ids_without_crossing_sections():
    for name in ("bns", "ipc"):
        path = PROCESSED / f"{name}_sections.jsonl"
        if not path.exists():
            continue
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        docs = build_documents(records)
        ids = [c["_id"] for c in docs["chunks"]]
        assert len(ids) == len(set(ids))
        assert {c["chunking_version"] for c in docs["chunks"]} == {CHUNKING_VERSION}
        assert all(c["chunk_id"].startswith(c["section_id"] + ":c") for c in docs["chunks"])
