"""Single source of truth for collection names, fields, IDs, validators and indexes (Story 2.2).

Nothing else in the project hard-codes a collection name, field name or ID format.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field
from pymongo.database import Database
from pymongo.errors import CollectionInvalid
from pymongo.operations import SearchIndexModel

from building_with_rag.contracts import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_VERSION,
)

# ---- collections and index ----

SOURCES = "sources"
SECTIONS = "sections"
CHUNKS = "chunks"
EMBEDDINGS = "embeddings"
VECTOR_INDEX_NAME = "embeddings_vector_index"
# Story 4.1: Atlas Search (keyword) index on the stored chunk text.
KEYWORD_INDEX_NAME = "chunks_keyword_index"
KEYWORD_TEXT_FIELD = "text"
# Scope filters applied inside $search (token fields); the same names the vector index filters on,
# minus `chapter`, which no mode filters on.
KEYWORD_FILTER_FIELDS = ["act", "status", "access_level"]

ACCESS_LEVEL_PUBLIC = "public"

# Fields used as vector filters (must also appear on the embeddings documents).
VECTOR_FILTER_FIELDS = ["act", "chapter", "status", "access_level"]
VECTOR_FIELD = "vector"

# ---- chunking ----

CHUNK_SIZE = 2048
CHUNK_OVERLAP = 256
SPLITTER = "paragraph-sentence-word"
CHUNKING_VERSION = f"chars-{CHUNK_SIZE}-{CHUNK_OVERLAP}-v1"
# Exact format of embedding_input_text, recorded with the chunking version.
EMBEDDING_INPUT_FORMAT = "{act_label} | {section_id} | {heading}\n\n{chunk_text}"


# ---- Pydantic models (shared document model) ----


class SourceDoc(BaseModel):
    source_id: str
    source_pdf: str
    source_sha256: str
    act: str
    act_label: str
    status: str
    parser: str
    parser_version: str
    source_status_version: str | None = None


class SectionDoc(BaseModel):
    section_id: str
    source_id: str
    act: str
    chapter: str
    section_number: int
    heading: str
    text: str
    source_pdf: str
    source_sha256: str
    page_start: int | None = None
    page_end: int | None = None
    needs_review: bool
    access_level: str = ACCESS_LEVEL_PUBLIC


class ChunkDoc(BaseModel):
    chunk_id: str
    section_id: str
    chunk_index: int
    chunking_version: str
    text: str
    embedding_input_text: str
    act: str
    chapter: str
    status: str
    access_level: str = ACCESS_LEVEL_PUBLIC
    splitter: dict[str, Any] = Field(default_factory=dict)


class EmbeddingDoc(BaseModel):
    embedding_id: str
    chunk_id: str
    model: str = EMBEDDING_MODEL
    model_version: str = EMBEDDING_MODEL_VERSION
    dimensions: int = EMBEDDING_DIMENSIONS
    vector: list[float]
    act: str
    chapter: str
    status: str
    access_level: str = ACCESS_LEVEL_PUBLIC


# ---- deterministic IDs ----


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_id_for(source_sha256: str) -> str:
    return f"src:{source_sha256[:16]}"


def section_id_for(section_id: str) -> str:
    """Section ids are reused unchanged from the JSONL (`bns:103`)."""
    return section_id


def chunk_id_for(section_id: str, chunk_index: int) -> str:
    return f"{section_id}:c{chunk_index}"


def embedding_id_for(
    chunk_id: str,
    embedding_input_text: str,
    model: str = EMBEDDING_MODEL,
    model_version: str = EMBEDDING_MODEL_VERSION,
    dimensions: int = EMBEDDING_DIMENSIONS,
) -> str:
    key = f"{chunk_id}|{model}|{model_version}|{dimensions}|{_sha(embedding_input_text)}"
    return f"emb:{_sha(key)[:32]}"


def build_embedding_input(act_label: str, section_id: str, heading: str, chunk_text: str) -> str:
    return EMBEDDING_INPUT_FORMAT.format(
        act_label=act_label, section_id=section_id, heading=heading, chunk_text=chunk_text
    )


# ---- validators (required fields and types only) ----


def _validator(required: dict[str, str]) -> dict[str, Any]:
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": list(required),
            "properties": {name: {"bsonType": t} for name, t in required.items()},
        }
    }


VALIDATORS: dict[str, dict[str, Any]] = {
    SOURCES: _validator({
        "_id": "string", "source_pdf": "string", "source_sha256": "string", "act": "string",
        "act_label": "string", "status": "string", "parser": "string", "parser_version": "string",
    }),
    SECTIONS: _validator({
        "_id": "string", "section_id": "string", "source_id": "string", "act": "string",
        "chapter": "string", "section_number": "int", "heading": "string", "text": "string",
        "source_pdf": "string", "source_sha256": "string", "needs_review": "bool",
        "access_level": "string",
    }),
    CHUNKS: _validator({
        "_id": "string", "chunk_id": "string", "section_id": "string", "chunk_index": "int",
        "chunking_version": "string", "text": "string", "embedding_input_text": "string",
        "act": "string", "chapter": "string", "status": "string", "access_level": "string",
        "splitter": "object",
    }),
    EMBEDDINGS: _validator({
        "_id": "string", "embedding_id": "string", "chunk_id": "string", "model": "string",
        "model_version": "string", "dimensions": "int", "vector": "array", "act": "string",
        "chapter": "string", "status": "string", "access_level": "string",
    }),
}

# (collection, field, unique)
ORDINARY_INDEXES: list[tuple[str, str, bool]] = [
    (SOURCES, "source_sha256", True),
    (SECTIONS, "section_id", True),
    (CHUNKS, "chunk_id", True),
    (CHUNKS, "section_id", False),
    (EMBEDDINGS, "embedding_id", True),
    (EMBEDDINGS, "chunk_id", False),
]


def ensure_collections(db: Database) -> None:
    """Create collections with validators and ordinary indexes. Safe to repeat."""
    existing = set(db.list_collection_names())
    for name, validator in VALIDATORS.items():
        if name in existing:
            db.command("collMod", name, validator=validator)
        else:
            try:
                db.create_collection(name, validator=validator)
            except CollectionInvalid:  # created concurrently
                db.command("collMod", name, validator=validator)
    for coll, field, unique in ORDINARY_INDEXES:
        db[coll].create_index([(field, 1)], unique=unique)


def vector_index_definition() -> dict[str, Any]:
    fields: list[dict[str, Any]] = [
        {
            "type": "vector",
            "path": VECTOR_FIELD,
            "numDimensions": EMBEDDING_DIMENSIONS,
            "similarity": "cosine",
        }
    ]
    fields += [{"type": "filter", "path": f} for f in VECTOR_FILTER_FIELDS]
    return {"fields": fields}


def vector_index_model() -> SearchIndexModel:
    return SearchIndexModel(
        definition=vector_index_definition(), name=VECTOR_INDEX_NAME, type="vectorSearch"
    )


def keyword_index_definition() -> dict[str, Any]:
    """Static Atlas Search mappings: analysed chunk text plus token fields for scope filters.

    `embedding_input_text` is deliberately not indexed (it carries the section id and heading
    prefix built for embedding, not the legal text) and `dynamic` stays false.
    """
    fields: dict[str, Any] = {KEYWORD_TEXT_FIELD: {"type": "string"}}
    fields.update({f: {"type": "token"} for f in KEYWORD_FILTER_FIELDS})
    return {"mappings": {"dynamic": False, "fields": fields}}


def keyword_index_model() -> SearchIndexModel:
    return SearchIndexModel(
        definition=keyword_index_definition(), name=KEYWORD_INDEX_NAME, type="search"
    )
