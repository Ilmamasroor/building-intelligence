"""Story 2.2: capability check, collections, validators, ordinary indexes and the vector index.

    uv run python -m building_with_rag.ingestion.setup_db
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from pymongo.database import Database
from pymongo.errors import OperationFailure

from building_with_rag.contracts import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from building_with_rag.ingestion.db import (
    PrerequisiteError,
    connect,
    require_voyage_key,
)
from building_with_rag.ingestion.mongodb_schema import (
    CHUNKS,
    EMBEDDINGS,
    KEYWORD_INDEX_NAME,
    SECTIONS,
    VECTOR_FIELD,
    VECTOR_INDEX_NAME,
    ensure_collections,
    keyword_index_definition,
    keyword_index_model,
    vector_index_definition,
    vector_index_model,
)
from building_with_rag.settings import get_settings

INDEX_WAIT_SECONDS = 300
INDEX_POLL_SECONDS = 10


def check_atlas_capability(db: Database) -> None:
    """Stop with a clear message unless Atlas Search index management is available."""
    if EMBEDDINGS not in db.list_collection_names():
        db.create_collection(EMBEDDINGS)
    try:
        list(db[EMBEDDINGS].list_search_indexes())
    except OperationFailure as exc:
        if exc.code == 13 or "not authorized" in str(exc).lower():
            raise PrerequisiteError(
                "user lacks the Atlas Search index privilege "
                "(needs at least readWrite plus search-index access)"
            ) from None
        raise PrerequisiteError("deployment is not Atlas (no Atlas Search index support)") from None
    print("Atlas Search index management: available")


def _covers(existing: dict[str, Any], wanted: dict[str, Any]) -> bool:
    return all(existing.get(k) == v for k, v in wanted.items())


def definition_differences(existing_def: dict[str, Any]) -> list[str]:
    """Differences between an existing index definition and ours (extra server defaults ignored)."""
    have = existing_def.get("fields", [])
    return [
        f"missing or different field: {w}"
        for w in vector_index_definition()["fields"]
        if not any(_covers(h, w) for h in have)
    ]


def ensure_vector_index(db: Database) -> str:
    """Create or reuse the vector index. Returns 'created' or 'reused'. Never drops or replaces."""
    coll = db[EMBEDDINGS]
    found = list(coll.list_search_indexes(VECTOR_INDEX_NAME))
    if found:
        idx = found[0]
        diffs = definition_differences(idx.get("latestDefinition") or idx.get("definition") or {})
        if diffs:
            raise PrerequisiteError(
                f"index {VECTOR_INDEX_NAME} exists but differs from the expected definition: "
                + "; ".join(diffs)
                + ". Not changing it; drop or fix it in Atlas yourself."
            )
        print(f"Vector index {VECTOR_INDEX_NAME}: existing index reused")
        return "reused"
    coll.create_search_index(vector_index_model())
    print(f"Vector index {VECTOR_INDEX_NAME}: created")
    return "created"


def wait_until_queryable(db: Database, timeout: int = INDEX_WAIT_SECONDS) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        found = list(db[EMBEDDINGS].list_search_indexes(VECTOR_INDEX_NAME))
        status = found[0].get("status") if found else "MISSING"
        if found and (found[0].get("queryable") or status == "READY"):
            print(f"Vector index {VECTOR_INDEX_NAME}: queryable (status {status})")
            return True
        if time.monotonic() >= deadline:
            print(
                f"Vector index {VECTOR_INDEX_NAME}: pending (status {status}) after "
                f"{timeout}s; re-run setup_db later to confirm it is queryable"
            )
            return False
        print(f"  waiting for index (status {status}) ...")
        time.sleep(INDEX_POLL_SECONDS)


def keyword_definition_differences(existing_def: dict[str, Any]) -> list[str]:
    """Differences between an existing Atlas Search definition and ours (defaults ignored)."""
    want = keyword_index_definition()["mappings"]
    have = existing_def.get("mappings", {})
    diffs = []
    if have.get("dynamic", False) != want["dynamic"]:
        diffs.append(f"mappings.dynamic should be {want['dynamic']}")
    for name, spec in want["fields"].items():
        got = (have.get("fields") or {}).get(name)
        if not isinstance(got, dict) or not _covers(got, spec):
            diffs.append(f"missing or different field {name!r}: {spec}")
    return diffs


def ensure_keyword_index(db: Database, timeout: int = INDEX_WAIT_SECONDS) -> str:
    """Create or reuse the Atlas Search keyword index on chunks. Never drops or replaces.

    Returns 'ready' or 'pending'. Raises PrerequisiteError if an index of that name differs.
    """
    coll = db[CHUNKS]
    found = list(coll.list_search_indexes(KEYWORD_INDEX_NAME))
    if found:
        idx = found[0]
        diffs = keyword_definition_differences(
            idx.get("latestDefinition") or idx.get("definition") or {}
        )
        if diffs:
            raise PrerequisiteError(
                f"index {KEYWORD_INDEX_NAME} exists but differs from the expected definition: "
                + "; ".join(diffs)
                + ". Not changing it; drop or fix it in Atlas yourself."
            )
        action = "existing index reused"
    else:
        coll.create_search_index(keyword_index_model())
        action = "created"
    deadline = time.monotonic() + timeout
    while True:
        found = list(coll.list_search_indexes(KEYWORD_INDEX_NAME))
        status = found[0].get("status") if found else "MISSING"
        if found and (found[0].get("queryable") or status == "READY"):
            print(f"Keyword search index {KEYWORD_INDEX_NAME}: ready ({action})")
            return "ready"
        if time.monotonic() >= deadline:
            print(
                f"Keyword search index {KEYWORD_INDEX_NAME}: pending (status {status}) after "
                f"{timeout}s; re-run setup_db later to confirm it is queryable"
            )
            return "pending"
        print(f"  waiting for keyword index (status {status}) ...")
        time.sleep(INDEX_POLL_SECONDS)


def verify_sample(db: Database, query: str = "punishment for murder") -> None:
    """One real $vectorSearch, resolved through chunks and sections. Prints, never raises."""
    try:
        if db[EMBEDDINGS].estimated_document_count() == 0:
            print("Sample query: verification pending (no embeddings stored yet)")
            return
        import voyageai

        vector = voyageai.Client(api_key=require_voyage_key(), max_retries=0).embed(
            [query],
            model=EMBEDDING_MODEL,
            input_type="query",
            output_dimension=EMBEDDING_DIMENSIONS,
        ).embeddings[0]
        pipeline = [
            {"$vectorSearch": {
                "index": VECTOR_INDEX_NAME, "path": VECTOR_FIELD, "queryVector": vector,
                "numCandidates": 50, "limit": 3,
            }},
            {"$project": {"chunk_id": 1, "score": {"$meta": "vectorSearchScore"}}},
        ]
        hits = list(db[EMBEDDINGS].aggregate(pipeline))
        if not hits:
            print("Sample query: verification pending (no results; index may still be syncing)")
            return
        top = hits[0]
        chunk = db[CHUNKS].find_one({"chunk_id": top["chunk_id"]})
        section = db[SECTIONS].find_one({"section_id": chunk["section_id"]}) if chunk else None
        if not chunk or not section:
            print(f"Sample query: ERROR chunk_id {top['chunk_id']} does not resolve to a section")
            return
        print(
            f"Sample query {query!r}: chunk_id={chunk['chunk_id']} "
            f"section_id={section['section_id']} heading={section['heading']!r} "
            f"score={top['score']:.4f}"
        )
    except PrerequisiteError as exc:
        print(f"Sample query: verification pending ({exc})")
    except Exception as exc:  # services or index not ready; report the type only
        print(f"Sample query: verification pending ({type(exc).__name__}: {str(exc)[:200]})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-sample", action="store_true", help="skip the sample $vectorSearch")
    args = ap.parse_args()
    try:
        client, db = connect()
    except PrerequisiteError as exc:
        print(f"setup_db: {exc}", file=sys.stderr)
        return 1
    try:
        print(f"Connected; database: {get_settings().mongodb_db_name}")
        check_atlas_capability(db)
        ensure_collections(db)
        print("Collections, validators and ordinary indexes: ready")
        ensure_vector_index(db)
        vector_ready = wait_until_queryable(db)
        # The keyword step is reported on its own and never alters the vector index.
        try:
            ensure_keyword_index(db)
        except PrerequisiteError as exc:
            print(f"Keyword search index: not ready: {exc}", file=sys.stderr)
        except Exception as exc:  # e.g. M0 search-index limit or privilege; report the type
            print(f"Keyword search index: failed ({type(exc).__name__}: {str(exc)[:200]})")
        if vector_ready and not args.no_sample:
            verify_sample(db)
    except PrerequisiteError as exc:
        print(f"setup_db: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
