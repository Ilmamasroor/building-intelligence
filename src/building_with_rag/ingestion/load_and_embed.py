"""Story 2.2: load sources and sections, chunk, embed with Voyage, store. Resumable and idempotent.

    uv run python -m building_with_rag.ingestion.load_and_embed [--act bns|ipc] [--prune]

A first full run takes about 40 minutes on a free Voyage key (rate limited); the script prints
its own estimate. A repeat run on unchanged data takes seconds and makes no Voyage calls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from pymongo import ReplaceOne
from pymongo.database import Database

from building_with_rag.contracts import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_VERSION,
)
from building_with_rag.ingestion.chunking import split_text
from building_with_rag.ingestion.db import PrerequisiteError, connect, require_voyage_key
from building_with_rag.ingestion.mongodb_schema import (
    ACCESS_LEVEL_PUBLIC,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKING_VERSION,
    CHUNKS,
    EMBEDDINGS,
    SECTIONS,
    SOURCES,
    SPLITTER,
    build_embedding_input,
    chunk_id_for,
    embedding_id_for,
    ensure_collections,
    section_id_for,
    source_id_for,
)

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"
ACT_FILES = {"bns": "bns_sections.jsonl", "ipc": "ipc_sections.jsonl"}

BATCH_SIZE = 8
BATCH_DELAY_SECONDS = 20
MAX_RETRIES = 6
BACKOFF_START_SECONDS = 30


# ---- build documents from the JSONL ----


def read_records(key: str) -> list[dict[str, Any]]:
    path = PROCESSED / ACT_FILES[key]
    if not path.exists():
        raise PrerequisiteError(
            f"{path.relative_to(ROOT)} is missing; run `uv run python -m building_with_rag.ingest`"
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_documents(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return source, section and chunk documents (with embedding ids) for one act."""
    first = records[0]
    source_id = source_id_for(first["source_sha256"])
    sources = [{
        "_id": source_id,
        "source_id": source_id,
        "source_pdf": first["source_pdf"],
        "source_sha256": first["source_sha256"],
        "act": first["act"],
        "act_label": first["act_label"],
        "status": first["status"],
        "parser": first["parser"],
        "parser_version": first["parser_version"],
        "source_status_version": first.get("source_status_version"),
    }]
    sections: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    splitter = {"name": SPLITTER, "chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP}
    for r in records:
        sid = section_id_for(r["section_id"])
        sections.append({
            "_id": sid,
            "section_id": sid,
            "source_id": source_id,
            "act": r["act"],
            "chapter": r["chapter"],
            "section_number": r["section_number"],
            "heading": r["heading"],
            "text": r["text"],
            "source_pdf": r["source_pdf"],
            "source_sha256": r["source_sha256"],
            "page_start": r.get("page_start"),
            "page_end": r.get("page_end"),
            "needs_review": r["needs_review"],
            "access_level": ACCESS_LEVEL_PUBLIC,
        })
        for i, text in enumerate(split_text(r["text"])):
            cid = chunk_id_for(sid, i)
            input_text = build_embedding_input(r["act_label"], sid, r["heading"], text)
            chunks.append({
                "_id": cid,
                "chunk_id": cid,
                "section_id": sid,
                "chunk_index": i,
                "chunking_version": CHUNKING_VERSION,
                "text": text,
                "embedding_input_text": input_text,
                "act": r["act"],
                "chapter": r["chapter"],
                "status": r["status"],
                "access_level": ACCESS_LEVEL_PUBLIC,
                "splitter": splitter,
                "_embedding_id": embedding_id_for(cid, input_text),
            })
    return {"sources": sources, "sections": sections, "chunks": chunks}


# ---- upserts and stale reporting ----


def upsert(db: Database, collection: str, docs: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Replace-upsert by `_id`. Returns (new, changed, unchanged)."""
    if not docs:
        return 0, 0, 0
    ops = [ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in docs]
    res = db[collection].bulk_write(ops, ordered=False)
    new, changed = res.upserted_count, res.modified_count
    return new, changed, len(docs) - new - changed


def stale_ids(db: Database, collection: str, acts: list[str], keep: set[str]) -> list[str]:
    cursor = db[collection].find({"act": {"$in": acts}}, {"_id": 1})
    return [d["_id"] for d in cursor if d["_id"] not in keep]


# ---- embedding ----


def embed_batch(client: Any, texts: list[str]) -> list[list[float]]:
    import voyageai.error as verr

    delay = BACKOFF_START_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.embed(
                texts,
                model=EMBEDDING_MODEL,
                input_type="document",
                output_dimension=EMBEDDING_DIMENSIONS,
            ).embeddings
        except verr.RateLimitError:
            if attempt == MAX_RETRIES:
                raise
            print(f"  rate limited; retry {attempt}/{MAX_RETRIES - 1} in {delay}s")
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


def embed_missing(
    db: Database, chunks: list[dict[str, Any]], existing: set[str]
) -> tuple[int, int]:
    todo = [c for c in chunks if c["_embedding_id"] not in existing]
    skipped = len(chunks) - len(todo)
    if not todo:
        print(f"Embeddings: nothing to embed ({skipped} already stored); no Voyage calls")
        return 0, skipped
    import voyageai

    client = voyageai.Client(api_key=require_voyage_key(), max_retries=0)
    batches = math.ceil(len(todo) / BATCH_SIZE)
    est_min = round(batches * BATCH_DELAY_SECONDS / 60)
    print(
        f"Embedding {len(todo)} chunks in {batches} batches of {BATCH_SIZE}; "
        f"expected duration about {est_min} minutes (resumable: re-run to continue)"
    )
    done = 0
    for n in range(batches):
        batch = todo[n * BATCH_SIZE:(n + 1) * BATCH_SIZE]
        vectors = embed_batch(client, [c["embedding_input_text"] for c in batch])
        docs = [{
            "_id": c["_embedding_id"],
            "embedding_id": c["_embedding_id"],
            "chunk_id": c["chunk_id"],
            "model": EMBEDDING_MODEL,
            "model_version": EMBEDDING_MODEL_VERSION,
            "dimensions": EMBEDDING_DIMENSIONS,
            "vector": v,
            "act": c["act"],
            "chapter": c["chapter"],
            "status": c["status"],
            "access_level": c["access_level"],
        } for c, v in zip(batch, vectors, strict=True)]
        upsert(db, EMBEDDINGS, docs)
        done += len(batch)
        print(f"batch {n + 1}/{batches}, embedded {done}/{len(todo)}, skipped {skipped}")
        if n + 1 < batches:
            time.sleep(BATCH_DELAY_SECONDS)
    return done, skipped


# ---- command ----


def run(db: Database, keys: list[str], prune: bool) -> None:
    ensure_collections(db)
    all_docs: dict[str, list[dict[str, Any]]] = {"sources": [], "sections": [], "chunks": []}
    acts: list[str] = []
    for key in keys:
        records = read_records(key)
        docs = build_documents(records)
        for name in all_docs:
            all_docs[name] += docs[name]
        acts.append(records[0]["act"])
    chunks = all_docs["chunks"]
    print(f"Corpus: {len(all_docs['sections'])} sections -> {len(chunks)} chunks "
          f"(chunking_version {CHUNKING_VERSION})")

    for coll, name in ((SOURCES, "sources"), (SECTIONS, "sections")):
        new, changed, same = upsert(db, coll, all_docs[name])
        print(f"{coll}: {new} new, {changed} changed, {same} unchanged")
    stored_chunks = [{k: v for k, v in c.items() if k != "_embedding_id"} for c in chunks]
    new, changed, same = upsert(db, CHUNKS, stored_chunks)
    print(f"{CHUNKS}: {new} new, {changed} changed, {same} unchanged")

    expected_emb = {c["_embedding_id"] for c in chunks}
    existing = {d["_id"] for d in db[EMBEDDINGS].find({"act": {"$in": acts}}, {"_id": 1})}
    embedded, skipped = embed_missing(db, chunks, existing & expected_emb)

    stale = {
        SOURCES: stale_ids(db, SOURCES, acts, {d["_id"] for d in all_docs["sources"]}),
        SECTIONS: stale_ids(db, SECTIONS, acts, {d["_id"] for d in all_docs["sections"]}),
        CHUNKS: stale_ids(db, CHUNKS, acts, {c["_id"] for c in chunks}),
        EMBEDDINGS: stale_ids(db, EMBEDDINGS, acts, expected_emb),
    }
    total_stale = sum(len(v) for v in stale.values())
    print(f"Embeddings: {embedded} new, {skipped} unchanged")
    print(f"Stale rows: {total_stale} "
          + ", ".join(f"{k}={len(v)}" for k, v in stale.items()))
    for coll, ids in stale.items():
        if ids:
            print(f"  stale {coll}: {ids[:10]}{' ...' if len(ids) > 10 else ''}")
    if total_stale and prune:
        for coll, ids in stale.items():
            if ids:
                db[coll].delete_many({"_id": {"$in": ids}})
        print(f"Pruned {total_stale} stale rows")
    elif total_stale:
        print("Stale rows were NOT deleted; they must not be queried as current. "
              "Re-run with --prune to remove them.")
    for coll in (SOURCES, SECTIONS, CHUNKS, EMBEDDINGS):
        print(f"count {coll}: {db[coll].count_documents({})}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--act", choices=["bns", "ipc"], help="load one act only")
    ap.add_argument("--prune", action="store_true", help="delete stale rows (default: only list)")
    args = ap.parse_args()
    keys = [args.act] if args.act else list(ACT_FILES)
    started = time.monotonic()
    try:
        require_voyage_key()
        client, db = connect()
    except PrerequisiteError as exc:
        print(f"load_and_embed: {exc}", file=sys.stderr)
        return 1
    try:
        run(db, keys, args.prune)
    except PrerequisiteError as exc:
        print(f"load_and_embed: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()
    print(f"Done in {time.monotonic() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
