# Story 2.2 — MongoDB collections, embeddings and vector index

Status: code complete; learner runs and checks pending

## Purpose

Take the section-level corpus from Story 2.1 and make it searchable evidence in one MongoDB Atlas database: four collections (`sources`, `sections`, `chunks`, `embeddings`), Voyage embeddings, and one Atlas `vectorSearch` index on the embeddings. This is preparation only. Later stories query the index; this story does not.

**Who runs what.** The implementer (the coding assistant) only writes the code and documentation. The assistant must **not** run `setup_db` or `load_and_embed`, connect to MongoDB, call Voyage, or read `.env`. It may run `uv run ruff check .` and `uv run pytest -q`. The learner runs the two commands with their own credentials, does the completion checks, and records the results in the handover.

**Out of scope (do not build):** retrieval endpoints or any change to `/v1/query`, `/v1/chat/completions` or the mode registry; answer generation; hybrid or keyword search; re-ranking; a separate vector store; broad validation, access control or deployment work.

## Prerequisites

- Stories 1.1 and 2.1 are complete. Reuse the existing app in `src/building_with_rag/`, `docs/architecture.md`, and the corpus `data/processed/bns_sections.jsonl` and `data/processed/ipc_sections.jsonl`. Keep every existing route, contract and JSONL field as it is. Extend additively only.
- No `docs/config.yaml` existed when this story was written; use the repository root. If one now names a project path, use that path.
- Settings come from the existing `.env.example` contract, exactly as named: `MONGODB_URI`, `MONGODB_DB_NAME`, `MONGODB_TEST_DB_NAME`, `VOYAGE_API_KEY`. Load them through the existing `Settings` in `settings.py` from an untracked `.env`. Do not rename, remove or add variables, and never print, log or commit a credential or the full URI.
- A MongoDB Atlas free (M0) cluster with the participant's IP in the Atlas access list. Assume Atlas Search index management and `$vectorSearch` are available, but verify (Work item 2).
- A Voyage API key (rate limited; see item 5).
- Fixed embedding choice from `docs/architecture.md` rule 6: model `voyage-3.5`, version `voyage-3.5`, 1,024 dimensions. The constants already exist in `contracts.py` (`EMBEDDING_MODEL`, `EMBEDDING_MODEL_VERSION`, `EMBEDDING_DIMENSIONS`); import them, do not redefine or guess.
- Before starting, read `docs/architecture.md`, Stories 1.1 and 2.1, the existing code, and the generated corpus. **Check the corpus first:** when this story was written, the files held 115 BNS and only 7 IPC records, with the largest sections about 240,000 characters (a sign that section splitting under-split the PDFs). If the learner's corpus is like that, stop and report it so Story 2.1 can be fixed. Do not paper over it with chunking.
- If `MONGODB_URI` or `VOYAGE_API_KEY` is empty, or MongoDB is unreachable, stop with a clear prerequisite message (which setting or step, no secret values) and do nothing else.

## Work to do

### 1. Dependencies

Add only what is needed, with `uv add`: the official `voyageai` client, and a text splitter only if you choose one (a small hand-written splitter on paragraph/sentence boundaries is acceptable and needs no dependency). `pymongo` is already present. Do not add another database, vector store or ORM.

### 2. Connection and capability check

Add a small command (`uv run python -m building_with_rag.ingestion.setup_db`, see item 6) whose first step identifies the deployment and records the result:

- Connect with `MONGODB_URI`, ping, and report the database name used (`MONGODB_DB_NAME`; use `MONGODB_TEST_DB_NAME` only for tests).
- Confirm the deployment is Atlas and supports Atlas Search index management: call `list_search_indexes()` on the `embeddings` collection (create the collection first if absent). A non-Atlas server or a missing privilege raises an error; catch it and stop with a clear message ("deployment is not Atlas" or "user lacks the Atlas Search index privilege (needs at least readWrite plus search-index access)"). Do not fall back to a generic index, `knnVector`, or any other mechanism.
- Print the capability in one line, for example `Atlas Search index management: available`. Record it in `docs/architecture.md`.

### 3. Shared schema module: `src/building_with_rag/ingestion/mongodb_schema.py`

Create the package `src/building_with_rag/ingestion/` (with `__init__.py`). Leave `ingest.py` and its command untouched. All other code in this story imports from `mongodb_schema.py`; nothing hard-codes a collection name, field name or ID format elsewhere. It holds:

- **Collection constants:** `SOURCES = "sources"`, `SECTIONS = "sections"`, `CHUNKS = "chunks"`, `EMBEDDINGS = "embeddings"`, plus the vector index name `VECTOR_INDEX_NAME = "embeddings_vector_index"`.
- **Project-local validators** (JSON Schema `$jsonSchema`, applied with `collMod`/`create_collection(validator=...)`, plus matching Pydantic models for the shared document model). Keep them small: required fields and types only. Existing `PassageDocument` in `contracts.py` stays as it is (it is not used here, and must not be removed).
- **Deterministic IDs**, pure functions of their inputs: source id from the PDF SHA-256 (or `source_pdf` plus hash); section id reuses the JSONL `section_id` unchanged (`bns:103`; the display form `BNS-103` is a formatting concern for later stories, not a stored rename); chunk id `f"{section_id}:c{chunk_index}"` together with `chunking_version`; embedding id from `chunk_id`, model, version and dimensions plus a hash of the embedding input text.
- **Ordinary indexes** (unique `_id` is implicit): `sections.section_id` unique; `chunks.chunk_id` unique, `chunks.section_id`; `embeddings.embedding_id` unique, `embeddings.chunk_id`; `sources.source_sha256` unique. Created idempotently by a `ensure_collections()` helper.

### 4. Four collections, not merged

Load from the existing JSONL files; never rewrite them or `data/raw/`. The acts stay separate (`bns:103` and `ipc:103` are different sections).

| Collection | Holds |
|---|---|
| `sources` | source PDF file name, `source_sha256`, `act`, `act_label`, `status`, parser name and version, `source_status_version` |
| `sections` | `section_id` (act-qualified), `source_id` link, `act`, `chapter`, `section_number`, `heading`, `text`, provenance (`source_pdf`, `source_sha256`, pages, `needs_review`), `access_level="public"` |
| `chunks` | `chunk_id`, parent `section_id`, `chunk_index`, `chunking_version`, `text`, `embedding_input_text`, denormalized `act`, `chapter`, `status`, `access_level`, and `splitter` |
| `embeddings` | `embedding_id`, `chunk_id`, `model`, `model_version`, `dimensions`, `vector`, plus denormalized `act`, `chapter`, `status`, `access_level` for vector filtering |

Rules:
- Vectors live only in `embeddings.vector`, never on chunks.
- Every document carries `access_level="public"` where listed above. The field is kept so the document shape stays compatible with later access control; do not build any.
- Section number and source identifiers are **not** copied onto embeddings. They stay on chunks and sections and are applied after `chunk_id` is resolved.
- Intended later path: a vector query on `embeddings`, then resolve each returned `chunk_id` in `chunks` and its `section_id` in `sections`, then build a `RetrievedChunk` (`chunk_id`, `section_id`, `act`, `text`, `heading`, `score`, `source`). Do not implement that path here, only keep the data able to support it.
- `embedding_input_text` is the chunk text prefixed with the act label, section id and heading, so the vector carries its context. Record the exact format with the `chunking_version`.

### 5. Chunking and embedding

- **Chunking:** split each section's text on its own, never across sections or acts. Target about 512 tokens (about 2,048 characters) with about 256 characters of overlap, preferring paragraph, then sentence, then word boundaries. Short sections become one chunk. Record the splitter name and parameters in each chunk (`splitter`) and the version in `chunking_version` (for example `chars-2048-256-v1`).
- **Embedding:** Voyage `voyage-3.5`, 1,024 dimensions, `input_type="document"`, `VOYAGE_API_KEY` only. Embed in small batches (for example 8 chunks), with a fixed delay between batches (for example 20 seconds; the free key is rate limited) and retry with growing back-off on rate-limit errors. Print progress (`batch n/N, embedded x/y, skipped z`) without printing text or keys.
- **Resumable and idempotent:** before embedding, skip any chunk whose deterministic embedding id already exists. Re-running upserts by deterministic id, so unchanged data creates no duplicate chunks or embeddings and calls Voyage zero times.
- **Changes are visible, not silent:** if a section's text, the `chunking_version`, or the model settings change, the new chunk or embedding ids differ. The run reports counts of new, unchanged and stale records (rows whose ids no longer match the current corpus or settings) and **lists but does not delete** stale rows unless the learner passes an explicit `--prune` flag. Mixed versions must never be queried silently: the report states how many stale rows remain.
- **Expected duration:** state it up front in the script output and here: about 40 minutes for a first full embedding run on a free Voyage key (as in Story 1.1; the script prints its own estimate from the chunk count and delay). A repeat run takes seconds.

### 6. Vector index and commands

Commands (from the repository root):

- `uv run python -m building_with_rag.ingestion.setup_db` — capability check, collections, validators, ordinary indexes, vector index.
- `uv run python -m building_with_rag.ingestion.load_and_embed` — load sources/sections, chunk, embed (resumable). Flags: `--act bns|ipc`, `--prune`.

Vector index, created through PyMongo's documented search-index API (`create_search_index` with `SearchIndexModel(..., type="vectorSearch")`):

- Name: `embeddings_vector_index`, on collection `embeddings`.
- `definition` is `{"fields": [...]}` with one `vector` entry (`path="vector"`, `numDimensions=1024` taken from `EMBEDDING_DIMENSIONS`, `similarity="cosine"`) plus `filter` entries for `act`, `chapter`, `status`, `access_level`. Copy field names from `mongodb_schema.py`.
- Do not use the legacy `knnVector` or `mappings` syntax; that belongs to `search`-type indexes.
- Repeat-safe: if the index exists and its definition matches, reuse it; if it exists but differs, report the difference and stop. Never drop or replace it automatically.
- Wait a bounded time (for example up to 5 minutes) for the index to become queryable (`queryable` true / status `READY`). On timeout, report "pending", not success.
- When the index is ready and embeddings exist, run one real `$vectorSearch` sample (embed a short query such as "punishment for murder" with `input_type="query"`, `numCandidates` about 50, `limit` 3, optionally filtered on `act`), resolve the top result's `chunk_id` through `chunks` and `sections`, and print `chunk_id`, `section_id`, heading and score. If services or credentials are unavailable, print "verification pending" and why.

### 7. Update `docs/architecture.md` (do not overwrite)

Add a short section recording: the four collections and their roles; the chunking choice and `chunking_version`; the embedding decision (unchanged: `voyage-3.5`, version `voyage-3.5`, 1,024); the index name and definition summary; the two commands; the recorded Atlas capability; and limitations (M0 limits, rate-limited embedding time, stale rows only reported unless `--prune`, corpus quality inherited from Story 2.1, the older `PassageDocument` contract left in place and superseded by these collections for storage).

## Completion checks

Keep it classroom-sized. The learner runs these and records command and result. Do not build a large suite.

1. `setup_db` prints the Atlas capability line, and stops with a clear message when `MONGODB_URI` is blank or the deployment lacks the capability (try once with the URI blank).
2. After `load_and_embed`: one section's stored link is correct (`sections` record, its `chunks` by `section_id`, each `embeddings` record by `chunk_id`).
3. One stored embedding has `len(vector) == 1024`, with model and version matching `docs/architecture.md`.
4. Repeat both commands: no new documents, no Voyage calls, counts unchanged, the existing index reported as reused.
5. The vector index `embeddings_vector_index` is `queryable` (or the timeout is reported as pending).
6. The sample query prints the top result's `chunk_id`, `section_id`, heading and score, or "verification pending" with the reason.
7. Existing checks still pass: `uv run ruff check .`, `uv run pytest -q`. `data/raw/` and the two JSONL files are unchanged; `.env` is untracked and no credential appears in output.

## Handover

When done, update this story with: files created or changed (including `pyproject.toml`, `uv.lock`, `docs/architecture.md`); the Atlas capability line; collection and chunk/embedding counts; actual run duration; commands run and results (including the repeat run and the sample query result); any stale-row report; deviations with reasons.

Later stories (semantic retrieval onward) query `embeddings` through `embeddings_vector_index`, apply `act`/`chapter`/`status`/`access_level` as vector filters, then resolve `chunk_id` to `chunks` and `sections` to build each `RetrievedChunk`, using the constants and IDs from `mongodb_schema.py`.

## Implementation handover (assistant)

**Prerequisite fix (Story 2.1).** The corpus held 115 BNS and 7 IPC records, with `bns:116` and `ipc:9` about 240,000 characters each. Two bugs in `ingest.py`: the BNS section-start regex needed a space after the dot (`115.(1)` was missed), and the IPC footnote cleanup dropped all text after the `-----` separator, losing sections 10-14. Fixed both, widened the IPC section-number window to 8 (repealed sections have no body), and regenerated with `--force`. Result: 358 BNS records (none missing, largest 12,664 characters) and 480 IPC records up to `ipc:511` (largest 8,621; 65 numbers have no record, mostly repealed/omitted, printed by the command). 11 BNS and 90 IPC records are flagged `needs_review`. The earlier JSONL was not kept in the repo.

**Files created:** `src/building_with_rag/ingestion/{__init__,mongodb_schema,chunking,db,setup_db,load_and_embed}.py`, `tests/test_ingestion.py`.
**Changed:** `src/building_with_rag/ingest.py` (fixes above), `data/processed/*.jsonl` (regenerated), `pyproject.toml` and `uv.lock` (`voyageai` added), `docs/architecture.md` (new section), this story.

**Expected counts** (from the code, offline): 358 + 480 sections; 411 + 516 = 927 chunks and embeddings; 2 sources. First embedding run: 927 / 8 = 116 batches x 20 s, about 40 minutes.

**Checked by the assistant:** `uv run ruff check .` passes; `uv run pytest -q` passes (11 tests). The assistant did not connect to MongoDB or Voyage. By mistake `setup_db` was run once and stopped at the empty `MONGODB_URI` check, so no connection was made.

**Deviations:**
- `chunk_id` is exactly `<section_id>:c<n>` and `_id` equals it; `chunking_version` is a field, not part of the id. A new version overwrites chunks in place and the old embeddings become stale (their ids include the input text hash). Stale rows are reported either way.
- `setup_db` runs the sample `$vectorSearch` itself (one Voyage query call; `--no-sample` skips it), so check 4's "no Voyage calls" refers to document embedding.

**Still for the learner:** run `setup_db`, `load_and_embed`, the repeat runs and completion checks 1-6; record the Atlas capability line, counts, duration, sample result and any stale report here and in `docs/architecture.md`.
