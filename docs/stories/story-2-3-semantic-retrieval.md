# Story 2.3 — Semantic retrieval

Status: implemented

## Purpose

Build the first real course mode: `semantic`. A question is embedded, MongoDB vector search finds the closest stored chunks, and `POST /v1/query` returns them as source passages in `QueryResult.results`, with enough diagnostic detail to inspect each one.

This story returns passages only. LLM answer generation, citations, confidence and the Open WebUI answer experience belong to later stories. A similarity score is a ranking signal, not proof that a passage is correct or answers the question; nothing in this story may present it as one.

## Prerequisites

- Stories 1.1, 2.1 and 2.2 are complete: the corpus, `chunks`, `embeddings` and the ready `embeddings_vector_index` exist in MongoDB.
- Read first: `docs/architecture.md`, Story 1.1, Stories 2.1 and 2.2, `contracts.py`, `registry.py`, `pipeline.py`, `main.py`, `settings.py`, `ingestion/mongodb_schema.py`, `ingestion/db.py`, and the sample query in `ingestion/setup_db.py` (`verify_sample`). Preserve existing work and reuse its constants and helpers.
- No `docs/config.yaml` existed when this story was written, so use the repository root. If one now names a project path, use that path.
- `.env` holds `MONGODB_URI`, `MONGODB_DB_NAME` and `VOYAGE_API_KEY`, named exactly as in `.env.example`. Do not add or rename settings. Retrieval needs no generation, rerank or `CAPSTONE_API_KEY` values.
- If a prerequisite is missing, stop and report which one. Do not re-run ingestion or create indexes from this story.

## Contracts to preserve

Use these as they are and extend them additively only:

- `QueryRequest`, `QueryResult`, `RetrievedChunk`, `SemanticFilters` in `contracts.py`; the `/v1/query` route; the pattern registry (`MODES`, `rag-semantic`); `run_pattern` as the only path.
- MongoDB names and IDs from `mongodb_schema.py` (`EMBEDDINGS`, `CHUNKS`, `SECTIONS`, `SOURCES`, `VECTOR_INDEX_NAME`, `VECTOR_FIELD`, `VECTOR_FILTER_FIELDS`, `ACCESS_LEVEL_PUBLIC`). Do not hard-code collection or field names elsewhere.
- Embedding choice from `contracts.py`: `EMBEDDING_MODEL`, `EMBEDDING_MODEL_VERSION`, `EMBEDDING_DIMENSIONS` (`voyage-3.5`, 1,024).
- `.env.example` names, including `MONGODB_URI`, `MONGODB_DB_NAME`, `VOYAGE_API_KEY`, `WEBUI_DEMO_CALLER_ID`.

Section IDs are stored act-qualified in lower case (`bns:103`, `ipc:302`). Return the stored `section_id` unchanged; do not reformat it to `BNS-103` or create a second ID.

## Work to do

### 1. Retrieval module

Add a small module inside the package (for example `src/building_with_rag/retrieval/semantic.py`) and make `run_pattern` call it for `pattern == "semantic"`. Other patterns keep their `not_implemented` placeholder. The route, request and result models do not change shape.

### 2. Settings for retrieval (no new `.env` names)

Add one code-level constant for the default and maximum bounds. Use `QueryRequest.limit` (default 5, range 1–20) as the top-k the caller can ask for. Set `numCandidates` as a bounded multiple of the limit (for example `max(50, limit * 10)`, capped at the Atlas maximum of 1,000), kept as a named constant, not an environment variable.

### 3. Readiness check before retrieving

Before the first query, and cheaply on each request (counts only, no scans of large data), verify:

- `sections`, `chunks` and `embeddings` are non-empty;
- the vector index `embeddings_vector_index` exists and is queryable;
- embeddings carry the approved model, model version and 1,024 dimensions.

If any check fails, return a service/configuration failure (see "Outcomes") that names the failed check. Never fall back to keyword search or another model.

### 4. Query embedding

Embed the question with Voyage using the same `EMBEDDING_MODEL` and `EMBEDDING_DIMENSIONS` as the documents, with `input_type="query"` (documents used `"document"`). Use `VOYAGE_API_KEY` only. Check that the returned vector has length 1,024 before searching. Never log or return the vector or the key.

### 5. Vector search

Query `embeddings` with `$vectorSearch` on `embeddings_vector_index` and `path` `vector`, with the bounded `numCandidates` and `limit`, and project the score with `{"$meta": "vectorSearchScore"}`. Resolve each hit's `chunk_id` through `chunks` and `sections` (and `sources` where source details are needed) with batched lookups, not one query per hit. Drop and report (in `trace`) any hit that does not resolve; do not invent text or fields for it.

### 6. Filters (applied in the database)

`SemanticFilters` stays as is: `act`, `status` and `access_level`, each an optional list.

- A supplied list narrows the effective scope and is turned into a `$vectorSearch` `filter` using only the indexed filter fields (`$in` per field, combined with `$and`). The filter is applied inside `$vectorSearch`, never by filtering the results in Python afterwards.
- This course uses one fixed local demo caller (`WEBUI_DEMO_CALLER_ID`, default `demo-public`) with the public access level only. The effective `access_level` is always the intersection of the caller's `["public"]` and any requested list. A request for another level narrows to nothing and returns the no-result outcome; it never widens access. Do not build users, roles or authorization.
- Reject with a clear 422 (not a silent ignore): unknown filter fields, operators such as `$in`, `$ne` or `$or`, and any caller-supplied MongoDB fragment or non-list value. The contract's own list fields `act`, `status` and `access_level` are valid and must not be rejected. Make `SemanticFilters` forbid extra fields, and check list items are plain strings; if that changes the model, it is an additive tightening only.
- Validate `question` (after trimming, 1–4,000 characters, not blank) and `limit` (1–20). Ignore `required_acts` and `chapter` for this mode; report in `trace` that they were not applied.
- Do not invent source fields. If a field such as `chapter_title`, page numbers or `source_status_version` is not stored, leave it out or `null`.

### 7. Result shape

Return a normal `QueryResult` with `pattern="semantic"`, the outcome in `status`, a short `message`, `trace`, and `results` as `RetrievedChunk` items ordered by score, highest first, at most `limit` long. `generation` stays `null` and `generate_answer` has no effect in this story.

Each `RetrievedChunk` carries:

- `chunk_id`, `section_id` (stored, act-qualified), `act`, `text` (the matching chunk text), `heading` (the section title where available), `score` (the vector search score), and `source` with the available source details (`source_pdf`, `source_sha256`, `page_start`, `page_end`, `chapter`, `status`, `source_status_version`, whichever exist).
- `semantic_score` and `semantic_rank` (1-based), filled for this mode.

`trace` is a list of short, safe entries: selected mode (`semantic`), the query text, embedding model and dimensions, effective filters, `limit` and `numCandidates`, number of hits returned and number dropped. It never contains vectors, keys, the connection string or raw documents.

### 8. Outcomes (no result is not a failure)

Use these distinct `status` values:

| status | when | HTTP |
|---|---|---|
| `ok` | one or more passages returned | 200 |
| `no_results` | the search ran fine and nothing matched (including filters that match nothing); `results` is empty and `message` says so | 200 |
| `unavailable` | MongoDB, Voyage, the index, the corpus or a required setting is missing or failing; `message` names the category (for example "MONGODB_URI is empty", "vector index not ready", "embedding service unavailable") without secrets or stack traces | 503 |

A not-ready index or empty collection is `unavailable`, never `no_results`. Invalid input stays a 422 from request validation. Passage text is quoted data and never changes behaviour.

### 9. Models and Open WebUI

`semantic` is already listed by `/v1/models` as `rag-semantic`; keep it listed through the same registry, with no second registry or model name. `/v1/chat/completions` keeps its existing behaviour and placeholder text until answer generation; it must still map to the same `QueryRequest` and `run_pattern` path. Do not render retrieved passages as a chat answer and do not add a custom Open WebUI display.

### 10. Diagnostic command and architecture update

Add a small command that calls the same `run_pattern` path and prints a compact diagnostic, for example:

`uv run python -m building_with_rag.retrieval.semantic "What is the punishment for murder?" --limit 3 --act BNS_2023`

It prints status, selected mode, query, and per result: rank, score, `chunk_id`, `section_id`, heading, act and text length. It prints no vectors and no full passage text.

Update (do not overwrite) `docs/architecture.md` with a short section: the retrieval flow (validate → readiness check → embed query → `$vectorSearch` with filters → resolve chunks, sections, sources → `RetrievedChunk`), the three outcomes, the filter rules and fixed demo caller, the stored act-qualified ID form, the diagnostic command, and limitations (similarity is not correctness; no answer generation, hybrid or re-ranking yet).

## Token-safe rules

- Every verification or data check below is **one compact script** (a short file in the scratchpad or the diagnostic command above) that prints only counts, status values, field names and values, and vector lengths.
- Never print vectors, embeddings, full passage text, or raw documents; not from scripts, tests, logs, exceptions or the API trace. Where text is needed, print its length or, for the single inspection in check 4, a short excerpt of at most 200 characters.
- Do not paste large outputs back into the conversation or the handover.

## Out of scope

Hybrid or keyword search, re-ranking, query rewriting, HyDE, decomposition, LLM answer generation, citations, confidence, an evaluation harness, broad or load testing, multi-user authorization, new indexes or re-embedding, and changing Open WebUI.

## Completion checks

Few and direct. Run each, and record the actual command and a compact result.

1. **Readiness and a natural-language question.** One compact script prints: counts of `sections`, `chunks` and `embeddings`; index name and queryable flag; the embedding model, version and vector length of one stored embedding (length only). Then call `POST /v1/query` with `pattern: "semantic"` and one real question about BNS or IPC (for example "What is the punishment for murder?"). Expect `status: "ok"` and a non-empty `results` list with the fields in section 7 present.
2. **No-result question.** Send a question that should match nothing, for example with `filters` of `{"act": ["IPC_1860"], "status": ["in_force"]}` (the IPC records are `repealed`). Expect `status: "no_results"`, an empty `results` list, HTTP 200, and a `message` that differs from the `unavailable` message. Then clear `MONGODB_URI` for one run and confirm `status: "unavailable"` with HTTP 503 and no secret in the output; restore it afterwards.
3. **Ordering and limit.** With `limit: 3`, confirm at most three results, scores in descending order, ranks 1..n, and that every result's `act` obeys an `act` filter given in the request. Confirm an unknown filter field and a caller-shaped fragment (for example `{"act": {"$ne": "BNS_2023"}}`) get 422, while list-valued `act`, `status` and `access_level` are accepted.
4. **Passage against source.** Take one returned result and compare it with its stored source: print the `chunk_id`, `section_id`, heading, a 200-character excerpt and the source file name, then check the section's text in `data/processed/*.jsonl` or the source PDF contains that excerpt. Record whether it matches. Also record in one sentence whether the passage actually answers the question, to show the score does not prove it.
5. Existing checks pass: `uv run ruff check .`, `uv run pytest -q`; `GET /v1/models` still lists `rag-semantic` and the other five models; `/v1/chat/completions` still returns its placeholder; `data/raw/` and the JSONL files are unchanged; `.env` is untracked and no credential appears in any output.

## Handover

When done, update this story with: files created or changed (including `docs/architecture.md`); the commands run for checks 1–5 with compact results; the diagnostic command and one sample output; the `numCandidates` rule chosen; any deviation with its reason, including any additive tightening of `SemanticFilters`; and anything the next stories need to know.

Later stories (hybrid, re-ranking, structured, decomposition, HyDE, generation) reuse this retrieval function, the `RetrievedChunk` fields filled here, and the three outcomes. Answer generation (Story 3.1) will consume `QueryResult.results` and must treat `no_results` as `insufficient_evidence` and `unavailable` as a failure, not as an empty answer.

## Handover (implemented)

**Files.** Created: `src/building_with_rag/retrieval/__init__.py`, `retrieval/semantic.py`, `tests/test_semantic.py`. Changed: `pipeline.py` (semantic → `semantic_retrieve`), `main.py` (`/v1/query` returns 503 on `unavailable`; chat still calls `run_pattern` but returns the placeholder text), `contracts.py`, `tests/test_api.py`, `docs/architecture.md`.

**Checks (compact results).**
1. Counts: sections 838, chunks 927, embeddings 927; `embeddings_vector_index` queryable=True, READY; stored embedding `voyage-3.5` / `voyage-3.5` / 1024, vector length 1024. `POST /v1/query` "What is the punishment for murder?" → 200, `ok`, 5 results, all section 7 fields present; source keys: chapter, page_start, page_end, source_pdf, source_sha256, source_status_version, status.
2. `act=[IPC_1860], status=[in_force]` → 200 `no_results`, message "The search ran successfully but no passages matched." `access_level=[restricted]` → 200 `no_results`. `MONGODB_URI=` empty → 503 `unavailable`, "Retrieval unavailable: MONGODB_URI is empty." (no secret).
3. `limit=3`, `act=[IPC_1860]` → 3 results, ranks [1,2,3], scores [0.767, 0.76, 0.756], all `IPC_1860`. Unknown field and `{"act": {"$ne": ...}}` → 422; list-valued act/status/access_level accepted.
4. `ipc:378:c1`, section `ipc:378`, heading "Theft", `IPC_1860_bare_act.pdf`; the 200-char excerpt is found in the section text in `ipc_sections.jsonl`: match. The excerpt (illustrations (c)/(d) of theft) is a real part of the theft section but does not state the punishment, so it does not answer "what is theft punishment"-style questions: the score shows relevance, not an answer.
5. `ruff` clean; `pytest` 20 passed; `/v1/models` lists 6 models incl. `rag-semantic`; chat placeholder unchanged; `data/` untouched by this work; `.env` is git-ignored; no credential printed.

**Diagnostic sample.** `uv run python -m building_with_rag.retrieval.semantic "What is the punishment for murder?" --limit 3 --act BNS_2023` →
`#1 score=0.8086 bns:103:c0 bns:103 BNS_2023 'Punishment for murder.' text_len=428`, `#2 0.8078 bns:104:c0`, `#3 0.7769 bns:109:c0`.

**numCandidates rule.** `min(1000, max(50, limit * 10))` (named constants in `semantic.py`).

**Deviations / tightening.** `SemanticFilters` now forbids extra fields; `QueryRequest.question` is stripped and rejected if blank (additive). The existing chat test and the old "semantic not implemented" API test were adjusted (chat test stubs `run_pattern`; the placeholder test now uses `hybrid`).

**Notes for next stories.** The Voyage account is rate limited (quick successive queries return `unavailable`/"embedding service unavailable"); pace calls, or retry in later stories. Readiness runs per request; the full model-consistency scan runs once per process. Story 3.1 should map `no_results` → `insufficient_evidence` and `unavailable` → failure. Chat calls `run_pattern` for semantic, so each chat message makes a real retrieval call.
