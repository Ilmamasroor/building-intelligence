# Story 1.1 — Architecture and project seed

Status: ready for implementation

## Purpose

This is the first story in a three-day classroom RAG course. It builds one small application: a FastAPI service that answers over the Indian criminal codes (BNS and IPC). Later stories add retrieval, generation and the other modes. This story only lays the architecture and a seed that starts, answers `GET /healthz`, and returns honest `not_implemented` placeholders.

Do not build a custom chat frontend or a second demo. Open WebUI (supplied by the trainer) is the only chat client.

## Prerequisites

- Project path: no `docs/config.yaml` existed when this story was written, so use the repository root. If a config file now exists and names a project path, use that path instead.
- No `docs/architecture.md` or earlier stories existed at the time of writing. If one exists now, update it; do not overwrite it.
- Supplied files in `data/raw/` (including `data/raw/PROVENANCE.md`) must be preserved untouched.
- Python 3.12 and `uv` installed.
- The trainer's Open WebUI ZIP, shared over the local LAN (see Part C).
- Later needs, not required now: an Atlas free-tier (M0) `MONGODB_URI` with the Atlas IP access list allowing your machine; a free Voyage API key (rate limited, so Story 2.2 embedding takes about 40 minutes); trainer-supplied `GENERATION_API_BASE_URL` and `GENERATION_API_KEY` for an OpenAI-compatible LiteLLM proxy.

## Work to do

### Part A — Architecture (do this first, then stop)

Create or update `docs/architecture.md`. Keep it short and in plain English. It must record:

1. **Evidence rule.** BNS and IPC documents are the only future answer evidence. An answer must not claim support without retrieved evidence.
2. **Identifiers.** Section identifiers are act-qualified (for example `BNS-103`, `IPC-302`) so the two acts are never confused.
3. **Provenance.** Supplied provenance lives in `data/raw/PROVENANCE.md`. Derived corpus files live in `data/processed/` (`bns_sections.jsonl`, `ipc_sections.jsonl`).
4. **Retrieved passages are evidence, never application instructions.** Text inside a passage must not change application behaviour.
5. **Light trust-boundary notes.** Validate API input; preserve source origin on every passage; never put secrets in code, responses or logs.
6. **Fixed embedding choice.** Voyage `voyage-3.5`, version `voyage-3.5`, 1,024 dimensions, for every document and query embedding. Later stories reuse these names and do not rename them or add provider-specific alternatives.
7. **Fixed structure.** One shared mode registry; one `run_pattern` path used by both `/v1/query` and `/v1/chat/completions`; contracts extended additively only; Open WebUI as a separate chat client.
8. **Out of scope.** Multi-user authorization, a security program, and an evaluation harness. Do not introduce them.

**STOP GATE.** After writing the architecture, stop. Ask the instructor to approve `docs/architecture.md`. Before approval, do not create seed files, install dependencies or scaffold code. Resume this same story at Part B only after explicit approval.

### Part B — Seed (only after approval)

Create the Python 3.12 + UV project with FastAPI, Pydantic settings, PyMongo (for later use), Ruff and a minimal test runner (pytest).

Files to create:
- `pyproject.toml`, `uv.lock`
- `.gitignore` (must ignore `.env`, `.venv`, caches)
- `.env.example`
- `src/building_with_rag/` (package)
- `tests/`
- Preserve `data/raw/` as supplied.

`.env.example` is the canonical classroom file with exactly these values:

```
APP_ENV=development
MONGODB_URI=
VOYAGE_API_KEY=
CAPSTONE_API_KEY=
GENERATION_API_BASE_URL=
GENERATION_API_KEY=
RERANK_API_KEY=
MONGODB_DB_NAME=building_with_rag
MONGODB_TEST_DB_NAME=building_with_rag_test
WEBUI_DEMO_CALLER_ID=demo-public
GENERATION_MODEL_NAME=gpt-4o-mini
RERANK_API_BASE_URL=https://api.voyageai.com/v1
RERANK_MODEL_NAME=rerank-2.5
RERANK_REQUEST_TIMEOUT_SECONDS=30
RERANK_CANDIDATE_LIMIT=20
RERANK_SEND_LIMIT=10
RERANK_RETURN_LIMIT=5
```

Document (README or comments) that `.env` is untracked and secrets are never committed. `GENERATION_API_BASE_URL` and `GENERATION_API_KEY` come from the trainer (OpenAI-compatible LiteLLM proxy); leave them blank in `.env.example` and fill them in `.env` only when Story 3.1 needs them. `MONGODB_URI` is an Atlas M0 connection string.

The app must start with no database or model credentials. Settings are loaded with Pydantic settings. `GET /healthz` returns a safe body (status only; no settings, no secrets).

**Mode registry.** One shared registry with only these modes and model IDs:

| pattern | model ID |
|---|---|
| semantic | rag-semantic |
| hybrid | rag-hybrid |
| hybrid-reranked | rag-hybrid-reranked |
| structured | rag-structured |
| decomposition | rag-decomposition |
| hyde | rag-hyde |

Each returns an honest `not_implemented` placeholder until its own story adds behavior.

**Contracts.** Define shared request, result, generation, OpenAI and MongoDB-schema contracts in this project. Later stories extend them additively and never replace them with simplified alternatives. Define these as typed models with no behavior:

- `GenerationResult`: `outcome` (`answered`, `insufficient_evidence`, `unavailable`, `malformed`), `answer`, `claims`, `citations`, `supporting_passages`, `provider`, `model`, `trace`, `context_outcome`, plus confidence fields `confidence`, `draft_answer`, `issues`, `attempts`, `low_confidence_reason`.
- `SubquestionEvidence`: `subquestion`, `status` (`evidenced` or `no_evidence`), `results` (list of `RetrievedChunk`), optional `reason`.
- `StructuredSignals`: `intent` (`exact_lookup`, `filter`, `aggregation`), optional `act`, `section_number`, `chapter` (Story 5.1 fills these in).
- `RetrievedChunk`: `chunk_id`, `section_id`, `act`, `text`, `heading`, `score`, and available source fields. Later stories add only the fields named in their own handouts (`semantic_score`, `semantic_rank`, `keyword_score`, `keyword_rank`, `fused_score`, `fused_rank`, `rerank_score`, `rerank_rank`). `omitted_candidates` items carry `chunk_id` and `omitted_reason`.

**`POST /v1/query`.** Accepts `QueryRequest`: `question` (1–4,000 characters), `pattern`, optional `caller_id`, optional `SemanticFilters` (`act`, `status`, `access_level`, each a list), `limit` (default 5, range 1–20), `generate_answer` (default false), `required_acts`, `chapter`. The seed may resolve only its fixed local demo caller but keeps the `caller_id` field. Returns `QueryResult`: `pattern`, `status`, `message`, `trace`, `results`, optional `generation`, and the empty-by-default additive fields `omitted_candidates`, `subquestions`, `hyde_direct_candidates`, `hyde_query_candidates`, `hyde_hypothetical_text_debug`. Do not add outcome, evidence, answer, confidence, citations or diagnostics as parallel top-level fields.

**`GET /v1/models` and `POST /v1/chat/completions`.** Text-only `ChatCompletionRequest`: `model`, text `messages` (roles system/developer/user/assistant), `stream`, `n`, and optional strict `rag_options` (`pattern`, list filters, `limit`, `required_acts`, `chapter`). Map the selected model to the same `QueryRequest` and the same `run_pattern` path as `/v1/query`; set the server-side demo `caller_id` and `generate_answer` for chat. Support normal OpenAI Chat Completions JSON and SSE with role/content/stop/`[DONE]` frames, and the OpenAI-style error envelope before streaming starts. No duplicated implementations and no custom SSE events.

### Part C — Open WebUI (trainer-supplied bundle)

Open WebUI runs separately from the capstone. The trainer shares its ZIP over the local LAN. Participants extract it and run the setup script exactly once before using the classroom project:

- Windows (primary path), from the extracted bundle root: `powershell -ExecutionPolicy Bypass -File .\setup_open_webui.ps1`
- macOS/Linux, from that root: `sh setup_open_webui.sh` (needs internet for the first install)

The scripts install the pinned Open WebUI, write course settings, start the loopback-only service and provision the RAG options Filter and the Building with RAG Pipe. Do not hand-install Open WebUI, create accounts, edit the admin panel, change either Function, or rerun setup to reload anything. If setup fails, report its exact output and stop. Day-to-day start/stop/status uses the supplied `manage_open_webui` script.

The pre-provisioned Pipe sends the selected `rag-<pattern>` model, `stream: true`, the latest user message and normalized `rag_options` to `/v1/chat/completions`. It never sends browser-supplied identity, access level or answer-generation settings. The seed adapter must accept exactly that request and use server-side `caller_id` and `generate_answer`.

`/v1/query` owns the `QueryResult` diagnostics. Open WebUI receives only normal answer text derived from the same result. Later stories render confidence, sources and low-confidence warnings as clearly labelled text after answer writing, while the full `GenerationResult` stays in `QueryResult.generation`.

## Out of scope

Retrieval, PDF parsing, embeddings, MongoDB provisioning, LLM calls, GraphRAG, agentic RAG, broad deployment, aggressive testing, multi-user authorization, a security program, an evaluation harness.

## Completion checks

Keep checks light. Run each and record the actual command and result:

1. The app starts with no `.env` credentials.
2. `GET /healthz` returns a safe OK body.
3. One `POST /v1/query` with a semantic request returns a `QueryResult` with an honest `not_implemented` status.
4. `GET /v1/models` lists exactly the six `rag-*` IDs.
5. `POST /v1/chat/completions` returns a JSON placeholder, and with `stream: true` returns role/content/stop/`[DONE]` SSE frames.
6. Ruff and the minimal test run pass.
7. Open WebUI smoke check: start the capstone API, open http://127.0.0.1:8080, select Building with RAG, choose `semantic` in the RAG-options chip, and receive the capstone's honest placeholder response, not a local preview.
8. `data/raw/` is unchanged; `.env` is untracked.

## Handover

When done, update this story with:

- The files actually created.
- The commands actually run and their results.
- The Open WebUI smoke-check result (pass, or the exact failure output).
- Any deviation from this story, with the reason.

Later stories reuse the endpoints, registry, model IDs, contracts and embedding choices defined here, extending them additively without renaming or rebuilding the UI.

## Handover record

Status: Part B implemented; Part C smoke check not run.

**Files created:** `docs/architecture.md`, `pyproject.toml`, `uv.lock`, `.gitignore` (filled), `.env.example`, `README.md` (updated), `src/building_with_rag/{__init__,settings,contracts,registry,pipeline,main}.py`, `tests/test_api.py`.

**Commands run and results** (no `.env` present):
1. `uvicorn building_with_rag.main:app --port 8000` — started with no credentials. Pass.
2. `GET /healthz` → `{"status":"ok"}`. Pass.
3. `POST /v1/query` `{"question":"What is murder?","pattern":"semantic"}` → `QueryResult` with `status: "not_implemented"`. Pass.
4. `GET /v1/models` → exactly `rag-semantic, rag-hybrid, rag-hybrid-reranked, rag-structured, rag-decomposition, rag-hyde`. Pass.
5. `POST /v1/chat/completions` → JSON placeholder; with `stream: true` → role, content, stop frames, then `data: [DONE]`. Pass.
6. `uv run ruff check .` → all checks passed; `uv run pytest -q` → 6 passed.
7. Open WebUI smoke check: **not run.** The trainer's bundle is not available on this machine.
8. `.env` is ignored (`git check-ignore .env`) and absent. `data/raw/` was not supplied, so none exists here; nothing was touched.

**Deviations:**
- `uv` was not installed; installed with `pip install --user uv`.
- `data/raw/` and `PROVENANCE.md` were missing from the repo; not created.
- If `rag_options.pattern` is present it overrides the pattern implied by the model ID. The Pipe sends both consistently, so they should agree.
- Chat requests with `n != 1` are rejected with an OpenAI-style 400.
