# building-intelligence

FastAPI service answering over the Indian criminal codes (BNS and IPC). See `docs/architecture.md`.

## Run

```
uv sync
cp .env.example .env     # optional; the app starts with no credentials
uv run uvicorn building_with_rag.main:app --port 8000
```

`.env` is untracked and secrets are never committed. `MONGODB_URI` is an Atlas M0 connection
string. `GENERATION_API_BASE_URL` and `GENERATION_API_KEY` come from the trainer (an
OpenAI-compatible LiteLLM proxy). If `CAPSTONE_API_KEY` is set, `/v1/query`, `/v1/models` and
`/v1/chat/completions` require `Authorization: Bearer <key>`.

Open WebUI is supplied separately by the trainer (see the story, Part C). Start the API first,
open `http://127.0.0.1:8080`, pick Building with RAG and the `semantic` or (explicitly selected) `hybrid`, `hybrid-reranked`, `structured`, `decomposition`, `hyde` or `agentic` mode: answers stream as a
labelled DRAFT and end with confidence and sources. Run the setup script once; do not edit the
Pipe, Filter or admin settings. `hybrid` also needs the keyword search index: run
`uv run python -m building_with_rag.ingestion.setup_db` once (it creates `chunks_keyword_index`
beside the vector index and is safe to repeat); until it is queryable `hybrid` returns `unavailable`. `hybrid-reranked` re-ranks the hybrid candidates with one extra provider call and needs `RERANK_API_KEY` (without it the mode returns `unavailable`; there is no fallback).

## Checks

```
uv run ruff check . && uv run pytest
```

`structured` (explicitly selected only) looks up one section exactly, for example "BNS section 103" or "What does IPC s. 302 say?". It supports only `BNS`/`IPC` plus a plain integer section number; it asks for clarification when the act or section is missing or ambiguous (including lettered sections such as 153A) and recommends `hybrid` or `semantic` for anything else. It needs only `MONGODB_URI` and the loaded corpus. Check it directly with `uv run python -m building_with_rag.retrieval.structured "BNS section 103"`.

`decomposition` and `hyde` (explicitly selected only) each add one model step in front of the existing semantic retrieval; both need the existing `VOYAGE_API_KEY` and `GENERATION_*` settings and return `unavailable` / `hyde_unavailable` (503) without them. `decomposition` splits a compound comparison question into one to three subquestions, retrieves evidence for each and answers only what every part supports (`ok`, `partial_answer`, `clarify`). `hyde` embeds one hypothetical passage only to find real passages (never as evidence). Check them directly with `uv run python -m building_with_rag.retrieval.decomposition "How does the BNS punish murder, and how does that compare with the IPC?"` and `uv run python -m building_with_rag.retrieval.hyde "What happens if someone makes me hand over my property by threatening to hurt me?"`.

`agentic` (explicitly selected only, `rag-agentic`) is a fixed read-only controller, not an LLM planner: it validates the question, then chooses between the existing structured lookup, semantic search and (for a small table of everyday-wording questions) one HyDE call, with at most two adapter calls. It uses the existing `VOYAGE_API_KEY` and `GENERATION_*` settings and one new optional variable, `AGENTIC_DEADLINE_SECONDS` (default 30, valid 1-120). Statuses: `ready_for_synthesis`, `partial_result`, `clarify` (200) and `unavailable` (503). Check it with `uv run python -m building_with_rag.retrieval.agentic "BNS section 103"`. See `docs/architecture.md`.
