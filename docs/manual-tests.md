# Manual tests

## Story 1.1 — Architecture and project seed

**What it adds:** a FastAPI seed with `/healthz`, six `rag-*` modes, `/v1/query` and an OpenAI-compatible chat adapter, all returning honest `not_implemented` placeholders.

Start the API first (no `.env` credentials needed): `uv run uvicorn building_with_rag.main:app --port 8000`

```bash
# Health
curl -s http://127.0.0.1:8000/healthz
```

Expected: `{"status":"ok"}` and nothing else.

```bash
# Model list
curl -s http://127.0.0.1:8000/v1/models
```

Expected: exactly six IDs: `rag-semantic`, `rag-hybrid`, `rag-hybrid-reranked`, `rag-structured`, `rag-decomposition`, `rag-hyde`.

```bash
# Query: one request per mode
for p in semantic hybrid hybrid-reranked structured decomposition hyde; do
  curl -s -X POST http://127.0.0.1:8000/v1/query \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"What is the punishment for murder?\", \"pattern\": \"$p\"}"
  echo
done
```

Expected: each response is a `QueryResult` with `"pattern"` set to that mode, `"status": "not_implemented"`, and empty `results`.

```bash
# Query: invalid input (empty question)
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "", "pattern": "semantic"}'
```

Expected: HTTP 422 with a `detail` list naming `question`.

```bash
# Chat completion (JSON)
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-semantic", "messages": [{"role": "user", "content": "What is murder?"}]}'

# Chat completion (streaming)
curl -sN -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-semantic", "stream": true, "messages": [{"role": "user", "content": "What is murder?"}]}'

# Chat completion: unknown model
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-unknown", "messages": [{"role": "user", "content": "hi"}]}'
```

Expected: the JSON call returns a `chat.completion` whose message says the `semantic` pattern is not implemented yet. The streaming call sends a role frame, a content frame and a stop frame, then ends with `data: [DONE]`. The unknown model returns HTTP 404 with `{"error": {"code": "model_not_found", ...}}`.


## Story 2.3 — Semantic retrieval

**What it adds:** the real `semantic` mode: the question is embedded with Voyage, MongoDB `$vectorSearch` finds the closest chunks, and `/v1/query` returns them as source passages (no generated answer yet).

Prerequisites: ingestion and the vector index from Stories 2.1 and 2.2 are done, `.env` has `MONGODB_URI`, `MONGODB_DB_NAME` and `VOYAGE_API_KEY`, and the API is running (`uv run uvicorn building_with_rag.main:app --port 8000`).

```bash
# Success: natural-language question
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for murder?", "pattern": "semantic", "limit": 3}'
```

Expected: HTTP 200, `"pattern": "semantic"`, `"status": "ok"`, and up to 3 `results` ordered by `score` (highest first), each with `chunk_id`, act-qualified lower-case `section_id` (e.g. `bns:103`), `act`, `heading`, `text`, `source`, `semantic_score` and `semantic_rank` starting at 1. `trace` lists the mode, embedding model and the effective filters; `generation` is `null`.

```bash
# Filtered search: restrict to one act
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for murder?", "pattern": "semantic", "filters": {"act": ["BNS_2023"]}}'

# No match: IPC records are repealed, so this filter matches nothing
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for murder?", "pattern": "semantic", "filters": {"act": ["IPC_1860"], "status": ["in_force"]}}'
```

Expected: the first call returns `"status": "ok"` with every result's `act` equal to `BNS_2023`. The second returns HTTP 200, `"status": "no_results"` and an empty `results` list; this is not an error.

```bash
# Invalid input: unknown filter field
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is murder?", "pattern": "semantic", "filters": {"$or": []}}'
```

Expected: HTTP 422 with a `detail` list naming the unexpected filter field. If `MONGODB_URI` is empty or the index is not ready, a valid request returns HTTP 503 with `"status": "unavailable"` and a message naming the failed prerequisite (never a secret).

```bash
# Diagnostic command (same code path, no HTTP)
uv run python -m building_with_rag.retrieval.semantic "What is the punishment for murder?" --limit 3 --act BNS_2023
```

Expected: prints `status: ok`, `mode: semantic`, the query, and one line per result with rank, score, `chunk_id`, `section_id`, act, heading and text length (no vectors or full text).


## Story 3.1 — Grounded answer generation

**What it adds:** `POST /v1/query` with `"pattern": "semantic"` and `"generate_answer": true` retrieves passages, asks the generation model to answer only from them, and returns the validated result in `generation` (non-streaming).

Prerequisites: everything from Story 2.3, plus `GENERATION_API_BASE_URL` and `GENERATION_API_KEY` set in `.env` and `GENERATION_MODEL_NAME` (a model your provider serves, for example `openai/gpt-oss-120b` on Groq), and the API running (`uv run uvicorn building_with_rag.main:app --port 8000`).

```bash
# Success: grounded answer with citations
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for murder?", "pattern": "semantic", "generate_answer": true, "limit": 5, "filters": {"act": ["BNS_2023"]}}'
```

Expected: HTTP 200, `"pattern": "semantic"`, `"status": "ok"`, retrieved `results` as in Story 2.3, and `generation.outcome` = `"answered"` with a non-empty `answer`, `claims` (each with `evidence` labels such as `E1`), `citations` (each with `label`, `chunk_id`, `section_id`, `act`, `heading`, `source`), `supporting_passages`, `provider` and `model`. Every citation matches a passage in `supporting_passages`.

```bash
# No match: IPC records are repealed, so retrieval finds nothing and no answer is guessed
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for murder?", "pattern": "semantic", "generate_answer": true, "filters": {"act": ["IPC_1860"], "status": ["in_force"]}}'

# Out-of-corpus question: the model must refuse rather than guess
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?", "pattern": "semantic", "generate_answer": true, "limit": 3}'
```

Expected: the first call returns HTTP 200, `"status": "no_results"`, empty `results`, and `generation.outcome` = `"insufficient_evidence"` with `context_outcome` = `"no_results"` and no `answer`. The second returns `"status": "ok"` with `generation.outcome` = `"insufficient_evidence"`, no `answer` and no `citations` (the trace holds a `model reason:` line).

```bash
# Default is unchanged: no generate_answer means retrieval only
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for murder?", "pattern": "semantic", "limit": 3}'

# Invalid input: generate_answer must be a boolean
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is murder?", "pattern": "semantic", "generate_answer": "maybe"}'
```

Expected: the first call returns `"status": "ok"` with `"generation": null`. The second returns HTTP 422 with a `detail` list naming `generate_answer`. If the generation proxy is unreachable or unconfigured, retrieval still succeeds and `generation.outcome` is `"unavailable"` (no `answer`, no secret in the message).

```bash
# Diagnostic command (same code path, no HTTP)
uv run python -m building_with_rag.generation.answer "What is the punishment for murder?" --limit 5 --act BNS_2023
```

Expected: prints `retrieval status: ok`, `generation.outcome: answered`, one line per evidence passage (`E1`, `E2`, ...), the `answer`, each `claim [E1]: ...` and each `citation` with chunk, section, heading and source PDF pages.


## Story 4.1 — Hybrid search (keyword + semantic)

**What it adds:** the real `hybrid` mode: the question runs through both the vector search and an Atlas Search keyword search on `chunks.text`, and the two lists are merged by Reciprocal Rank Fusion (`RRF_K = 60`); answers, citations and streaming are the unchanged Story 3.1/3.2 path.

Prerequisites: everything from Story 3.1, the one-time keyword index created with `uv run python -m building_with_rag.ingestion.setup_db` (it prints `Keyword search index chunks_keyword_index: ready`), and the API running (`uv run uvicorn building_with_rag.main:app --port 8000`).

```bash
# Success: exact legal term plus semantic meaning
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is culpable homicide?", "pattern": "hybrid", "limit": 5, "filters": {"act": ["BNS_2023"]}}'
```

Expected: HTTP 200, `"pattern": "hybrid"`, `"status": "ok"`, and up to 5 `results` with `fused_score` and `fused_rank` set (`score` = fused score) plus `semantic_score`/`semantic_rank` and/or `keyword_score`/`keyword_rank`; a field is `null` when that route did not return the passage. `trace` shows the fusion rule, the keyword index `chunks_keyword_index`, route counts (semantic, keyword, both, only-one-route) and one line per passage; `generation` is `null`.

```bash
# No match: IPC records are repealed, so this filter matches nothing
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is culpable homicide?", "pattern": "hybrid", "filters": {"act": ["IPC_1860"], "status": ["in_force"]}}'

# Invalid input: empty question
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "", "pattern": "hybrid"}'
```

Expected: the first call returns HTTP 200, `"status": "no_results"` and empty `results` (not an error). The second returns HTTP 422 with a `detail` list naming `question`. If the keyword index is missing or not queryable (or either route fails), a valid request returns HTTP 503 with `"status": "unavailable"` naming the failed prerequisite, never a one-route result and never a secret.

```bash
# Grounded answer from the fused passages
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is culpable homicide?", "pattern": "hybrid", "generate_answer": true, "limit": 5, "filters": {"act": ["BNS_2023"]}}'
```

Expected: HTTP 200, `"pattern": "hybrid"`, `"status": "ok"`, hybrid `results` as above, and `generation.outcome` = `"answered"` with `answer`, `claims`, and `citations` that each match a passage in `supporting_passages`. Needs `GENERATION_*` settings as in Story 3.1.

```bash
# Chat completion with the hybrid model (streaming)
curl -sN -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-hybrid", "stream": true, "messages": [{"role": "user", "content": "What is culpable homicide?"}]}'
```

Expected: an SSE stream for the `hybrid` pattern with the grounded answer text, ending with `data: [DONE]`.

```bash
# Semantic is unchanged
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is culpable homicide?", "pattern": "semantic", "limit": 3}'
```

Expected: `"pattern": "semantic"`, `"status": "ok"`, results with `semantic_score`/`semantic_rank` and no `keyword_*` or `fused_*` values.

```bash
# Diagnostic command (same code path, no HTTP)
uv run python -m building_with_rag.retrieval.hybrid "What is culpable homicide?" --limit 5 --act BNS_2023
```

Expected: prints the status, mode, the fusion rule, and one line per result with its semantic, keyword and fused ranks/scores, route label (`both`, `semantic only` or `keyword only`) and text length.


## Story 4.2 — Hybrid-reranked search (re-ranking the hybrid candidates)

**What it adds:** the real `hybrid-reranked` mode: the hybrid candidates from Story 4.1 are cut by `RERANK_CANDIDATE_LIMIT` (20) and `RERANK_SEND_LIMIT` (10), re-ranked by one provider call, and the top `RERANK_RETURN_LIMIT` (5) are returned; answers, citations and streaming are the unchanged Story 3.1/3.2 path.

Prerequisites: everything from Story 4.1, `RERANK_API_KEY` set in `.env` (the other `RERANK_*` defaults are in `.env.example`), and the API running (`uv run uvicorn building_with_rag.main:app --port 8000`).

```bash
# Success: re-ranked hybrid candidates
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the difference between culpable homicide and murder?", "pattern": "hybrid-reranked", "limit": 5, "filters": {"act": ["IPC_1860"]}}'
```

Expected: HTTP 200, `"pattern": "hybrid-reranked"`, `"status": "ok"`, and up to 5 `results` ordered by `rerank_score` (`score` equals it) with `rerank_rank` starting at 1, plus the hybrid `fused_score`/`fused_rank` fields. `omitted_candidates` lists each cut candidate with its `omitted_reason`; `trace` shows the order before and after re-ranking; `generation` is `null`.

```bash
# No match: nothing to re-rank, so the provider is not called
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is culpable homicide?", "pattern": "hybrid-reranked", "filters": {"act": ["IPC_1860"], "status": ["in_force"]}}'

# Invalid input: empty question
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "", "pattern": "hybrid-reranked"}'
```

Expected: the first call returns HTTP 200, `"status": "no_results"` and empty `results` (not an error). The second returns HTTP 422 with a `detail` list naming `question`. If `RERANK_API_KEY` is unset (or the provider fails), a valid request returns HTTP 503, `"status": "unavailable"`, a message naming the failed prerequisite (e.g. `missing setting: RERANK_API_KEY`), empty `results`, and no hybrid order passed off as re-ranked.

```bash
# Grounded answer from the re-ranked passages
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the difference between culpable homicide and murder?", "pattern": "hybrid-reranked", "generate_answer": true, "filters": {"act": ["IPC_1860"]}}'
```

Expected: HTTP 200, `"pattern": "hybrid-reranked"`, `"status": "ok"`, re-ranked `results` as above, and `generation.outcome` = `"answered"` with `answer`, `claims`, and `citations` that each match a passage in `supporting_passages`. Needs `GENERATION_*` settings as in Story 3.1.

```bash
# Chat completion with the hybrid-reranked model (streaming)
curl -sN -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-hybrid-reranked", "stream": true, "messages": [{"role": "user", "content": "What is the difference between culpable homicide and murder?"}]}'
```

Expected: an SSE stream for the `hybrid-reranked` pattern with the grounded answer text, ending with `data: [DONE]`.

```bash
# Hybrid is unchanged
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the difference between culpable homicide and murder?", "pattern": "hybrid", "limit": 5, "filters": {"act": ["IPC_1860"]}}'
```

Expected: `"pattern": "hybrid"`, `"status": "ok"`, results with `fused_*` values and `rerank_score`/`rerank_rank` all `null`.

```bash
# Diagnostic command (same code path, no HTTP)
uv run python -m building_with_rag.retrieval.hybrid_reranked "What is the difference between culpable homicide and murder?" --limit 5 --act IPC_1860
```

Expected: prints the status, mode, query, the three limits and model name, counts, the before- and after-re-ranking lists, and the omitted candidates with reasons (no vectors, passage text or key). Exits with code 1 only when `unavailable`.

## Story 5.1 — Structured exact retrieval (BNS/IPC section lookup)

**What it adds:** the real `structured` mode: a rule-based classifier (no LLM) turns an exact request such as "BNS section 103" into a read-only MongoDB lookup of that one section, returned as ordinary `results` chunks; anything else gets a recommendation or clarification with no database call.

Prerequisites: the ingested corpus from Stories 2.1–2.2 (`MONGODB_*` set in `.env`) and the API running (`uv run uvicorn building_with_rag.main:app --port 8000`).

```bash
# Success: exact section lookup
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "BNS section 103", "pattern": "structured"}'
```

Expected: HTTP 200, `"pattern": "structured"`, `"status": "ok"`, and `results` holding only the chunks of BNS section 103 (`section_id` for that section, ordered by chunk). `message` and `trace` show the matched `BNS 103` rule and the recorded source status; `generation` is `null`.

```bash
# Not found: valid reference, no such section
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "BNS section 9999", "pattern": "structured"}'

# Clarification: no act named
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does section 103 say?", "pattern": "structured"}'

# Recommendation: not an exact lookup (no database call)
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many sections are in the BNS?", "pattern": "structured"}'
```

Expected: the first returns HTTP 200, `"status": "not_found"`, a message naming `BNS 9999` and empty `results`. The second returns `"status": "needs_clarification"` with the message "Say which act: BNS or IPC." and empty `results`. The third returns the recommendation status (`"recommendation"`) pointing to the hybrid or semantic mode, empty `results`, and `trace` containing `database lookup skipped`. If MongoDB or the corpus is unavailable, a valid exact request returns HTTP 503 and `"status": "unavailable"`.

```bash
# Grounded answer from the exact section
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does IPC section 302 say?", "pattern": "structured", "generate_answer": true}'
```

Expected: HTTP 200, `"status": "ok"`, `results` from IPC section 302, and `generation.outcome` = `"answered"` with `answer`, `claims` and `citations` matching those passages. Needs `GENERATION_*` settings as in Story 3.1.

```bash
# Chat completion with the structured model (streaming)
curl -sN -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-structured", "stream": true, "messages": [{"role": "user", "content": "What does IPC section 302 say?"}]}'
```

Expected: an SSE stream for the `structured` pattern with the grounded answer text and sources, ending with `data: [DONE]`.

```bash
# Diagnostic command (same code path, no HTTP)
uv run python -m building_with_rag.retrieval.structured "BNS section 103"
```

Expected: prints `status: ok`, the message, the classifier signals (`intent=exact_lookup act=BNS section_number=103`), the rule, and one line per returned chunk. Exits with code 1 only when `unavailable`.


## Story 5.2 — Query decomposition and HyDE

**What it adds:** the real `decomposition` mode (one model step splits a comparison question into 1–3 subquestions, each retrieved separately) and the real `hyde` mode (one model step writes a hypothetical passage whose embedding is used to find real passages); answers, citations and streaming are the unchanged Story 3.1/3.2 path.

Prerequisites: everything from Story 3.1 (`MONGODB_*`, `VOYAGE_API_KEY`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`, `GENERATION_MODEL_NAME` in `.env`) and the API running (`uv run uvicorn building_with_rag.main:app --port 8000`).

```bash
# Decomposition: compound comparison question
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does the BNS punish murder, and how does that compare with the IPC?", "pattern": "decomposition", "limit": 5}'
```

Expected: HTTP 200, `"pattern": "decomposition"`, `"status": "ok"`, a `subquestions` list (1–3 entries, each with its own `results` and `"status": "evidenced"`), and `results` holding the union of their passages. `generation` is `null`.

```bash
# Decomposition: missing evidence for one subquestion (IPC scope is filtered out)
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does the BNS punish murder, and how does that compare with the IPC?", "pattern": "decomposition", "filters": {"act": ["BNS_2023"]}}'

# Decomposition: invalid input (empty question)
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "", "pattern": "decomposition"}'
```

Expected: the first call returns HTTP 200 and `"status": "partial_answer"` (or `"clarify"`), never `ok`; the IPC subquestion has `"status": "no_evidence"` with a `reason`. The second returns HTTP 422 with a `detail` list naming `question`. If the generation or embedding service is unavailable, a valid request returns HTTP 503 with `"status": "unavailable"` and no secret.

```bash
# Decomposition: grounded answer
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does the BNS punish murder, and how does that compare with the IPC?", "pattern": "decomposition", "generate_answer": true}'
```

Expected: HTTP 200, `"status": "ok"`, and `generation.outcome` = `"answered"` with `answer`, `claims` and `citations` matching `supporting_passages`. With `partial_answer`, the answer is limited to the evidenced subquestions.

```bash
# HyDE: lay-worded question
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What happens if someone makes me hand over my property by threatening to hurt me?", "pattern": "hyde", "limit": 5}'
```

Expected: HTTP 200, `"pattern": "hyde"`, `"status": "ok"`, final `results`, plus `hyde_direct_candidates` (from the question) and `hyde_query_candidates` (from the hypothetical passage), each with `semantic_rank`. `hyde_hypothetical_text_debug` holds the hypothetical text, which appears nowhere else in the response and is never a source; `trace` has a `final order:` line with each passage's route.

```bash
# HyDE: no match
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What happens if someone makes me hand over my property by threatening to hurt me?", "pattern": "hyde", "filters": {"act": ["IPC_1860"], "status": ["in_force"]}}'
```

Expected: HTTP 200, `"status": "no_results"` and empty `results` (not an error). If the hypothetical text cannot be produced (for example `GENERATION_API_KEY` unset), the response is HTTP 503 with `"status": "hyde_unavailable"` naming the failure, with no fallback to plain semantic results.

```bash
# Chat completion with each new model (streaming)
curl -sN -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-decomposition", "stream": true, "messages": [{"role": "user", "content": "How does the BNS punish murder, and how does that compare with the IPC?"}]}'

curl -sN -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-hyde", "stream": true, "messages": [{"role": "user", "content": "What happens if someone makes me hand over my property by threatening to hurt me?"}]}'
```

Expected: each is an SSE stream for its pattern with the grounded answer, a confidence line and `Sources:`, ending with `data: [DONE]`. The hypothetical text is not shown.

```bash
# Diagnostic commands (same code path, no HTTP)
uv run python -m building_with_rag.retrieval.decomposition "How does the BNS punish murder, and how does that compare with the IPC?" --limit 5
uv run python -m building_with_rag.retrieval.hyde "What happens if someone makes me hand over my property by threatening to hurt me?" --limit 5
```

Expected: the first prints `status`, `message`, and per subquestion its status, reason, scope and one line per chunk. The second prints `status`, `message`, the hypothetical text length (not the text), direct, hypothetical and final candidates with ranks, scores and route. Each exits with code 1 only when unavailable.


## Story 6.1 — Bounded agentic RAG

**What it adds:** the real `agentic` mode (`rag-agentic`): a fixed, read-only controller that validates the question, then makes at most two calls among the structured, semantic and HyDE adapters (HyDE only for the prepared lay-wording phrases when the direct result is weak) and checks the evidence; answers, citations and streaming are the unchanged Story 3.1/3.2 path.

Prerequisites: everything from Story 5.2 (`MONGODB_*`, `VOYAGE_API_KEY`, `GENERATION_*` in `.env`), optionally `AGENTIC_DEADLINE_SECONDS` (default 30), and the API running (`uv run uvicorn building_with_rag.main:app --port 8000`; restart it to load this story).

```bash
# Success: act-qualified section request (structured path)
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "BNS section 103", "pattern": "agentic"}'

# Success: ordinary question (semantic path)
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is culpable homicide?", "pattern": "agentic", "limit": 5}'
```

Expected: both return HTTP 200, `"pattern": "agentic"`, `"status": "ready_for_synthesis"` and real `results` (the first only `bns:103` chunks, the second up to 5 passages); `generation` is `null`. `trace` lists `mode: agentic`, `validation: accepted`, `path: structured` / `path: semantic`, one `call 1: ...` line (`structuredLookup` / `semanticRetrieve`), `calls: 1 of 2` and `stop: evidence_adequate`. The trace never contains the question or passage text.

```bash
# Vocabulary mismatch: lay wording, HyDE allowed only if the direct result is weak
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What happens if someone makes me hand over my property by threatening to hurt me?", "pattern": "agentic", "limit": 5}'
```

Expected: HTTP 200. `call 1` is `semanticRetrieve`; `hydeRetrieve` appears as `call 2` (`calls: 2 of 2`, `path: semantic+hyde`) only when the `adequacy: vocabulary mismatch (direct)` line says `not adequate`. Status is `ready_for_synthesis` or `partial_result`; `hyde_direct_candidates`, `hyde_query_candidates` and `hyde_hypothetical_text_debug` stay empty/null. If HyDE is unavailable (e.g. `GENERATION_API_KEY` unset) the result is `partial_result` with the direct passages and `stop: tool_unavailable`.

```bash
# Clarify: bare section number, no adapter call
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "section 103", "pattern": "agentic"}'

# Clarify: valid reference, no such section (no fall-through to semantic search)
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "BNS section 9999", "pattern": "agentic"}'

# Narrowed scope: filters reach the adapter unchanged
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is culpable homicide?", "pattern": "agentic", "filters": {"access_level": ["restricted"]}}'
```

Expected: the first returns HTTP 200, `"status": "clarify"`, a generic hint message, empty `results`, `validation: rejected (...)`, `stop: validation_rejected` and no `call` lines. The second returns `"status": "clarify"`, empty `results`, `stop: record_not_found`. The third returns HTTP 200 and `"status": "partial_result"` with empty `results` and `stop: no_usable_evidence` (never a wider result). If MongoDB or Voyage is unavailable and nothing was retained, a valid request returns HTTP 503 with `"status": "unavailable"` naming the adapter and a category, never a secret.

```bash
# Invalid input: empty question
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "", "pattern": "agentic"}'
```

Expected: HTTP 422 with a `detail` list naming `question`.

```bash
# Grounded answer
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is culpable homicide?", "pattern": "agentic", "generate_answer": true, "limit": 5}'
```

Expected: HTTP 200, `"status": "ready_for_synthesis"`, and `generation.outcome` = `"answered"` with `answer`, `claims` and `citations` that each match a passage in `results`. `clarify` and `unavailable` results never generate (`generation` stays `null`).

```bash
# Chat completion with the agentic model (streaming)
curl -sN -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-agentic", "stream": true, "messages": [{"role": "user", "content": "What is culpable homicide?"}]}'
```

Expected: an SSE stream for the `agentic` pattern with the grounded answer text, a confidence line and `Sources:`, ending with `data: [DONE]`.

```bash
# Diagnostic command (same code path, no HTTP)
uv run python -m building_with_rag.retrieval.agentic "BNS section 103"
```

Expected: prints the status, message, every trace line, and one line per result with `chunk_id`, `section_id`, heading and text length (it does not echo the question). Exits with code 1 only when `unavailable`.
