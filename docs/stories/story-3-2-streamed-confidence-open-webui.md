# Story 3.2 — Streamed answer with confidence in Open WebUI

Status: implemented offline; live smoke checks pending participant

## Purpose

Connect the real RAG answer (Story 3.1) to the Open WebUI streaming path that Story 1.1 already established. After this story:

- `POST /v1/chat/completions` (what the Open WebUI Pipe calls) and `POST /v1/query` run the **same selected pattern** through the **same `run_pattern` path** and share one `QueryResult` with its nested `GenerationResult`: the same retrieval, context assembly, answer validation, citations and final outcome. There is no second chat path.
- Only **answer writing** streams. Retrieval and context assembly stay non-streamed and finish before the first text piece.
- Each request performs **one** generation/validation operation (a bounded number of attempts inside it). Its text events feed the stream, and the final `QueryResult.generation` describes that same operation. The stream and the result are never two independent LLM calls.
- Open WebUI cannot retract text it has already shown, so every attempt is shown as a clearly labelled **DRAFT** until it passes validation. The reply ends with confidence and source/check information as readable text. `/v1/query` stays the richer, learner-facing inspection surface.

Not in this story: agentic behavior, new retrieval modes, hybrid / re-ranking / structured / decomposition / HyDE generation, new Open WebUI plugins, Filters or Pipes, a custom frontend, conversation memory, multi-user authorization, an evaluation harness, broad test suites, new `.env` names.

## Prerequisites

- Stories 1.1, 2.1, 2.2, 2.3 and 3.1 are complete. `/v1/query` with `pattern: "semantic"` and `generate_answer: true` returns a real `generation` against the learner's MongoDB, Voyage and generation keys. If 3.1's live checks are still "pending", say so in the handover; do not try to complete them here.
- Read first: `docs/config.yaml` (it did not exist when this story was written, so use the repository root; if one now names a project path, use that path), `docs/architecture.md`, Story 1.1 (especially Part C, the Pipe description), Stories 2.1–2.3 and 3.1 with their handovers, `contracts.py`, `settings.py`, `registry.py`, `pipeline.py`, `main.py` (the chat adapter and `_sse`), `generation/{context,provider,answer}.py`, `tests/test_api.py` and `tests/test_generation.py`.
- The supplied **Open WebUI course bundle notes** (the trainer's ZIP: setup, manage and Pipe/Filter notes). They were not on this machine when the story was written; Story 1.1 Part C is the only description available. If the bundle notes are present, read them and confirm: the Pipe sends `rag-<pattern>`, `stream: true`, the latest user message and normalized `rag_options` to `/v1/chat/completions`, and how it authenticates to the capstone. If they contradict this story (for example the Pipe does not send `Authorization: Bearer <CAPSTONE_API_KEY>`), stop and report the exact difference. Do not edit the bundle, the Pipe or the Filter.
- Settings are used by their exact names in `.env.example`: `CAPSTONE_API_KEY`, `WEBUI_DEMO_CALLER_ID`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`, `GENERATION_MODEL_NAME`. **Do not add, rename, re-default or replace any setting; `.env.example` stays unchanged.** Tuning values are named code constants. `settings.py` already defines all five.
- If a prerequisite is missing, stop and report which one. Do not re-run ingestion, change the index or alter retrieval.

## Contracts and behavior to preserve

Extend additively only.

- `QueryRequest`, `QueryResult`, `RetrievedChunk` and the nested `GenerationResult` with all existing fields and meanings: `outcome` (`answered`, `insufficient_evidence`, `unavailable`, `malformed`), `answer`, `claims`, `citations`, `supporting_passages`, `provider`, `model`, `trace`, `context_outcome`, and the confidence extension `confidence`, `draft_answer`, `issues`, `attempts`, `low_confidence_reason`. Add no parallel top-level confidence, draft or outcome fields and no new outcome values.
- `/v1/query` shape and status codes (200 / 503 for retrieval `unavailable` / 422), the semantic outcomes `ok` / `no_results` / `unavailable`, the six `rag-*` model IDs, the mode registry, `/v1/models` and `/healthz`.
- Story 3.1 rules: `answer` is set only for `answered`; labels the model invents are never dropped silently; citations are resolved from the assembled context, never from model-supplied identifiers; secrets, the base URL and headers never appear in results, traces, logs, errors or stream text; context budget constants and the prompt rules (evidence-only, cite labels, untrusted passages) stay.
- Chat request validation, the OpenAI-style error envelope **before** streaming starts (unknown model 404, `n != 1`, blank question), `rag_options` (strict), server-side `caller_id` taken from `WEBUI_DEMO_CALLER_ID`, and the placeholder text for patterns that are not implemented.
- `GET /v1/models` lists exactly the six IDs. `.env.example`, `data/raw/` and `data/processed/` unchanged. `.env` untracked.

## Work to do

### 1. One path: `run_pattern` produces events, both routes consume them

Keep `run_pattern(request) -> QueryResult` as the one public entry point used by `/v1/query`. Add, in `pipeline.py`, one sibling generator (for example `run_pattern_events(request)`) that performs the whole operation once and yields small typed events, ending with a final event that carries the finished `QueryResult`. `run_pattern` simply drains that generator and returns the final `QueryResult`, so `/v1/query` and chat execute identical code: same retrieval call, same `assemble_context`, same generation and validation, same citations, same final `GenerationResult`.

Rules:

- Retrieval and context assembly run exactly once, non-streamed, before any text event. Generation receives the retrieval `QueryResult`; it never retrieves again.
- Chat sets `generate_answer=True` server-side (replace the Story 3.1 `generate_answer=False` placeholder line). The browser/Pipe never controls it, `caller_id`, or access level.
- Only `pattern == "semantic"` generates. Other patterns keep the existing `not_implemented` result; chat shows the existing placeholder text for them, through the same event path.
- Retrieval `unavailable` makes no model call and leaves `generation` `null` (Story 3.1). Retrieval `no_results` makes no model call and gives `insufficient_evidence`.
- The final `QueryResult` is the single source of truth. Chat text is rendered **from** it and from the events of the same run; it is not recomputed.
- Event types stay small and internal (a text piece; an attempt started; an attempt's check finished with pass/fail and issues; final). They are not a public API and do not appear in any response.

### 2. Streamed provider call

In `generation/provider.py` add one function (for example `stream_chat_completion(messages) -> Iterator[str]`) that sends `POST <GENERATION_API_BASE_URL>/chat/completions` with `Authorization: Bearer <GENERATION_API_KEY>`, the same `model` (`GENERATION_MODEL_NAME`), `messages` and temperature, and `"stream": true`, and yields the content text pieces from the OpenAI-compatible `data:` lines until `[DONE]`. Rules:

- `httpx` streaming, explicit timeout constant, no vendor SDK, no provider-specific shapes, no hidden retry loop. Join the URL as in 3.1.
- Failures use the existing `ProviderUnavailable` with the same safe categories (missing setting names, timeout, connection error, non-2xx status, malformed stream, no content) plus an "interrupted" category when the stream breaks after pieces were received. Never echo the URL, key, headers or response body.
- Leave the existing `chat_completion` and its tests untouched (Story 3.1 behavior is preserved). Generation uses the streamed function for every caller, so `/v1/query` and chat share one provider path; note in the handover that `chat_completion` is no longer used by generation and was deliberately not deleted here.

### 3. Stream only the answer text

The model still replies with the Story 3.1 JSON object. Do not stream raw JSON to the user.

- Add a small incremental extractor that, as pieces arrive, emits the decoded characters of the JSON `"answer"` string value (handle escapes and pieces split mid-escape). It emits nothing before the answer value starts, nothing for `claims` / `reason`, and nothing if the reply is not JSON.
- The **full** accumulated reply is parsed and validated after the stream ends, exactly as in 3.1. The extractor only decides what is shown early; it never decides the outcome.
- If the reply is `insufficient_evidence`, no answer text was emitted; the chat text is the honest insufficient-evidence message (with the model's short reason if present).
- Keep the prompt change minimal: ask that `"answer"` comes first in the JSON object. Do not weaken any existing prompt rule.

### 4. Bounded attempts and validation after each attempt

Add named constants (for example `MAX_ATTEMPTS = 2`) in `generation/answer.py`. After **each** attempt, once its reply is complete, validate:

1. the reply parses and has an `answered` outcome with a non-blank answer and claims (the 3.1 checks);
2. every claim cites at least one label and every label is in the supplied set (an invented label is a failed check, never stripped);
3. **support:** each claim has a coarse, deterministic grounding check against the passages it cites (for example a named minimum share of the claim's meaningful words appearing in the cited passage text; thresholds are named constants). This is a coarse signal, not proof. It makes no extra LLM call. Record the limitation in the architecture notes.

Handling:

- All checks pass → `outcome="answered"`, `answer` = validated text, `citations` resolved as in 3.1, **`confidence` high** (a named constant, for example `0.9`, labelled "high" in chat text), `attempts` = attempts used, `issues` = every failed check from earlier attempts (empty if the first attempt passed), `draft_answer` and `low_confidence_reason` unset.
- A check fails and attempts remain → record each failed check with its plain-language detail (for example "E7 is not one of the supplied passages", "claim 2 is not supported by E1"), then retry once. The retry is a new streamed attempt inside the same operation: original messages plus the previous reply and a short plain-language list of what failed (no passage text, no secrets). Its text is streamed too.
- Final attempt fails → no validated answer. `outcome="malformed"`, `answer=None`, `draft_answer` = the last attempt's text exactly as streamed (invalid labels left in place, not stripped), `confidence` low (a named constant), `attempts`, **every** failed check from **every** attempt in `issues`, and `low_confidence_reason` = one short sentence. Keep `claims` as the model gave them where parseable. `citations` stay empty (nothing was validated) and each invalid label is named in `issues`. `trace` gets one line per attempt. Never put the draft in `answer`.
- No answer text could be judged (not JSON, no `answer` value, provider failed before any text, `insufficient_evidence`) → `confidence` is `None`, `draft_answer` is `None`, and the result reports the real outcome (`malformed`, `unavailable`, `insufficient_evidence`) with its reason in `trace` / `issues`.
- Provider failure is never retried (3.1: no retry loop for provider errors); only failed validation triggers the second attempt.
- Update `pipeline._GENERATION_NOTE` / `message` additively so a low-confidence result is distinguishable from a plain malformed reply (for example "Answer failed evidence checks (low confidence)").

### 5. Provider fails after text began

If `ProviderUnavailable` is raised after one or more text pieces were already streamed: stop that attempt, set `outcome="unavailable"`, `answer=None`, `draft_answer` = the partial text, `confidence=None`, record the safe category in `issues`/`trace`, and have chat emit one final clearly labelled line, for example `UNAVAILABLE — the answer service stopped while writing; the text above is incomplete and not validated.` Then still send the normal stop chunk and `data: [DONE]`. Any other unexpected exception after streaming started is handled the same way with a generic safe line (no stack trace, no secrets). The stream must always close.

### 6. Chat adapter: OpenAI SSE and readable Open WebUI text

In `main.py` (the adapter, plus a small formatting helper module if that keeps it clean), keep `_sse` and the frame order: first chunk `{"role": "assistant"}`, then `{"content": ...}` chunks, then a chunk with an empty delta and `finish_reason: "stop"`, then `data: [DONE]`. No custom SSE event names, no extra fields. Retrieval and context assembly run inside the response generator after the role chunk, so a slow retrieval does not delay headers. Pre-stream validation errors still return the OpenAI error envelope as today. A retrieval `unavailable` is shown as a labelled `UNAVAILABLE — retrieval: <safe message>` assistant line followed by stop and `[DONE]` (HTTP 200), not a dropped connection.

Rendered text (Markdown, plain enough to read raw; exact wording is yours but the labels below are required):

- Each attempt starts with the line `DRAFT — checking evidence`, then the streamed answer text. This applies to every attempt, including a first attempt that later passes, because nothing is validated until its text is complete.
- Retry: after a failed attempt that will be retried, one line such as `Check failed (attempt 1 of 2): <plain detail>. Retrying…`, then the new attempt starting with `DRAFT — checking evidence`.
- Validated: a short footer, for example `Evidence check passed — confidence: high (attempt 1 of 2).` followed by `Sources:` as one line per citation: label, act-qualified section ID, heading, `source_pdf` and pages (from the citation dicts). Also state that the draft text above is now the validated answer.
- Failed final check: `DRAFT — low confidence, not the final answer.` plus the short `low_confidence_reason` and the failed-check details (compact). No sources are presented as support.
- `insufficient_evidence`: plain message, no draft label, no confidence.
- Non-streaming chat (`stream: false`) uses the same event/result path and returns the same rendered text in one `chat.completion` message (including the DRAFT labels for a failed final check; a validated answer is returned without a DRAFT prefix because nothing was shown before validation).
- Never put full passage text, claims JSON, the prompt, the base URL or any key in chat text. Chat shows what a reader needs; `/v1/query` remains the place for `claims`, `supporting_passages`, `trace`, `issues` and `attempts`.

### 7. Auth with `CAPSTONE_API_KEY`

`CAPSTONE_API_KEY` is in `.env.example` and `settings.py` but nothing enforces it yet. Add one small shared FastAPI dependency (no users, roles or sessions): when `CAPSTONE_API_KEY` is set, `/v1/query`, `/v1/models` and `/v1/chat/completions` require `Authorization: Bearer <CAPSTONE_API_KEY>` (constant-time compare). Missing/wrong → `/v1/query` returns 401 with a `detail`; the chat and models routes return the OpenAI error envelope (401, code `invalid_api_key`). `/healthz` stays open. When `CAPSTONE_API_KEY` is empty or unset the app behaves exactly as today, so the existing tests and credential-free start keep working. The key never appears in a response, trace or log. If the bundle notes show the Pipe authenticates differently, stop and report instead of guessing.

### 8. Tests (light, offline only)

Add a few fast tests with stubbed retrieval and a stubbed streaming provider. No network, no database. Keep the existing tests passing (update only stubs that targeted `chat_completion` for generation). Cover:

- one `/v1/query` and one streamed chat request each perform exactly one retrieval and one generation run (counted stubs), and the chat stream and the `QueryResult` come from the same stubbed provider reply (the streamed text equals `draft_answer`/`answer` of the result);
- `/v1/query` and chat return the same outcome, confidence, citations and counts for the same stubbed input;
- SSE shape: role chunk, content chunks, stop chunk, `[DONE]`; valid JSON in every `data:` line;
- first attempt valid → `answered`, high confidence, `attempts == 1`, no issues, DRAFT label then footer with sources;
- first attempt invalid (invented label or unsupported claim), second valid → `answered`, `attempts == 2`, the first attempt's check kept in `issues`, retry line in the stream;
- both attempts invalid → `outcome="malformed"`, `answer is None`, `draft_answer` set verbatim, low confidence, all issues kept, invalid label not stripped, stream ends with `DRAFT — low confidence, not the final answer`;
- non-JSON / no answer text → `confidence is None`, no draft, honest outcome;
- provider failure after some text → labelled `UNAVAILABLE` line, stop and `[DONE]` still sent, `outcome="unavailable"`; provider failure before any text → `unavailable`, same closing;
- `insufficient_evidence` and `no_results` make no draft text; `no_results` makes no provider call;
- split JSON pieces and escaped characters in the answer extractor;
- auth: key set → missing/wrong Bearer is 401 on the three routes, correct Bearer works, `/healthz` open; key unset → unchanged;
- no secret (generation key, capstone key, base URL) appears in any result or stream text.

### 9. Documentation

Update (do not overwrite) `docs/architecture.md` with a "Streamed answers and confidence in Open WebUI (Story 3.2)" section: the single event path, what streams and what does not, the one-operation rule, attempt/validation/support checks and their limits, the confidence fields and their meanings, outcome mapping for failed checks and mid-stream provider failure, the rendered Open WebUI labels (`DRAFT — checking evidence`, retry line, footer, `DRAFT — low confidence, not the final answer`, `UNAVAILABLE`), the auth rule, and limitations (draft text cannot be retracted; the support check is coarse; a model can still misread a passage; no legal-applicability claim beyond the documents; non-semantic modes still placeholders). Fix the Story 3.1 limitation lines that say chat is still a placeholder, and the Story 2.3 line likewise, by pointing to this section. Add an Open WebUI usage note: start the capstone API first, open `http://127.0.0.1:8080`, select Building with RAG and a RAG-options `semantic` mode; the setup script is run once only; do not edit the Pipe, Filter or admin settings. Update the README's Open WebUI/Run lines accordingly. Do not touch `docs/manual-tests.md` (it has its own master prompt).

## Token-safe and assistant-safety rules

- **The coding assistant (Claude/Codex) must not test with `curl`, Postman, a browser, or any live HTTP or server call.** Do not start `uvicorn`, do not call the capstone API, the generation provider, Voyage or MongoDB. Verify only with the offline tests in section 8 and `uv run ruff check .`.
- The live smoke checks below are run by the **participant** in the Open WebUI interface (not Postman, which would only repeat Story 3.1's checks). The assistant leaves them as **pending participant**.
- Never print or log `GENERATION_API_KEY`, `CAPSTONE_API_KEY`, the base URL's credentials or the `Authorization` header. Do not print full passage text or the full prompt.

## Out of scope

Agentic or multi-step behavior, tool use, new retrieval modes or generation for modes other than `semantic`, UI plugins, a Pipe/Filter/Open WebUI change, a custom frontend, token-by-token confidence, a second LLM judge call, conversation memory, multi-user authorization, an evaluation harness, broad or load testing, provider fallbacks or retries beyond the bounded validation retry, new `.env` names.

## Completion checks

### A. Assistant (offline only)

1. `uv run ruff check .` passes.
2. `uv run pytest -q` passes, including the new tests above and all earlier tests. Report the pass count.
3. Review of the diff confirms: one retrieval and one generation run per request, `/v1/query` and chat share `run_pattern`, no second chat path, `.env.example` / `data/raw/` / `data/processed/` unchanged, no new settings, no secret in any output.

### B. Participant live smoke checks (run in Open WebUI; the assistant does not run these)

Setup: fill `.env` with `MONGODB_URI`, `VOYAGE_API_KEY`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`, `CAPSTONE_API_KEY` (and optionally `GENERATION_MODEL_NAME`); start the API (`uv run uvicorn building_with_rag.main:app --port 8000`); open `http://127.0.0.1:8080`, select Building with RAG, choose `semantic` in the RAG-options chip. Pace questions (the Voyage key is rate limited).

| # | Do in Open WebUI | Expected |
|---|---|---|
| 1 | Supported answer: ask "What is the punishment for murder under the BNS?" | Text streams in pieces under `DRAFT — checking evidence`; ends with `Evidence check passed — confidence: high`; `Sources:` lists labelled sections (expect `bns:103`); no error text; the reply finishes (no spinner left). |
| 2 | Unsupported: ask "What is the penalty for hacking under the Information Technology Act, 2000?" | Honest insufficient-evidence message; no draft, no confidence, no sources presented as support. |
| 3 | Invalid-output case: with the participant's own deliberate setup that makes validation fail (for example temporarily set `GENERATION_MODEL_NAME` to a weak or non-JSON-following model, or ask a question built to provoke an unsupported answer), then ask the murder question again. Restore `.env` and restart afterwards. | Either a retry line (`Check failed (attempt 1 of 2) … Retrying…`) followed by a second DRAFT and a passed check, or, if both attempts fail, the reply ends with `DRAFT — low confidence, not the final answer` plus a short reason; the draft is never presented as validated. |
| 4 | Stream shape: watch the reply as it is written. | Text arrives progressively (not all at once at the end); retrieval wait is before the first text; the reply always closes. |
| 5 | Auth: with `CAPSTONE_API_KEY` set in `.env` and the API restarted, repeat check 1. | Works as in check 1 (the supplied Pipe is authorised). If the reply is an authentication error, record the exact message and report it; do not edit the Pipe. |
| 6 | Provider failure: temporarily blank `GENERATION_API_KEY`, restart the API, ask the murder question; restore afterwards. | A labelled `UNAVAILABLE` line, no answer, the reply still closes. |

Participant records, for each check: status (pass/fail), outcome shown, confidence shown, source count, and any exact error text.

## Handover

When done, update this story with:

- Files created or changed (including `docs/architecture.md`, `README.md`), and confirmation that `.env.example`, `settings.py` names, `data/raw/` and `data/processed/` are untouched.
- Offline results actually run: `uv run ruff check .` and `uv run pytest -q` with the pass count.
- The named constants chosen (`MAX_ATTEMPTS`, the high/low confidence values, support-check thresholds, provider timeout), the prompt change (answer first), and the exact rendered labels.
- How the single-operation rule is met (the one event generator, how `run_pattern` drains it, what the stream and `QueryResult` share), and the auth behavior with and without `CAPSTONE_API_KEY`.
- Deviations with reasons (for example the Story 3.1 generation tests re-stubbed to the streamed provider; `chat_completion` kept but unused by generation; the bundle notes' presence or absence).
- Live smoke checks 1–6 reported as **pending participant**, one line each, with the expected result copied from the table above. The assistant must not mark them passed or failed, and must state that it ran no curl or live server calls.
- Anything the next story needs to know.

Later stories reuse the same event path and `QueryResult.generation` when other patterns gain generation.

---

## Handover (implementation)

**Files.** Created: `generation/{events,extractor}.py`, `chat_format.py`, `tests/conftest.py`, `tests/test_streaming.py`. Changed: `pipeline.py` (`run_pattern_events`, `run_pattern` drains it), `generation/answer.py` (attempts, validation, `stream_grounded_answer`; `generate_grounded_answer` now drains it), `generation/provider.py` (`stream_chat_completion`), `main.py` (chat adapter, auth), `tests/test_generation.py`, `tests/test_api.py`, `docs/architecture.md`, `README.md`, this story. `.env.example`, `settings.py` names, `contracts.py`, `data/raw/`, `data/processed/` untouched; no new settings.

**Offline results.** `uv run ruff check .` clean; `uv run pytest -q` → 56 passed. No curl, server, provider, Voyage or MongoDB calls were made.

**Constants.** `MAX_ATTEMPTS=2`, `CONFIDENCE_HIGH=0.9`, `CONFIDENCE_LOW=0.2`, `SUPPORT_MIN_WORD_LEN=4`, `SUPPORT_STEM_LEN=5`, `SUPPORT_MIN_SHARE=0.5` (`answer.py`); `REQUEST_TIMEOUT_SECONDS=60` (`provider.py`). Prompt change: `"answer"` is the first key of the JSON template, with a sentence saying so; no rule weakened.

**Labels.** `DRAFT — checking evidence`; `Check failed (attempt n of 2): <detail>. Retrying…`; `Evidence check passed — confidence: high (attempt n of 2). The draft text above is now the validated answer.` + `Sources:`; `DRAFT — low confidence, not the final answer.`; `UNAVAILABLE — the answer service stopped while writing; the text above is incomplete and not validated.` / `UNAVAILABLE — retrieval: <message>`.

**Single operation.** `run_pattern_events` runs retrieval once, then `stream_grounded_answer` once (up to 2 attempts), yielding events and ending with `Final(QueryResult)`. `run_pattern` drains it; chat renders the same events (streaming) or the final result (non-streaming). The streamed text and `QueryResult.generation` come from the same provider replies.

**Auth.** `CAPSTONE_API_KEY` unset/empty: no enforcement. Set: Bearer required on the three routes (401 `detail` on `/v1/query`; OpenAI envelope `invalid_api_key` on chat and models); `/healthz` open.

**Deviations.** Story 3.1 generation tests re-stubbed to the streamed provider (shared stubs moved to `tests/conftest.py`; model-name assertions now read the configured setting because a local `.env` overrides the default). `chat_completion` kept but unused by generation. The DRAFT label is shown lazily at the first text of an attempt so `insufficient_evidence` has no draft label. Open WebUI bundle notes were **not present**; the Pipe's `Authorization: Bearer <CAPSTONE_API_KEY>` behaviour is unverified (smoke check 5 covers it). Story 3.1 live checks remain pending as before.

**Live smoke checks (all pending participant; not run by the assistant).**
1. Supported answer: pending participant. Expected: text streams under `DRAFT — checking evidence`; ends `Evidence check passed — confidence: high`; `Sources:` lists `bns:103`; reply finishes.
2. Unsupported (IT Act hacking): pending participant. Expected: honest insufficient-evidence message; no draft, confidence or sources.
3. Invalid-output case: pending participant. Expected: retry line then second DRAFT and passed check, or `DRAFT — low confidence, not the final answer` with a reason.
4. Stream shape: pending participant. Expected: progressive text after the retrieval wait; the reply always closes.
5. Auth with `CAPSTONE_API_KEY` set: pending participant. Expected: works as check 1; otherwise record the exact error.
6. Provider failure (blank `GENERATION_API_KEY`): pending participant. Expected: labelled `UNAVAILABLE` line, no answer, reply closes.

**For the next story.** Other patterns gain generation by yielding the same events from `run_pattern_events` and reusing `stream_grounded_answer`.
