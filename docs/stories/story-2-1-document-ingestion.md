# Story 2.1 — Document ingestion

Status: ready for implementation

## Purpose

Turn the supplied BNS and IPC PDFs into a small, inspectable, section-level JSONL corpus: one record per legal section, one file per act. This is the evidence base that later stories chunk, embed and retrieve from.

**Who runs what.** The implementer (the coding assistant) only writes the extraction script and the documentation. The assistant must **not** run the extraction and must not generate the JSONL files itself. The learner runs the script and produces `data/processed/*.jsonl`, then does the completion checks. The script is `src/building_with_rag/ingest.py`; if it already exists, review and improve it rather than replacing it.

This story only produces and checks that corpus. MongoDB, embeddings, vector indexes, retrieval and answer generation belong to later stories. Do not start them.

## Prerequisites

- Story 1.1 is complete. Reuse its application (`src/building_with_rag/`), its documents (`docs/architecture.md`, Story 1.1) and the existing `.env.example`. Do not add or rename settings; this story needs none.
- No `docs/config.yaml` existed when this story was written, so use the repository root. If one now names a project path, use that path.
- Supplied inputs, which must be preserved untouched:
  - `data/raw/BNS_2023_bare_act.pdf`
  - `data/raw/IPC_1860_bare_act.pdf`
  - `data/raw/PROVENANCE.md`
- Python 3.12 and `uv`.
- Before starting, read `docs/architecture.md`, Story 1.1, the existing code, and the three input files. Preserve existing work.
- If any input is missing, unreadable, or its SHA-256 does not match `PROVENANCE.md`, stop and report exactly which one. Do not substitute another source or invent provenance.

## Work to do

### 1. Choose and add a PDF parser

Pick a suitable PDF text-extraction library and add it with `uv add`, so `pyproject.toml` and `uv.lock` are updated. Prefer a plain-text extractor that gives page-level text and keeps reading order. Do not add OCR, database or embedding dependencies. Record the chosen package and version; these become `parser` and `parser_version` in each record.

### 2. Write the extraction command

Add a small module inside the existing package (`src/building_with_rag/ingest.py`) that the learner runs with one documented `uv run` command (`uv run python -m building_with_rag.ingest`). Keep it simple: no framework, no new settings. Write it, but do not execute it.

It must, for each act:

- Read the PDF from `data/raw/`, compute its SHA-256, and compare it with `PROVENANCE.md`.
- Extract the section text from the body of the act and split it into sections without rewriting their meaning. Keep the legal wording as printed.
- Detect chapter number and chapter title and attach them to every section in that chapter.
- Handle these conservatively:
  - **Repeated page headers, footers and page numbers:** remove only text that clearly repeats on many pages.
  - **Spacing and line breaks:** normalise whitespace only. Do not paraphrase, reorder or "correct" legal text.
  - **Multi-page sections:** join text across pages into one record.
  - **Obvious duplicate extraction** (the same section captured twice): keep one record.
- Ignore front matter that is not section text. The BNS PDF contains an index and a BNS-to-IPC correspondence table; the IPC PDF contains amendment footnotes. These must not become section records, and should not leak into section text as duplicate sections.
- **Flag, don't repair:** if a section is uncertain (unclear boundary, odd heading, unexpected number sequence, missing or garbled text, scanned/image-only pages), still emit the record where possible with `needs_review: true`. Never silently fix it. If a page has no extractable text, report it.

### 3. Output records

Write exactly these files, one JSON object per line, UTF-8:

- `data/processed/bns_sections.jsonl`
- `data/processed/ipc_sections.jsonl`

Every record has these fields, with exactly these value rules:

| Field | Rule |
|---|---|
| `section_id` | `bns:<number>` for BNS, `ipc:<number>` for IPC (lower-case prefix) |
| `act` | `BNS_2023` or `IPC_1860` |
| `act_label` | human-readable act name as printed in the source (for example "Bharatiya Nyaya Sanhita, 2023"); do not invent |
| `status` | `in_force` for BNS, `repealed` for IPC |
| `chapter` | string, for example `I` |
| `chapter_title` | chapter title as printed |
| `section_number` | integer |
| `heading` | section heading as printed |
| `text` | section text, never empty |
| `source_pdf` | file name of the source PDF in `data/raw/` |
| `source_sha256` | SHA-256 of that PDF |
| `parser` | parser package name |
| `parser_version` | parser package version |
| `source_status_version` | taken from `PROVENANCE.md` (the supplied status and retrieval information for that file). Use it as supplied; if it cannot be determined, do not guess: set `null` and `needs_review: true` |
| `needs_review` | boolean |

Page references (for example `page_start`, `page_end`) may be added as extra fields, but they do not replace any field above. Do not invent any provenance that is not in the inputs or computed from them.

Rules for the corpus as a whole:

- BNS and IPC records stay in separate files and are never merged, even where section numbers match (`bns:103` and `ipc:103` are different sections).
- Sections with lettered numbers in the source (for example 153A) must not be coerced into a wrong integer. If one cannot satisfy `section_number` as an integer without losing information, flag it with `needs_review: true` and report it rather than dropping or renumbering it.
- Do not overwrite a supplied valid corpus unnecessarily. If a processed file already exists and its records all carry the current `source_sha256`, leave it alone.
- If the source hash differs from the hash in existing records, regenerate that act's file **as one replacement**: write to a temporary file in the same folder, then replace the old file in a single step. Never append to or mix records from different source versions.
- Re-running must not append duplicates. Records are a function of the PDF, not of run history.

### 4. Update `docs/architecture.md`

Update (do not overwrite) the architecture file with a short section recording: the parser choice and version, the output format (JSONL, fields as above, file locations), the extraction command, and known limitations (for example flagged sections, footnote handling, anything the parser gets wrong).

### 5. Later chunking (to be explained, not built)

In the handover, explain in a few sentences how later chunking will keep the parent section and source information: each chunk must carry its parent `section_id`, `act`, `status`, `section_number`, `chapter`, `heading`, `source_pdf` and `source_sha256`, so any retrieved passage can be traced to one section of one act in one source file. Do not implement chunking here.

## Out of scope

MongoDB, embeddings, vector indexes, retrieval, answer generation, a comprehensive corpus-validation framework, reference-difference reports, broad security tests, synthetic permission records, modifying anything in `data/raw/`.

## Completion checks

Keep checks small and directly relevant. The **learner** runs each one and records the actual command and result. The assistant must not run the extraction; it may only tell the learner what to run and what to look for.

1. The extraction command runs and writes both JSONL files.
2. Every record in both files has all required fields with the specified types and values (`act`, `status`, `section_id` format, integer `section_number`, string `chapter`), and no `text` is empty.
3. Inspect representative records against each PDF: at least a few sections per act, including one section that spans pages, if present. Compare text and heading with the PDF and note the pages used.
4. List any records flagged `needs_review`, and any missing input or unextractable pages.
5. Run the extraction a second time. Confirm the files are not changed, no duplicate `section_id` appears, and record counts are identical.
6. `data/raw/` (PDFs and `PROVENANCE.md`) is unchanged; checksums still match `PROVENANCE.md`.
7. Existing Story 1.1 checks still pass (`uv run ruff check .`, `uv run pytest -q`).

Do not build a larger test suite for this story.

## Handover

The assistant's handover, after writing the script and docs, lists the files created and the exact command for the learner to run. The learner then updates this story with the results below.

When the learner has run the script, update this story with:

- **Inputs:** the exact files read, with their verified hashes.
- **Output:** the files written, with record counts per act.
- **Files changed or created**, including `pyproject.toml`/`uv.lock` and `docs/architecture.md`.
- **Parser:** the package and version chosen, and why.
- **Commands run** and their actual results, including the second run.
- **What was inspected** and the outcome, including any `needs_review` records and known limitations.
- **Chunking note:** how later chunking will preserve the parent section and source information (see Work to do, item 5).
- **Deviations** from this story, with the reason.

Later stories (2.2 onward) import the two JSONL files as they are and extend them additively, without renaming fields.
