# RAG Explorer: Baseline Assessment and Improvement Plan

> **Status update (2026-10-27):** The implementation items listed in this
> historical baseline assessment have since been completed to the extent
> described in the current [README](../README.md) and
> [architecture](./architecture.md). The app now has vector, keyword, and
> hybrid retrieval; an isolated PDF evaluation runner; metadata filters and
> citation validation; and optional read-only Jira plus Confluence indexing.
> The baseline description below records the state before that work and is not
> a description of the current application. Current checks: 29 backend unit
> tests and the frontend production build passed. No live Atlassian sync was
> performed for the Confluence implementation.

## Repository baseline

The application currently consists of a React/Vite frontend and a single FastAPI
backend. The working tree already contains uncommitted Jira authentication,
synchronization, UI, and documentation changes. Those changes predate this
improvement pass and must be preserved.

Baseline checks performed:

| Command | Result |
|---|---|
| `npm --prefix D:\RAG\Basic_RAG\frontend run build` | Passed (`tsc -b` and Vite production build). |
| `.venv\Scripts\python.exe -m compileall -q backend` | Passed. |
| `.venv\Scripts\python.exe -c "import fastapi, chromadb, pymupdf, sentence_transformers, groq"` | Passed. |
| `.venv\Scripts\python.exe -m pytest --version` | Not available; pytest is not installed. |
| Existing automated tests | No test files or test runner configuration were found. |
| Frontend lint/type-check scripts | No lint script is configured; the production build performs the TypeScript build check. |

Two initial baseline command attempts ran from the workspace parent instead of
`Basic_RAG` and failed to locate the project files. They were rerun with
absolute project paths; the results above are from the corrected commands.

## Current architecture

### Request and retrieval flow

`frontend/src/App.tsx` posts a question to `/api/query`. `backend/main.py`
embeds it with the Nomic query instruction `search_query:`, requests up to four
nearest chunks from a persistent ChromaDB collection configured for cosine
distance, and sends the question plus excerpts to Groq. Document vectors use
the matching `search_document:` instruction and normalized embeddings.

The API currently returns the answer and retrieved chunks. The model is asked
to cite PDF pages and Jira issue keys, but citation text is not validated
against the actual retrieved results. Retrieval is dense-vector-only; there is
no keyword index, reranker, metadata-filter API, or retrieval evaluation
command.

### PDF indexing and chunking

At startup, PyMuPDF extracts text page by page. `split_pages` divides each
page's whitespace-separated words into fixed 420-word chunks with 70-word
overlap. Chunks retain page and sequence numbers, but not stable content IDs,
section headings, document hashes, or parser/configuration versions. Ingestion
generates all embeddings before replacing existing PDF records, but deletes
the old records before upserting new ones; a storage failure at that point can
leave the PDF index incomplete.

### Jira synchronization

`backend/atlassian.py` validates Jira Cloud credentials, stores them in the OS
keyring, and uses read-only Jira REST calls. Issues are indexed in batches and
the connector prunes stale issues only after successful pagination. Existing
issue chunks are reused when timestamps and a subset of summary/project
metadata match. The flow has progress reporting and disconnect preserves local
Jira chunks. Retry/backoff, sync history, and detailed change counts are not
implemented. Integration with a live Jira account is not part of this
baseline.

### Storage, prompts, errors, and security-sensitive flows

PDF and Jira chunks share one persistent ChromaDB collection. PDF records use
page/chunk metadata; Jira records include issue key, project, summary, URL, and
updated timestamp. The generation prompt says retrieved content is untrusted
reference data, but sources are passed as plain text and answer citations are
not backend-validated.

Jira credentials travel from the UI to the local API and are stored by the
keyring connector. The frontend does not use browser storage for credentials.
The API CORS allowlist is limited to the local Vite origins; the README
instructs users to keep the application local. Retrieved excerpts and the
question are sent to Groq for answer generation. The `.env` file was not read
or printed during this assessment.

## Gaps and compatibility risks

1. There is no version-controlled evaluation dataset or repeatable retrieval
   benchmark, so retrieval changes cannot currently be compared.
2. Dense-only retrieval can miss exact requirement IDs, Jira keys, and rare
   technical tokens.
3. Similarity scores are presented as `1 - cosine distance`; they are not
   calibrated answerability probabilities.
4. Chunk IDs are position-based and the chunker discards paragraph, heading,
   and table structure.
5. PDF replacement is not failure-atomic, and the shared collection has no
   index-version migration strategy.
6. Answer citations are model-generated text and are not checked against the
   retrieved passages.
7. The API has basic question length validation, but no retrieval filters,
   request-level timing diagnostics, or structured error categories.
8. No backend or frontend unit/integration test suite is present. Pytest and a
   frontend test runner are not installed/configured.
9. Existing local ChromaDB data and Jira credentials are user data. Tests and
   evaluations must use isolated in-memory/temporary indexes and mocked
   external services.

## Implementation sequence

1. Add isolated, deterministic retrieval utilities and tests, then introduce
   configurable vector-only, keyword-only, and hybrid retrieval without
   changing the embedding model or Nomic encoding instructions.
2. Add a PDF-grounded evaluation dataset and a repeatable, read-only/isolated
   evaluation command with retrieval metrics and machine-readable reports.
3. Add exact local Jira-key lookup, validated metadata filters, structured
   result ranks, and request diagnostics where they fit the existing API.
4. Improve chunk structure and safe re-indexing only after tests capture the
   current parser/index behavior and the migration path is proven against an
   isolated collection.
5. Add citation validation, abstention tests, and source-focused UI only when
   the response contract can be maintained compatibly.
6. Extend Jira sync observability/recovery and add optional reranking only when
   measurements justify the added complexity.
7. Add any remaining CI, security/privacy, and architecture documentation
   based on implemented and verified behavior.

The evaluation corpus will use facts from the checked-in requirements PDF only
unless locally available Jira data can be evaluated without copying sensitive
issue content into version control. No live Jira write operations or mutations
of the production-local ChromaDB index are permitted during these stages.
