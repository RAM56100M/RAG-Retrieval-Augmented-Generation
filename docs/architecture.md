# RAG Explorer Architecture

## Ingestion and storage

The FastAPI application extracts page text from the configured PDF with
PyMuPDF. The chunker groups text under recognized requirement and numbered
headings, then uses configurable windows for oversized sections (default 420
words, 70-word overlap). Each chunk retains its page and section heading. The
Nomic Embed Text v1.5 model receives
`search_document:` for source chunks and `search_query:` for questions; both
vectors are normalized. PDF, Jira, and Confluence chunks are kept in the same
persistent ChromaDB collection using cosine distance.

PDF chunk IDs incorporate a hash of the PDF content and indexing configuration.
Metadata records the document hash, index version, parser/chunker versions,
embedding model, page, and chunk. New records are staged before stale PDF
records are removed; a failed upsert cleans up newly staged IDs and preserves
prior records. ChromaDB does not provide an atomic transaction across the
upsert and stale-record deletion, so a failed deletion can still require
operator inspection.

Jira Cloud access uses read-only REST API calls. The connector discovers
projects, pages through issue search results, indexes batches locally, and
prunes stale issues only after successful enumeration. Credentials are stored
through the operating-system keyring. Disconnecting removes credentials but
does not delete local Jira vectors.

Confluence pages are read from accessible spaces using the same read-only
Atlassian credentials. Page storage-format markup is converted to text and
indexed with stable page/chunk IDs, page and space metadata, content hashes,
and a same-site page URL. Page traversal is paginated, same-site links are
validated, and stale page cleanup occurs only after a complete traversal.
Confluence sync errors are reported separately so a lack of Confluence access
does not invalidate a successful Jira sync.

## Retrieval

The query endpoint supports:

- **Vector:** Nomic query embedding followed by ChromaDB cosine retrieval.
- **Keyword:** local BM25 over indexed chunk text. Tokenization preserves
  identifiers such as `PROJ-123`, `FR-CART-001`, `api/v2`, and configuration
  names.
- **Hybrid:** vector and BM25 candidate lists are merged with Reciprocal Rank
  Fusion. It does not compare the raw score scales of the two systems.

Candidate count, result count, default mode, chunk size/overlap, and RRF rank
constant are validated from environment configuration. A request may override
the retrieval mode and filter by PDF/Jira/Confluence source type, Jira project,
issue type, or status. Jira chunks include labels and available issue-link
relationships. A recognizable Jira issue key gets an exact local metadata
lookup before ordinary retrieval; there is no per-question Jira network
request. Each result includes source metadata and its vector/keyword rank when
available. Similarity values are not answerability probabilities.

## Generation, citations, and diagnostics

The backend sends the question and retrieved excerpts to Groq. The system
prompt explicitly treats the excerpts as untrusted evidence, not instructions,
and asks for abstention when the context does not establish an answer. Each
source is assigned a request-local citation ID (`[S1]`, `[S2]`, ...). The API
removes citation-looking page/issue references and source IDs that were not
retrieved, and returns only citations resolved against actual result objects.

The response includes the retrieval mode, candidate counts, request ID, model
names, and retrieval/generation/total timings. The frontend displays those
observable diagnostics and the actual retrieved excerpts, with links to Jira
issues and Confluence pages. It does not display hidden model reasoning.

## Remote frontend boundary

The Vite frontend reads its backend base URL from `VITE_API_BASE_URL`; its
development proxy is only a local convenience. For Vercel, host the static
build independently and route browser requests through a Cloudflare Tunnel to
FastAPI on the laptop. FastAPI remains bound to loopback, and ChromaDB, PDF
files, embedding model, and Atlassian keyring credentials remain local.

`FRONTEND_ORIGINS` is parsed as a comma-separated list of exact HTTPS origins
(plus the two loopback development origins). Wildcards, paths, credentials,
and non-HTTPS remote origins are rejected. The backend does not enable CORS
credentials. When `RAG_ACCESS_PASSWORD` is configured, a public health route
is the only unauthenticated API endpoint other than token issue; other API
routes require an in-memory, short-lived bearer session. Login/query/mutation
requests are rate-limited and declared API bodies over 1 MB are rejected.
These controls provide a simple single-user gate, not multi-user identity,
per-project authorization, or distributed abuse protection.

The frontend prompts for the access password and retains the issued session
token in memory only. Neither the password nor token is a Vercel environment
variable or embedded in the build. Vercel previews require their exact origins
to be explicitly added to backend CORS configuration. CORS does not protect
the API from direct non-browser clients; bearer authentication is required
for remote use.

## Evaluation

`backend.evaluate` embeds the configured PDF into an in-memory ephemeral
ChromaDB collection and compares vector, keyword, and hybrid retrieval against
page-level relevance labels in `Data/evaluation/golden.jsonl`. It never uses
the production collection and does not call Groq or Jira. See
[`evaluation.md`](./evaluation.md) for metrics and limitations.

## Current boundaries

The chunker recognizes common requirement IDs and numbered headings, but it
does not parse PDF tables into structured rows and can split oversized
sections. PDF metadata records an index version, but there is no
active-version pointer and Atlassian embedding-space migrations are not
isolated.
There is no reranker, query rewriting, conversation history, streaming,
feedback persistence, or background-job manager. The shared collection does
not yet isolate PDF and Atlassian migrations. Jira/Confluence sync retry and
backoff, sync history, and a background-job manager are not implemented.
