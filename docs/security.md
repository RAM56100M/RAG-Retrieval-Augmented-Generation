# Security and Privacy Notes

## Credentials and local API

Atlassian Cloud credentials are submitted to the local FastAPI backend and persisted
using the operating-system keyring. The frontend clears the API token after
the connection attempt and does not use browser local/session storage for it.
The token is not intended for `.env` or application logs. Disconnect removes
saved credentials while retaining locally indexed Jira issues and Confluence pages.

The Vite development server binds to loopback. Keep the API local as well;
there is no application login or user authorization. CORS is restricted to the
local Vite origins, but CORS is not authentication. Do not expose the app
through a public tunnel or bind it to a shared network without adding
authentication, authorization, request limits, and deployment hardening.

## Data sent to external services

PDF text, Jira issue data, and accessible Confluence page data are indexed
locally in ChromaDB. Atlassian API calls are read-only and use the connected
account's permissions; a Confluence permission failure does not block Jira
indexing. When a question is submitted, the question and the retrieved excerpts
are sent to Groq for answer generation. The app is therefore local-first, not
fully offline. Embedding model downloads may also contact the model host when
the model is not cached.

Queries use the local index and do not automatically make Jira or Confluence
network requests.

## Prompt injection and citations

Retrieved PDF/Jira/Confluence content is untrusted data. The generation prompt tells the
model not to follow instructions in source text, disclose secrets, or claim
actions it did not perform. The model has no shell, filesystem, or arbitrary
tool access. Source citations are request-local IDs, and the backend removes
IDs and page/issue citation-looking markers that do not map to the actual
retrieved result set.

Prompt wording is a defense-in-depth measure, not a guarantee that a model
cannot produce unsupported prose. Citation validation establishes that a
referenced source was retrieved; it does not prove that every sentence is
entailed by that source.

## Logs, errors, and data lifecycle

The query path does not intentionally log prompts or retrieved document
contents. Groq failures return a generic error category rather than an SDK
exception string. Avoid adding raw credentials, authorization headers, full
document bodies, or full Jira issue descriptions to logs.

Disconnecting Atlassian preserves its locally indexed issues and pages. The application does
not currently provide a dedicated control to delete the Jira index, delete the
PDF index, or clear all derived local data. Re-indexing behavior and
transactional preservation of the active PDF index remain limitations; back
up important local data before manual index maintenance.

The evaluation runner uses a separate in-memory ChromaDB collection and makes
no writes to the production-local index. Generated evaluation reports are
written under the Git-ignored `artifacts/` directory.
