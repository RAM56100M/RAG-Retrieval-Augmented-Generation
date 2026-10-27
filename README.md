# RAG Explorer

A small React + FastAPI demo that makes a document RAG pipeline visible:

1. Extract text and page numbers from the product requirements PDF in `Data/`.
2. Split the text into overlapping chunks.
3. Generate document and query vectors with Nomic Embed Text v1.5.
4. Persist vectors locally in ChromaDB.
5. Retrieve candidates using dense vectors, local BM25 keyword search, or a
   hybrid rank that merges both lists with Reciprocal Rank Fusion (RRF).
6. Optionally connect Jira Cloud with an Atlassian account email and API token
   to sync issues from every project and Confluence pages from every space that
   account can access.
7. Resolve explicit Jira keys against the local index, apply source/project
   filters, and ask Groq's `openai/gpt-oss-120b` to answer using the retrieved
   excerpts. Citation IDs are validated against the returned sources.

## Run locally

### Backend

From `Basic_RAG` in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Set GROQ_API_KEY in .env
uvicorn backend.main:app --reload
```

The API ingests the PDF when it starts. The first run downloads the embedding
model. ChromaDB data is saved under `backend/chroma_db/`; ingestion state is
available from `http://localhost:8000/api/status`.

The default retrieval mode is `hybrid`. Override it in `.env` with
`RETRIEVAL_MODE=vector`, `keyword`, or `hybrid`. The playground also lets you
select a mode and filter by source, Jira project, issue type, or status. Chunk
size and overlap can be set with `CHUNK_WORDS` and `CHUNK_OVERLAP`; the defaults
remain 420 and 70. The query API validates the filters. Exact Jira issue keys
are looked up only in the local index; questions do not trigger Jira API calls.

## Evaluation and tests

Run backend unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The GitHub Actions workflow runs the Python syntax/unit checks and frontend
production build on pushes and pull requests. The full model-backed retrieval
benchmark is intentionally kept out of fast CI because it may download model
weights and is evaluated separately.

Run the retrieval benchmark:

```powershell
.\.venv\Scripts\python.exe -m backend.evaluate
```

The benchmark uses `Data/evaluation/golden.jsonl`, embeds the configured PDF,
and creates an isolated in-memory ChromaDB collection. It does not read or
modify the production-local index and does not call Groq or Jira. It reports
Recall@K, Precision@K, MRR@K, and binary nDCG@K for vector-only, keyword-only,
and hybrid retrieval, plus deltas relative to vector-only. The first run may
load the embedding model. Its JSON report is written under `artifacts/`, which
is ignored by Git.

To evaluate externally prepared answers without sending data to an LLM, provide
a JSON object mapping evaluation case IDs to answer strings:

```powershell
.\.venv\Scripts\python.exe -m backend.evaluate --answers path\to\answers.json
```

Answer checks are deterministic proxies based on labeled facts, source IDs,
lexical overlap, and abstention wording. They are not a semantic judge and
cannot identify every unsupported claim. No LLM-judge result is produced.

### Optional Atlassian sources

Start the backend and frontend as above, then enter your Atlassian Cloud site
URL, account email, and API token in the **Atlassian sources** card. Create an
[Atlassian API token](https://id.atlassian.com/manage-profile/security/api-tokens)
in your account settings. Use an API token, not your account password. The
account needs permission to browse the Jira projects/issues and Confluence
spaces/pages you want indexed. Confluence indexing is optional; lack of
Confluence access is reported separately and does not prevent Jira syncing.

The token is sent to the local backend over the app's local connection and
stored in the operating system keyring (Windows Credential Manager on Windows);
it is never written to the project `.env` file or browser local storage. After
the token is validated, the backend reads accessible Jira projects/issues and
Confluence spaces/pages using read-only APIs, embeds their content with Nomic,
and stores it alongside PDF vectors in local ChromaDB. Saved credentials are
used to refresh both Atlassian indexes on backend startup. The first sync can
take several minutes because embeddings are generated locally on the CPU;
source cards report progress and indexed counts. Later syncs reuse embeddings
for unchanged issues and pages. **Disconnect** removes the saved Atlassian
credentials but deliberately keeps already indexed Atlassian content in local
ChromaDB. Source filters let you search PDF, Jira, or Confluence content.
Vectors are stored locally; the question and its retrieved excerpts are sent
to Groq to generate an answer.

The local frontend and API are not protected by application login. Keep the
frontend bound to loopback as configured; do not expose it through a public
tunnel without adding authentication and request limits, since visitors could
query indexed Jira/PDF content and use the configured Groq quota.

## Current limitations

- The PDF chunker groups common requirement/numbered headings, but it does not
  parse PDF tables into structured rows and may split oversized sections. The
  evaluation dataset is grounded in the checked-in PDF.
- Retrieval evaluation uses page-level relevance labels and reports search
  ranking, not answer correctness. Answer quality is measured only when a
  supplied answer file is provided.
- No local reranker or LLM-judge evaluation is configured.
- The application has no login, conversation history, feedback persistence,
  or streaming response support.
- PDF IDs now carry a content/configuration version and failed upserts preserve
  the previous records. ChromaDB replacement is not fully atomic, and Jira
  embedding-space migration, background indexing jobs, and detailed sync
  history remain follow-up work.

### Frontend

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL (usually `http://localhost:5173`). Add your Groq API key to
`.env` before asking questions. See `.env.example` for optional model and path
configuration.
