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

## Publish the frontend to Vercel; keep the backend local

The React/Vite frontend can be deployed as a static Vercel project. FastAPI,
the PDF files, the Nomic model cache, Jira credentials, and ChromaDB remain on
your Windows laptop. An ngrok tunnel provides HTTPS connectivity from the
public website to FastAPI without router port forwarding. The laptop must stay
powered on, awake, and connected to the internet for queries to work.

### Configure the local backend

In the project-root `.env` (never commit this file), set:

```dotenv
GROQ_API_KEY=your-server-side-Groq-key
FRONTEND_ORIGINS=https://your-project.vercel.app
RAG_ACCESS_PASSWORD=replace-with-a-random-password-at-least-16-characters
RAG_ACCESS_TOKEN_TTL_SECONDS=3600
```

Set `FRONTEND_ORIGINS` to the exact HTTPS origin shown by Vercel, without a
path or trailing slash. If using a custom frontend domain, list that exact
origin instead. Multiple explicitly allowed origins may be comma-separated.
The backend validates the origins and refuses a remote origin unless a
password of at least 16 characters is configured. Generate a password locally
with:

```powershell
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
```

Paste the generated value into the laptop's `.env`; do not put it in Vercel or
in any `VITE_*` setting. The frontend prompts for this password at runtime.
FastAPI issues a short-lived bearer session, kept only in the open browser tab.
It is not saved in local storage; refresh/reopen requires signing in again.
Signing out revokes the session. Keep the existing Atlassian API token in the
OS keyring and `GROQ_API_KEY` in the backend `.env`.

The API protects status, query, indexing, Jira connect/sync/disconnect, and
other `/api` endpoints with the bearer session whenever a backend password is
configured. It also limits login, query, mutation, and read request rates,
rejects request bodies larger than 1 MB when a content length is supplied, and
validates API payload fields. `/api/health` is intentionally public and returns
only service availability and whether authentication is enabled. CORS allows
only the local development origins plus exact origins in
`FRONTEND_ORIGINS`; credentialed CORS and wildcard origins are disabled.
CORS is a browser policy, **not authentication**.

### Set up ngrok on Windows

Install the ngrok agent from the
[official Windows download page](https://ngrok.com/downloads/windows), create
an ngrok account, and copy your auth token from its dashboard. In PowerShell,
save the token to ngrok's local user configuration:

```powershell
ngrok config add-authtoken YOUR_NGROK_AUTHTOKEN
```

Keep the auth token private; do not commit it or put it in Vercel. Start the
backend as described below, then open a second PowerShell window and run:

```powershell
ngrok http 8000
```

Use the HTTPS forwarding address ngrok prints as the API base in Vercel, with
`/api` appended. Free ngrok URLs can change when restarted; update
`VITE_API_BASE_URL` in Vercel and `FRONTEND_ORIGINS` in the laptop `.env` to
match, then redeploy the frontend and restart the backend.

### Start and stop FastAPI

From `Basic_RAG` in PowerShell, after configuring the `.env`:

```powershell
.\scripts\Start-RagServices.ps1
```

This starts FastAPI bound to `127.0.0.1:8000` and checks `/api/health`. Keep
ngrok running in its separate PowerShell window. Logs and process IDs are
written to the Git-ignored `.run/` folder.

Stop only the processes started by the launcher with:

```powershell
.\scripts\Stop-RagServices.ps1
```

Run the start command again after a Windows reboot. Startup checks the PDF
index version and reuses existing vectors when the same PDF and indexing
configuration are already indexed; it does not clear or reset ChromaDB. Avoid
the **Re-index PDF** action unless you intend to index the source again.

### Deploy the frontend to Vercel

1. Import `RAM56100M/RAG-Retrieval-Augmented-Generation` into Vercel.
2. Set **Root Directory** to `frontend` (the Git repository root is
   `Basic_RAG/`).
3. Use the Vite framework preset, install command `npm install`, build command
   `npm run build`, and output directory `dist`.
4. Add the production environment variable
   `VITE_API_BASE_URL=https://YOUR-NGROK-HOST/api`, replacing the example with
   the HTTPS forwarding hostname shown by ngrok.
5. Deploy. Copy the exact Vercel production origin, such as
   `https://your-project.vercel.app`, into `FRONTEND_ORIGINS` in the laptop's
   `.env`, then restart the backend.

`VITE_API_BASE_URL` is public browser configuration; it must contain only the
API URL, never a password or API key. The frontend has no client-side routes, so
no custom Vercel rewrite or serverless API function is required. Vite's local
development proxy remains available when the variable is unset. Redeploying
the static frontend does not deploy or modify the laptop's backend, source PDF,
keyring credentials, or ChromaDB index.

Vercel preview deployments use different origins. They are blocked by CORS
unless their exact HTTPS origins are added to `FRONTEND_ORIGINS` and the backend
is restarted. Prefer testing with the production domain; do not allow broad
Vercel wildcard origins.

### Troubleshooting and security boundaries

- **Backend unavailable / tunnel disconnected:** keep the laptop awake; check
  `http://127.0.0.1:8000/api/health`, then inspect `.run\backend-error.log` or
  the ngrok terminal.
- **Browser CORS error:** verify that `FRONTEND_ORIGINS` exactly matches the
  deployed scheme and hostname, and restart FastAPI after editing `.env`.
- **401 / sign-in expired:** enter the backend access password again. Sessions
  expire after the configured TTL and are invalidated when FastAPI restarts.
- **Vercel 404 or API URL errors:** confirm the frontend root directory and
  `VITE_API_BASE_URL` includes the `/api` suffix, then redeploy.
- **No results:** confirm the local backend reports ready and the source index
  exists; errors are not fixed by rebuilding the frontend.

The browser-to-backend session is a single shared-password, single-user access
gate, not per-user authorization. Anyone who learns the password can query the
local PDF/Jira/Confluence index and use the configured Groq quota. Rate limits
are in-memory best-effort protection, not a distributed abuse-prevention
system. Do not publish the password, enable untrusted preview origins, or use
this as a multi-user/production service. Cloudflare Tunnel avoids direct router
port forwarding, but the backend is still internet-reachable through its
authenticated tunnel hostname.

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
