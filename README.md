# RAG Explorer

A small React + FastAPI demo that makes a document RAG pipeline visible:

1. Extract text and page numbers from the product requirements PDF in `Data/`.
2. Split the text into overlapping chunks.
3. Generate document and query vectors with Nomic Embed Text v1.5.
4. Persist vectors locally in ChromaDB.
5. Retrieve the four most relevant chunks for a question.
6. Ask Groq's `openai/gpt-oss-120b` to answer using those excerpts.

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
