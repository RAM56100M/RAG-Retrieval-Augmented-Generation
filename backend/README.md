# RAG Explorer API

The API reads the product requirements PDF from `Data/`, splits page text into
overlapping chunks, embeds them with Nomic Embed Text v1.5, and persists the
vectors in `backend/chroma_db/`. Ingestion starts automatically when the API
starts. The React app can also trigger a re-ingestion.

## Run

From the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Add your Groq API key to .env
uvicorn backend.main:app --reload
```

The first ingestion downloads the Nomic embedding model and may take a few
minutes. The API reports progress at `http://localhost:8000/api/status`.
