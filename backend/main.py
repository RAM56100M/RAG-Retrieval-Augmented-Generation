import asyncio
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import chromadb
import fitz
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PDF_PATH = Path(os.getenv("PDF_PATH", "Data/Shopping_Cart_BRD.pdf"))
if not PDF_PATH.is_absolute():
    PDF_PATH = ROOT / PDF_PATH
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "backend/chroma_db"))
if not CHROMA_PATH.is_absolute():
    CHROMA_PATH = ROOT / CHROMA_PATH
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
COLLECTION_NAME = "vwo_product_requirements"
CHUNK_WORDS = 420
CHUNK_OVERLAP = 70

status: dict[str, Any] = {
    "stage": "starting",
    "message": "Preparing the document pipeline",
    "file_name": PDF_PATH.name,
    "page_count": 0,
    "chunk_count": 0,
    "embedding_model": EMBEDDING_MODEL,
    "vector_store": "ChromaDB",
    "error": None,
}
status_lock = threading.Lock()
pipeline_lock = threading.Lock()
model: SentenceTransformer | None = None
collection: Any = None
client: chromadb.PersistentClient | None = None


def set_status(stage: str, message: str, **extra: Any) -> None:
    with status_lock:
        status.update(stage=stage, message=message, **extra)


def find_pdf() -> Path:
    if PDF_PATH.is_file():
        return PDF_PATH
    data_dir = ROOT / "Data"
    candidates = sorted(data_dir.glob("*.pdf")) if data_dir.is_dir() else []
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No PDF found at {PDF_PATH} or in the Data folder. Add the source PDF and restart."
    )


def split_pages(pages: list[tuple[int, str]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for page_number, text in pages:
        words = text.split()
        start = 0
        while start < len(words):
            end = min(start + CHUNK_WORDS, len(words))
            chunk_text = " ".join(words[start:end]).strip()
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number,
                        "chunk": len(chunks) + 1,
                    }
                )
            if end == len(words):
                break
            start = end - CHUNK_OVERLAP
    return chunks


def get_model() -> SentenceTransformer:
    global model
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    return model


def ingest_document() -> None:
    global client, collection
    with pipeline_lock:
        try:
            set_status("reading", "Reading and extracting PDF text", error=None)
            source = find_pdf()
            with fitz.open(source) as document:
                page_count = len(document)
                pages = [
                    (page_number, page.get_text("text"))
                    for page_number, page in enumerate(document, start=1)
                ]

            chunks = split_pages(pages)
            if not chunks:
                raise ValueError("The PDF contains no extractable text.")
            set_status(
                "chunking",
                f"Split {page_count} pages into {len(chunks)} overlapping chunks",
                file_name=source.name,
                page_count=page_count,
                chunk_count=len(chunks),
            )

            set_status("embedding", f"Generating vectors with {EMBEDDING_MODEL}")
            embedder = get_model()
            vectors = embedder.encode(
                [f"search_document: {chunk['text']}" for chunk in chunks],
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            set_status("storing", "Saving vectors to the local ChromaDB collection")
            CHROMA_PATH.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(CHROMA_PATH))
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            existing = collection.get(where={"source": source.name}, include=[])
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
            collection.add(
                ids=[f"chunk-{index:04d}" for index in range(len(chunks))],
                documents=[chunk["text"] for chunk in chunks],
                embeddings=vectors,
                metadatas=[
                    {
                        "source": source.name,
                        "page": chunk["page"],
                        "chunk": chunk["chunk"],
                    }
                    for chunk in chunks
                ],
            )
            set_status(
                "ready",
                "Document embedded and stored. Ready for questions.",
                file_name=source.name,
                page_count=page_count,
                chunk_count=len(chunks),
                error=None,
            )
        except Exception as exc:
            set_status("error", "Document ingestion failed", error=str(exc))
            raise


def retrieve(query: str) -> list[dict[str, Any]]:
    if collection is None:
        raise HTTPException(status_code=409, detail="The document is not ingested yet.")
    query_vector = get_model().encode(
        [f"search_query: {query}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(4, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        {
            "text": text,
            "page": metadata["page"],
            "chunk": metadata["chunk"],
            "source": metadata["source"],
            "score": round(max(0.0, 1.0 - distance), 4),
        }
        for text, metadata, distance in zip(documents, metadatas, distances)
    ]


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(asyncio.to_thread(ingest_document))
    yield
    if not task.done():
        task.cancel()


app = FastAPI(title="RAG Explorer API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    with status_lock:
        return dict(status)


@app.post("/api/ingest")
async def start_ingestion() -> dict[str, Any]:
    try:
        await asyncio.to_thread(ingest_document)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return get_status()


@app.post("/api/query")
def ask_question(request: QueryRequest) -> dict[str, Any]:
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="Set GROQ_API_KEY in the project .env file to generate answers.",
        )
    chunks = retrieve(request.question.strip())
    if not chunks:
        raise HTTPException(status_code=409, detail="No document chunks are available.")

    evidence = "\n\n".join(
        f"[Page {chunk['page']}, chunk {chunk['chunk']}]\n{chunk['text']}"
        for chunk in chunks
    )
    try:
        response = Groq(api_key=os.environ["GROQ_API_KEY"]).chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question using only the supplied document "
                        "excerpts. If the excerpts do not contain the answer, say so. "
                        "Cite supporting information using its [Page N] label."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Document excerpts:\n{evidence}\n\nQuestion: {request.question}",
                },
            ],
            temperature=0.2,
            max_tokens=1200,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Groq answer generation failed: {exc}") from exc

    return {
        "answer": response.choices[0].message.content,
        "model": GROQ_MODEL,
        "chunks": chunks,
    }
