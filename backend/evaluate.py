import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
import pymupdf

from backend.evaluation import (
    evaluate_answer_cases,
    evaluate_retrieval_cases,
    load_answer_file,
)
from backend.main import (
    CHUNK_OVERLAP,
    CHUNK_WORDS,
    EMBEDDING_MODEL,
    PDF_PATH,
    get_model,
    split_pdf_pages,
)
from backend.retrieval import keyword_search, reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "Data" / "evaluation" / "golden.jsonl"
DEFAULT_REPORT = ROOT / "artifacts" / "rag-evaluation.json"


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid evaluation JSON at line {line_number}.") from exc
        required_fields = {
            "id",
            "question",
            "expected_answer",
            "relevant_source_ids",
            "answerable",
            "category",
            "expected_behavior",
        }
        if not isinstance(case, dict) or not required_fields.issubset(case):
            raise ValueError(
                f"Evaluation case on line {line_number} is missing required fields."
            )
        if not isinstance(case["relevant_source_ids"], list):
            raise ValueError(
                f"Evaluation case on line {line_number} has invalid source IDs."
            )
        cases.append(case)
    if not cases:
        raise ValueError("The evaluation dataset contains no cases.")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Evaluation case IDs must be unique.")
    return cases


def read_pdf_chunks(path: Path) -> list[dict[str, Any]]:
    with pymupdf.open(path) as document:
        pages = [
            (page_number, page.get_text("text"))
            for page_number, page in enumerate(document, start=1)
        ]
    chunks = split_pdf_pages(pages)
    return [
        {
            "id": f"eval-pdf-{index:05d}",
            "text": chunk["text"],
            "metadata": {
                "source_type": "pdf",
                "source": path.name,
                "page": chunk["page"],
                "chunk": chunk["chunk"],
            },
        }
        for index, chunk in enumerate(chunks, start=1)
    ]


def evaluate(
    *,
    dataset_path: Path,
    pdf_path: Path,
    candidate_count: int,
    answers_path: Path | None,
    answer_mode: str,
) -> dict[str, Any]:
    cases = load_cases(dataset_path)
    chunks = read_pdf_chunks(pdf_path)
    if not chunks:
        raise ValueError("The PDF contains no extractable text to evaluate.")
    with pymupdf.open(pdf_path) as document:
        page_count = len(document)

    model = get_model()
    document_vectors = model.encode(
        [f"search_document: {chunk['text']}" for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
    query_vectors = model.encode(
        [f"search_query: {case['question']}" for case in cases],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    client = chromadb.EphemeralClient()
    collection_name = f"rag_eval_{hashlib.sha256(str(dataset_path).encode()).hexdigest()[:12]}"
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    try:
        collection.upsert(
            ids=[chunk["id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            embeddings=document_vectors,
            metadatas=[chunk["metadata"] for chunk in chunks],
        )
        stored = collection.get(include=["documents", "metadatas"])
        keyword_documents = [
            {"id": identifier, "text": text, "metadata": metadata}
            for identifier, text, metadata in zip(
                stored["ids"],
                stored["documents"] or [],
                stored["metadatas"] or [],
            )
        ]
        results_by_mode: dict[str, dict[str, list[dict[str, Any]]]] = {
            "vector": {},
            "keyword": {},
            "hybrid": {},
        }

        for case, query_vector in zip(cases, query_vectors):
            vector_response = collection.query(
                query_embeddings=[query_vector],
                n_results=min(candidate_count, len(chunks)),
                include=["documents", "metadatas", "distances"],
            )
            vector_results = [
                {
                    "id": identifier,
                    "text": text,
                    "metadata": metadata,
                    "distance": distance,
                    "score": round(max(0.0, 1.0 - distance), 4),
                    "vector_rank": rank,
                }
                for rank, (identifier, text, metadata, distance) in enumerate(
                    zip(
                        vector_response["ids"][0],
                        vector_response["documents"][0],
                        vector_response["metadatas"][0],
                        vector_response["distances"][0],
                    ),
                    start=1,
                )
            ]
            keyword_results = keyword_search(
                case["question"],
                keyword_documents,
                limit=candidate_count,
            )
            results_by_mode["vector"][case["id"]] = vector_results
            results_by_mode["keyword"][case["id"]] = keyword_results
            results_by_mode["hybrid"][case["id"]] = reciprocal_rank_fusion(
                vector_results,
                keyword_results,
                limit=candidate_count,
            )

        mode_reports = {
            mode: evaluate_retrieval_cases(cases, results)
            for mode, results in results_by_mode.items()
        }
        comparisons = {
            mode: {
                k: {
                    metric: round(
                        metrics[metric] - mode_reports["vector"]["metrics_at_k"][k][metric],
                        6,
                    )
                    for metric in ("recall", "precision", "mrr", "ndcg")
                }
                for k, metrics in report["metrics_at_k"].items()
            }
            for mode, report in mode_reports.items()
            if mode != "vector"
        }

        answer_report = {
            "status": "not_measured",
            "reason": "No answer file supplied; retrieval evaluation does not call Groq.",
        }
        if answers_path:
            answers = load_answer_file(str(answers_path))
            answer_report = evaluate_answer_cases(
                cases,
                answers,
                results_by_mode[answer_mode],
            )
            answer_report["status"] = "measured_from_supplied_answers"
            answer_report["answer_mode"] = answer_mode

        dataset_bytes = dataset_path.read_bytes()
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset": {
                "path": str(dataset_path.relative_to(ROOT))
                if dataset_path.is_relative_to(ROOT)
                else dataset_path.name,
                "sha256": hashlib.sha256(dataset_bytes).hexdigest(),
                "case_count": len(cases),
            },
            "corpus": {
                "pdf_name": pdf_path.name,
                "chunk_count": len(chunks),
                "page_count": page_count,
            },
            "configuration": {
                "embedding_model": EMBEDDING_MODEL,
                "document_instruction": "search_document:",
                "query_instruction": "search_query:",
                "chunk_words": CHUNK_WORDS,
                "chunk_overlap_words": CHUNK_OVERLAP,
                "chunker_version": "heading-aware-v1",
                "vector_metric": "cosine",
                "candidate_count": candidate_count,
                "result_limit": candidate_count,
                "generation_model": "not used",
                "index": "isolated ChromaDB ephemeral collection",
            },
            "retrieval": mode_reports,
            "delta_vs_vector_baseline": comparisons,
            "answer_quality": answer_report,
            "errors": [],
        }
    finally:
        client.delete_collection(collection_name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate PDF retrieval without touching the production ChromaDB index."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--answers", type=Path)
    parser.add_argument(
        "--answer-mode",
        choices=("vector", "keyword", "hybrid"),
        default="hybrid",
        help="Retrieval results used to check supplied answer annotations.",
    )
    args = parser.parse_args()
    if args.candidate_count <= 0:
        parser.error("--candidate-count must be positive.")
    if not args.dataset.is_file():
        parser.error(f"Evaluation dataset not found: {args.dataset}")
    if not args.pdf.is_file():
        parser.error(f"PDF not found: {args.pdf}")
    if args.answers and not args.answers.is_file():
        parser.error(f"Answer file not found: {args.answers}")

    try:
        report = evaluate(
            dataset_path=args.dataset,
            pdf_path=args.pdf,
            candidate_count=args.candidate_count,
            answers_path=args.answers,
            answer_mode=args.answer_mode,
        )
    except Exception as exc:
        print(
            f"RAG evaluation failed during setup or execution ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Evaluation report: {args.report}")
    print(f"Dataset cases: {report['dataset']['case_count']}")
    print(f"PDF chunks: {report['corpus']['chunk_count']}")
    print("Retrieval metrics (answerable cases only):")
    for mode, result in report["retrieval"].items():
        metrics = result["metrics_at_k"].get("5", {})
        print(
            f"  {mode:7} "
            f"Recall@5={metrics.get('recall', 0):.3f} "
            f"Precision@5={metrics.get('precision', 0):.3f} "
            f"MRR@5={metrics.get('mrr', 0):.3f} "
            f"nDCG@5={metrics.get('ndcg', 0):.3f}"
        )
    if report["answer_quality"]["status"] == "not_measured":
        print("Answer quality: not measured (no answer file supplied).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
