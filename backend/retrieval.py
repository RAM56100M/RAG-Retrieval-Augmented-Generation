import math
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+(?:[-./:][A-Za-z0-9_]+)*")


def tokenize(text: str) -> list[str]:
    """Keep Jira keys, requirement IDs, and technical names intact."""
    return [token.casefold() for token in _TOKEN_PATTERN.findall(text)]


def source_id_from_metadata(metadata: dict[str, Any]) -> str:
    source_type = metadata.get("source_type", "pdf")
    if source_type == "jira":
        return f"jira:{metadata.get('issue_key') or metadata.get('source', '')}"
    if source_type == "confluence":
        return f"confluence:{metadata.get('page_id', '')}"
    return f"pdf:{metadata.get('source', '')}#page={metadata.get('page', '')}"


def keyword_search(
    query: str,
    documents: Iterable[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Return BM25-ranked documents with stable, deterministic ordering."""
    if limit <= 0:
        return []

    query_terms = set(tokenize(query))
    if not query_terms:
        return []

    corpus: list[tuple[dict[str, Any], list[str]]] = []
    document_frequency: Counter[str] = Counter()
    for document in documents:
        tokens = tokenize(str(document.get("text", "")))
        corpus.append((document, tokens))
        document_frequency.update(set(tokens))

    if not corpus:
        return []

    average_length = sum(len(tokens) for _, tokens in corpus) / len(corpus) or 1.0
    k1 = 1.5
    b = 0.75
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for document, tokens in corpus:
        frequencies = Counter(tokens)
        length_norm = k1 * (1 - b + b * len(tokens) / average_length)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            doc_freq = document_frequency[term]
            inverse_frequency = math.log(
                1 + (len(corpus) - doc_freq + 0.5) / (doc_freq + 0.5)
            )
            score += (
                inverse_frequency
                * frequency
                * (k1 + 1)
                / (frequency + length_norm)
            )
        if score > 0:
            stable_id = str(document.get("id", ""))
            ranked.append((score, stable_id, document))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [
        {
            **document,
            "keyword_score": score,
            "keyword_rank": rank,
        }
        for rank, (score, _, document) in enumerate(ranked[:limit], start=1)
    ]


def reciprocal_rank_fusion(
    vector_results: Iterable[dict[str, Any]],
    keyword_results: Iterable[dict[str, Any]],
    limit: int,
    *,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    if rank_constant <= 0:
        raise ValueError("rank_constant must be greater than zero")

    fused: dict[str, dict[str, Any]] = {}
    for channel, results in (
        ("vector", vector_results),
        ("keyword", keyword_results),
    ):
        rank_key = f"{channel}_rank"
        for position, result in enumerate(results, start=1):
            identifier = str(
                result.get("id")
                or result.get("chunk_id")
                or source_id_from_metadata(result.get("metadata") or {})
                + f"#{result.get('text', '')}"
            )
            record = fused.setdefault(identifier, {**result, "id": identifier})
            rank = result.get(rank_key, position)
            record[rank_key] = int(rank)
            record[f"{channel}_score"] = result.get(
                f"{channel}_score",
                result.get("score") if channel == "vector" else None,
            )
            record["rrf_score"] = record.get("rrf_score", 0.0) + (
                1.0 / (rank_constant + int(rank))
            )

    sorted_results = sorted(
        fused.values(),
        key=lambda result: (
            -result["rrf_score"],
            result.get("vector_rank", math.inf),
            result.get("keyword_rank", math.inf),
            str(result["id"]),
        ),
    )
    return [
        {**result, "rank": rank}
        for rank, result in enumerate(sorted_results[:limit], start=1)
    ]


def retrieval_metrics(
    results: Iterable[dict[str, Any]],
    relevant_source_ids: set[str],
    *,
    k: int,
) -> dict[str, float]:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    ranked_results: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for result in results:
        source_id = source_id_from_metadata(result.get("metadata") or {})
        if source_id in seen_sources:
            continue
        seen_sources.add(source_id)
        ranked_results.append(result)
        if len(ranked_results) == k:
            break
    relevant_count = len(relevant_source_ids)
    if relevant_count == 0:
        return {
            f"recall@{k}": 0.0,
            f"precision@{k}": 0.0,
            f"mrr@{k}": 0.0,
            f"ndcg@{k}": 0.0,
        }

    relevance = [
        source_id_from_metadata(result.get("metadata") or {}) in relevant_source_ids
        for result in ranked_results
    ]
    relevant_retrieved = sum(relevance)
    reciprocal_rank = next(
        (1.0 / rank for rank, is_relevant in enumerate(relevance, start=1) if is_relevant),
        0.0,
    )
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, is_relevant in enumerate(relevance, start=1)
        if is_relevant
    )
    ideal_relevant = min(relevant_count, k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    return {
        f"recall@{k}": relevant_retrieved / relevant_count,
        f"precision@{k}": relevant_retrieved / k,
        f"mrr@{k}": reciprocal_rank,
        f"ndcg@{k}": dcg / ideal_dcg if ideal_dcg else 0.0,
    }
