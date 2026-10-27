import re
from collections.abc import Iterable
from typing import Any

from backend.retrieval import retrieval_metrics, source_id_from_metadata, tokenize


_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")
_ABSTENTION_PATTERN = re.compile(
    r"\b(?:not specified|does not specify|doesn't specify|not established|"
    r"cannot determine|can't determine|insufficient evidence|not enough evidence)\b",
    re.IGNORECASE,
)
_STOP_WORDS = {
    "about",
    "after",
    "are",
    "does",
    "from",
    "have",
    "how",
    "into",
    "the",
    "their",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def evaluate_retrieval_cases(
    cases: list[dict[str, Any]],
    results_by_case: dict[str, list[dict[str, Any]]],
    *,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    answerable_cases = [case for case in cases if case.get("answerable") is True]
    metrics_by_k: dict[str, dict[str, float]] = {}
    for k in ks:
        per_case = [
            retrieval_metrics(
                results_by_case[case["id"]],
                set(case.get("relevant_source_ids") or []),
                k=k,
            )
            for case in answerable_cases
            if case.get("relevant_source_ids")
        ]
        metrics_by_k[str(k)] = {
            metric: (
                sum(result[f"{metric}@{k}"] for result in per_case) / len(per_case)
                if per_case
                else 0.0
            )
            for metric in ("recall", "precision", "mrr", "ndcg")
        }
    return {
        "answerable_case_count": len(answerable_cases),
        "unanswerable_case_count": len(cases) - len(answerable_cases),
        "metrics_at_k": metrics_by_k,
    }


def evaluate_answer_cases(
    cases: list[dict[str, Any]],
    answers: dict[str, str],
    results_by_case: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    measured_cases: list[dict[str, float | None]] = []
    skipped_case_ids: list[str] = []
    for case in cases:
        answer = answers.get(case["id"])
        if answer is None:
            skipped_case_ids.append(case["id"])
            continue

        answer_folded = answer.casefold()
        results = results_by_case.get(case["id"], [])
        valid_source_ids = {
            source_id_from_metadata(result.get("metadata") or {})
            for result in results
        }
        source_id_by_citation = {
            f"S{rank}": source_id_from_metadata(result.get("metadata") or {})
            for rank, result in enumerate(results, start=1)
        }
        citations = [match.group(1).upper() for match in _CITATION_PATTERN.finditer(answer)]
        valid_citations = sum(
            citation in source_id_by_citation
            and source_id_by_citation[citation] in valid_source_ids
            for citation in citations
        )
        required_facts = case.get("required_answer_facts") or []
        present_facts = [
            fact for fact in required_facts if str(fact).casefold() in answer_folded
        ]
        retrieved_text = " ".join(
            str(result.get("text", "")) for result in results
        ).casefold()
        supported_facts = [
            fact
            for fact in required_facts
            if str(fact).casefold() in answer_folded
            and str(fact).casefold() in retrieved_text
        ]
        question_terms = {
            token
            for token in tokenize(case["question"])
            if token not in _STOP_WORDS and len(token) > 2
        }
        answer_terms = set(tokenize(answer))
        relevance = (
            len(question_terms & answer_terms) / len(question_terms)
            if question_terms
            else 0.0
        )
        answerable = case.get("answerable") is True
        measured_cases.append(
            {
                "fact_coverage": (
                    len(present_facts) / len(required_facts)
                    if required_facts
                    else None
                ),
                "labeled_fact_support": (
                    len(supported_facts) / len(present_facts)
                    if present_facts
                    else None
                ),
                "lexical_relevance": relevance,
                "citation_validity": (
                    (valid_citations / len(citations) if citations else 0.0)
                    if answerable else None
                ),
                "citation_coverage": (
                    1.0 if citations else (0.0 if answerable else 1.0)
                ),
                "appropriate_abstention": (
                    float(bool(_ABSTENTION_PATTERN.search(answer)))
                    if not answerable
                    else None
                ),
            }
        )

    metric_names = (
        "fact_coverage",
        "labeled_fact_support",
        "lexical_relevance",
        "citation_validity",
        "citation_coverage",
        "appropriate_abstention",
    )
    return {
        "measured_case_count": len(measured_cases),
        "skipped_case_ids": skipped_case_ids,
        "deterministic_metrics": {
            name: (
                sum(value for value in values if value is not None)
                / len([value for value in values if value is not None])
                if (values := [result[name] for result in measured_cases if result[name] is not None])
                else None
            )
            for name in metric_names
        },
        "limitations": [
            "Labeled fact support checks only dataset facts; it cannot detect every unsupported claim.",
            "Lexical relevance is a token-overlap indicator, not a semantic judge.",
            "No LLM-judge score is produced.",
        ],
    }


def load_answer_file(path: str) -> dict[str, str]:
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in data.items()
    ):
        raise ValueError("Answer file must be a JSON object mapping case IDs to strings.")
    return data


def merge_retrieval_results(
    cases: Iterable[dict[str, Any]],
    results_by_case: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    return {case["id"]: results_by_case.get(case["id"], []) for case in cases}
