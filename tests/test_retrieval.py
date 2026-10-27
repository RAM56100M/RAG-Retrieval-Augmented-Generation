import unittest
import math

from backend.citations import resolve_citations
from backend.evaluation import evaluate_answer_cases, evaluate_retrieval_cases
from backend.retrieval import (
    keyword_search,
    reciprocal_rank_fusion,
    retrieval_metrics,
    source_id_from_metadata,
    tokenize,
)


class RetrievalTests(unittest.TestCase):
    def test_tokenize_preserves_jira_and_requirement_identifiers(self) -> None:
        tokens = tokenize("Fix PROJ-123 and FR-CART-001; check api/v2.")
        self.assertIn("proj-123", tokens)
        self.assertIn("fr-cart-001", tokens)
        self.assertIn("api/v2", tokens)

    def test_keyword_search_ranks_exact_identifier_deterministically(self) -> None:
        documents = [
            {"id": "b", "text": "Generic checkout behavior"},
            {"id": "a", "text": "FR-CART-001 opens the shopping cart"},
            {"id": "c", "text": "FR-CART-002 lists products"},
        ]

        results = keyword_search("FR-CART-001", documents, limit=3)

        self.assertEqual(results[0]["id"], "a")
        self.assertEqual(results[0]["keyword_rank"], 1)
        self.assertEqual([item["id"] for item in results], ["a"])

    def test_keyword_search_handles_empty_query_and_nonpositive_limit(self) -> None:
        documents = [{"id": "a", "text": "requirements"}]
        self.assertEqual(keyword_search("   ", documents, limit=3), [])
        self.assertEqual(keyword_search("requirements", documents, limit=0), [])

    def test_rrf_deduplicates_channels_and_retains_ranks(self) -> None:
        vector = [
            {"id": "a", "text": "a", "vector_rank": 1, "score": 0.8},
            {"id": "b", "text": "b", "vector_rank": 2, "score": 0.7},
        ]
        keyword = [
            {"id": "b", "text": "b", "keyword_rank": 1, "keyword_score": 2.0},
            {"id": "c", "text": "c", "keyword_rank": 2, "keyword_score": 1.0},
        ]

        results = reciprocal_rank_fusion(vector, keyword, limit=3)

        self.assertEqual([item["id"] for item in results], ["b", "a", "c"])
        self.assertEqual(results[0]["vector_rank"], 2)
        self.assertEqual(results[0]["keyword_rank"], 1)
        self.assertEqual(len({item["id"] for item in results}), 3)

    def test_rrf_rejects_invalid_rank_constant(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], [], limit=3, rank_constant=0)

    def test_source_id_uses_page_and_issue_metadata(self) -> None:
        self.assertEqual(
            source_id_from_metadata({"source_type": "pdf", "source": "brd.pdf", "page": 5}),
            "pdf:brd.pdf#page=5",
        )
        self.assertEqual(
            source_id_from_metadata({"source_type": "jira", "issue_key": "PROJ-123"}),
            "jira:PROJ-123",
        )
        self.assertEqual(
            source_id_from_metadata(
                {"source_type": "confluence", "page_id": "12345"}
            ),
            "confluence:12345",
        )

    def test_metrics_calculate_recall_precision_mrr_and_ndcg(self) -> None:
        results = [
            {"metadata": {"source_type": "pdf", "source": "brd.pdf", "page": 2}},
            {"metadata": {"source_type": "pdf", "source": "brd.pdf", "page": 5}},
        ]
        metrics = retrieval_metrics(
            results,
            {"pdf:brd.pdf#page=5", "pdf:brd.pdf#page=7"},
            k=2,
        )
        self.assertEqual(metrics["recall@2"], 0.5)
        self.assertEqual(metrics["precision@2"], 0.5)
        self.assertEqual(metrics["mrr@2"], 0.5)
        expected_ndcg = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(metrics["ndcg@2"], expected_ndcg)

    def test_metrics_return_zero_for_unanswerable_reference_set(self) -> None:
        metrics = retrieval_metrics([{"metadata": {}}], set(), k=1)
        self.assertEqual(metrics["recall@1"], 0.0)
        self.assertEqual(metrics["precision@1"], 0.0)
        self.assertEqual(metrics["mrr@1"], 0.0)

    def test_page_level_metrics_do_not_count_duplicate_chunks_multiple_times(self) -> None:
        same_page = {
            "metadata": {"source_type": "pdf", "source": "brd.pdf", "page": 5}
        }
        metrics = retrieval_metrics(
            [same_page, same_page],
            {"pdf:brd.pdf#page=5"},
            k=2,
        )

        self.assertEqual(metrics["recall@2"], 1.0)
        self.assertEqual(metrics["precision@2"], 0.5)

    def test_retrieval_case_summary_excludes_unanswerable_questions(self) -> None:
        cases = [
            {
                "id": "known",
                "answerable": True,
                "relevant_source_ids": ["pdf:brd.pdf#page=5"],
            },
            {"id": "unknown", "answerable": False, "relevant_source_ids": []},
        ]
        results = {
            "known": [
                {"metadata": {"source_type": "pdf", "source": "brd.pdf", "page": 5}}
            ],
            "unknown": [],
        }

        report = evaluate_retrieval_cases(cases, results)

        self.assertEqual(report["answerable_case_count"], 1)
        self.assertEqual(report["unanswerable_case_count"], 1)
        self.assertEqual(report["metrics_at_k"]["1"]["recall"], 1.0)

    def test_citation_resolution_removes_unretrieved_source_references(self) -> None:
        sources = [
            {
                "citation_id": "S1",
                "source_id": "pdf:brd.pdf#page=5",
                "page": 5,
            }
        ]

        answer, citations = resolve_citations(
            "The cart opens from Inventory [S1]. Unsupported [S9] [Page 99].",
            sources,
        )

        self.assertEqual(answer, "The cart opens from Inventory [S1]. Unsupported  .")
        self.assertEqual([item["citation_id"] for item in citations], ["S1"])

    def test_answer_evaluation_checks_facts_support_citations_and_abstention(self) -> None:
        cases = [
            {
                "id": "known",
                "question": "What is the quantity?",
                "answerable": True,
                "required_answer_facts": ["fixed at 1"],
            },
            {
                "id": "unknown",
                "question": "Which payment gateway is used?",
                "answerable": False,
                "required_answer_facts": [],
            },
        ]
        results = {
            "known": [
                {
                    "text": "Quantity is fixed at 1.",
                    "metadata": {"source_type": "pdf", "source": "brd.pdf", "page": 6},
                }
            ],
            "unknown": [],
        }
        answers = {
            "known": "Quantity is fixed at 1 [S1].",
            "unknown": "The document does not specify the payment gateway.",
        }

        report = evaluate_answer_cases(cases, answers, results)
        metrics = report["deterministic_metrics"]

        self.assertEqual(metrics["fact_coverage"], 1.0)
        self.assertEqual(metrics["labeled_fact_support"], 1.0)
        self.assertEqual(metrics["citation_validity"], 1.0)
        self.assertEqual(metrics["appropriate_abstention"], 1.0)


if __name__ == "__main__":
    unittest.main()
