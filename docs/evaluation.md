# Retrieval Evaluation Guide

## Dataset

`Data/evaluation/golden.jsonl` contains 55 hand-curated cases grounded in the
checked-in `Shopping_Cart_BRD.pdf`. Cases cover direct facts, exact values and
requirement IDs, multi-fact questions, security and performance requirements,
and questions the source does not answer. No Jira issue contents are copied
into the dataset. Unanswerable cases have an empty `relevant_source_ids` list
and an expected behavior of `abstain`.

Answerable cases identify relevant PDF pages with IDs such as
`pdf:Shopping_Cart_BRD.pdf#page=5`. This is intentionally page-level relevance;
chunk IDs are stable for a given content/configuration version, but evaluation
labels are not tied to particular chunk boundaries.

Each JSONL object includes `id`, `question`, `expected_answer`,
`relevant_source_ids`, `answerable`, `category`, `expected_behavior`, and
`notes`; cases with deterministic answer annotations also include
`required_answer_facts`.

## Run and compare

From `Basic_RAG`:

```powershell
.\.venv\Scripts\python.exe -m backend.evaluate
```

The evaluator uses the configured local PDF and Nomic model, creates an
in-memory ChromaDB collection, and writes a JSON report to
`artifacts/rag-evaluation.json`. To change the candidate count, report path,
dataset, or PDF:

```powershell
.\.venv\Scripts\python.exe -m backend.evaluate --candidate-count 20 --report artifacts\run.json
```

It reports Recall@K, Precision@K, MRR@K, and binary nDCG@K at K=1, 3, and 5
for vector-only, keyword-only, and hybrid retrieval, plus per-metric deltas
against vector-only. Unanswerable questions are reported separately and are
not counted in ranking metrics. Retrieval uses fixed settings from the report
and makes no Groq/Jira requests. It does not open or modify
`backend/chroma_db/`.

## Answer checks

Answer generation is not automatic. To avoid unexpected external API costs,
the evaluator accepts an optional JSON file mapping case IDs to answer strings:

```json
{
  "pdf-001": "The Shopping Cart icon on the Inventory page opens the cart. [S1]",
  "pdf-048": "The document does not specify a payment processor."
}
```

```powershell
.\.venv\Scripts\python.exe -m backend.evaluate --answers path\to\answers.json --answer-mode hybrid
```

The deterministic report includes labeled-fact coverage, whether labeled
facts appear in retrieved excerpts, lexical question/answer overlap, citation
validity/coverage, and abstention wording. These checks are transparent
heuristics: lexical overlap is not semantic relevance, and checking labeled
facts cannot detect every unsupported claim. No LLM judge is run. `not
measured` is reported when no answer file is provided.

## Adding cases and interpreting results

Add one JSON object per line. Use source IDs only for pages that genuinely
support the expected answer. Use an empty relevance list for truly
unanswerable questions and mark them `answerable: false`; do not invent facts
to fill gaps. Keep Jira evaluation content out of the committed dataset unless
it is approved, non-sensitive, and safe to distribute.

The delta table compares retrieval modes only. A positive change in one metric
does not prove an overall quality improvement. The current report does not
measure generation latency, answer correctness from a live model, or
per-query performance percentiles. Treat the first measured report as the
baseline and retain its JSON report outside Git if a historical comparison is
needed.
