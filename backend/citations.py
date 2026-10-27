import re
from typing import Any


_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")
_SOURCE_LIKE_PATTERN = re.compile(
    r"\[(?:S\d+|Page\s+\d+|[A-Z][A-Z0-9_]*-\d+)\]",
    re.IGNORECASE,
)


def resolve_citations(
    answer: str,
    sources: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Keep only citation IDs assigned to sources retrieved for this response."""
    by_id = {
        str(source["citation_id"]): source
        for source in sources
        if source.get("citation_id")
    }
    cited_ids: list[str] = []
    for match in _CITATION_PATTERN.finditer(answer):
        citation_id = match.group(1).upper()
        if citation_id in by_id and citation_id not in cited_ids:
            cited_ids.append(citation_id)

    def replace_match(match: re.Match[str]) -> str:
        token = match.group(0)
        citation_id = token[1:-1].upper()
        if citation_id.startswith("S") and citation_id in by_id:
            return f"[{citation_id}]"
        return ""

    sanitized = _SOURCE_LIKE_PATTERN.sub(replace_match, answer).strip()
    citations = [
        {"citation_id": citation_id, **by_id[citation_id]}
        for citation_id in cited_ids
    ]
    return sanitized, citations
