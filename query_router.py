import re
from dataclasses import dataclass
from typing import Literal


@dataclass
class SearchIntent:
    semantic_text: str | None
    price_order: Literal["asc", "desc"] | None


ASC_PATTERNS = [
    r"\bcheapest\b",
    r"\blowest(?:\s+price)?\b",
    r"\bleast\s+expensive\b",
    r"\bmost\s+affordable\b",
    r"\baffordable\b",
    r"\bbudget\b",
]

DESC_PATTERNS = [
    r"\bmost\s+expensive\b",
    r"\bpriciest\b",
    r"\bhighest(?:\s+price)?\b",
    r"\bpremium\b",
]


def _strip_patterns(text: str, patterns: list[str]) -> str:
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_search_intent(user_query: str) -> SearchIntent:
    query = user_query.strip()
    all_patterns = ASC_PATTERNS + DESC_PATTERNS

    has_asc = any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in ASC_PATTERNS)
    has_desc = any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in DESC_PATTERNS)

    price_order: Literal["asc", "desc"] | None = None
    if has_asc and not has_desc:
        price_order = "asc"
    elif has_desc and not has_asc:
        price_order = "desc"

    semantic_text = _strip_patterns(query, all_patterns)
    if not semantic_text:
        semantic_text = None

    return SearchIntent(semantic_text=semantic_text, price_order=price_order)
