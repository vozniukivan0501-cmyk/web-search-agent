"""Lightweight lexical reranker for filtering search results without heavy ML frameworks."""
import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def lexical_rerank(
    query: str,
    results: List[Dict[str, str]],
    top_k: int = 3
) -> List[Dict[str, str]]:
    """
    Lightweight, fast lexical reranker based on keyword and phrase matching in title and snippet.
    Consumes ~0 MB RAM, avoiding out-of-memory errors on constrained environments (e.g., Render free tier).
    """
    if not results:
        return []

    tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 1]
    if not tokens:
        tokens = query.lower().split()

    clean_query = " ".join(tokens)

    scored = []
    for idx, item in enumerate(results):
        title = item.get("title", "").lower()
        snippet = item.get("snippet", "").lower()

        score = 0.0
        # Exact query match bonuses
        if clean_query and clean_query in title:
            score += 15.0
        elif clean_query and clean_query in snippet:
            score += 8.0

        # Term overlap scoring (title has higher weight)
        for term in tokens:
            if term in title:
                score += 3.0 * title.count(term)
            if term in snippet:
                score += 1.0 * snippet.count(term)

        # Use -idx as tie-breaker to preserve original search engine ranking order for equal scores
        scored.append((score, -idx, item))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [item for _, _, item in scored[:top_k]]


def rerank_results(
    query: str,
    results: List[Dict[str, str]],
    model_name: Optional[str] = None,
    top_k: int = 3
) -> List[Dict[str, str]]:
    """
    Rerank search results to save LLM tokens and select the top relevant pages.
    Uses lightweight lexical reranker to guarantee low memory footprint (~0 MB RAM).
    """
    if not results:
        return []

    reranked = lexical_rerank(query, results, top_k=top_k)
    logger.info(
        "Reranked %d results to top %d using lightweight lexical reranker.",
        len(results), len(reranked)
    )
    return reranked
