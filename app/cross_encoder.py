"""Cross-Encoder reranker for filtering search results and saving LLM tokens."""
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

_cross_encoder_instance = None
_model_failed = False


def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    Lazy-load and cache the cross-encoder model.
    Optimized for low-memory environments by restricting PyTorch execution threads.
    """
    global _cross_encoder_instance, _model_failed
    if _model_failed:
        return None

    if _cross_encoder_instance is None:
        try:
            import torch
            # Restrict PyTorch to a single thread to avoid memory spikes on small instances
            torch.set_num_threads(1)
            from sentence_transformers import CrossEncoder

            logger.info("Loading Cross-Encoder model: %s", model_name)
            _cross_encoder_instance = CrossEncoder(model_name)
            logger.info("Cross-Encoder model loaded successfully.")
        except Exception as e:
            logger.warning(
                "Could not load CrossEncoder model due to memory/system limits (%s). "
                "Falling back to lightweight lexical reranking.",
                e
            )
            _model_failed = True
            return None

    return _cross_encoder_instance


def lexical_rerank(
    query: str,
    results: List[Dict[str, str]],
    top_k: int = 3
) -> List[Dict[str, str]]:
    """
    Lightweight fallback reranker based on keyword relevance in title and snippet.
    Used when memory limits prevent transformer execution.
    """
    terms = set(query.lower().split())

    def compute_score(item: Dict[str, str]) -> float:
        content = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
        score = sum(content.count(term) * 2 if term in item.get('title', '').lower() else content.count(term) for term in terms)
        return float(score)

    scored = sorted(results, key=compute_score, reverse=True)
    return scored[:top_k]


def rerank_results(
    query: str,
    results: List[Dict[str, str]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 3
) -> List[Dict[str, str]]:
    """
    Rerank search results using Cross-Encoder or fallback heuristic to save prompt tokens.
    Returns the top_k most relevant result dictionaries.
    """
    if not results:
        return []

    if len(results) <= top_k:
        return results

    model = get_cross_encoder(model_name)
    if model is None:
        logger.info("Using lexical fallback reranking for %d search results.", len(results))
        return lexical_rerank(query, results, top_k=top_k)

    try:
        # Construct query-document pairs for relevance scoring
        pairs = []
        for r in results:
            doc_text = f"{r.get('title', '')}. {r.get('snippet', '')}"
            pairs.append((query, doc_text))

        scores = model.predict(pairs)

        scored_results: List[Tuple[float, Dict[str, str]]] = [
            (float(score), result) for score, result in zip(scores, results)
        ]
        scored_results.sort(key=lambda x: x[0], reverse=True)

        reranked = [item[1] for item in scored_results[:top_k]]
        logger.info(
            "Reranked %d results to top %d. Score range: %.3f - %.3f",
            len(results), top_k, scored_results[0][0], scored_results[-1][0]
        )
        return reranked

    except Exception as e:
        logger.warning(
            "CrossEncoder prediction failed (%s). Falling back to lexical ranking.",
            e
        )
        return lexical_rerank(query, results, top_k=top_k)
