"""Cross-Encoder reranker for filtering search results and saving tokens."""
from sentence_transformers import CrossEncoder
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

_cross_encoder_instance: CrossEncoder | None = None


def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """Lazy-load and cache the cross-encoder model."""
    global _cross_encoder_instance
    if _cross_encoder_instance is None:
        logger.info(f"Loading Cross-Encoder model: {model_name}")
        _cross_encoder_instance = CrossEncoder(model_name)
        logger.info("Cross-Encoder model loaded successfully.")
    return _cross_encoder_instance


def rerank_results(
    query: str,
    results: List[Dict[str, str]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 3
) -> List[Dict[str, str]]:
    """Rerank search results using Cross-Encoder and return top_k most relevant.
    
    This saves tokens by filtering out irrelevant results before sending to LLM.
    Each result dict should have 'title', 'url', 'snippet' keys.
    """
    if not results:
        return []
    
    if len(results) <= top_k:
        return results
    
    model = get_cross_encoder(model_name)
    
    # Build query-document pairs for scoring
    pairs = []
    for r in results:
        doc_text = f"{r.get('title', '')}. {r.get('snippet', '')}"
        pairs.append((query, doc_text))
    
    # Get relevance scores
    scores = model.predict(pairs)
    
    # Sort by score descending and take top_k
    scored_results: List[Tuple[float, Dict[str, str]]] = [
        (float(score), result) for score, result in zip(scores, results)
    ]
    scored_results.sort(key=lambda x: x[0], reverse=True)
    
    reranked = [item[1] for item in scored_results[:top_k]]
    
    logger.info(
        f"Reranked {len(results)} results to top {top_k}. "
        f"Score range: {scored_results[0][0]:.3f} - {scored_results[-1][0]:.3f}"
    )
    
    return reranked
