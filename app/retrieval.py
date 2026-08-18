"""Retrieval against Azure AI Search: hybrid (BM25 + vector) with optional
semantic reranking."""

from __future__ import annotations

import logging

from azure.search.documents.models import VectorizedQuery

from app.azure_clients import embed_texts, get_search_client
from app.config import get_settings
from app.index_setup import SEMANTIC_CONFIG_NAME

logger = logging.getLogger(__name__)


def retrieve(question: str, top_k: int | None = None, source: str | None = None) -> list[dict]:
    """Return the top_k chunks for a question, best first.

    Hybrid search is the default because pure vector search missed exact-token
    queries (error codes, policy names) that BM25 nails, while pure keyword
    search missed paraphrased questions.
    """
    settings = get_settings()
    k = top_k or settings.top_k

    query_vector = embed_texts([question])[0]
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=max(k * 2, 10),
        fields="content_vector",
    )

    kwargs: dict = {
        "search_text": question,
        "vector_queries": [vector_query],
        "select": ["id", "content", "source", "title", "chunk_index"],
        "top": k,
    }
    if source:
        kwargs["filter"] = f"source eq '{source}'"
    if settings.use_semantic_ranker:
        kwargs["query_type"] = "semantic"
        kwargs["semantic_configuration_name"] = SEMANTIC_CONFIG_NAME

    try:
        results = get_search_client().search(**kwargs)
        hits = [_to_hit(r) for r in results]
    except Exception:
        if not settings.use_semantic_ranker:
            raise
        # Semantic ranking is tier-gated; fall back to plain hybrid rather than
        # failing the request.
        logger.warning("semantic ranking failed, falling back to hybrid", exc_info=True)
        kwargs.pop("query_type", None)
        kwargs.pop("semantic_configuration_name", None)
        hits = [_to_hit(r) for r in get_search_client().search(**kwargs)]

    return [h for h in hits if h["score"] >= settings.min_score][:k]


def _to_hit(result) -> dict:
    # The semantic reranker score is on a different (0-4) scale than the hybrid
    # RRF score, so prefer it when present and keep both out of the same bucket.
    score = result.get("@search.reranker_score") or result.get("@search.score") or 0.0
    return {
        "content": result["content"],
        "source": result["source"],
        "title": result.get("title") or result["source"],
        "chunk_index": result.get("chunk_index", 0),
        "score": float(score),
    }
