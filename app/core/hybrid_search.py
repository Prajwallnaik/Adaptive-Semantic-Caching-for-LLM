"""
Hybrid search with Reciprocal Rank Fusion (RRF).

Combines dense (semantic) and sparse (BM25-style) retrieval results from
Qdrant into a single ranked list. This is the core of what makes the cache
"semantic but safe":

  - Dense search catches paraphrases and meaning-equivalent queries
  - Sparse search catches exact terms, numbers, dates, and entities that
    dense embeddings blur (e.g. "Q1 2024" vs "Q1 2023")

Fusion via RRF is preferred over raw score combination because dense and
sparse scores live on different scales and are not directly comparable.
RRF is rank-based, so it's scale-agnostic.
"""

import logging
from dataclasses import dataclass, field

from qdrant_client.http.models import Filter

from app.config import settings
from app.cache.store import search_dense, search_sparse
from app.core.embeddings import embed_query, extract_sparse_terms, normalize_text
from app.cache.store import terms_to_sparse_vector
from app.core.metadata_filter import build_filter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single fused search result with payload and score."""
    point_id: str
    score: float  # Fused RRF score (or weighted score)
    query_text: str
    answer: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    payload: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

# RRF constant k — controls how much lower-ranked results are penalized.
# k=60 is the standard value from the original RRF paper (Cormack et al., 2009).
_RRF_K = 60


def _reciprocal_rank_fusion(
    dense_results: list,
    sparse_results: list,
    k: int = _RRF_K,
) -> list[SearchResult]:
    """
    Fuse dense and sparse result lists using Reciprocal Rank Fusion.

    RRF score for each document = sum of 1/(k + rank) across all result lists
    where the document appears. Rank is 1-indexed.

    This is scale-agnostic — it doesn't matter that dense cosine scores are
    in [0, 1] while sparse BM25 scores can be any positive float.

    Args:
        dense_results: Scored points from dense vector search.
        sparse_results: Scored points from sparse vector search.
        k: RRF constant (default 60).

    Returns:
        List of SearchResult sorted by fused score (descending).
    """
    # Collect scores and payloads by point ID
    fused: dict[str, dict] = {}

    # Process dense results
    for rank, point in enumerate(dense_results, start=1):
        pid = str(point.id)
        if pid not in fused:
            fused[pid] = {
                "score": 0.0,
                "dense_score": 0.0,
                "sparse_score": 0.0,
                "payload": point.payload or {},
            }
        rrf_contribution = 1.0 / (k + rank)
        fused[pid]["score"] += rrf_contribution
        fused[pid]["dense_score"] = point.score

    # Process sparse results
    for rank, point in enumerate(sparse_results, start=1):
        pid = str(point.id)
        if pid not in fused:
            fused[pid] = {
                "score": 0.0,
                "dense_score": 0.0,
                "sparse_score": 0.0,
                "payload": point.payload or {},
            }
        rrf_contribution = 1.0 / (k + rank)
        fused[pid]["score"] += rrf_contribution
        fused[pid]["sparse_score"] = point.score

    # Build sorted results
    results = [
        SearchResult(
            point_id=pid,
            score=data["score"],
            query_text=data["payload"].get("query_text", ""),
            answer=data["payload"].get("answer", ""),
            dense_score=data["dense_score"],
            sparse_score=data["sparse_score"],
            payload=data["payload"],
        )
        for pid, data in fused.items()
    ]

    results.sort(key=lambda r: r.score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Weighted linear combination (alternative fusion)
# ---------------------------------------------------------------------------

def _weighted_fusion(
    dense_results: list,
    sparse_results: list,
    alpha: float = 0.7,
) -> list[SearchResult]:
    """
    Fuse dense and sparse results via weighted linear combination.

    score = alpha * dense_score + (1 - alpha) * sparse_score

    Less robust than RRF because it requires scores to be on comparable scales,
    but can work well when the dense/sparse score distributions are understood.

    Args:
        dense_results: Scored points from dense vector search.
        sparse_results: Scored points from sparse vector search.
        alpha: Weight for dense scores (0-1). Default 0.7 favors semantic match.

    Returns:
        List of SearchResult sorted by fused score (descending).
    """
    scores: dict[str, dict] = {}

    for point in dense_results:
        pid = str(point.id)
        scores[pid] = {
            "dense_score": point.score,
            "sparse_score": 0.0,
            "payload": point.payload or {},
        }

    for point in sparse_results:
        pid = str(point.id)
        if pid not in scores:
            scores[pid] = {
                "dense_score": 0.0,
                "sparse_score": 0.0,
                "payload": point.payload or {},
            }
        scores[pid]["sparse_score"] = point.score

    results = []
    for pid, data in scores.items():
        fused_score = alpha * data["dense_score"] + (1 - alpha) * data["sparse_score"]
        results.append(
            SearchResult(
                point_id=pid,
                score=fused_score,
                query_text=data["payload"].get("query_text", ""),
                answer=data["payload"].get("answer", ""),
                dense_score=data["dense_score"],
                sparse_score=data["sparse_score"],
                payload=data["payload"],
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Main hybrid search entry point
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    user_id: str | None = None,
    model: str | None = None,
    context_version: str | None = None,
    top_k: int = 5,
    fusion: str = "rrf",
    alpha: float = 0.7,
) -> list[SearchResult]:
    """
    Execute a full hybrid search: embed → filter → dual search → fuse.

    This is the main entry point for cache retrieval. It:
      1. Normalizes the query text
      2. Produces dense and sparse representations
      3. Builds metadata pre-filters (user_id, model, context_version, TTL)
      4. Runs both dense and sparse searches against Qdrant
      5. Fuses results via RRF (default) or weighted combination

    Args:
        query: Raw query text from the user.
        user_id: Tenant identifier for isolation.
        model: LLM model identifier for scoping.
        context_version: Prompt/context version for invalidation.
        top_k: Number of results per search method (before fusion).
        fusion: Fusion strategy — "rrf" (default) or "weighted".
        alpha: Weight for dense scores when using weighted fusion.

    Returns:
        List of SearchResult sorted by fused score (descending).
        May be empty if no results pass the metadata filter.
    """
    # 1. Normalize
    normalized_query = normalize_text(query)

    # 2. Embed
    dense_vector = embed_query(normalized_query)
    sparse_terms = extract_sparse_terms(normalized_query)
    sparse_indices, sparse_values = terms_to_sparse_vector(sparse_terms)

    # 3. Build metadata filter (pre-filtering, not post-filtering)
    query_filter = build_filter(
        user_id=user_id,
        model=model,
        context_version=context_version,
        exclude_expired=True,
    )

    # 4. Dual search
    dense_results = search_dense(
        dense_vector=dense_vector,
        query_filter=query_filter,
        top_k=top_k,
    )

    sparse_results = []
    if sparse_indices:  # Only search sparse if we have terms
        sparse_results = search_sparse(
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            query_filter=query_filter,
            top_k=top_k,
        )

    logger.info(
        "Hybrid search for '%.50s...': %d dense results, %d sparse results",
        normalized_query,
        len(dense_results),
        len(sparse_results),
    )

    # 5. Fuse
    if fusion == "weighted":
        fused = _weighted_fusion(dense_results, sparse_results, alpha=alpha)
    else:
        fused = _reciprocal_rank_fusion(dense_results, sparse_results)

    if fused:
        logger.info(
            "Top result: score=%.4f, dense=%.4f, sparse=%.4f, query='%.50s...'",
            fused[0].score,
            fused[0].dense_score,
            fused[0].sparse_score,
            fused[0].query_text,
        )

    return fused
