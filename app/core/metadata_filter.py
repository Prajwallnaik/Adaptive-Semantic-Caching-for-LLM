"""
Metadata filter construction for Qdrant pre-filtering.

Builds Qdrant Filter objects that are applied BEFORE/DURING retrieval,
not after ranking. This is a key design decision: filtering after ranking
wastes top-k slots on candidates that get discarded. Pre-filtering shrinks
the search space and is both cheaper and safer.

Filters enforce:
  - Tenant isolation (user_id): user A's cache never leaks to user B
  - Model scoping (model): one model's cached answer is not served for another
  - Context versioning (context_version): old entries from a different
    prompt/context are excluded, regardless of age or usage
  - TTL enforcement (expires_at): expired entries are skipped at query time
    (belt-and-suspenders with the periodic eviction sweep)
"""

import logging
from datetime import datetime, timezone

from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)

logger = logging.getLogger(__name__)


def build_filter(
    user_id: str | None = None,
    model: str | None = None,
    context_version: str | None = None,
    exclude_expired: bool = True,
) -> Filter | None:
    """
    Construct a Qdrant Filter with `must` conditions for metadata pre-filtering.

    Only non-None parameters are included in the filter. If all parameters are
    None and exclude_expired is False, returns None (no filtering).

    Args:
        user_id: Tenant identifier. If set, only entries from this user are returned.
        model: LLM model identifier. If set, only entries for this model are returned.
        context_version: Prompt/context version tag. If set, only matching versions
            are returned (enables instant invalidation on context change).
        exclude_expired: If True, adds a condition to skip entries where
            expires_at < now. Defaults to True.

    Returns:
        A Qdrant Filter object, or None if no conditions apply.
    """
    conditions: list[FieldCondition] = []

    if user_id is not None:
        conditions.append(
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id),
            )
        )

    if model is not None:
        conditions.append(
            FieldCondition(
                key="model",
                match=MatchValue(value=model),
            )
        )

    if context_version is not None:
        conditions.append(
            FieldCondition(
                key="context_version",
                match=MatchValue(value=context_version),
            )
        )

    if exclude_expired:
        now = datetime.now(timezone.utc).timestamp()
        conditions.append(
            FieldCondition(
                key="expires_at",
                range=Range(gt=now),
            )
        )

    if not conditions:
        return None

    filter_obj = Filter(must=conditions)

    logger.debug(
        "Built metadata filter with %d conditions: user_id=%s, model=%s, "
        "context_version=%s, exclude_expired=%s",
        len(conditions),
        user_id,
        model,
        context_version,
        exclude_expired,
    )

    return filter_obj
