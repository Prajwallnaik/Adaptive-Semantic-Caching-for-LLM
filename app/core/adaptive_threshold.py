"""
Adaptive Semantic Confidence Thresholding.

Dynamically calibrates the cache similarity threshold based on four
independent signals, each contributing a small delta to the base
threshold configured in settings:

  1. **Query Complexity** — longer, multi-clause queries are harder to
     match accurately, so the threshold is raised to prevent false hits.

  2. **Entity Sensitivity** — queries containing numbers, dates, version
     strings, or proper nouns are at high risk of factual drift (e.g.
     "Q1 2023" vs "Q1 2024"), so the threshold is raised.

  3. **Latency Pressure** — when the LLM backend is slow (high rolling
     average response time), the threshold is lowered to favor cache
     hits and maintain responsiveness.

  4. **Hit-Rate Feedback** — if the session hit-rate is very low, the
     threshold is lowered slightly to recover cacheability; if very
     high, it's raised slightly to tighten quality.

Each signal produces a delta in the range [-0.05, +0.05]. The final
threshold is clamped to [threshold_floor, threshold_ceiling] from config.

Also contains the three-way cache decision gate (hit / verify / miss)
that was previously in threshold.py.

Usage:
    from app.core.adaptive_threshold import compute_adaptive_threshold, decide, Decision
    threshold = compute_adaptive_threshold(query, base_threshold)
    decision = decide(score, threshold, band)
"""

import re
import logging
from enum import Enum

from app.config import settings
from app.monitoring.metrics import get_avg_llm_latency, get_session_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision gate — three-way cache hit / miss / verify classification
# ---------------------------------------------------------------------------

class Decision(str, Enum):
    """Cache decision outcome."""
    HIT = "hit"
    VERIFY = "verify"
    MISS = "miss"


def decide(
    score: float,
    threshold: float,
    band: float,
) -> Decision:
    """
    Classify a similarity score into hit, verify, or miss.

    The decision boundary looks like:

        |<--- MISS --->|<--- VERIFY --->|<--- HIT --->|
        0        threshold - band    threshold        1

    Args:
        score: The fused similarity score from hybrid search.
        threshold: The minimum score for a confident cache hit.
        band: The tolerance band below the threshold. Scores within
            [threshold - band, threshold) are sent for verification.

    Returns:
        Decision.HIT, Decision.VERIFY, or Decision.MISS.

    Examples:
        >>> decide(0.90, threshold=0.82, band=0.05)
        <Decision.HIT: 'hit'>

        >>> decide(0.80, threshold=0.82, band=0.05)
        <Decision.VERIFY: 'verify'>

        >>> decide(0.70, threshold=0.82, band=0.05)
        <Decision.MISS: 'miss'>
    """
    if score >= threshold:
        logger.debug("Score %.4f >= threshold %.4f → HIT", score, threshold)
        return Decision.HIT

    lower_bound = threshold - band
    if score >= lower_bound:
        logger.debug(
            "Score %.4f in verification band [%.4f, %.4f) → VERIFY",
            score,
            lower_bound,
            threshold,
        )
        return Decision.VERIFY

    logger.debug(
        "Score %.4f < lower bound %.4f → MISS",
        score,
        lower_bound,
    )
    return Decision.MISS


# ---------------------------------------------------------------------------
# Signal 1: Query Complexity
# ---------------------------------------------------------------------------

# Clause-boundary markers — commas, semicolons, conjunctions that typically
# introduce a new informational requirement in the query.
_CLAUSE_MARKERS = re.compile(r",|;|\band\b|\bor\b|\bbut\b|\bthen\b", re.IGNORECASE)


def _query_complexity_delta(query: str) -> float:
    """
    Estimate query complexity from word count and clause count.

    Short, single-clause queries are simple and can tolerate a lower
    threshold.  Long, multi-clause queries need a higher threshold to
    avoid matching against a cached answer that only covers part of
    the question.

    Returns:
        A delta in [-0.03, +0.04].
    """
    words = query.split()
    word_count = len(words)
    clause_count = len(_CLAUSE_MARKERS.findall(query)) + 1  # at least 1

    # Short & simple → lower threshold (easier to match)
    if word_count <= 5 and clause_count == 1:
        return -0.03

    # Medium → no change
    if word_count <= 15:
        return 0.0

    # Long query → raise threshold
    delta = 0.02
    if word_count > 30:
        delta = 0.03

    # Multi-clause adds extra strictness
    if clause_count >= 3:
        delta += 0.01

    return min(delta, 0.04)


# ---------------------------------------------------------------------------
# Signal 2: Entity Sensitivity
# ---------------------------------------------------------------------------

# Patterns that detect "critical" tokens embeddings tend to blur:
#   - Years and dates:   2023, 2024-01-15
#   - Decimal numbers:   3.14, 99.9
#   - Version strings:   v1, v2.3.1
#   - Quarters:          Q1, Q2
#   - Currency amounts:  $100, €50
_ENTITY_PATTERNS = [
    re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b"),    # ISO dates
    re.compile(r"\b(?:19|20)\d{2}\b"),               # Years 1900-2099
    re.compile(r"\bQ[1-4]\b", re.IGNORECASE),        # Quarters
    re.compile(r"\bv\d+(?:\.\d+)*\b", re.IGNORECASE),  # Versions
    re.compile(r"[$€£¥]\s?\d+(?:[.,]\d+)?"),         # Currency
    re.compile(r"\b\d+\.\d+\b"),                     # Decimal numbers
]


def _entity_sensitivity_delta(query: str) -> float:
    """
    Detect entity-rich queries that are prone to factual drift.

    If the query contains numbers, dates, versions, or currency amounts,
    the threshold is raised because even a tiny mismatch in these tokens
    (e.g. "Q1 2023" vs "Q1 2024") makes the cached answer wrong.

    Returns:
        A delta in [0.0, +0.05].
    """
    entity_count = 0
    for pattern in _ENTITY_PATTERNS:
        entity_count += len(pattern.findall(query))

    if entity_count == 0:
        return 0.0
    if entity_count == 1:
        return 0.02
    if entity_count == 2:
        return 0.03
    # 3+ entities → maximum strictness boost
    return 0.05


# ---------------------------------------------------------------------------
# Signal 3: Latency Pressure
# ---------------------------------------------------------------------------

def _latency_pressure_delta() -> float:
    """
    Lower the threshold when the LLM backend is slow.

    If the rolling average LLM latency is high, it's better to serve
    a slightly-less-perfect cached answer quickly than to wait for
    a slow LLM call.

    Returns:
        A delta in [-0.05, 0.0].
    """
    avg_latency = get_avg_llm_latency()

    if avg_latency is None:
        # No data yet — no adjustment
        return 0.0

    # Thresholds for "slow" and "very slow" LLM responses
    if avg_latency > 8.0:
        return -0.05  # Very slow — aggressively prefer cache
    if avg_latency > 5.0:
        return -0.03  # Slow — moderately prefer cache
    if avg_latency > 3.0:
        return -0.01  # Slightly slow — minor nudge

    return 0.0


# ---------------------------------------------------------------------------
# Signal 4: Hit-Rate Feedback
# ---------------------------------------------------------------------------

def _hit_rate_feedback_delta() -> float:
    """
    Nudge the threshold based on the current session hit-rate.

    If the hit-rate is very low, the threshold might be too strict —
    lower it slightly to recover cacheability.  If the hit-rate is
    very high, raise it slightly to tighten quality control (we can
    afford to be pickier).

    Returns:
        A delta in [-0.02, +0.02].
    """
    stats = get_session_stats()
    total = stats.get("total_queries", 0)

    if total < 10:
        # Not enough data to make a meaningful adjustment
        return 0.0

    hit_rate = stats.get("cache_hits", 0) / total

    if hit_rate < 0.15:
        return -0.02  # Very low hit-rate — loosen up
    if hit_rate < 0.30:
        return -0.01  # Low hit-rate — slight nudge
    if hit_rate > 0.85:
        return 0.02   # Very high hit-rate — tighten quality
    if hit_rate > 0.70:
        return 0.01   # High hit-rate — slight tighten

    return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_adaptive_threshold(
    query: str,
    base_threshold: float,
) -> float:
    """
    Compute an adaptive similarity threshold for the given query.

    Combines four independent signals — query complexity, entity
    sensitivity, LLM latency pressure, and hit-rate feedback — into
    a single calibrated threshold value.

    If adaptive thresholding is disabled in config, returns the
    base_threshold unchanged.

    Args:
        query: The raw user query text.
        base_threshold: The static threshold from config (e.g. 0.82).

    Returns:
        The adjusted threshold, clamped to [threshold_floor, threshold_ceiling].
    """
    if not settings.adaptive_threshold_enabled:
        return base_threshold

    # Compute each signal's contribution
    complexity = _query_complexity_delta(query)
    entity = _entity_sensitivity_delta(query)
    latency = _latency_pressure_delta()
    hit_rate = _hit_rate_feedback_delta()

    total_delta = complexity + entity + latency + hit_rate
    adaptive = base_threshold + total_delta

    # Clamp to safe bounds
    adaptive = max(settings.threshold_floor, min(settings.threshold_ceiling, adaptive))

    logger.info(
        "Adaptive threshold: base=%.3f → %.3f "
        "(complexity=%+.3f, entity=%+.3f, latency=%+.3f, hit_rate=%+.3f)",
        base_threshold,
        adaptive,
        complexity,
        entity,
        latency,
        hit_rate,
    )

    return round(adaptive, 4)
