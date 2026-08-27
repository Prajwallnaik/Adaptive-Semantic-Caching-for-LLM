"""
Verification module — secondary check for borderline cache hits.

When the hybrid search score falls within the verification band (close to
the threshold but not confidently above it), we don't trust the score
blindly. Instead, we run a cheap secondary check to decide if the cached
answer is truly appropriate.

Two strategies are implemented:

  1. **Rule-based** (default, free, fast):
     - Token overlap ratio between the incoming query and the cached query
     - Exact-match check on numbers, dates, and entity-like tokens
     - If overlap is high AND critical tokens match → pass

  2. **LLM judge** (optional, costs one API call):
     - Short prompt to the same Nemotron model asking
       "Are these two queries asking the same thing?"
     - More accurate but uses a free-tier API call

The mode is selected via VERIFICATION_MODE in config.
"""

import re
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Pattern to detect "critical" tokens — numbers, dates, version strings,
# and capitalized entities that embeddings tend to blur.
_CRITICAL_TOKEN_PATTERN = re.compile(
    r"\b("
    r"\d+(?:\.\d+)?"          # Numbers: 2024, 3.14
    r"|[A-Z][a-zA-Z]*"        # Capitalized words (potential entities)
    r"|v\d+(?:\.\d+)*"        # Version strings: v1, v2.3
    r"|Q[1-4]"                # Quarters: Q1, Q2
    r")\b"
)


def _extract_critical_tokens(text: str) -> set[str]:
    """Extract numbers, entities, and version-like tokens from text."""
    return set(_CRITICAL_TOKEN_PATTERN.findall(text))


def _tokenize_simple(text: str) -> set[str]:
    """Simple lowercased word tokenization for overlap computation."""
    return set(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()))


# ---------------------------------------------------------------------------
# Rule-based verification
# ---------------------------------------------------------------------------

def _verify_rule_based(
    incoming_query: str,
    cached_query: str,
    token_overlap_threshold: float = 0.65,
) -> bool:
    """
    Rule-based verification for borderline cache hits.

    Checks two things:
      1. General token overlap ratio (Jaccard-like) between the two queries.
         High overlap means the queries are about the same topic.
      2. Critical token exact-match — numbers, dates, entities, versions.
         If the incoming query has critical tokens not present in the cached
         query, the match is rejected regardless of overall overlap.

    This catches the classic failure case: "Q1 2024 revenue" vs "Q1 2023
    revenue" have high embedding similarity and decent token overlap, but
    differ on a critical token (the year).

    Args:
        incoming_query: The new query from the user.
        cached_query: The query text stored in the cache entry.
        token_overlap_threshold: Minimum token overlap ratio to pass.

    Returns:
        True if the match is verified, False if it should be rejected.
    """
    # Token overlap
    incoming_tokens = _tokenize_simple(incoming_query)
    cached_tokens = _tokenize_simple(cached_query)

    if not incoming_tokens or not cached_tokens:
        return False

    intersection = incoming_tokens & cached_tokens
    union = incoming_tokens | cached_tokens
    overlap_ratio = len(intersection) / len(union) if union else 0.0

    # Critical token check
    incoming_critical = _extract_critical_tokens(incoming_query)
    cached_critical = _extract_critical_tokens(cached_query)

    # If the incoming query has critical tokens not in the cached query,
    # this is likely a factually different question.
    missing_critical = incoming_critical - cached_critical

    logger.debug(
        "Rule-based verification: overlap=%.2f (threshold=%.2f), "
        "incoming_critical=%s, cached_critical=%s, missing=%s",
        overlap_ratio,
        token_overlap_threshold,
        incoming_critical,
        cached_critical,
        missing_critical,
    )

    if missing_critical:
        logger.info(
            "Verification FAILED: critical tokens missing from cached query: %s",
            missing_critical,
        )
        return False

    if overlap_ratio < token_overlap_threshold:
        logger.info(
            "Verification FAILED: token overlap %.2f below threshold %.2f",
            overlap_ratio,
            token_overlap_threshold,
        )
        return False

    logger.info("Verification PASSED (rule-based): overlap=%.2f", overlap_ratio)
    return True


# ---------------------------------------------------------------------------
# LLM judge verification
# ---------------------------------------------------------------------------

def _verify_llm_judge(
    incoming_query: str,
    cached_query: str,
    cached_answer: str,
) -> bool:
    """
    LLM-based verification using a short, cheap prompt.

    Asks the Nemotron model whether the two queries are semantically
    equivalent. This is more accurate than rule-based but costs one
    API call from the free-tier quota.

    Args:
        incoming_query: The new query from the user.
        cached_query: The query text stored in the cache entry.
        cached_answer: The cached answer (provided for context).

    Returns:
        True if the LLM judge says the queries are equivalent.
    """
    # Import here to avoid circular dependency (llm_client is Phase 3)
    try:
        from app.core.llm_client import call_llm_raw
    except ImportError:
        logger.warning(
            "LLM client not available, falling back to rule-based verification"
        )
        return _verify_rule_based(incoming_query, cached_query)

    judge_prompt = (
        "You are a semantic similarity judge. Determine if these two queries "
        "are asking for the EXACT SAME information. Consider numbers, dates, "
        "entities, and specific details — not just general topic.\n\n"
        f"Query A: {incoming_query}\n"
        f"Query B: {cached_query}\n\n"
        "Answer ONLY 'YES' or 'NO'."
    )

    try:
        response = call_llm_raw(judge_prompt)
        answer = response.strip().upper()
        is_match = answer.startswith("YES")

        logger.info(
            "LLM judge verdict: %s (raw: '%s') for queries: '%.40s...' vs '%.40s...'",
            "PASS" if is_match else "FAIL",
            answer[:10],
            incoming_query,
            cached_query,
        )
        return is_match

    except Exception as e:
        logger.error("LLM judge call failed: %s. Falling back to rule-based.", e)
        return _verify_rule_based(incoming_query, cached_query)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_match(
    incoming_query: str,
    cached_query: str,
    cached_answer: str = "",
) -> bool:
    """
    Verify whether a borderline cache hit is a genuine match.

    Dispatches to the configured verification strategy (rule_based or llm_judge).
    The strategy is set via VERIFICATION_MODE in the environment/config.

    Args:
        incoming_query: The new query from the user.
        cached_query: The query text stored in the cache entry.
        cached_answer: The cached answer (used by LLM judge for context).

    Returns:
        True if the match is verified, False if it should be rejected
        (and the query should be sent to the LLM instead).
    """
    mode = settings.verification_mode.lower()

    if mode == "llm_judge":
        return _verify_llm_judge(incoming_query, cached_query, cached_answer)
    else:
        # Default to rule-based (free, fast, no API calls)
        return _verify_rule_based(incoming_query, cached_query)
