"""
LLM client — NVIDIA Nemotron via NIM API (OpenAI-compatible).

Uses the `openai` Python SDK with a custom base_url pointing to
NVIDIA's NIM endpoint. This is NOT OpenRouter — it's a direct
connection to `https://integrate.api.nvidia.com/v1`.

Includes:
  - Retry logic with exponential backoff for rate limits (429)
  - A raw call variant for the verification judge (Phase 2)
  - Token usage tracking for cost estimation
"""

import time
import logging

from openai import OpenAI, RateLimitError, APIError

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Return the OpenAI client singleton configured for NVIDIA NIM."""
    global _client
    if _client is None:
        if not settings.nvidia_api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set. Please set it in .env or environment."
            )
        _client = OpenAI(
            base_url=settings.nvidia_base_url,
            api_key=settings.nvidia_api_key,
        )
        logger.info(
            "NVIDIA NIM client initialized: base_url=%s, model=%s",
            settings.nvidia_base_url,
            settings.nvidia_model,
        )
    return _client


# ---------------------------------------------------------------------------
# Core LLM call with retry
# ---------------------------------------------------------------------------

def call_llm(
    query: str,
    model: str | None = None,
    max_retries: int = 3,
    initial_backoff: float = 1.0,
) -> tuple[str, dict]:
    """
    Call the LLM and return the answer with usage metadata.

    Args:
        query: The user's query text.
        model: Override model name (uses config default if None).
        max_retries: Number of retry attempts on rate limit errors.
        initial_backoff: Initial backoff time in seconds (doubles each retry).

    Returns:
        Tuple of (answer_text, usage_info) where usage_info contains
        token counts for cost tracking.

    Raises:
        APIError: If all retries are exhausted.
    """
    client = _get_client()
    model_name = model or settings.nvidia_model
    backoff = initial_backoff

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": query}],
            )

            answer = response.choices[0].message.content or ""
            usage_info = {}
            if response.usage:
                usage_info = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            logger.info(
                "LLM call succeeded: model=%s, tokens=%s, answer_len=%d",
                model_name,
                usage_info.get("total_tokens", "?"),
                len(answer),
            )
            return answer, usage_info

        except RateLimitError as e:
            if attempt < max_retries:
                logger.warning(
                    "Rate limited (attempt %d/%d), backing off %.1fs: %s",
                    attempt + 1,
                    max_retries,
                    backoff,
                    e,
                )
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff
            else:
                logger.error("Rate limit exceeded after %d retries", max_retries)
                raise

        except APIError as e:
            logger.error("LLM API error: %s", e)
            raise


# ---------------------------------------------------------------------------
# Raw call (for verification judge — no tuple wrapping)
# ---------------------------------------------------------------------------

def call_llm_raw(
    prompt: str,
    model: str | None = None,
) -> str:
    """
    Simple LLM call returning just the answer text.

    Used by the verification judge (app/core/verification.py) which
    only needs the text response, not usage metadata.

    Args:
        prompt: The full prompt text to send.
        model: Override model name (uses config default if None).

    Returns:
        The LLM's response text.
    """
    answer, _ = call_llm(prompt, model=model)
    return answer
