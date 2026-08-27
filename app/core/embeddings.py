"""
Embedding module — dense vectors and sparse term extraction.

Dense: sentence-transformers `all-MiniLM-L6-v2` (384-dim, local, free).
Sparse: simple TF-based term weighting for BM25-style retrieval in Qdrant.

The embedding model is loaded lazily as a singleton to avoid repeated
initialization overhead.
"""

import re
import math
import logging
from collections import Counter
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """
    Lazy-load the sentence-transformers model (singleton).

    The first call downloads / loads the model; subsequent calls return
    the cached instance.
    """
    logger.info("Loading embedding model: %s", settings.embedding_model_name)
    model = SentenceTransformer(settings.embedding_model_name)
    logger.info("Embedding model loaded (dim=%d)", model.get_sentence_embedding_dimension())
    return model


def embed_query(text: str) -> list[float]:
    """
    Produce a dense embedding vector for the given text.

    Args:
        text: The query string to embed.

    Returns:
        A list of floats representing the dense vector (384-dim for MiniLM-L6-v2).
    """
    model = _get_model()
    # SentenceTransformer.encode returns a numpy array; convert to plain list
    # for JSON serialization and Qdrant compatibility.
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


# ---------------------------------------------------------------------------
# Sparse term extraction (BM25-style)
# ---------------------------------------------------------------------------

# Minimal stop-word set — enough to remove noise without a heavy dependency.
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "about", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than", "too",
    "very", "just", "because", "if", "when", "while", "where", "how",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their",
})

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def normalize_text(text: str) -> str:
    """
    Lightweight query normalization.

    - Lowercases
    - Collapses whitespace
    - Strips leading/trailing whitespace

    Intentionally simple — the embedding model handles semantic normalization;
    this is just for consistent sparse-term extraction and cache key stability.
    """
    return " ".join(text.lower().split())


def extract_sparse_terms(text: str) -> dict[str, float]:
    """
    Extract sparse term weights from text using log-scaled TF weighting.

    This produces the "sparse vector" component for hybrid search. Terms are
    lowercased, stop-words removed, and weighted by `1 + log(tf)` so that
    repeated terms get diminishing returns rather than linear boosting.

    Numbers and decimal values (e.g. "2024", "3.14") are preserved as tokens —
    this is critical because embeddings blur numbers, but sparse matching
    catches them exactly (the whole point of hybrid search).

    Args:
        text: Raw query text.

    Returns:
        Dict mapping each term to its TF weight. Empty dict for empty input.
    """
    normalized = normalize_text(text)
    tokens = _TOKEN_PATTERN.findall(normalized)

    # Remove stop words but keep numbers (numbers are never stop words)
    meaningful_tokens = [t for t in tokens if t not in _STOP_WORDS]

    if not meaningful_tokens:
        return {}

    # Log-scaled TF weighting: 1 + log(count)
    counts = Counter(meaningful_tokens)
    weights = {
        term: round(1.0 + math.log(count), 4)
        for term, count in counts.items()
    }

    return weights
