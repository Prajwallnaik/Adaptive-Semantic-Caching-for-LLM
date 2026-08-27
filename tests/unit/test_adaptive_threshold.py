"""
Unit tests for adaptive semantic confidence thresholding.

Tests each signal independently and the combined compute function,
including edge cases and the feature-flag toggle.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.core.adaptive_threshold import (
    compute_adaptive_threshold,
    _query_complexity_delta,
    _entity_sensitivity_delta,
    _latency_pressure_delta,
    _hit_rate_feedback_delta,
)


# ---------------------------------------------------------------------------
# Signal 1: Query Complexity
# ---------------------------------------------------------------------------

class TestQueryComplexityDelta:
    """Tests for _query_complexity_delta."""

    def test_short_simple_query_lowers_threshold(self):
        """Short queries (<=5 words, 1 clause) → negative delta."""
        delta = _query_complexity_delta("What is Python?")
        assert delta == -0.03

    def test_medium_query_no_change(self):
        """Medium-length queries (6-15 words) → zero delta."""
        delta = _query_complexity_delta("Explain the difference between lists and tuples in Python")
        assert delta == 0.0

    def test_long_query_raises_threshold(self):
        """Long queries (>15 words) → positive delta."""
        query = "Explain how to implement a binary search tree with self-balancing properties and discuss the time complexity of insertion"
        delta = _query_complexity_delta(query)
        assert delta > 0.0

    def test_very_long_query_higher_delta(self):
        """Very long queries (>30 words) → higher delta."""
        query = " ".join(["word"] * 35)
        delta = _query_complexity_delta(query)
        assert delta >= 0.03

    def test_multi_clause_adds_strictness(self):
        """Queries with 3+ clauses get additional strictness."""
        query = "Explain A, and describe B, but also compare C, then summarize the overall impact of all three approaches"
        delta = _query_complexity_delta(query)
        assert delta > 0.0


# ---------------------------------------------------------------------------
# Signal 2: Entity Sensitivity
# ---------------------------------------------------------------------------

class TestEntitySensitivityDelta:
    """Tests for _entity_sensitivity_delta."""

    def test_no_entities_no_change(self):
        """Queries without entities → zero delta."""
        delta = _entity_sensitivity_delta("explain how caching works")
        assert delta == 0.0

    def test_single_year_raises(self):
        """A year like 2024 → positive delta."""
        delta = _entity_sensitivity_delta("revenue in 2024")
        assert delta == 0.02

    def test_quarter_and_year_raises_more(self):
        """Q1 + 2024 → higher delta (2 entities)."""
        delta = _entity_sensitivity_delta("Q1 2024 revenue report")
        assert delta == 0.03

    def test_many_entities_max_delta(self):
        """3+ entities → maximum delta."""
        delta = _entity_sensitivity_delta("Compare Q1 2023 vs Q2 2024 revenue at $100M")
        assert delta == 0.05

    def test_version_string_detected(self):
        """Version strings like v2.3 are entities."""
        delta = _entity_sensitivity_delta("changes in v2.3")
        assert delta > 0.0

    def test_iso_date_detected(self):
        """ISO dates like 2024-01-15 are entities."""
        delta = _entity_sensitivity_delta("events on 2024-01-15")
        assert delta > 0.0

    def test_currency_detected(self):
        """Currency amounts like $100 are entities."""
        delta = _entity_sensitivity_delta("budget is $500")
        assert delta > 0.0


# ---------------------------------------------------------------------------
# Signal 3: Latency Pressure
# ---------------------------------------------------------------------------

class TestLatencyPressureDelta:
    """Tests for _latency_pressure_delta."""

    @patch("app.core.adaptive_threshold.get_avg_llm_latency", return_value=None)
    def test_no_data_no_change(self, mock_latency):
        """No latency data → zero delta."""
        assert _latency_pressure_delta() == 0.0

    @patch("app.core.adaptive_threshold.get_avg_llm_latency", return_value=1.5)
    def test_fast_llm_no_change(self, mock_latency):
        """Fast LLM (< 3s) → zero delta."""
        assert _latency_pressure_delta() == 0.0

    @patch("app.core.adaptive_threshold.get_avg_llm_latency", return_value=4.0)
    def test_slightly_slow_minor_nudge(self, mock_latency):
        """Slightly slow (3-5s) → small negative delta."""
        assert _latency_pressure_delta() == -0.01

    @patch("app.core.adaptive_threshold.get_avg_llm_latency", return_value=6.0)
    def test_slow_moderate_pressure(self, mock_latency):
        """Slow (5-8s) → moderate negative delta."""
        assert _latency_pressure_delta() == -0.03

    @patch("app.core.adaptive_threshold.get_avg_llm_latency", return_value=10.0)
    def test_very_slow_aggressive_pressure(self, mock_latency):
        """Very slow (>8s) → maximum negative delta."""
        assert _latency_pressure_delta() == -0.05


# ---------------------------------------------------------------------------
# Signal 4: Hit-Rate Feedback
# ---------------------------------------------------------------------------

class TestHitRateFeedbackDelta:
    """Tests for _hit_rate_feedback_delta."""

    @patch(
        "app.core.adaptive_threshold.get_session_stats",
        return_value={"total_queries": 5, "cache_hits": 2},
    )
    def test_insufficient_data_no_change(self, mock_stats):
        """Fewer than 10 queries → zero delta (not enough data)."""
        assert _hit_rate_feedback_delta() == 0.0

    @patch(
        "app.core.adaptive_threshold.get_session_stats",
        return_value={"total_queries": 100, "cache_hits": 10},
    )
    def test_very_low_hit_rate_lowers(self, mock_stats):
        """10% hit rate → negative delta (loosen up)."""
        assert _hit_rate_feedback_delta() == -0.02

    @patch(
        "app.core.adaptive_threshold.get_session_stats",
        return_value={"total_queries": 100, "cache_hits": 25},
    )
    def test_low_hit_rate_slight_lower(self, mock_stats):
        """25% hit rate → slight negative delta."""
        assert _hit_rate_feedback_delta() == -0.01

    @patch(
        "app.core.adaptive_threshold.get_session_stats",
        return_value={"total_queries": 100, "cache_hits": 50},
    )
    def test_normal_hit_rate_no_change(self, mock_stats):
        """50% hit rate → zero delta (healthy range)."""
        assert _hit_rate_feedback_delta() == 0.0

    @patch(
        "app.core.adaptive_threshold.get_session_stats",
        return_value={"total_queries": 100, "cache_hits": 75},
    )
    def test_high_hit_rate_slight_raise(self, mock_stats):
        """75% hit rate → slight positive delta (tighten quality)."""
        assert _hit_rate_feedback_delta() == 0.01

    @patch(
        "app.core.adaptive_threshold.get_session_stats",
        return_value={"total_queries": 100, "cache_hits": 90},
    )
    def test_very_high_hit_rate_raises(self, mock_stats):
        """90% hit rate → positive delta (tighten quality)."""
        assert _hit_rate_feedback_delta() == 0.02


# ---------------------------------------------------------------------------
# Combined: compute_adaptive_threshold
# ---------------------------------------------------------------------------

class TestComputeAdaptiveThreshold:
    """Tests for the main compute_adaptive_threshold function."""

    @patch("app.core.adaptive_threshold.settings")
    def test_disabled_returns_base(self, mock_settings):
        """Feature flag disabled → return base threshold unchanged."""
        mock_settings.adaptive_threshold_enabled = False
        result = compute_adaptive_threshold("any query", 0.82)
        assert result == 0.82

    @patch("app.core.adaptive_threshold.get_avg_llm_latency", return_value=None)
    @patch(
        "app.core.adaptive_threshold.get_session_stats",
        return_value={"total_queries": 0, "cache_hits": 0},
    )
    @patch("app.core.adaptive_threshold.settings")
    def test_short_query_lowers_threshold(self, mock_settings, mock_stats, mock_latency):
        """A short, entity-free query should lower the threshold."""
        mock_settings.adaptive_threshold_enabled = True
        mock_settings.threshold_floor = 0.65
        mock_settings.threshold_ceiling = 0.95

        result = compute_adaptive_threshold("What is Python?", 0.82)
        assert result < 0.82

    @patch("app.core.adaptive_threshold.get_avg_llm_latency", return_value=None)
    @patch(
        "app.core.adaptive_threshold.get_session_stats",
        return_value={"total_queries": 0, "cache_hits": 0},
    )
    @patch("app.core.adaptive_threshold.settings")
    def test_entity_query_raises_threshold(self, mock_settings, mock_stats, mock_latency):
        """A query with dates and quarters should raise the threshold."""
        mock_settings.adaptive_threshold_enabled = True
        mock_settings.threshold_floor = 0.65
        mock_settings.threshold_ceiling = 0.95

        result = compute_adaptive_threshold("Q1 2024 revenue vs Q2 2023", 0.82)
        assert result > 0.82

    @patch("app.core.adaptive_threshold.get_avg_llm_latency", return_value=None)
    @patch(
        "app.core.adaptive_threshold.get_session_stats",
        return_value={"total_queries": 0, "cache_hits": 0},
    )
    @patch("app.core.adaptive_threshold.settings")
    def test_result_clamped_to_floor(self, mock_settings, mock_stats, mock_latency):
        """Threshold should never go below the configured floor."""
        mock_settings.adaptive_threshold_enabled = True
        mock_settings.threshold_floor = 0.80
        mock_settings.threshold_ceiling = 0.95

        # Short query would try to lower, but floor prevents it
        result = compute_adaptive_threshold("Hi", 0.80)
        assert result >= 0.80

    @patch("app.core.adaptive_threshold.get_avg_llm_latency", return_value=None)
    @patch(
        "app.core.adaptive_threshold.get_session_stats",
        return_value={"total_queries": 0, "cache_hits": 0},
    )
    @patch("app.core.adaptive_threshold.settings")
    def test_result_clamped_to_ceiling(self, mock_settings, mock_stats, mock_latency):
        """Threshold should never go above the configured ceiling."""
        mock_settings.adaptive_threshold_enabled = True
        mock_settings.threshold_floor = 0.65
        mock_settings.threshold_ceiling = 0.90

        # Heavy entity query on a high base would try to exceed ceiling
        result = compute_adaptive_threshold("Q1 2024 at $500M vs Q2 2023 at $300M on 2024-01-15", 0.90)
        assert result <= 0.90

    @patch("app.core.adaptive_threshold.get_avg_llm_latency", return_value=10.0)
    @patch(
        "app.core.adaptive_threshold.get_session_stats",
        return_value={"total_queries": 100, "cache_hits": 10},
    )
    @patch("app.core.adaptive_threshold.settings")
    def test_high_latency_and_low_hitrate_lowers_significantly(
        self, mock_settings, mock_stats, mock_latency
    ):
        """Combined pressure from slow LLM + low hit rate → significant decrease."""
        mock_settings.adaptive_threshold_enabled = True
        mock_settings.threshold_floor = 0.65
        mock_settings.threshold_ceiling = 0.95

        result = compute_adaptive_threshold("What is caching?", 0.82)
        # -0.03 (complexity) + 0.0 (entity) + -0.05 (latency) + -0.02 (hit_rate) = -0.10
        assert result < 0.75
