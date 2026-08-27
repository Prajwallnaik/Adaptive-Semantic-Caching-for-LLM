"""
Unit tests for the threshold decision logic.

Tests the three-way gate: HIT / VERIFY / MISS, including edge cases
at exact boundary values and various threshold/band combinations.
"""

import pytest
from app.core.adaptive_threshold import decide, Decision


class TestDecide:
    """Tests for the decide() function."""

    # --- Standard threshold/band (0.82 / 0.05) ---

    def test_confident_hit_above_threshold(self):
        assert decide(0.90, threshold=0.82, band=0.05) == Decision.HIT

    def test_exact_threshold_is_hit(self):
        """Score exactly at threshold should be a HIT."""
        assert decide(0.82, threshold=0.82, band=0.05) == Decision.HIT

    def test_max_score_is_hit(self):
        assert decide(1.00, threshold=0.82, band=0.05) == Decision.HIT

    def test_just_above_threshold_is_hit(self):
        assert decide(0.8200001, threshold=0.82, band=0.05) == Decision.HIT

    def test_just_below_threshold_is_verify(self):
        assert decide(0.8199999, threshold=0.82, band=0.05) == Decision.VERIFY

    def test_mid_band_is_verify(self):
        assert decide(0.80, threshold=0.82, band=0.05) == Decision.VERIFY

    def test_lower_bound_of_band_is_verify(self):
        """Score exactly at (threshold - band) should be VERIFY."""
        assert decide(0.77, threshold=0.82, band=0.05) == Decision.VERIFY

    def test_just_below_band_is_miss(self):
        assert decide(0.7699999, threshold=0.82, band=0.05) == Decision.MISS

    def test_clear_miss(self):
        assert decide(0.50, threshold=0.82, band=0.05) == Decision.MISS

    def test_zero_score_is_miss(self):
        assert decide(0.00, threshold=0.82, band=0.05) == Decision.MISS

    # --- Decision enum values ---

    def test_hit_value(self):
        assert decide(0.90, 0.82, 0.05).value == "hit"

    def test_verify_value(self):
        assert decide(0.80, 0.82, 0.05).value == "verify"

    def test_miss_value(self):
        assert decide(0.50, 0.82, 0.05).value == "miss"

    # --- Zero band (no verification zone) ---

    def test_zero_band_hit(self):
        """With band=0, there's no verify zone — it's either hit or miss."""
        assert decide(0.82, threshold=0.82, band=0.0) == Decision.HIT

    def test_zero_band_miss(self):
        assert decide(0.8199999, threshold=0.82, band=0.0) == Decision.MISS

    # --- Different threshold values ---

    def test_low_threshold(self):
        assert decide(0.50, threshold=0.40, band=0.05) == Decision.HIT
        assert decide(0.37, threshold=0.40, band=0.05) == Decision.VERIFY
        assert decide(0.30, threshold=0.40, band=0.05) == Decision.MISS

    def test_high_threshold(self):
        assert decide(0.96, threshold=0.95, band=0.03) == Decision.HIT
        assert decide(0.93, threshold=0.95, band=0.03) == Decision.VERIFY
        assert decide(0.91, threshold=0.95, band=0.03) == Decision.MISS

    # --- Wide band ---

    def test_wide_band(self):
        """A very wide band means most scores fall into VERIFY."""
        assert decide(0.82, threshold=0.82, band=0.50) == Decision.HIT
        assert decide(0.50, threshold=0.82, band=0.50) == Decision.VERIFY
        assert decide(0.32, threshold=0.82, band=0.50) == Decision.VERIFY
        assert decide(0.31, threshold=0.82, band=0.50) == Decision.MISS

    # --- Negative score (shouldn't happen, but test defensively) ---

    def test_negative_score(self):
        assert decide(-0.1, threshold=0.82, band=0.05) == Decision.MISS
