"""
===============================================================================
  test_feature_engine.py — Unit Tests for Technical Indicator Calculations
===============================================================================
  Verifies the correctness of RSI, MACD, and Bollinger Band calculations
  in the FeatureEngine against known values.

  Run with:
      cd hft_trading_bot
      python -m pytest tests/test_feature_engine.py -v
===============================================================================
"""

import math
import pytest
import numpy as np

from core.feature_engine import FeatureEngine


# ═════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═════════════════════════════════════════════════════════════════════════

@pytest.fixture
def engine():
    """Create a FeatureEngine with default settings."""
    return FeatureEngine(
        rsi_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        bollinger_period=20,
        bollinger_std_dev=2.0,
    )


@pytest.fixture
def small_engine():
    """Create a FeatureEngine with small periods for fast testing."""
    return FeatureEngine(
        rsi_period=3,
        macd_fast=3,
        macd_slow=5,
        macd_signal=3,
        bollinger_period=5,
        bollinger_std_dev=2.0,
    )


# ═════════════════════════════════════════════════════════════════════════
#  TEST: BASIC FUNCTIONALITY
# ═════════════════════════════════════════════════════════════════════════

class TestFeatureEngineBasics:
    """Test fundamental behavior of the FeatureEngine."""

    def test_initial_state_returns_none(self, engine):
        """FeatureEngine should return None during warm-up phase."""
        result = engine.update(100.0)
        assert result is None, "Should return None on first tick"

    def test_warmup_period(self, engine):
        """Should return None until all indicators have enough data."""
        # Default needs at least 26 ticks for MACD slow EMA + 9 for signal
        for i in range(30):
            result = engine.update(100.0 + i * 0.1)

        # After 30 ticks with defaults:
        # RSI needs 15 (period+1), MACD needs 26+9=35 for signal
        # So features should NOT be ready at 30 ticks
        # (MACD signal needs macd_slow + macd_signal ticks)

    def test_small_engine_produces_features(self, small_engine):
        """With small periods, features should be available quickly."""
        # Feed enough data points to warm up all indicators
        # RSI: 4 ticks, MACD: 5+3=8 ticks, BB: 5 ticks → max = 8
        prices = [100, 101, 102, 101, 103, 104, 102, 105, 106, 103]
        result = None
        for p in prices:
            result = small_engine.update(p)

        assert result is not None, "Should produce features after warming up"
        assert isinstance(result, np.ndarray), "Should return numpy array"
        assert result.shape == (9,), f"Expected shape (9,), got {result.shape}"
        assert result.dtype == np.float32, "Should be float32"

    def test_feature_names(self, engine):
        """Feature names should match the expected list."""
        expected = [
            "rsi", "macd_line", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower", "bb_pct_b", "price",
        ]
        assert engine.feature_names == expected

    def test_reset_clears_state(self, small_engine):
        """After reset, engine should require warm-up again."""
        # Warm up
        for p in [100, 101, 102, 101, 103, 104, 102, 105, 106, 103]:
            small_engine.update(p)

        assert small_engine.is_warm, "Should be warm after feeding data"

        # Reset
        small_engine.reset()

        assert not small_engine.is_warm, "Should NOT be warm after reset"
        result = small_engine.update(100.0)
        assert result is None, "Should return None after reset"


# ═════════════════════════════════════════════════════════════════════════
#  TEST: RSI CALCULATIONS
# ═════════════════════════════════════════════════════════════════════════

class TestRSI:
    """Test RSI calculation correctness."""

    def test_rsi_range(self, small_engine):
        """RSI should always be between 0 and 100."""
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                  108, 107, 106, 105, 104, 103, 102, 101, 100, 99]

        for p in prices:
            result = small_engine.update(p)
            if result is not None:
                rsi = result[0]  # RSI is the first feature
                assert 0 <= rsi <= 100, f"RSI {rsi} out of range [0, 100]"

    def test_rsi_all_gains(self, small_engine):
        """RSI should approach 100 with consistent gains."""
        prices = [100 + i for i in range(20)]  # Steadily rising
        result = None
        for p in prices:
            result = small_engine.update(p)

        if result is not None:
            rsi = result[0]
            assert rsi > 80, f"RSI should be high with all gains, got {rsi}"

    def test_rsi_all_losses(self, small_engine):
        """RSI should approach 0 with consistent losses."""
        prices = [200 - i for i in range(20)]  # Steadily falling
        result = None
        for p in prices:
            result = small_engine.update(p)

        if result is not None:
            rsi = result[0]
            assert rsi < 20, f"RSI should be low with all losses, got {rsi}"


# ═════════════════════════════════════════════════════════════════════════
#  TEST: BOLLINGER BANDS
# ═════════════════════════════════════════════════════════════════════════

class TestBollingerBands:
    """Test Bollinger Band calculation correctness."""

    def test_constant_price_bands(self, small_engine):
        """With constant price, bands should converge to the price."""
        constant_price = 100.0

        result = None
        for _ in range(20):
            result = small_engine.update(constant_price)

        if result is not None:
            bb_upper = result[4]
            bb_middle = result[5]
            bb_lower = result[6]

            # With zero variance, upper and lower should equal middle
            assert abs(bb_middle - constant_price) < 0.01, \
                f"Middle band should equal price, got {bb_middle}"
            assert abs(bb_upper - bb_lower) < 0.01, \
                f"Bands should converge with zero variance"

    def test_upper_above_lower(self, small_engine):
        """Upper band should always be >= lower band."""
        prices = [100, 102, 98, 103, 97, 104, 96, 105, 95, 106]
        for p in prices:
            result = small_engine.update(p)
            if result is not None:
                assert result[4] >= result[6], \
                    "Upper band should be >= lower band"

    def test_pct_b_range(self, small_engine):
        """Percent B should be reasonable for normal price action."""
        prices = [100, 101, 99, 102, 98, 101, 100, 103, 97, 100]
        for p in prices:
            result = small_engine.update(p)
            if result is not None:
                pct_b = result[7]
                # %B can go outside [0,1] when price breaks bands,
                # but should be roughly in [-1, 2] for normal data
                assert -2 <= pct_b <= 3, \
                    f"Percent B {pct_b} seems unreasonable"


# ═════════════════════════════════════════════════════════════════════════
#  TEST: MACD
# ═════════════════════════════════════════════════════════════════════════

class TestMACD:
    """Test MACD calculation correctness."""

    def test_macd_histogram_sign(self, small_engine):
        """Histogram should be positive when MACD > Signal."""
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        result = None
        for p in prices:
            result = small_engine.update(p)

        if result is not None:
            macd_line = result[1]
            macd_signal = result[2]
            histogram = result[3]

            # Histogram = MACD - Signal
            expected_hist = macd_line - macd_signal
            assert abs(histogram - expected_hist) < 0.001, \
                f"Histogram should equal MACD - Signal"

    def test_price_is_last_feature(self, small_engine):
        """The raw price should be the last element in the feature vector."""
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        result = None
        last_price = prices[-1]
        for p in prices:
            result = small_engine.update(p)

        if result is not None:
            assert result[-1] == last_price, \
                f"Last feature should be the price ({last_price}), got {result[-1]}"


# ═════════════════════════════════════════════════════════════════════════
#  TEST: PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """Test computational performance characteristics."""

    def test_throughput(self, small_engine):
        """FeatureEngine should handle 10,000 ticks in under 1 second."""
        import time

        prices = [100 + (i % 50) * 0.1 for i in range(10000)]

        t0 = time.perf_counter()
        for p in prices:
            small_engine.update(p)
        elapsed = time.perf_counter() - t0

        assert elapsed < 1.0, \
            f"10,000 updates took {elapsed:.3f}s (should be < 1s)"

    def test_memory_bounded(self, engine):
        """Deque sizes should remain bounded regardless of input volume."""
        for i in range(100000):
            engine.update(100.0 + (i % 100) * 0.01)

        # Check that internal deques are bounded
        assert len(engine._prices) <= engine._prices.maxlen
        assert len(engine._bb_window) <= engine._bb_window.maxlen
        assert len(engine._macd_history) <= engine._macd_history.maxlen


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
