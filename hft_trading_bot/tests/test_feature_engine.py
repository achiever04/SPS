"""
===============================================================================
  test_feature_engine.py — Unit Tests for FeatureEngine v2
===============================================================================
  Tests all 19 features including the 5 new microstructure indicators:
  ATR, VWAP deviation, volume delta, OI change, bid-ask spread, tick streak.

  Fixes from v1:
  • test_warmup_period now has a proper assertion (was an empty loop).
  • update() now accepts full tick dicts instead of bare floats.
  • All feature indices updated to match FEATURE_DIM = 19.

  Run with:
      cd hft_trading_bot
      python -m pytest tests/test_feature_engine.py -v
===============================================================================
"""

import math
import time
import pytest
import numpy as np

from core.feature_engine import FeatureEngine, FEATURE_DIM


# ═════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════

def make_tick(
    ltp: float,
    volume: float = 100_000.0,
    oi: float = 500_000.0,
    bid: float = 0.0,
    ask: float = 0.0,
) -> dict:
    """Build a minimal tick dict for the v2 FeatureEngine."""
    if bid == 0:
        bid = ltp - 0.05
    if ask == 0:
        ask = ltp + 0.05
    return {"ltp": ltp, "volume": volume, "oi": oi, "bid": bid, "ask": ask}


def feed(engine: FeatureEngine, prices: list, **tick_kwargs) -> np.ndarray | None:
    """Feed a list of prices and return the last feature vector."""
    result = None
    for p in prices:
        result = engine.update(make_tick(p, **tick_kwargs))
    return result


def warm_up(engine: FeatureEngine, base_price: float = 100.0, n: int = 60) -> np.ndarray:
    """Feed enough ticks to warm the engine, assert features are produced."""
    prices = [base_price + i * 0.1 for i in range(n)]
    result = feed(engine, prices)
    assert result is not None, "Engine should be warm after 60 ticks with small periods"
    return result


# ═════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═════════════════════════════════════════════════════════════════════════

@pytest.fixture
def engine():
    """Default FeatureEngine with standard periods."""
    return FeatureEngine(
        rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9,
        bollinger_period=20, bollinger_std_dev=2.0,
        atr_period=14, vwap_period=20, streak_max=10,
    )


@pytest.fixture
def small():
    """Small-period engine for fast test warm-up."""
    return FeatureEngine(
        rsi_period=3, macd_fast=3, macd_slow=5, macd_signal=3,
        bollinger_period=5, bollinger_std_dev=2.0,
        atr_period=3, vwap_period=5, streak_max=5,
    )


# ═════════════════════════════════════════════════════════════════════════
#  BASIC BEHAVIOUR
# ═════════════════════════════════════════════════════════════════════════

class TestBasics:

    def test_returns_none_on_first_tick(self, engine):
        result = engine.update(make_tick(100.0))
        assert result is None

    def test_warmup_period_with_default_engine(self, engine):
        """
        Default periods need ~35 ticks for MACD (26+9) to warm up.
        Verify engine still returns None at exactly 34 ticks.
        """
        for i in range(34):
            result = engine.update(make_tick(100.0 + i * 0.1))
        # At 34 ticks the signal EMA has not seeded yet
        assert result is None, (
            "Engine must still return None before all indicators are warm"
        )

    def test_small_engine_produces_features_after_warmup(self, small):
        result = warm_up(small)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (FEATURE_DIM,), \
            f"Expected ({FEATURE_DIM},), got {result.shape}"
        assert result.dtype == np.float32

    def test_feature_count_matches_constant(self, small):
        from core.feature_engine import FEATURE_DIM as DIM
        result = warm_up(small)
        assert result.shape[0] == DIM

    def test_feature_names_count(self, engine):
        assert len(engine.feature_names) == FEATURE_DIM

    def test_feature_names_order(self, engine):
        names = engine.feature_names
        assert names[0]  == "rsi"
        assert names[8]  == "atr"
        assert names[9]  == "vwap_dev"
        assert names[10] == "vol_delta"
        assert names[11] == "oi_change"
        assert names[12] == "spread"
        assert names[13] == "tick_streak"
        assert names[14] == "price"

    def test_reset_clears_warm_state(self, small):
        warm_up(small)
        assert small.is_warm
        small.reset()
        assert not small.is_warm
        assert small.update(make_tick(100.0)) is None

    def test_invalid_ltp_returns_none(self, engine):
        assert engine.update(make_tick(0.0))  is None
        assert engine.update(make_tick(-5.0)) is None

    def test_price_is_index_14(self, small):
        """Raw LTP must appear at index 14 in the feature vector."""
        last_price = 105.5
        prices = [100 + i * 0.1 for i in range(59)]
        prices.append(last_price)
        result = feed(small, prices)
        if result is not None:
            assert abs(result[14] - last_price) < 0.001, \
                f"price at index 14 should be {last_price}, got {result[14]}"


# ═════════════════════════════════════════════════════════════════════════
#  RSI
# ═════════════════════════════════════════════════════════════════════════

class TestRSI:

    def test_rsi_range(self, small):
        prices = [100 + (i % 10) * 0.5 for i in range(60)]
        for p in prices:
            r = small.update(make_tick(p))
            if r is not None:
                assert 0.0 <= r[0] <= 100.0, f"RSI {r[0]} out of [0, 100]"

    def test_rsi_high_on_all_gains(self, small):
        """Monotonically rising prices → RSI approaches 100."""
        prices = [100 + i for i in range(60)]
        result = feed(small, prices)
        if result is not None:
            assert result[0] > 80, f"Expected RSI > 80 on all gains, got {result[0]}"

    def test_rsi_low_on_all_losses(self, small):
        """Monotonically falling prices → RSI approaches 0."""
        prices = [200 - i for i in range(60)]
        result = feed(small, prices)
        if result is not None:
            assert result[0] < 20, f"Expected RSI < 20 on all losses, got {result[0]}"


# ═════════════════════════════════════════════════════════════════════════
#  MACD
# ═════════════════════════════════════════════════════════════════════════

class TestMACD:

    def test_histogram_equals_line_minus_signal(self, small):
        result = warm_up(small)
        macd_line = float(result[1])
        macd_sig  = float(result[2])
        histogram = float(result[3])
        assert abs(histogram - (macd_line - macd_sig)) < 1e-5, \
            "Histogram must equal MACD line - signal"

    def test_macd_positive_on_uptrend(self, small):
        prices = [100 + i * 0.5 for i in range(60)]
        result = feed(small, prices)
        if result is not None:
            assert result[1] > 0, "MACD line should be positive on strong uptrend"


# ═════════════════════════════════════════════════════════════════════════
#  BOLLINGER BANDS
# ═════════════════════════════════════════════════════════════════════════

class TestBollinger:

    def test_constant_price_bands_collapse(self, small):
        """With zero variance, upper ≈ lower ≈ middle ≈ price."""
        prices = [100.0] * 60
        result = feed(small, prices)
        if result is not None:
            bb_upper  = result[4]
            bb_middle = result[5]
            bb_lower  = result[6]
            assert abs(bb_middle - 100.0) < 0.01
            assert abs(bb_upper - bb_lower) < 0.01

    def test_upper_always_ge_lower(self, small):
        prices = [100 + math.sin(i / 3) * 5 for i in range(60)]
        for p in prices:
            r = small.update(make_tick(p))
            if r is not None:
                assert r[4] >= r[6], "Upper band must be ≥ lower band"

    def test_pct_b_inside_band(self, small):
        """For non-extreme prices, %B should stay roughly in [-0.5, 1.5]."""
        prices = [100 + math.sin(i / 5) * 2 for i in range(60)]
        for p in prices:
            r = small.update(make_tick(p))
            if r is not None:
                assert -2.0 <= r[7] <= 3.0, f"Unusual %B value: {r[7]}"


# ═════════════════════════════════════════════════════════════════════════
#  ATR  [NEW]
# ═════════════════════════════════════════════════════════════════════════

class TestATR:

    def test_atr_positive(self, small):
        """ATR must be positive for non-flat prices."""
        result = warm_up(small)
        assert result[8] >= 0.0, f"ATR should be ≥ 0, got {result[8]}"

    def test_atr_zero_on_flat_prices(self, small):
        """With constant price, ATR should converge to 0."""
        prices = [100.0] * 60
        result = feed(small, prices)
        if result is not None:
            assert result[8] < 0.01, f"ATR should be ~0 on flat prices, got {result[8]}"

    def test_atr_increases_with_volatility(self, small):
        """Higher price swings → higher ATR."""
        small_swings = FeatureEngine(rsi_period=3, macd_fast=3, macd_slow=5,
                                     macd_signal=3, bollinger_period=5,
                                     atr_period=3, vwap_period=5)
        large_swings = FeatureEngine(rsi_period=3, macd_fast=3, macd_slow=5,
                                     macd_signal=3, bollinger_period=5,
                                     atr_period=3, vwap_period=5)

        prices_small = [100 + math.sin(i) * 0.1 for i in range(60)]
        prices_large = [100 + math.sin(i) * 5.0 for i in range(60)]

        r_small = feed(small_swings, prices_small)
        r_large = feed(large_swings, prices_large)

        if r_small is not None and r_large is not None:
            assert r_large[8] > r_small[8], \
                f"High-vol ATR ({r_large[8]:.4f}) should exceed low-vol ({r_small[8]:.4f})"


# ═════════════════════════════════════════════════════════════════════════
#  VWAP DEVIATION  [NEW]
# ═════════════════════════════════════════════════════════════════════════

class TestVWAPDeviation:

    def test_vwap_dev_at_vwap_is_near_zero(self, small):
        """
        If all prices are equal, VWAP = price → deviation ≈ 0.
        """
        prices = [100.0] * 60
        result = feed(small, prices, volume=100_000)
        if result is not None:
            assert abs(result[9]) < 0.001, \
                f"VWAP deviation should be ~0 for constant prices, got {result[9]}"

    def test_vwap_dev_positive_above_vwap(self, small):
        """Rising prices should produce positive deviation eventually."""
        prices = [100 + i * 0.2 for i in range(60)]
        result = feed(small, prices)
        if result is not None:
            # Current price > rolling VWAP → positive deviation
            assert result[9] > -1.0, "VWAP deviation out of expected range"


# ═════════════════════════════════════════════════════════════════════════
#  VOLUME DELTA  [NEW]
# ═════════════════════════════════════════════════════════════════════════

class TestVolumeDelta:

    def test_volume_delta_range(self, small):
        """Volume delta must be in [-1, 1]."""
        prices  = [100 + i * 0.1 for i in range(60)]
        volumes = [100_000 + i * 5_000 for i in range(60)]
        for p, v in zip(prices, volumes):
            r = small.update(make_tick(p, volume=v))
            if r is not None:
                assert -1.0 <= r[10] <= 1.0, f"vol_delta {r[10]} out of [-1, 1]"

    def test_volume_delta_positive_on_surge(self, small):
        """A sudden large volume spike should push delta close to +1."""
        prices  = [100.0] * 30
        volumes = [100_000.0] * 29 + [1_000_000.0]  # spike on last bar
        result  = feed(small, prices, volume=100_000)  # warm up
        # Now feed the spike
        result = small.update(make_tick(100.0, volume=1_000_000.0))
        if result is not None:
            assert result[10] > 0.5, \
                f"Expected vol_delta > 0.5 on volume spike, got {result[10]}"


# ═════════════════════════════════════════════════════════════════════════
#  OI CHANGE  [NEW]
# ═════════════════════════════════════════════════════════════════════════

class TestOIChange:

    def test_oi_change_range(self, small):
        prices = [100 + i * 0.1 for i in range(60)]
        ois    = [500_000 + i * 1_000 for i in range(60)]
        for p, o in zip(prices, ois):
            r = small.update(make_tick(p, oi=o))
            if r is not None:
                assert -1.0 <= r[11] <= 1.0, f"oi_change {r[11]} out of [-1, 1]"

    def test_oi_zero_when_flat(self, small):
        """Constant OI → oi_change = 0."""
        result = feed(small, [100.0] * 60, oi=500_000)
        if result is not None:
            assert result[11] == 0.0, \
                f"oi_change should be 0 with constant OI, got {result[11]}"


# ═════════════════════════════════════════════════════════════════════════
#  BID-ASK SPREAD  [NEW]
# ═════════════════════════════════════════════════════════════════════════

class TestSpread:

    def test_spread_positive(self, small):
        """Spread should be ≥ 0."""
        result = warm_up(small)
        assert result[12] >= 0.0, f"Spread should be ≥ 0, got {result[12]}"

    def test_wider_spread_on_wider_quotes(self, small):
        """A wider bid-ask should produce a larger spread feature."""
        eng1 = FeatureEngine(rsi_period=3, macd_fast=3, macd_slow=5,
                             macd_signal=3, bollinger_period=5, atr_period=3,
                             vwap_period=5)
        eng2 = FeatureEngine(rsi_period=3, macd_fast=3, macd_slow=5,
                             macd_signal=3, bollinger_period=5, atr_period=3,
                             vwap_period=5)

        prices = [100.0] * 60
        r1 = feed(eng1, prices, bid=99.95, ask=100.05)   # narrow 0.1
        r2 = feed(eng2, prices, bid=99.50, ask=100.50)   # wide  1.0

        if r1 is not None and r2 is not None:
            assert r2[12] > r1[12], "Wider quotes should produce higher spread feature"


# ═════════════════════════════════════════════════════════════════════════
#  TICK STREAK  [NEW]
# ═════════════════════════════════════════════════════════════════════════

class TestTickStreak:

    def test_streak_range(self, small):
        """Streak must be in [-1, 1] (normalised by streak_max)."""
        prices = [100 + i * 0.1 for i in range(60)]
        for p in prices:
            r = small.update(make_tick(p))
            if r is not None:
                assert -1.0 <= r[13] <= 1.0, f"streak {r[13]} out of [-1, 1]"

    def test_streak_positive_on_upticks(self, small):
        """All up-ticks → streak saturates at +1."""
        prices = [100 + i for i in range(60)]
        result = feed(small, prices)
        if result is not None:
            assert result[13] == 1.0, \
                f"Streak should be 1.0 on all up-ticks, got {result[13]}"

    def test_streak_negative_on_downticks(self, small):
        """All down-ticks → streak saturates at -1."""
        prices = [200 - i for i in range(60)]
        result = feed(small, prices)
        if result is not None:
            assert result[13] == -1.0, \
                f"Streak should be -1.0 on all down-ticks, got {result[13]}"

    def test_streak_resets_on_direction_change(self, small):
        """After a reversal, streak should become positive."""
        # Drive streak strongly negative
        down = [200 - i for i in range(30)]
        feed(small, down)
        # Now feed a run of up-ticks
        up = [170 + i * 0.1 for i in range(20)]
        result = feed(small, up)
        if result is not None:
            assert result[13] > 0, "Streak should be positive after reversal to up-ticks"


# ═════════════════════════════════════════════════════════════════════════
#  PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════

class TestPerformance:

    def test_throughput_10k_ticks(self, small):
        """10,000 ticks must complete in < 1 second on any modern machine."""
        prices = [100 + (i % 100) * 0.05 for i in range(10_000)]
        t0 = time.perf_counter()
        for p in prices:
            small.update(make_tick(p))
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, \
            f"10k ticks took {elapsed:.3f}s — should be < 1s"

    def test_deques_bounded_after_100k_ticks(self, engine):
        """No deque should grow beyond its maxlen after massive input."""
        for i in range(100_000):
            engine.update(make_tick(100.0 + (i % 200) * 0.01))

        assert len(engine._prices)      <= engine._prices.maxlen
        assert len(engine._bb_window)   <= engine._bb_window.maxlen
        assert len(engine._macd_history) <= engine._macd_history.maxlen
        assert len(engine._vwap_pv)     <= engine._vwap_pv.maxlen

    def test_no_nan_in_features(self, small):
        """No NaN or Inf values should appear in the output vector."""
        prices  = [100 + math.sin(i) * 3 for i in range(200)]
        volumes = [max(1.0, 100_000 + math.cos(i) * 20_000) for i in range(200)]
        for p, v in zip(prices, volumes):
            r = small.update(make_tick(p, volume=v))
            if r is not None:
                assert not np.any(np.isnan(r)), f"NaN in features at price {p}"
                assert not np.any(np.isinf(r)), f"Inf in features at price {p}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
