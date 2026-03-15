"""
===============================================================================
  feature_engine.py — Incremental Technical Indicator Calculator
===============================================================================
  Computes RSI, MACD, and Bollinger Bands on-the-fly from streaming tick data
  using ONLY collections.deque and pure NumPy/math operations.

  ⚡ ZERO PANDAS — Every calculation is incremental (O(1) per tick).
  ⚡ FIXED MEMORY — All state lives in bounded deques.

  Indicators computed:
  ┌─────────────────────┬────────────────────────────────────────────┐
  │ Indicator           │ Method                                     │
  ├─────────────────────┼────────────────────────────────────────────┤
  │ RSI (14)            │ Wilder's smoothed avg gain/loss            │
  │ MACD (12, 26, 9)    │ Incremental EMA with multiplier            │
  │ Bollinger Bands     │ Welford's online mean/variance algorithm    │
  └─────────────────────┴────────────────────────────────────────────┘

  Architecture:
      tick_queue → FeatureEngine.update(ltp) → numpy feature vector → ONNX
===============================================================================
"""

import math
import time
from collections import deque
from typing import Optional

import numpy as np

from utils.logger import get_logger, log_latency

# ── Module logger ───────────────────────────────────────────────────────
logger = get_logger("FEATURE")


class FeatureEngine:
    """
    High-performance incremental feature calculator.

    Maintains internal state for each indicator using bounded deques.
    Each call to update() ingests a single price tick and recomputes
    all features in O(1) amortized time.

    Attributes:
        rsi_period:         Lookback period for RSI calculation.
        macd_fast:          Fast EMA period for MACD.
        macd_slow:          Slow EMA period for MACD.
        macd_signal:        Signal line EMA period for MACD.
        bollinger_period:   Lookback window for Bollinger Bands.
        bollinger_std_dev:  Standard deviation multiplier for bands.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bollinger_period: int = 20,
        bollinger_std_dev: float = 2.0,
    ):
        """
        Initialize the feature engine with configurable indicator periods.

        Args:
            rsi_period:       RSI lookback (default 14).
            macd_fast:        Fast EMA period (default 12).
            macd_slow:        Slow EMA period (default 26).
            macd_signal:      Signal line period (default 9).
            bollinger_period: Bollinger window (default 20).
            bollinger_std_dev: Band width multiplier (default 2.0).
        """
        # ── Configuration ────────────────────────────────────────────────
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bollinger_period = bollinger_period
        self.bollinger_std_dev = bollinger_std_dev

        # ── Price history (bounded) ──────────────────────────────────────
        # We only store enough prices to bootstrap the slowest indicator
        self._prices: deque = deque(maxlen=max(macd_slow, bollinger_period) + 5)

        # ── RSI State (Wilder's smoothed method) ─────────────────────────
        self._rsi_avg_gain: float = 0.0
        self._rsi_avg_loss: float = 0.0
        self._rsi_initialized: bool = False
        self._rsi_tick_count: int = 0

        # ── MACD State (Incremental EMA) ─────────────────────────────────
        self._ema_fast: float = 0.0
        self._ema_slow: float = 0.0
        self._ema_signal: float = 0.0
        self._ema_fast_multiplier: float = 2.0 / (macd_fast + 1)
        self._ema_slow_multiplier: float = 2.0 / (macd_slow + 1)
        self._ema_signal_multiplier: float = 2.0 / (macd_signal + 1)
        self._macd_initialized: bool = False
        self._macd_tick_count: int = 0
        self._macd_history: deque = deque(maxlen=macd_signal + 5)

        # ── Bollinger Bands State (Welford's online algorithm) ───────────
        self._bb_window: deque = deque(maxlen=bollinger_period)
        self._bb_sum: float = 0.0
        self._bb_sum_sq: float = 0.0

        # ── Total update count ───────────────────────────────────────────
        self._tick_count: int = 0

        logger.info(
            f"FeatureEngine initialized | "
            f"RSI({rsi_period}) MACD({macd_fast},{macd_slow},{macd_signal}) "
            f"BB({bollinger_period},{bollinger_std_dev})"
        )

    # ═════════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ═════════════════════════════════════════════════════════════════════

    def update(self, price: float) -> Optional[np.ndarray]:
        """
        Ingest a new price tick and recompute all indicators.

        Args:
            price: The latest traded price (LTP).

        Returns:
            A flat numpy array of features [rsi, macd, macd_signal,
            macd_histogram, bb_upper, bb_middle, bb_lower, bb_pct_b, price]
            or None if not enough data has been accumulated for all indicators.

        Performance:
            Amortized O(1) per call. No heap allocations in steady state.
        """
        t_start = time.perf_counter_ns()
        self._tick_count += 1

        # ── Store the price ──────────────────────────────────────────────
        self._prices.append(price)

        # ── Update each indicator incrementally ──────────────────────────
        rsi = self._update_rsi(price)
        macd_line, macd_sig, macd_hist = self._update_macd(price)
        bb_upper, bb_middle, bb_lower, bb_pct_b = self._update_bollinger(price)

        # ── Check if all indicators are warmed up ────────────────────────
        if rsi is None or macd_line is None or bb_upper is None:
            if self._tick_count % 100 == 0:
                logger.debug(
                    f"Warming up... tick {self._tick_count} | "
                    f"RSI: {'✅' if rsi is not None else '⏳'} "
                    f"MACD: {'✅' if macd_line is not None else '⏳'} "
                    f"BB: {'✅' if bb_upper is not None else '⏳'}"
                )
            return None

        # ── Assemble feature vector ──────────────────────────────────────
        # Shape: (9,) — flat array ready for ONNX inference
        features = np.array([
            rsi,
            macd_line,
            macd_sig,
            macd_hist,
            bb_upper,
            bb_middle,
            bb_lower,
            bb_pct_b,
            price,
        ], dtype=np.float32)

        # ── Log latency periodically ─────────────────────────────────────
        if self._tick_count % 500 == 0:
            log_latency(logger, "FEATURE_COMPUTE", t_start)

        return features

    @property
    def feature_names(self) -> list[str]:
        """Return ordered list of feature names matching the output vector."""
        return [
            "rsi", "macd_line", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower", "bb_pct_b", "price",
        ]

    @property
    def is_warm(self) -> bool:
        """Return True if all indicators have enough data to compute."""
        return (
            self._rsi_initialized
            and self._macd_initialized
            and len(self._bb_window) >= self.bollinger_period
        )

    # ═════════════════════════════════════════════════════════════════════
    #  RSI — Wilder's Smoothed Average Gain/Loss (Incremental)
    # ═════════════════════════════════════════════════════════════════════

    def _update_rsi(self, price: float) -> Optional[float]:
        """
        Incrementally update RSI using Wilder's smoothing method.

        Phase 1 (Initialization): Accumulate first `rsi_period + 1` prices,
        then compute the initial average gain and average loss.

        Phase 2 (Incremental): For each subsequent tick, update averages
        using the smoothing formula:
            avg_gain = (prev_avg_gain * (n-1) + current_gain) / n
            avg_loss = (prev_avg_loss * (n-1) + current_loss) / n

        Returns:
            RSI value (0–100) or None if still warming up.
        """
        self._rsi_tick_count += 1

        if self._rsi_tick_count < 2:
            return None  # Need at least 2 prices for a change

        # ── Calculate price change ───────────────────────────────────────
        prev_price = self._prices[-2]
        change = price - prev_price
        gain = max(change, 0.0)
        loss = abs(min(change, 0.0))

        if not self._rsi_initialized:
            # ── Phase 1: Accumulating initial values ─────────────────────
            self._rsi_avg_gain += gain
            self._rsi_avg_loss += loss

            if self._rsi_tick_count == self.rsi_period + 1:
                # First valid RSI — compute initial averages
                self._rsi_avg_gain /= self.rsi_period
                self._rsi_avg_loss /= self.rsi_period
                self._rsi_initialized = True
                logger.info(f"RSI initialized after {self._rsi_tick_count} ticks")
            else:
                return None
        else:
            # ── Phase 2: Incremental smoothing ───────────────────────────
            n = self.rsi_period
            self._rsi_avg_gain = (self._rsi_avg_gain * (n - 1) + gain) / n
            self._rsi_avg_loss = (self._rsi_avg_loss * (n - 1) + loss) / n

        # ── Compute RSI ──────────────────────────────────────────────────
        if self._rsi_avg_loss == 0:
            return 100.0  # No losses → maximum strength

        rs = self._rsi_avg_gain / self._rsi_avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    # ═════════════════════════════════════════════════════════════════════
    #  MACD — Incremental EMA(12), EMA(26), Signal EMA(9)
    # ═════════════════════════════════════════════════════════════════════

    def _update_macd(
        self, price: float
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Incrementally update MACD using EMA multipliers.

        EMA update formula:
            ema = (price - prev_ema) * multiplier + prev_ema

        Returns:
            Tuple of (macd_line, signal_line, histogram) or (None, None, None).
        """
        self._macd_tick_count += 1

        if self._macd_tick_count == 1:
            # ── Seed EMAs with the first price ───────────────────────────
            self._ema_fast = price
            self._ema_slow = price
            return None, None, None

        # ── Update fast and slow EMAs ────────────────────────────────────
        self._ema_fast = (
            (price - self._ema_fast) * self._ema_fast_multiplier + self._ema_fast
        )
        self._ema_slow = (
            (price - self._ema_slow) * self._ema_slow_multiplier + self._ema_slow
        )

        # ── Need at least `macd_slow` ticks for valid slow EMA ───────────
        if self._macd_tick_count < self.macd_slow:
            return None, None, None

        # ── MACD line = Fast EMA - Slow EMA ──────────────────────────────
        macd_line = self._ema_fast - self._ema_slow
        self._macd_history.append(macd_line)

        # ── Signal line (EMA of MACD line) ───────────────────────────────
        if not self._macd_initialized:
            if len(self._macd_history) >= self.macd_signal:
                # Seed signal EMA with SMA of first `macd_signal` MACD values
                self._ema_signal = sum(self._macd_history) / len(self._macd_history)
                self._macd_initialized = True
                logger.info(f"MACD initialized after {self._macd_tick_count} ticks")
            else:
                return None, None, None
        else:
            self._ema_signal = (
                (macd_line - self._ema_signal) * self._ema_signal_multiplier
                + self._ema_signal
            )

        # ── Histogram = MACD line - Signal line ──────────────────────────
        histogram = macd_line - self._ema_signal

        return macd_line, self._ema_signal, histogram

    # ═════════════════════════════════════════════════════════════════════
    #  BOLLINGER BANDS — Online Mean/Variance via Running Sums
    # ═════════════════════════════════════════════════════════════════════

    def _update_bollinger(
        self, price: float
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Update Bollinger Bands using a running sum and sum-of-squares
        approach for O(1) incremental mean and standard deviation.

        When the deque is full, the oldest value is subtracted from
        the running sums before the new value is added — no recomputation.

        Returns:
            Tuple of (upper_band, middle_band, lower_band, %B) or all None.
        """
        # ── Remove oldest value from sums if window is full ──────────────
        if len(self._bb_window) == self.bollinger_period:
            oldest = self._bb_window[0]  # Will be popped by deque maxlen
            self._bb_sum -= oldest
            self._bb_sum_sq -= oldest * oldest

        # ── Add new value ────────────────────────────────────────────────
        self._bb_window.append(price)
        self._bb_sum += price
        self._bb_sum_sq += price * price

        # ── Need full window ─────────────────────────────────────────────
        n = len(self._bb_window)
        if n < self.bollinger_period:
            return None, None, None, None

        # ── Compute mean (middle band) ───────────────────────────────────
        mean = self._bb_sum / n

        # ── Compute standard deviation ───────────────────────────────────
        # Variance = E[X²] - (E[X])²
        variance = (self._bb_sum_sq / n) - (mean * mean)

        # Guard against floating-point negative variance
        std_dev = math.sqrt(max(variance, 0.0))

        # ── Compute bands ────────────────────────────────────────────────
        upper = mean + (self.bollinger_std_dev * std_dev)
        lower = mean - (self.bollinger_std_dev * std_dev)

        # ── %B = (Price - Lower) / (Upper - Lower) ──────────────────────
        band_width = upper - lower
        pct_b = (price - lower) / band_width if band_width > 0 else 0.5

        return upper, mean, lower, pct_b

    # ═════════════════════════════════════════════════════════════════════
    #  RESET
    # ═════════════════════════════════════════════════════════════════════

    def reset(self) -> None:
        """
        Reset all internal state. Useful for session boundaries
        (e.g., market open/close transitions).
        """
        self._prices.clear()
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        self._rsi_initialized = False
        self._rsi_tick_count = 0
        self._ema_fast = 0.0
        self._ema_slow = 0.0
        self._ema_signal = 0.0
        self._macd_initialized = False
        self._macd_tick_count = 0
        self._macd_history.clear()
        self._bb_window.clear()
        self._bb_sum = 0.0
        self._bb_sum_sq = 0.0
        self._tick_count = 0
        logger.info("FeatureEngine state reset")
