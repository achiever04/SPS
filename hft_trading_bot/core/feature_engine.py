"""
===============================================================================
  feature_engine.py — Incremental Technical Indicator Calculator  (v2)
===============================================================================
  Computes all indicators on-the-fly from streaming tick data using ONLY
  collections.deque and pure NumPy/math.  Zero Pandas, O(1) per tick.

  ┌─────────────────────────┬────────────────────────────────────────────┐
  │ Indicator               │ Method                                     │
  ├─────────────────────────┼────────────────────────────────────────────┤
  │ RSI (14)                │ Wilder's smoothed avg gain/loss            │
  │ MACD (12, 26, 9)        │ Incremental EMA with multiplier            │
  │ Bollinger Bands (20)    │ Running sum / sum-of-squares               │
  │ ATR (14)        [NEW]   │ Wilder's smoothed true range               │
  │ VWAP deviation  [NEW]   │ (ltp - vwap) / vwap rolling window         │
  │ Volume delta    [NEW]   │ (vol_now - vol_prev) normalised            │
  │ OI change       [NEW]   │ (oi_now  - oi_prev)  normalised            │
  │ Bid-ask spread  [NEW]   │ (ask - bid) / ltp                          │
  │ Tick streak     [NEW]   │ signed consecutive up/down tick count       │
  └─────────────────────────┴────────────────────────────────────────────┘

  Feature vector (19 elements):
  [rsi, macd_line, macd_signal, macd_hist,
   bb_upper, bb_middle, bb_lower, bb_pct_b,
   atr, vwap_dev, vol_delta, oi_change, spread, tick_streak,
   price, volume, oi, bid, ask]

  Architecture:
      tick_queue → FeatureEngine.update(tick) → numpy (19,) → ONNX
===============================================================================
"""

import math
import time
from collections import deque
from typing import Optional

import numpy as np

from utils.logger import get_logger, log_latency

logger = get_logger("FEATURE")

# Number of features in the output vector
FEATURE_DIM = 19


class FeatureEngine:
    """
    High-performance incremental feature calculator (v2).

    Accepts full tick dicts (ltp, volume, oi, bid, ask) so that
    microstructure features (spread, volume delta, OI change) can be
    computed alongside the price-based indicators.

    All state lives in bounded deques — memory usage is O(max_period).
    """

    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bollinger_period: int = 20,
        bollinger_std_dev: float = 2.0,
        atr_period: int = 14,
        vwap_period: int = 20,
        streak_max: int = 10,
    ):
        # ── Configuration ────────────────────────────────────────────────
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bollinger_period = bollinger_period
        self.bollinger_std_dev = bollinger_std_dev
        self.atr_period = atr_period
        self.vwap_period = vwap_period
        self.streak_max = streak_max

        _maxlen = max(macd_slow, bollinger_period, atr_period, vwap_period) + 10

        # ── Price history ─────────────────────────────────────────────────
        self._prices: deque = deque(maxlen=_maxlen)

        # ── RSI state (Wilder's) ─────────────────────────────────────────
        self._rsi_avg_gain: float = 0.0
        self._rsi_avg_loss: float = 0.0
        self._rsi_initialized: bool = False
        self._rsi_tick_count: int = 0

        # ── MACD state ───────────────────────────────────────────────────
        self._ema_fast: float = 0.0
        self._ema_slow: float = 0.0
        self._ema_signal: float = 0.0
        self._ema_fast_k: float = 2.0 / (macd_fast + 1)
        self._ema_slow_k: float = 2.0 / (macd_slow + 1)
        self._ema_signal_k: float = 2.0 / (macd_signal + 1)
        self._macd_initialized: bool = False
        self._macd_tick_count: int = 0
        self._macd_history: deque = deque(maxlen=macd_signal + 5)

        # ── Bollinger state ──────────────────────────────────────────────
        self._bb_window: deque = deque(maxlen=bollinger_period)
        self._bb_sum: float = 0.0
        self._bb_sum_sq: float = 0.0

        # ── ATR state (Wilder's smoothed true range) ─────────────────────
        self._atr_value: float = 0.0
        self._atr_initialized: bool = False
        self._atr_tick_count: int = 0
        self._prev_close: float = 0.0        # previous ltp used as prev close
        self._atr_sum: float = 0.0           # accumulator during init phase

        # ── VWAP state (rolling window) ──────────────────────────────────
        # Stores (price * volume, volume) tuples for rolling VWAP
        self._vwap_pv: deque = deque(maxlen=vwap_period)   # price*volume
        self._vwap_v: deque = deque(maxlen=vwap_period)    # volume
        self._vwap_pv_sum: float = 0.0
        self._vwap_v_sum: float = 0.0

        # ── Volume / OI state ─────────────────────────────────────────────
        self._prev_volume: float = 0.0
        self._prev_oi: float = 0.0
        # Running max for normalisation (to keep delta in [-1, 1])
        self._vol_delta_max: float = 1.0
        self._oi_change_max: float = 1.0

        # ── Bid-ask spread state ─────────────────────────────────────────
        # Exponential moving average of spread for smoothing
        self._spread_ema: float = 0.0
        self._spread_k: float = 0.1          # small k → heavy smoothing

        # ── Tick direction streak ────────────────────────────────────────
        self._streak: int = 0               # positive = up-ticks, negative = down
        self._prev_ltp: float = 0.0

        # ── Counters ─────────────────────────────────────────────────────
        self._tick_count: int = 0

        logger.info(
            f"FeatureEngine v2 initialized | "
            f"RSI({rsi_period}) MACD({macd_fast},{macd_slow},{macd_signal}) "
            f"BB({bollinger_period}) ATR({atr_period}) VWAP({vwap_period})"
        )

    # ═════════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ═════════════════════════════════════════════════════════════════════

    def update(self, tick: dict) -> Optional[np.ndarray]:
        """
        Ingest a full tick dict and return a feature vector.

        Args:
            tick: Dict with keys: ltp, volume, oi, bid, ask.
                  Missing keys default to 0 gracefully.

        Returns:
            numpy float32 array of shape (FEATURE_DIM,) = (19,)
            or None during the warm-up phase.
        """
        t_start = time.perf_counter_ns()
        self._tick_count += 1

        # ── Extract tick fields ──────────────────────────────────────────
        ltp: float = float(tick.get("ltp", 0.0))
        volume: float = float(tick.get("volume", 0.0))
        oi: float = float(tick.get("oi", 0.0))
        bid: float = float(tick.get("bid", 0.0))
        ask: float = float(tick.get("ask", 0.0))

        if ltp <= 0:
            return None

        self._prices.append(ltp)

        # ── Update each indicator ────────────────────────────────────────
        rsi = self._update_rsi(ltp)
        macd_line, macd_sig, macd_hist = self._update_macd(ltp)
        bb_upper, bb_mid, bb_lower, bb_pct_b = self._update_bollinger(ltp)
        atr = self._update_atr(ltp)
        vwap_dev = self._update_vwap(ltp, volume)
        vol_delta = self._update_volume_delta(volume)
        oi_change = self._update_oi_change(oi)
        spread = self._update_spread(ltp, bid, ask)
        streak = self._update_streak(ltp)

        # ── Gate: wait until every indicator has enough data ─────────────
        if any(v is None for v in [rsi, macd_line, bb_upper, atr, vwap_dev]):
            if self._tick_count % 100 == 0:
                logger.debug(
                    f"Warming up tick {self._tick_count} | "
                    f"RSI={'✅' if rsi is not None else '⏳'} "
                    f"MACD={'✅' if macd_line is not None else '⏳'} "
                    f"BB={'✅' if bb_upper is not None else '⏳'} "
                    f"ATR={'✅' if atr is not None else '⏳'} "
                    f"VWAP={'✅' if vwap_dev is not None else '⏳'}"
                )
            return None

        # ── Assemble feature vector (19 elements) ────────────────────────
        features = np.array([
            rsi,            # 0  — momentum (0–100)
            macd_line,      # 1  — trend
            macd_sig,       # 2  — signal
            macd_hist,      # 3  — histogram
            bb_upper,       # 4  — volatility band
            bb_mid,         # 5
            bb_lower,       # 6
            bb_pct_b,       # 7  — position within band
            atr,            # 8  — volatility regime [NEW]
            vwap_dev,       # 9  — mean-reversion signal [NEW]
            vol_delta,      # 10 — volume momentum [NEW]
            oi_change,      # 11 — open interest flow [NEW]
            spread,         # 12 — liquidity signal [NEW]
            float(streak),  # 13 — tick momentum [NEW]
            ltp,            # 14 — raw price
            volume,         # 15 — raw volume
            oi,             # 16 — raw OI
            bid,            # 17 — raw bid
            ask,            # 18 — raw ask
        ], dtype=np.float32)

        if self._tick_count % 500 == 0:
            log_latency(logger, "FEATURE_COMPUTE", t_start)

        return features

    @property
    def feature_names(self) -> list[str]:
        """Return ordered feature names matching the output vector."""
        return [
            "rsi", "macd_line", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower", "bb_pct_b",
            "atr", "vwap_dev", "vol_delta", "oi_change",
            "spread", "tick_streak",
            "price", "volume", "oi", "bid", "ask",
        ]

    @property
    def is_warm(self) -> bool:
        """True when all indicators have accumulated enough data."""
        return (
            self._rsi_initialized
            and self._macd_initialized
            and len(self._bb_window) >= self.bollinger_period
            and self._atr_initialized
            and len(self._vwap_v) >= self.vwap_period
        )

    # ═════════════════════════════════════════════════════════════════════
    #  RSI — Wilder's Smoothed Average Gain/Loss
    # ═════════════════════════════════════════════════════════════════════

    def _update_rsi(self, price: float) -> Optional[float]:
        """
        Incremental RSI using Wilder's smoothing.

        Phase 1: Accumulate raw gains/losses for rsi_period + 1 ticks,
                 then compute initial averages.
        Phase 2: Rolling exponential smoothing on each new tick.

        Returns RSI (0–100) or None during warm-up.
        """
        self._rsi_tick_count += 1

        if self._rsi_tick_count < 2:
            return None

        prev = self._prices[-2]
        change = price - prev
        gain = max(change, 0.0)
        loss = abs(min(change, 0.0))

        if not self._rsi_initialized:
            self._rsi_avg_gain += gain
            self._rsi_avg_loss += loss

            if self._rsi_tick_count == self.rsi_period + 1:
                self._rsi_avg_gain /= self.rsi_period
                self._rsi_avg_loss /= self.rsi_period
                self._rsi_initialized = True
                logger.info(f"RSI initialized after {self._rsi_tick_count} ticks")
            else:
                return None
        else:
            n = self.rsi_period
            self._rsi_avg_gain = (self._rsi_avg_gain * (n - 1) + gain) / n
            self._rsi_avg_loss = (self._rsi_avg_loss * (n - 1) + loss) / n

        if self._rsi_avg_loss == 0:
            return 100.0

        rs = self._rsi_avg_gain / self._rsi_avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    # ═════════════════════════════════════════════════════════════════════
    #  MACD — Incremental EMA(fast), EMA(slow), Signal EMA
    # ═════════════════════════════════════════════════════════════════════

    def _update_macd(
        self, price: float
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Incremental MACD.  Seed EMAs with first price, then apply
        multiplier formula each tick.

        Returns (macd_line, signal_line, histogram) or (None, None, None).
        """
        self._macd_tick_count += 1

        if self._macd_tick_count == 1:
            self._ema_fast = price
            self._ema_slow = price
            return None, None, None

        self._ema_fast += (price - self._ema_fast) * self._ema_fast_k
        self._ema_slow += (price - self._ema_slow) * self._ema_slow_k

        if self._macd_tick_count < self.macd_slow:
            return None, None, None

        macd_line = self._ema_fast - self._ema_slow
        self._macd_history.append(macd_line)

        if not self._macd_initialized:
            if len(self._macd_history) >= self.macd_signal:
                # Seed signal EMA with SMA of first macd_signal MACD values
                self._ema_signal = sum(self._macd_history) / len(self._macd_history)
                self._macd_initialized = True
                logger.info(f"MACD initialized after {self._macd_tick_count} ticks")
            else:
                return None, None, None
        else:
            self._ema_signal += (macd_line - self._ema_signal) * self._ema_signal_k

        return macd_line, self._ema_signal, macd_line - self._ema_signal

    # ═════════════════════════════════════════════════════════════════════
    #  BOLLINGER BANDS — Running Sum/Sum-of-Squares
    # ═════════════════════════════════════════════════════════════════════

    def _update_bollinger(
        self, price: float
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Bollinger Bands via running sum and sum-of-squares.
        O(1) per tick, no recomputation.

        Returns (upper, middle, lower, %B) or all None.
        """
        if len(self._bb_window) == self.bollinger_period:
            oldest = self._bb_window[0]
            self._bb_sum -= oldest
            self._bb_sum_sq -= oldest * oldest

        self._bb_window.append(price)
        self._bb_sum += price
        self._bb_sum_sq += price * price

        n = len(self._bb_window)
        if n < self.bollinger_period:
            return None, None, None, None

        mean = self._bb_sum / n
        variance = (self._bb_sum_sq / n) - (mean * mean)
        std_dev = math.sqrt(max(variance, 0.0))

        upper = mean + self.bollinger_std_dev * std_dev
        lower = mean - self.bollinger_std_dev * std_dev
        band_w = upper - lower
        pct_b = (price - lower) / band_w if band_w > 0 else 0.5

        return upper, mean, lower, pct_b

    # ═════════════════════════════════════════════════════════════════════
    #  ATR — Average True Range (Wilder's Smoothing)  [NEW]
    # ═════════════════════════════════════════════════════════════════════

    def _update_atr(self, price: float) -> Optional[float]:
        """
        Incremental ATR using Wilder's smoothing.

        True Range = max(high - low, |high - prev_close|, |low - prev_close|)
        For tick data we approximate: high = low = ltp, so:
            TR = |ltp - prev_ltp|

        Phase 1 (atr_period ticks): sum raw TRs.
        Phase 2: Wilder's smooth: ATR = (prev_ATR * (n-1) + TR) / n

        Returns ATR value or None during warm-up.
        """
        self._atr_tick_count += 1

        if self._atr_tick_count < 2:
            self._prev_close = price
            return None

        tr = abs(price - self._prev_close)

        if not self._atr_initialized:
            self._atr_sum += tr
            if self._atr_tick_count == self.atr_period + 1:
                self._atr_value = self._atr_sum / self.atr_period
                self._atr_initialized = True
                logger.info(f"ATR initialized after {self._atr_tick_count} ticks")
            else:
                self._prev_close = price
                return None
        else:
            n = self.atr_period
            self._atr_value = (self._atr_value * (n - 1) + tr) / n

        self._prev_close = price
        return self._atr_value

    # ═════════════════════════════════════════════════════════════════════
    #  VWAP DEVIATION — Rolling Window  [NEW]
    # ═════════════════════════════════════════════════════════════════════

    def _update_vwap(self, price: float, volume: float) -> Optional[float]:
        """
        Rolling VWAP deviation: (price - vwap) / vwap

        Uses a sliding window of vwap_period ticks.  Removes the oldest
        (price * volume, volume) pair when the window is full.

        Volume = 0 ticks (e.g. first tick) use a minimal weight of 1
        to avoid division by zero.

        Returns normalised deviation or None until window fills.
        """
        v = max(volume, 1.0)      # guard against zero volume ticks
        pv = price * v

        if len(self._vwap_v) == self.vwap_period:
            # Remove the oldest values from running sums
            old_pv = self._vwap_pv[0]
            old_v = self._vwap_v[0]
            self._vwap_pv_sum -= old_pv
            self._vwap_v_sum -= old_v

        self._vwap_pv.append(pv)
        self._vwap_v.append(v)
        self._vwap_pv_sum += pv
        self._vwap_v_sum += v

        if len(self._vwap_v) < self.vwap_period:
            return None

        vwap = self._vwap_pv_sum / self._vwap_v_sum
        if vwap == 0:
            return 0.0
        return (price - vwap) / vwap

    # ═════════════════════════════════════════════════════════════════════
    #  VOLUME DELTA — Normalised Tick-to-Tick Volume Change  [NEW]
    # ═════════════════════════════════════════════════════════════════════

    def _update_volume_delta(self, volume: float) -> float:
        """
        Compute normalised volume delta.

        delta = (current_volume - prev_volume) / max_observed_delta

        Clipped to [-1, 1] after normalisation.  The running max prevents
        stale small values from dominating early in the session.
        """
        delta = volume - self._prev_volume
        self._prev_volume = volume

        abs_delta = abs(delta)
        if abs_delta > self._vol_delta_max:
            self._vol_delta_max = abs_delta

        if self._vol_delta_max == 0:
            return 0.0

        return max(-1.0, min(1.0, delta / self._vol_delta_max))

    # ═════════════════════════════════════════════════════════════════════
    #  OI CHANGE — Normalised Open Interest Change  [NEW]
    # ═════════════════════════════════════════════════════════════════════

    def _update_oi_change(self, oi: float) -> float:
        """
        Normalised open interest change per tick.

        Positive → new positions being opened (trend confirmation).
        Negative → positions being closed (trend exhaustion).

        Returns value in [-1, 1].
        """
        change = oi - self._prev_oi
        self._prev_oi = oi

        abs_change = abs(change)
        if abs_change > self._oi_change_max:
            self._oi_change_max = abs_change

        if self._oi_change_max == 0:
            return 0.0

        return max(-1.0, min(1.0, change / self._oi_change_max))

    # ═════════════════════════════════════════════════════════════════════
    #  BID-ASK SPREAD  [NEW]
    # ═════════════════════════════════════════════════════════════════════

    def _update_spread(self, ltp: float, bid: float, ask: float) -> float:
        """
        Smoothed relative bid-ask spread: (ask - bid) / ltp

        Uses an EMA for stability; raw spread can spike on bad ticks.
        Returns 0 when bid/ask not available.
        """
        if bid <= 0 or ask <= 0 or ltp <= 0:
            return self._spread_ema

        raw_spread = (ask - bid) / ltp
        if self._spread_ema == 0:
            self._spread_ema = raw_spread
        else:
            self._spread_ema += (raw_spread - self._spread_ema) * self._spread_k

        return self._spread_ema

    # ═════════════════════════════════════════════════════════════════════
    #  TICK DIRECTION STREAK  [NEW]
    # ═════════════════════════════════════════════════════════════════════

    def _update_streak(self, price: float) -> float:
        """
        Consecutive tick-direction streak.

        +N = N consecutive up-ticks.
        -N = N consecutive down-ticks.
        Capped to [-streak_max, +streak_max] and normalised to [-1, 1].
        """
        if self._prev_ltp == 0:
            self._prev_ltp = price
            return 0.0

        if price > self._prev_ltp:
            self._streak = min(self._streak + 1, self.streak_max) if self._streak >= 0 else 1
        elif price < self._prev_ltp:
            self._streak = max(self._streak - 1, -self.streak_max) if self._streak <= 0 else -1
        # equal price → streak unchanged

        self._prev_ltp = price
        return self._streak / self.streak_max   # normalise to [-1, 1]

    # ═════════════════════════════════════════════════════════════════════
    #  RESET
    # ═════════════════════════════════════════════════════════════════════

    def reset(self) -> None:
        """
        Reset all internal state.
        Call at session boundaries (market open / close transitions).
        """
        self._prices.clear()
        # RSI
        self._rsi_avg_gain = self._rsi_avg_loss = 0.0
        self._rsi_initialized = False
        self._rsi_tick_count = 0
        # MACD
        self._ema_fast = self._ema_slow = self._ema_signal = 0.0
        self._macd_initialized = False
        self._macd_tick_count = 0
        self._macd_history.clear()
        # Bollinger
        self._bb_window.clear()
        self._bb_sum = self._bb_sum_sq = 0.0
        # ATR
        self._atr_value = self._atr_sum = 0.0
        self._atr_initialized = False
        self._atr_tick_count = 0
        self._prev_close = 0.0
        # VWAP
        self._vwap_pv.clear()
        self._vwap_v.clear()
        self._vwap_pv_sum = self._vwap_v_sum = 0.0
        # Microstructure
        self._prev_volume = self._prev_oi = 0.0
        self._vol_delta_max = self._oi_change_max = 1.0
        self._spread_ema = 0.0
        self._streak = 0
        self._prev_ltp = 0.0
        # Counter
        self._tick_count = 0
        logger.info("FeatureEngine v2 state reset")
