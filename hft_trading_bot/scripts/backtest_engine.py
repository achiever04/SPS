"""
===============================================================================
  backtest_engine.py — Offline Strategy Backtester  [NEW in v2]
===============================================================================
  Replays a CSV of historical OHLCV bars through the exact same
  FeatureEngine and InferenceEngine used in live trading, then simulates
  order fills with configurable slippage and commission.

  Metrics computed:
  • Total return (%)
  • Sharpe ratio (annualised, risk-free = 6 % for India)
  • Maximum drawdown (%)
  • Win rate (%)
  • Average win / average loss (₹)
  • Profit factor (gross profit / gross loss)
  • Total trades
  • Signal distribution (BUY / SELL / HOLD counts)

  Usage:
      python scripts/backtest_engine.py \
          --data data/NIFTY_1min.csv \
          --model models/model.onnx \
          --scaler models/scaler.pkl \
          --slippage 0.02 \
          --commission 20.0

  Output:
      • Prints metrics table to stdout.
      • Saves equity curve to logs/backtest_equity.csv
      • Saves trade log to logs/backtest_trades.csv
===============================================================================
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.feature_engine import FeatureEngine
from core.inference_engine import InferenceEngine
from utils.logger import get_logger

logger = get_logger("BACKTEST")

# ── Risk-free rate for Sharpe (India: ~6 % annualised) ───────────────────
RISK_FREE_DAILY = (1.06 ** (1 / 252)) - 1


# ═════════════════════════════════════════════════════════════════════════
#  DATA MODELS
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestTrade:
    """Record of one completed simulated trade."""
    entry_bar:   int
    exit_bar:    int
    side:        str            # "BUY" or "SELL"
    entry_price: float
    exit_price:  float
    quantity:    int
    pnl:         float
    exit_reason: str            # "TP", "SL", "EOD", "TRAIL"
    confidence:  float


@dataclass
class BacktestResult:
    """Aggregated backtest metrics."""
    total_return_pct:   float
    sharpe_ratio:       float
    max_drawdown_pct:   float
    win_rate:           float
    avg_win:            float
    avg_loss:           float
    profit_factor:      float
    total_trades:       int
    signal_buy:         int
    signal_sell:        int
    signal_hold:        int
    trades:             list = field(default_factory=list)
    equity_curve:       list = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Simulates the live trading pipeline offline.

    The engine steps through each bar, feeds the tick dict into
    FeatureEngine, runs InferenceEngine, and applies the same
    signal-gating / risk rules as ExecutionEngine — but fills
    are simulated with slippage instead of real broker calls.

    Attributes:
        slippage_pct:  Fill price adjusted by ±slippage_pct % of LTP.
        commission:    Flat per-trade commission in rupees.
        stop_loss_pct: SL distance from entry (%).
        tp_pct:        TP distance from entry (%).
        trailing_step: Trailing SL step size (%).
        lot_size:      Fixed quantity per trade.
        cooldown_bars: Minimum bars between consecutive entries.
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        slippage_pct: float = 0.02,
        commission: float = 20.0,
        stop_loss_pct: float = 0.5,
        tp_pct: float = 0.8,
        trailing_step: float = 0.2,
        lot_size: int = 1,
        cooldown_bars: int = 5,
        initial_capital: float = 1_000_000.0,
        min_confidence: float = 0.60,
        dry_run: bool = True,
    ):
        self.slippage_pct    = slippage_pct
        self.commission      = commission
        self.stop_loss_pct   = stop_loss_pct
        self.tp_pct          = tp_pct
        self.trailing_step   = trailing_step
        self.lot_size        = lot_size
        self.cooldown_bars   = cooldown_bars
        self.initial_capital = initial_capital
        self.min_confidence  = min_confidence

        # ── Feature + inference engines ──────────────────────────────────
        self.feature_engine = FeatureEngine(
            rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9,
            bollinger_period=20, bollinger_std_dev=2.0,
            atr_period=14, vwap_period=20, streak_max=10,
        )

        self.inference_engine = InferenceEngine(
            model_path=model_path,
            scaler_path=scaler_path,
            min_confidence=min_confidence,
            dry_run=dry_run,
        )

    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        prices:  np.ndarray,
        volumes: np.ndarray,
        oi_arr:  np.ndarray,
    ) -> BacktestResult:
        """
        Run backtest on a price/volume/OI time series.

        Args:
            prices:  (N,) array of close prices.
            volumes: (N,) array of volumes.
            oi_arr:  (N,) array of open interest.

        Returns:
            BacktestResult with all metrics and trade log.
        """
        self.feature_engine.reset()

        capital     = self.initial_capital
        equity      = [capital]
        trades: list[BacktestTrade] = []

        # ── Signal counters ───────────────────────────────────────────────
        sig_buy = sig_sell = sig_hold = 0

        # ── Open position state ───────────────────────────────────────────
        in_trade      = False
        trade_side    = ""
        entry_price   = 0.0
        entry_bar     = 0
        sl_price      = 0.0
        tp_price      = 0.0
        tsl_price     = 0.0
        best_price    = 0.0
        trade_conf    = 0.0
        last_entry_bar = -self.cooldown_bars  # allow first trade immediately

        n = len(prices)

        for i in range(n):
            price  = float(prices[i])
            vol    = float(volumes[i])
            oi     = float(oi_arr[i])

            tick = {
                "ltp":    price,
                "volume": vol,
                "oi":     oi,
                "bid":    price - 0.05,
                "ask":    price + 0.05,
            }

            # ── Update features ───────────────────────────────────────────
            features = self.feature_engine.update(tick)

            # ── Check open position exit conditions ───────────────────────
            if in_trade:
                exit_reason = None

                if trade_side == "BUY":
                    if price >= tp_price:
                        exit_reason = "TP"
                    elif price <= tsl_price:
                        exit_reason = "TRAIL" if tsl_price > sl_price else "SL"
                    else:
                        # Ratchet trailing SL up
                        if price > best_price:
                            step_thr = best_price * (1 + self.trailing_step / 100)
                            if price >= step_thr:
                                new_tsl = price * (1 - self.stop_loss_pct / 100)
                                if new_tsl > tsl_price:
                                    tsl_price  = new_tsl
                                    best_price = price

                else:  # SELL
                    if price <= tp_price:
                        exit_reason = "TP"
                    elif price >= tsl_price:
                        exit_reason = "TRAIL" if tsl_price < sl_price else "SL"
                    else:
                        if price < best_price:
                            step_thr = best_price * (1 - self.trailing_step / 100)
                            if price <= step_thr:
                                new_tsl = price * (1 + self.stop_loss_pct / 100)
                                if new_tsl < tsl_price:
                                    tsl_price  = new_tsl
                                    best_price = price

                # Last bar — force close (end-of-day)
                if i == n - 1 and exit_reason is None:
                    exit_reason = "EOD"

                if exit_reason is not None:
                    # Simulate fill with slippage
                    slip_dir   = -1 if trade_side == "BUY" else 1
                    exit_fill  = price * (1 + slip_dir * self.slippage_pct / 100)

                    if trade_side == "BUY":
                        pnl = (exit_fill - entry_price) * self.lot_size - self.commission
                    else:
                        pnl = (entry_price - exit_fill) * self.lot_size - self.commission

                    capital += pnl
                    equity.append(capital)

                    trades.append(BacktestTrade(
                        entry_bar=entry_bar,
                        exit_bar=i,
                        side=trade_side,
                        entry_price=entry_price,
                        exit_price=exit_fill,
                        quantity=self.lot_size,
                        pnl=pnl,
                        exit_reason=exit_reason,
                        confidence=trade_conf,
                    ))

                    in_trade   = False
                    last_entry_bar = i

            # ── Check entry signal ────────────────────────────────────────
            if features is None or in_trade:
                continue

            if i - last_entry_bar < self.cooldown_bars:
                continue

            pred = self.inference_engine.predict(features)
            sig  = pred["signal"]
            conf = pred["confidence"]

            if sig == 1:
                sig_buy += 1
            elif sig == -1:
                sig_sell += 1
            else:
                sig_hold += 1

            if sig == 0 or conf < self.min_confidence:
                continue

            # Simulate entry fill with slippage
            slip_dir    = 1 if sig == 1 else -1
            entry_fill  = price * (1 + slip_dir * self.slippage_pct / 100)

            sl_off    = entry_fill * self.stop_loss_pct / 100
            tp_off    = entry_fill * self.tp_pct / 100

            trade_side  = "BUY" if sig == 1 else "SELL"
            entry_price = entry_fill
            entry_bar   = i
            trade_conf  = conf
            best_price  = entry_fill

            if trade_side == "BUY":
                sl_price  = entry_fill - sl_off
                tp_price  = entry_fill + tp_off
                tsl_price = sl_price
            else:
                sl_price  = entry_fill + sl_off
                tp_price  = entry_fill - tp_off
                tsl_price = sl_price

            in_trade      = True
            last_entry_bar = i
            capital       -= self.commission  # entry commission

        # ── Compute metrics ────────────────────────────────────────────────
        result = self._compute_metrics(
            equity=equity,
            trades=trades,
            sig_buy=sig_buy,
            sig_sell=sig_sell,
            sig_hold=sig_hold,
        )
        return result

    # ─────────────────────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        equity: list[float],
        trades: list[BacktestTrade],
        sig_buy: int,
        sig_sell: int,
        sig_hold: int,
    ) -> BacktestResult:
        """Derive all summary statistics from the equity curve and trade log."""
        eq = np.array(equity, dtype=np.float64)

        # ── Total return ──────────────────────────────────────────────────
        total_return_pct = (eq[-1] / eq[0] - 1.0) * 100.0

        # ── Sharpe ratio (using bar-to-bar equity returns) ────────────────
        if len(eq) > 2:
            bar_returns = np.diff(eq) / eq[:-1]
            excess      = bar_returns - RISK_FREE_DAILY
            sharpe = (
                float(np.mean(excess) / np.std(excess) * np.sqrt(252))
                if np.std(excess) > 0 else 0.0
            )
        else:
            sharpe = 0.0

        # ── Maximum drawdown ──────────────────────────────────────────────
        peak = np.maximum.accumulate(eq)
        dd   = (eq - peak) / peak
        max_dd_pct = float(np.min(dd)) * 100.0

        # ── Trade statistics ──────────────────────────────────────────────
        pnls     = [t.pnl for t in trades]
        wins     = [p for p in pnls if p > 0]
        losses   = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100.0 if pnls else 0.0
        avg_win  = float(np.mean(wins))  if wins   else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0

        gross_profit = sum(wins)
        gross_loss   = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return BacktestResult(
            total_return_pct=round(total_return_pct, 2),
            sharpe_ratio=round(sharpe, 3),
            max_drawdown_pct=round(max_dd_pct, 2),
            win_rate=round(win_rate, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(profit_factor, 3),
            total_trades=len(trades),
            signal_buy=sig_buy,
            signal_sell=sig_sell,
            signal_hold=sig_hold,
            trades=trades,
            equity_curve=list(eq),
        )

    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def print_report(result: BacktestResult) -> None:
        """Print a formatted backtest report to stdout."""
        print("\n" + "=" * 55)
        print("  BACKTEST REPORT")
        print("=" * 55)
        print(f"  Total return     : {result.total_return_pct:>10.2f} %")
        print(f"  Sharpe ratio     : {result.sharpe_ratio:>10.3f}")
        print(f"  Max drawdown     : {result.max_drawdown_pct:>10.2f} %")
        print(f"  Win rate         : {result.win_rate:>10.2f} %")
        print(f"  Avg win          : ₹{result.avg_win:>9,.2f}")
        print(f"  Avg loss         : ₹{result.avg_loss:>9,.2f}")
        print(f"  Profit factor    : {result.profit_factor:>10.3f}")
        print(f"  Total trades     : {result.total_trades:>10,}")
        print("-" * 55)
        print(f"  Signal BUY       : {result.signal_buy:>10,}")
        print(f"  Signal SELL      : {result.signal_sell:>10,}")
        print(f"  Signal HOLD      : {result.signal_hold:>10,}")
        print("=" * 55)

    @staticmethod
    def save_outputs(result: BacktestResult, out_dir: str = "logs") -> None:
        """Save equity curve and trade log as CSVs."""
        import csv
        out = Path(out_dir)
        out.mkdir(exist_ok=True)

        # ── Equity curve ──────────────────────────────────────────────────
        eq_path = out / "backtest_equity.csv"
        with open(eq_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bar", "equity"])
            for i, v in enumerate(result.equity_curve):
                writer.writerow([i, round(v, 2)])
        print(f"   Equity curve → {eq_path}")

        # ── Trade log ─────────────────────────────────────────────────────
        tr_path = out / "backtest_trades.csv"
        with open(tr_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "entry_bar", "exit_bar", "side",
                "entry_price", "exit_price", "quantity",
                "pnl", "exit_reason", "confidence",
            ])
            for t in result.trades:
                writer.writerow([
                    t.entry_bar, t.exit_bar, t.side,
                    round(t.entry_price, 2), round(t.exit_price, 2),
                    t.quantity, round(t.pnl, 2), t.exit_reason,
                    round(t.confidence, 4),
                ])
        print(f"   Trade log     → {tr_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest HFT model v2")
    parser.add_argument("--data",       type=str,   required=False, default=None,
                        help="OHLCV CSV (date,open,high,low,close,volume[,oi])")
    parser.add_argument("--model",      type=str,   default="models/model.onnx")
    parser.add_argument("--scaler",     type=str,   default="models/scaler.pkl")
    parser.add_argument("--slippage",   type=float, default=0.02,
                        help="Slippage as %% of price (default 0.02%%)")
    parser.add_argument("--commission", type=float, default=20.0,
                        help="Flat per-trade commission ₹ (default 20)")
    parser.add_argument("--sl",         type=float, default=0.5,   help="Stop-loss %%")
    parser.add_argument("--tp",         type=float, default=0.8,   help="Take-profit %%")
    parser.add_argument("--trailing",   type=float, default=0.2,   help="Trailing SL step %%")
    parser.add_argument("--capital",    type=float, default=1e6,   help="Starting capital ₹")
    parser.add_argument("--confidence", type=float, default=0.60,  help="Min confidence gate")
    parser.add_argument("--synthetic",  action="store_true",
                        help="Use synthetic data (quick test only)")
    args = parser.parse_args()

    # ── Load price data ───────────────────────────────────────────────────
    if args.synthetic or args.data is None:
        print("⚠️  Using synthetic data.  Pass --data for a real backtest.")
        from scripts.train_model import generate_synthetic_data
        prices, volumes, oi_arr = generate_synthetic_data(n_samples=5000)
    else:
        from scripts.train_model import load_csv_data
        prices, volumes, oi_arr = load_csv_data(args.data)

    # ── Run backtest ──────────────────────────────────────────────────────
    engine = BacktestEngine(
        model_path=str(PROJECT_ROOT / args.model),
        scaler_path=str(PROJECT_ROOT / args.scaler),
        slippage_pct=args.slippage,
        commission=args.commission,
        stop_loss_pct=args.sl,
        tp_pct=args.tp,
        trailing_step=args.trailing,
        initial_capital=args.capital,
        min_confidence=args.confidence,
        dry_run=True,   # always True for backtest
    )

    result = engine.run(prices, volumes, oi_arr)

    BacktestEngine.print_report(result)
    BacktestEngine.save_outputs(result, out_dir=str(PROJECT_ROOT / "logs"))
