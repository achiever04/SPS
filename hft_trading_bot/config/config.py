"""
===============================================================================
  config.py — Centralized Configuration Manager
===============================================================================
  Loads environment variables from .env, validates required keys, and exposes
  a frozen Settings dataclass for type-safe, read-only access across all
  modules.

  New in v2:
  • TAKE_PROFIT_PERCENT       — target profit per trade
  • SIGNAL_COOLDOWN_SECONDS   — minimum gap between consecutive signals
  • TRAILING_STOP_ENABLED     — activate trailing stop-loss
  • TRAILING_STOP_STEP        — step size for trailing stop updates
  • VWAP_PERIOD               — window for VWAP deviation feature
  • ATR_PERIOD                — window for Average True Range feature
  • BARRIER_LOOKAHEAD         — bars ahead for triple-barrier labeling
  • MODEL_TYPE                — "lightgbm" | "xgboost" | "sklearn"
  • DRY_RUN                   — simulate orders without real API calls

  Usage:
      from config.config import settings
      print(settings.TAKE_PROFIT_PERCENT)
===============================================================================
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv


# ── Resolve project root (one level up from /config) ────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Load .env file from project root ────────────────────────────────────
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
else:
    print(f"[CONFIG] WARNING: .env file not found at {_env_path}")


def _get_env(key: str, default: str | None = None, required: bool = False) -> str:
    """
    Retrieve an environment variable with validation.

    Args:
        key:      Environment variable name.
        default:  Fallback value if not set.
        required: If True, raises SystemExit when missing.

    Returns:
        The environment variable value as a string.
    """
    value = os.getenv(key, default)
    if required and (value is None or value.strip() == ""):
        print(f"[CONFIG] FATAL: Required environment variable '{key}' is missing.")
        sys.exit(1)
    return value or ""


def _bool_env(key: str, default: bool = False) -> bool:
    """Parse a boolean environment variable (true/1/yes → True)."""
    raw = os.getenv(key, str(default)).strip().lower()
    return raw in ("true", "1", "yes")


@dataclass(frozen=True)
class Settings:
    """
    Immutable configuration container.
    All trading parameters are loaded once at startup and cannot be modified
    during runtime — this prevents accidental config mutation in hot paths.
    """

    # ── Broker Credentials ───────────────────────────────────────────────
    BROKER_API_KEY: str = field(default_factory=lambda: _get_env(
        "BROKER_API_KEY", required=True
    ))
    BROKER_API_SECRET: str = field(default_factory=lambda: _get_env(
        "BROKER_API_SECRET", required=True
    ))
    BROKER_ACCESS_TOKEN: str = field(default_factory=lambda: _get_env(
        "BROKER_ACCESS_TOKEN", default=""
    ))

    # ── WebSocket ────────────────────────────────────────────────────────
    WEBSOCKET_URL: str = field(default_factory=lambda: _get_env(
        "WEBSOCKET_URL", default="wss://localhost:8080/ws"
    ))

    # ── Trading Parameters ───────────────────────────────────────────────
    TRADE_SYMBOL: str = field(default_factory=lambda: _get_env(
        "TRADE_SYMBOL", default="NIFTY"
    ))
    TRADE_EXCHANGE: str = field(default_factory=lambda: _get_env(
        "TRADE_EXCHANGE", default="NFO"
    ))
    MAX_POSITION_SIZE: int = field(default_factory=lambda: int(_get_env(
        "MAX_POSITION_SIZE", default="50"
    )))
    STOP_LOSS_PERCENT: float = field(default_factory=lambda: float(_get_env(
        "STOP_LOSS_PERCENT", default="0.5"
    )))
    TAKE_PROFIT_PERCENT: float = field(default_factory=lambda: float(_get_env(
        "TAKE_PROFIT_PERCENT", default="0.8"
    )))
    MAX_OPEN_POSITIONS: int = field(default_factory=lambda: int(_get_env(
        "MAX_OPEN_POSITIONS", default="3"
    )))

    # ── Trailing Stop ────────────────────────────────────────────────────
    TRAILING_STOP_ENABLED: bool = field(default_factory=lambda: _bool_env(
        "TRAILING_STOP_ENABLED", default=True
    ))
    TRAILING_STOP_STEP: float = field(default_factory=lambda: float(_get_env(
        "TRAILING_STOP_STEP", default="0.2"
    )))  # Move SL by this % each time price improves by STEP

    # ── Signal Filtering ─────────────────────────────────────────────────
    SIGNAL_COOLDOWN_SECONDS: float = field(default_factory=lambda: float(_get_env(
        "SIGNAL_COOLDOWN_SECONDS", default="30.0"
    )))
    MIN_CONFIDENCE: float = field(default_factory=lambda: float(_get_env(
        "MIN_CONFIDENCE", default="0.60"
    )))  # Reject signals with model confidence below this

    # ── Risk Management ──────────────────────────────────────────────────
    MAX_DAILY_LOSS: float = field(default_factory=lambda: float(_get_env(
        "MAX_DAILY_LOSS", default="10000"
    )))
    MAX_ORDER_VALUE: float = field(default_factory=lambda: float(_get_env(
        "MAX_ORDER_VALUE", default="500000"
    )))

    # ── Dry-run / simulation mode ────────────────────────────────────────
    DRY_RUN: bool = field(default_factory=lambda: _bool_env(
        "DRY_RUN", default=False
    ))

    # ── Logging ──────────────────────────────────────────────────────────
    LOG_LEVEL: str = field(default_factory=lambda: _get_env(
        "LOG_LEVEL", default="INFO"
    ))
    LOG_FILE: str = field(default_factory=lambda: _get_env(
        "LOG_FILE", default="logs/trading_bot.log"
    ))

    # ── Model Paths (relative to project root) ───────────────────────────
    MODEL_PATH: str = field(default_factory=lambda: str(
        PROJECT_ROOT / "models" / "model.onnx"
    ))
    SCALER_PATH: str = field(default_factory=lambda: str(
        PROJECT_ROOT / "models" / "scaler.pkl"
    ))
    MODEL_TYPE: str = field(default_factory=lambda: _get_env(
        "MODEL_TYPE", default="lightgbm"
    ))  # "lightgbm" | "xgboost" | "sklearn"

    # ── WebSocket Reconnection ───────────────────────────────────────────
    WS_RECONNECT_BASE_DELAY: float = 1.0
    WS_RECONNECT_MAX_DELAY: float = 60.0
    WS_RECONNECT_MULTIPLIER: float = 2.0
    WS_HEARTBEAT_INTERVAL: float = 30.0

    # ── Feature Engine — Price Indicators ────────────────────────────────
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD_DEV: float = 2.0

    # ── Feature Engine — New Indicators ──────────────────────────────────
    ATR_PERIOD: int = 14          # Average True Range window
    VWAP_PERIOD: int = 20         # VWAP deviation rolling window
    STREAK_MAX: int = 10          # Cap for tick-direction streak counter

    # ── Training / Labeling ──────────────────────────────────────────────
    BARRIER_LOOKAHEAD: int = field(default_factory=lambda: int(_get_env(
        "BARRIER_LOOKAHEAD", default="10"
    )))  # Number of bars ahead for triple-barrier labeling

    # ── Order Execution ──────────────────────────────────────────────────
    ORDER_API_BASE_URL: str = field(default_factory=lambda: _get_env(
        "ORDER_API_BASE_URL", default="https://api.broker.com/v1"
    ))
    ORDER_TIMEOUT_SECONDS: float = 5.0

    # ── Backtest ─────────────────────────────────────────────────────────
    BACKTEST_SLIPPAGE_PCT: float = field(default_factory=lambda: float(_get_env(
        "BACKTEST_SLIPPAGE_PCT", default="0.02"
    )))  # Simulated fill slippage as % of price
    BACKTEST_COMMISSION: float = field(default_factory=lambda: float(_get_env(
        "BACKTEST_COMMISSION", default="20.0"
    )))  # Per-trade commission in rupees (flat)

    def validate(self) -> None:
        """Run post-init validation checks on critical parameters."""
        assert self.MAX_POSITION_SIZE > 0, "MAX_POSITION_SIZE must be > 0"
        assert 0 < self.STOP_LOSS_PERCENT < 100, "STOP_LOSS_PERCENT must be 0-100"
        assert 0 < self.TAKE_PROFIT_PERCENT < 100, "TAKE_PROFIT_PERCENT must be 0-100"
        assert self.TAKE_PROFIT_PERCENT > self.STOP_LOSS_PERCENT, (
            "TAKE_PROFIT_PERCENT must exceed STOP_LOSS_PERCENT for positive expectancy"
        )
        assert self.MAX_OPEN_POSITIONS > 0, "MAX_OPEN_POSITIONS must be > 0"
        assert self.RSI_PERIOD > 1, "RSI_PERIOD must be > 1"
        assert self.BOLLINGER_PERIOD > 1, "BOLLINGER_PERIOD must be > 1"
        assert 0.0 <= self.MIN_CONFIDENCE <= 1.0, "MIN_CONFIDENCE must be 0.0–1.0"
        assert self.SIGNAL_COOLDOWN_SECONDS >= 0, "SIGNAL_COOLDOWN_SECONDS must be >= 0"
        assert self.ATR_PERIOD > 1, "ATR_PERIOD must be > 1"
        assert self.BARRIER_LOOKAHEAD > 0, "BARRIER_LOOKAHEAD must be > 0"
        if self.DRY_RUN:
            print("[CONFIG] ⚠️  DRY_RUN=True — no real orders will be placed")


# ── Singleton instance — import this everywhere ──────────────────────────
settings = Settings()
