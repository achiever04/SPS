"""
===============================================================================
  config.py — Centralized Configuration Manager
===============================================================================
  Loads environment variables from .env, validates required keys, and exposes
  a frozen Settings dataclass for type-safe, read-only access across all modules.

  Usage:
      from config.config import settings
      print(settings.BROKER_API_KEY)
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
        key: Environment variable name.
        default: Fallback value if not set.
        required: If True, raises SystemExit when missing.

    Returns:
        The environment variable value as a string.
    """
    value = os.getenv(key, default)
    if required and (value is None or value.strip() == ""):
        print(f"[CONFIG] FATAL: Required environment variable '{key}' is missing.")
        sys.exit(1)
    return value or ""


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
    MAX_OPEN_POSITIONS: int = field(default_factory=lambda: int(_get_env(
        "MAX_OPEN_POSITIONS", default="3"
    )))

    # ── Risk Management ──────────────────────────────────────────────────
    MAX_DAILY_LOSS: float = field(default_factory=lambda: float(_get_env(
        "MAX_DAILY_LOSS", default="10000"
    )))
    MAX_ORDER_VALUE: float = field(default_factory=lambda: float(_get_env(
        "MAX_ORDER_VALUE", default="500000"
    )))

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

    # ── WebSocket Reconnection ───────────────────────────────────────────
    WS_RECONNECT_BASE_DELAY: float = 1.0     # Initial backoff delay (seconds)
    WS_RECONNECT_MAX_DELAY: float = 60.0     # Maximum backoff cap (seconds)
    WS_RECONNECT_MULTIPLIER: float = 2.0     # Exponential multiplier
    WS_HEARTBEAT_INTERVAL: float = 30.0      # Ping interval (seconds)

    # ── Feature Engine ───────────────────────────────────────────────────
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD_DEV: float = 2.0

    # ── Order Execution ──────────────────────────────────────────────────
    ORDER_API_BASE_URL: str = field(default_factory=lambda: _get_env(
        "ORDER_API_BASE_URL", default="https://api.broker.com/v1"
    ))
    ORDER_TIMEOUT_SECONDS: float = 5.0       # HTTP timeout for order APIs

    def validate(self) -> None:
        """Run post-init validation checks on critical parameters."""
        assert self.MAX_POSITION_SIZE > 0, "MAX_POSITION_SIZE must be > 0"
        assert 0 < self.STOP_LOSS_PERCENT < 100, "STOP_LOSS_PERCENT must be 0-100"
        assert self.MAX_OPEN_POSITIONS > 0, "MAX_OPEN_POSITIONS must be > 0"
        assert self.RSI_PERIOD > 1, "RSI_PERIOD must be > 1"
        assert self.BOLLINGER_PERIOD > 1, "BOLLINGER_PERIOD must be > 1"


# ── Singleton instance — import this everywhere ─────────────────────────
settings = Settings()
