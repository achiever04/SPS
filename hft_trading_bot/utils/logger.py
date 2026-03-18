"""
===============================================================================
  logger.py — Production Logging Infrastructure
===============================================================================
  Provides a centralized logging factory with:
  • Dual output: RotatingFileHandler (10 MB / 5 backups) + StreamHandler
  • Microsecond-precision timestamps for latency measurement
  • Named loggers per module (WS, FEATURE, INFERENCE, EXECUTION, BACKTEST)
  • Thread-safe and async-compatible

  Usage:
      from utils.logger import get_logger, log_latency
      logger = get_logger("WEBSOCKET")
      logger.info("Connected to feed")
===============================================================================
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ── Module-level cache to prevent duplicate loggers ─────────────────────
_loggers: dict[str, logging.Logger] = {}

LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-12s | "
    "%(funcName)-20s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    log_file: str = "logs/trading_bot.log",
    log_level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Create or retrieve a named logger with rotating file + console handlers.

    Args:
        name:         Logger name (e.g., "WEBSOCKET", "INFERENCE").
        log_file:     Path to the log file.
        log_level:    Minimum log level string.
        max_bytes:    Maximum log file size before rotation.
        backup_count: Number of rotated log files to keep.

    Returns:
        Configured logging.Logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    level  = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    if logger.handlers:
        _loggers[name] = logger
        return logger

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── File handler ─────────────────────────────────────────────────────
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # ── Console handler ──────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    _loggers[name] = logger
    return logger


def log_latency(logger: logging.Logger, operation: str, start_ns: int) -> float:
    """
    Log elapsed time for an operation.

    Args:
        logger:    Logger instance.
        operation: Short label (e.g., "INFERENCE").
        start_ns:  Start time from time.perf_counter_ns().

    Returns:
        Elapsed time in milliseconds.

    Latency tiers:
        < 1 ms   → INFO  ⚡
        1–10 ms  → INFO  ✅
        10–100ms → WARNING ⚠️
        > 100 ms → ERROR  🐢
    """
    import time
    elapsed_ns = time.perf_counter_ns() - start_ns
    elapsed_ms = elapsed_ns / 1_000_000

    if elapsed_ms < 1.0:
        logger.info(f"⚡ {operation} in {elapsed_ms:.3f}ms")
    elif elapsed_ms < 10.0:
        logger.info(f"✅ {operation} in {elapsed_ms:.2f}ms")
    elif elapsed_ms < 100.0:
        logger.warning(f"⚠️  {operation} in {elapsed_ms:.1f}ms (slow)")
    else:
        logger.error(f"🐢 {operation} in {elapsed_ms:.0f}ms (CRITICAL LATENCY)")

    return elapsed_ms
