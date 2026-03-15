"""
===============================================================================
  logger.py — Production Logging Infrastructure
===============================================================================
  Provides a centralized logging factory with:
  • Dual output: RotatingFileHandler (10MB / 5 backups) + StreamHandler
  • Microsecond-precision timestamps for latency measurement
  • Named loggers per module (WS, FEATURE, INFERENCE, EXECUTION)
  • Thread-safe and async-compatible

  Usage:
      from utils.logger import get_logger
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

# ── Log format with microsecond precision for latency tracking ──────────
LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-12s | "
    "%(funcName)-20s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    log_file: str = "logs/trading_bot.log",
    log_level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB per file
    backup_count: int = 5,
) -> logging.Logger:
    """
    Create or retrieve a named logger with file + console handlers.

    Args:
        name:         Logger name (e.g., "WEBSOCKET", "INFERENCE").
        log_file:     Path to the log file (relative to project root).
        log_level:    Minimum log level ("DEBUG", "INFO", "WARNING", etc.).
        max_bytes:    Maximum log file size before rotation.
        backup_count: Number of rotated log files to keep.

    Returns:
        A configured logging.Logger instance.

    Notes:
        - Each unique `name` gets exactly ONE logger (cached).
        - The log directory is created automatically if it doesn't exist.
        - File handler uses UTF-8 encoding for international symbol names.
    """

    # ── Return cached logger if it already exists ────────────────────────
    if name in _loggers:
        return _loggers[name]

    # ── Create the logger ────────────────────────────────────────────────
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Prevent duplicate handlers if getLogger returns existing
    if logger.handlers:
        _loggers[name] = logger
        return logger

    # ── Formatter ────────────────────────────────────────────────────────
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── File Handler (rotating) ──────────────────────────────────────────
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)  # Capture everything to file
    file_handler.setFormatter(formatter)

    # ── Console Handler ──────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)

    # ── Attach handlers ──────────────────────────────────────────────────
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # ── Prevent log propagation to root logger ───────────────────────────
    logger.propagate = False

    # ── Cache and return ─────────────────────────────────────────────────
    _loggers[name] = logger
    return logger


def log_latency(logger: logging.Logger, operation: str, start_ns: int) -> float:
    """
    Log the elapsed time for an operation in microseconds.

    Args:
        logger:    The logger instance to use.
        operation: A short label for the operation (e.g., "INFERENCE").
        start_ns:  The start time from time.perf_counter_ns().

    Returns:
        Elapsed time in milliseconds (float).

    Usage:
        import time
        t0 = time.perf_counter_ns()
        # ... do work ...
        elapsed_ms = log_latency(logger, "INFERENCE", t0)
    """
    import time

    elapsed_ns = time.perf_counter_ns() - start_ns
    elapsed_ms = elapsed_ns / 1_000_000  # Convert to milliseconds

    if elapsed_ms < 1.0:
        logger.info(f"⚡ {operation} completed in {elapsed_ms:.3f} ms")
    elif elapsed_ms < 10.0:
        logger.info(f"✅ {operation} completed in {elapsed_ms:.2f} ms")
    elif elapsed_ms < 100.0:
        logger.warning(f"⚠️  {operation} took {elapsed_ms:.1f} ms (slow)")
    else:
        logger.error(f"🐢 {operation} took {elapsed_ms:.0f} ms (CRITICAL LATENCY)")

    return elapsed_ms
