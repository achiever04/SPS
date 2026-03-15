"""
===============================================================================
  main.py — HFT Trading Bot Entry Point & Orchestrator
===============================================================================
  Initializes all modules, creates the async tick pipeline, and runs
  concurrent tasks for WebSocket streaming, feature computation,
  model inference, and order execution.

  Pipeline Architecture:
  ┌──────────┐    ┌──────────────┐    ┌───────────────┐    ┌───────────────┐
  │ WebSocket │───▶│ Feature      │───▶│ Inference     │───▶│ Execution     │
  │ Client    │    │ Engine       │    │ Engine (ONNX) │    │ Engine        │
  │           │    │ (RSI/MACD/BB)│    │               │    │ (Orders + SL) │
  └──────────┘    └──────────────┘    └───────────────┘    └───────────────┘
       │                                                          │
       └──────────── asyncio.Queue ────────────────────────────────┘

  Usage:
      python main.py

  Graceful Shutdown:
      Press Ctrl+C or send SIGTERM. The bot will:
      1. Stop the WebSocket client.
      2. Close all open positions with market orders.
      3. Cancel pending stop-loss orders.
      4. Flush logs and exit cleanly.
===============================================================================
"""

import asyncio
import signal
import sys
import time
from typing import Optional

# ── Internal imports ─────────────────────────────────────────────────────
from config.config import settings
from utils.logger import get_logger, log_latency
from core.websocket_client import WebSocketClient
from core.feature_engine import FeatureEngine
from core.inference_engine import InferenceEngine
from core.execution_engine import ExecutionEngine

# ── Module logger ────────────────────────────────────────────────────────
logger = get_logger("MAIN", log_file=settings.LOG_FILE, log_level=settings.LOG_LEVEL)


# ═════════════════════════════════════════════════════════════════════════
#  TICK PROCESSING PIPELINE
# ═════════════════════════════════════════════════════════════════════════

async def process_ticks(
    tick_queue: asyncio.Queue,
    feature_engine: FeatureEngine,
    inference_engine: InferenceEngine,
    execution_engine: ExecutionEngine,
    shutdown_event: asyncio.Event,
) -> None:
    """
    Main tick processing loop — the heart of the trading bot.

    For each incoming tick:
    1. Extract LTP from the tick dict.
    2. Feed it into the FeatureEngine to compute indicators.
    3. If features are ready (warm-up complete), run ONNX inference.
    4. If inference returns a BUY/SELL signal, execute the trade.

    This coroutine runs indefinitely until the shutdown event is set.

    Args:
        tick_queue:        Queue receiving ticks from WebSocketClient.
        feature_engine:    Incremental indicator calculator.
        inference_engine:  ONNX model for signal prediction.
        execution_engine:  Order placement and risk management.
        shutdown_event:    asyncio.Event to signal graceful shutdown.
    """
    logger.info("🚀 Tick processing pipeline started")
    ticks_processed: int = 0
    signals_generated: int = 0

    while not shutdown_event.is_set():
        try:
            # ── Fetch tick from queue with timeout ───────────────────────
            # Timeout allows periodic shutdown checks
            try:
                tick = await asyncio.wait_for(
                    tick_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue  # Check shutdown_event and retry

            t_pipeline_start = time.perf_counter_ns()
            ticks_processed += 1

            # ── Step 1: Extract LTP ──────────────────────────────────────
            ltp = tick.get("ltp", 0.0)
            if ltp <= 0:
                logger.warning(f"Invalid LTP: {ltp} — skipping tick")
                continue

            # ── Step 2: Compute features ─────────────────────────────────
            features = feature_engine.update(ltp)

            if features is None:
                # Still in warm-up phase — skip inference
                continue

            # ── Step 3: Run inference ────────────────────────────────────
            prediction = inference_engine.predict(features)
            signal_value = prediction["signal"]
            confidence = prediction["confidence"]

            # ── Step 4: Execute if actionable ────────────────────────────
            if signal_value != 0:
                signals_generated += 1

                position = await execution_engine.execute_signal(
                    signal=signal_value,
                    current_price=ltp,
                    confidence=confidence,
                )

                if position:
                    logger.info(
                        f"📊 Trade #{signals_generated} | "
                        f"Signal: {'BUY' if signal_value > 0 else 'SELL'} | "
                        f"LTP: ₹{ltp:,.2f} | "
                        f"Confidence: {confidence:.2%}"
                    )

            # ── Log pipeline latency periodically ────────────────────────
            if ticks_processed % 500 == 0:
                pipeline_ms = log_latency(logger, "FULL_PIPELINE", t_pipeline_start)
                logger.info(
                    f"📈 Pipeline stats | "
                    f"Ticks: {ticks_processed} | "
                    f"Signals: {signals_generated} | "
                    f"Queue depth: {tick_queue.qsize()} | "
                    f"Pipeline latency: {pipeline_ms:.3f}ms"
                )

        except asyncio.CancelledError:
            logger.info("Tick processing cancelled")
            break
        except Exception as e:
            logger.exception(f"Error in tick processing: {e}")
            # Don't crash the pipeline on individual tick errors
            await asyncio.sleep(0.01)

    logger.info(
        f"🏁 Tick processing stopped | "
        f"Total ticks: {ticks_processed} | "
        f"Signals generated: {signals_generated}"
    )


# ═════════════════════════════════════════════════════════════════════════
#  SYSTEM STATUS MONITOR
# ═════════════════════════════════════════════════════════════════════════

async def status_monitor(
    ws_client: WebSocketClient,
    inference_engine: InferenceEngine,
    execution_engine: ExecutionEngine,
    shutdown_event: asyncio.Event,
    interval: float = 60.0,
) -> None:
    """
    Periodic system health monitor — logs stats from all modules.

    Args:
        ws_client:        WebSocket client instance.
        inference_engine: Inference engine instance.
        execution_engine: Execution engine instance.
        shutdown_event:   Shutdown signal.
        interval:         Seconds between status reports.
    """
    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(interval)

            ws_stats = ws_client.stats
            inf_stats = inference_engine.stats
            exec_stats = execution_engine.stats

            logger.info(
                f"\n{'=' * 60}\n"
                f"  SYSTEM STATUS REPORT\n"
                f"{'=' * 60}\n"
                f"  WebSocket   | Connected: {ws_stats['connected']} | "
                f"Ticks: {ws_stats['total_ticks']} | "
                f"Dropped: {ws_stats['dropped_packets']}\n"
                f"  Inference   | Total: {inf_stats['total_inferences']} | "
                f"Avg latency: {inf_stats['avg_latency_ms']:.3f}ms\n"
                f"  Execution   | Open: {exec_stats['open_positions']} | "
                f"Closed: {exec_stats['closed_positions']} | "
                f"Daily P&L: ₹{exec_stats['daily_pnl']:,.2f}\n"
                f"  Orders      | Total: {exec_stats['total_orders']} | "
                f"Rejected: {exec_stats['rejected_orders']} | "
                f"Halted: {exec_stats['trading_halted']}\n"
                f"{'=' * 60}"
            )

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Status monitor error: {e}")


# ═════════════════════════════════════════════════════════════════════════
#  GRACEFUL SHUTDOWN HANDLER
# ═════════════════════════════════════════════════════════════════════════

def setup_signal_handlers(
    shutdown_event: asyncio.Event, loop: asyncio.AbstractEventLoop
) -> None:
    """
    Register OS signal handlers for graceful shutdown.

    Handles:
        SIGINT (Ctrl+C) and SIGTERM (kill command).

    On Windows, signal handling is limited — only SIGINT is supported
    via the asyncio event loop directly.
    """
    def _signal_handler(sig_name: str) -> None:
        logger.warning(f"⚠️  Received {sig_name} — initiating graceful shutdown...")
        shutdown_event.set()

    if sys.platform == "win32":
        # Windows: use signal module directly
        signal.signal(signal.SIGINT, lambda s, f: _signal_handler("SIGINT"))
        signal.signal(signal.SIGTERM, lambda s, f: _signal_handler("SIGTERM"))
    else:
        # Unix: use loop.add_signal_handler for async-safe handling
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig, lambda s=sig: _signal_handler(s.name)
            )


# ═════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════

async def main() -> None:
    """
    Initialize all components and run the trading bot.

    Startup sequence:
    1. Validate configuration.
    2. Initialize FeatureEngine (stateless, CPU-bound).
    3. Initialize InferenceEngine (loads ONNX model + scaler).
    4. Initialize ExecutionEngine (creates HTTP session).
    5. Initialize WebSocketClient (connects to broker feed).
    6. Run all concurrent tasks.
    7. On shutdown, close positions and clean up.
    """
    logger.info("=" * 60)
    logger.info("  🤖 HFT TRADING BOT — STARTING")
    logger.info("=" * 60)
    logger.info(f"  Symbol:      {settings.TRADE_SYMBOL}")
    logger.info(f"  Exchange:    {settings.TRADE_EXCHANGE}")
    logger.info(f"  Max Pos:     {settings.MAX_OPEN_POSITIONS}")
    logger.info(f"  SL:          {settings.STOP_LOSS_PERCENT}%")
    logger.info(f"  Max Loss:    ₹{settings.MAX_DAILY_LOSS:,.0f}")
    logger.info("=" * 60)

    # ── Step 1: Validate configuration ───────────────────────────────────
    try:
        settings.validate()
        logger.info("✅ Configuration validated")
    except AssertionError as e:
        logger.critical(f"❌ Configuration error: {e}")
        sys.exit(1)

    # ── Step 2: Create shared asyncio.Queue ──────────────────────────────
    # Bounded queue to apply backpressure if processing falls behind
    tick_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)

    # ── Step 3: Initialize Feature Engine ────────────────────────────────
    feature_engine = FeatureEngine(
        rsi_period=settings.RSI_PERIOD,
        macd_fast=settings.MACD_FAST,
        macd_slow=settings.MACD_SLOW,
        macd_signal=settings.MACD_SIGNAL,
        bollinger_period=settings.BOLLINGER_PERIOD,
        bollinger_std_dev=settings.BOLLINGER_STD_DEV,
    )

    # ── Step 4: Initialize Inference Engine ──────────────────────────────
    inference_engine = InferenceEngine(
        model_path=settings.MODEL_PATH,
        scaler_path=settings.SCALER_PATH,
    )

    # ── Step 5: Initialize Execution Engine ──────────────────────────────
    execution_engine = ExecutionEngine(
        api_base_url=settings.ORDER_API_BASE_URL,
        api_key=settings.BROKER_API_KEY,
        access_token=settings.BROKER_ACCESS_TOKEN,
        symbol=settings.TRADE_SYMBOL,
        exchange=settings.TRADE_EXCHANGE,
        max_position_size=settings.MAX_POSITION_SIZE,
        stop_loss_pct=settings.STOP_LOSS_PERCENT,
        max_open_positions=settings.MAX_OPEN_POSITIONS,
        max_daily_loss=settings.MAX_DAILY_LOSS,
        max_order_value=settings.MAX_ORDER_VALUE,
        order_timeout=settings.ORDER_TIMEOUT_SECONDS,
    )
    await execution_engine.start()

    # ── Step 6: Initialize WebSocket Client ──────────────────────────────
    ws_client = WebSocketClient(
        ws_url=settings.WEBSOCKET_URL,
        tick_queue=tick_queue,
        api_key=settings.BROKER_API_KEY,
        access_token=settings.BROKER_ACCESS_TOKEN,
        heartbeat_interval=settings.WS_HEARTBEAT_INTERVAL,
        reconnect_base_delay=settings.WS_RECONNECT_BASE_DELAY,
        reconnect_max_delay=settings.WS_RECONNECT_MAX_DELAY,
        reconnect_multiplier=settings.WS_RECONNECT_MULTIPLIER,
    )

    # ── Step 7: Setup shutdown event and signal handlers ─────────────────
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    setup_signal_handlers(shutdown_event, loop)

    # ── Step 8: Launch all concurrent tasks ──────────────────────────────
    logger.info("🚀 Launching async tasks...")

    tasks = [
        asyncio.create_task(
            ws_client.start(),
            name="websocket_stream"
        ),
        asyncio.create_task(
            process_ticks(
                tick_queue, feature_engine, inference_engine,
                execution_engine, shutdown_event,
            ),
            name="tick_processor"
        ),
        asyncio.create_task(
            status_monitor(
                ws_client, inference_engine, execution_engine,
                shutdown_event, interval=60.0,
            ),
            name="status_monitor"
        ),
    ]

    # ── Wait for shutdown signal ─────────────────────────────────────────
    try:
        await shutdown_event.wait()
        logger.info("🔻 Shutdown signal received — cleaning up...")
    except Exception as e:
        logger.exception(f"Error waiting for shutdown: {e}")

    # ── Step 9: Graceful shutdown sequence ───────────────────────────────
    logger.info("Stopping WebSocket client...")
    await ws_client.stop()

    logger.info("Stopping execution engine...")
    await execution_engine.stop()

    # ── Cancel all running tasks ─────────────────────────────────────────
    for task in tasks:
        if not task.done():
            task.cancel()

    # Wait for tasks to finish cancellation
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("=" * 60)
    logger.info("  🏁 HFT TRADING BOT — SHUTDOWN COMPLETE")
    logger.info("=" * 60)


# ═════════════════════════════════════════════════════════════════════════
#  SCRIPT ENTRY
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user. Goodbye! 👋")
    except Exception as e:
        print(f"\n[MAIN] Fatal error: {e}")
        sys.exit(1)
