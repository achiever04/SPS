"""
===============================================================================
  main.py — HFT Trading Bot Entry Point & Orchestrator  (v2)
===============================================================================
  Pipeline (v2):
  ┌──────────────┐   tick dict    ┌──────────────┐   feature vec   ┌─────────────┐
  │  WebSocket   │ ─────────────▶ │   Feature    │ ──────────────▶ │  Inference  │
  │  Client      │                │   Engine v2  │                 │  Engine v2  │
  └──────────────┘                └──────────────┘                 └─────────────┘
                                                                         │ signal
                                                                         ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  Execution Engine v2                                                         │
  │  • execute_signal()       — open new position on BUY/SELL signal             │
  │  • check_open_positions() — evaluate TP / trailing-SL on EVERY tick          │
  └──────────────────────────────────────────────────────────────────────────────┘

  Key changes from v1:
  • process_ticks() passes the full tick dict to feature_engine.update()
  • check_open_positions(ltp) is called on every processed tick
  • DRY_RUN and MIN_CONFIDENCE are wired from settings
  • Status monitor shows win-rate and signal distribution

  Usage:
      python main.py

  Graceful shutdown:  Ctrl+C or SIGTERM.
===============================================================================
"""

import asyncio
import signal
import sys
import time

from config.config import settings
from utils.logger import get_logger, log_latency
from core.websocket_client import WebSocketClient
from core.feature_engine import FeatureEngine
from core.inference_engine import InferenceEngine
from core.execution_engine import ExecutionEngine

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
    Core tick-processing loop.

    For every tick:
    1. Extract ltp; validate.
    2. Feed full tick dict into FeatureEngine (v2 needs volume/oi/bid/ask).
    3. If warm, run inference.
    4. If signal ≠ HOLD and confidence meets threshold → execute_signal().
    5. ALWAYS call check_open_positions(ltp) to evaluate TP / trailing-SL.
    """
    logger.info("🚀 Tick processing pipeline started")
    ticks_processed = 0
    signals_generated = 0

    while not shutdown_event.is_set():
        try:
            try:
                tick = await asyncio.wait_for(tick_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            t0 = time.perf_counter_ns()
            ticks_processed += 1

            ltp: float = tick.get("ltp", 0.0)
            if ltp <= 0:
                logger.debug(f"Invalid LTP {ltp} — skipping")
                continue

            # ── Step 1: Compute features (full tick dict) ────────────────
            features = feature_engine.update(tick)

            # ── Step 2: Always check open positions on every tick ────────
            #    (This handles TP and trailing-SL independent of inference)
            await execution_engine.check_open_positions(ltp)

            if features is None:
                continue   # Still warming up

            # ── Step 3: Run inference ────────────────────────────────────
            prediction = inference_engine.predict(features)
            signal_val  = prediction["signal"]
            confidence  = prediction["confidence"]

            # ── Step 4: Execute non-HOLD signals ─────────────────────────
            if signal_val != 0:
                signals_generated += 1
                position = await execution_engine.execute_signal(
                    signal=signal_val,
                    current_price=ltp,
                    confidence=confidence,
                )
                if position:
                    logger.info(
                        f"📊 Trade #{signals_generated} | "
                        f"{'BUY' if signal_val > 0 else 'SELL'} | "
                        f"₹{ltp:,.2f} | {confidence:.2%}"
                    )

            # ── Periodic pipeline latency log ────────────────────────────
            if ticks_processed % 500 == 0:
                pipeline_ms = log_latency(logger, "FULL_PIPELINE", t0)
                logger.info(
                    f"📈 Pipeline | "
                    f"Ticks: {ticks_processed} | "
                    f"Signals: {signals_generated} | "
                    f"Queue: {tick_queue.qsize()} | "
                    f"Latency: {pipeline_ms:.3f}ms"
                )

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(f"Tick processing error: {e}")
            await asyncio.sleep(0.01)

    logger.info(
        f"🏁 Tick processing stopped | "
        f"Ticks: {ticks_processed} | Signals: {signals_generated}"
    )


# ═════════════════════════════════════════════════════════════════════════
#  STATUS MONITOR
# ═════════════════════════════════════════════════════════════════════════

async def status_monitor(
    ws_client: WebSocketClient,
    inference_engine: InferenceEngine,
    execution_engine: ExecutionEngine,
    shutdown_event: asyncio.Event,
    interval: float = 60.0,
) -> None:
    """Log a comprehensive status report every `interval` seconds."""
    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(interval)
            ws   = ws_client.stats
            inf  = inference_engine.stats
            exc  = execution_engine.stats

            sig_dist = inf.get("signal_distribution", {})

            logger.info(
                f"\n{'=' * 60}\n"
                f"  SYSTEM STATUS\n"
                f"{'=' * 60}\n"
                f"  WebSocket  | Connected: {ws['connected']} | "
                f"Ticks: {ws['total_ticks']} | "
                f"Dropped: {ws['dropped_packets']} | "
                f"Invalid: {ws.get('invalid_ticks', 0)}\n"
                f"  Inference  | Total: {inf['total_inferences']} | "
                f"Avg: {inf['avg_latency_ms']:.3f}ms | "
                f"p95: {inf['p95_latency_ms']:.3f}ms\n"
                f"  Signals    | BUY: {sig_dist.get(1, 0)} | "
                f"SELL: {sig_dist.get(-1, 0)} | "
                f"HOLD: {sig_dist.get(0, 0)}\n"
                f"  Execution  | Open: {exc['open_positions']} | "
                f"Closed: {exc['closed_positions']} | "
                f"Win rate: {exc['win_rate']:.1%}\n"
                f"  P&L        | Daily: ₹{exc['daily_pnl']:,.2f} | "
                f"Avg win: ₹{exc['avg_win']:,.2f} | "
                f"Avg loss: ₹{exc['avg_loss']:,.2f}\n"
                f"  Orders     | Total: {exc['total_orders']} | "
                f"Rejected: {exc['rejected_orders']} | "
                f"Halted: {exc['trading_halted']} | "
                f"DryRun: {exc['dry_run']}\n"
                f"{'=' * 60}"
            )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Status monitor error: {e}")


# ═════════════════════════════════════════════════════════════════════════
#  SIGNAL HANDLERS
# ═════════════════════════════════════════════════════════════════════════

def setup_signal_handlers(
    shutdown_event: asyncio.Event,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Register SIGINT / SIGTERM for graceful shutdown."""
    def _handler(sig_name: str) -> None:
        logger.warning(f"⚠️  {sig_name} received — shutting down...")
        shutdown_event.set()

    if sys.platform == "win32":
        signal.signal(signal.SIGINT,  lambda s, f: _handler("SIGINT"))
        signal.signal(signal.SIGTERM, lambda s, f: _handler("SIGTERM"))
    else:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: _handler(s.name))


# ═════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════

async def main() -> None:
    """
    Startup sequence:
    1. Validate config.
    2. Create shared asyncio.Queue.
    3. Initialise FeatureEngine, InferenceEngine, ExecutionEngine.
    4. Initialise WebSocketClient.
    5. Register signal handlers.
    6. Launch concurrent tasks.
    7. Await shutdown event.
    8. Graceful teardown.
    """
    logger.info("=" * 60)
    logger.info("  🤖 HFT TRADING BOT v2 — STARTING")
    logger.info("=" * 60)
    logger.info(f"  Symbol:       {settings.TRADE_SYMBOL}")
    logger.info(f"  Exchange:     {settings.TRADE_EXCHANGE}")
    logger.info(f"  Max Pos:      {settings.MAX_OPEN_POSITIONS}")
    logger.info(f"  SL / TP:      {settings.STOP_LOSS_PERCENT}% / {settings.TAKE_PROFIT_PERCENT}%")
    logger.info(f"  Trailing SL:  {'ON' if settings.TRAILING_STOP_ENABLED else 'OFF'}")
    logger.info(f"  Cooldown:     {settings.SIGNAL_COOLDOWN_SECONDS}s")
    logger.info(f"  Min Conf:     {settings.MIN_CONFIDENCE:.0%}")
    logger.info(f"  Max Loss:     ₹{settings.MAX_DAILY_LOSS:,.0f}")
    logger.info(f"  Dry Run:      {settings.DRY_RUN}")
    logger.info("=" * 60)

    # ── Validate config ──────────────────────────────────────────────────
    try:
        settings.validate()
        logger.info("✅ Configuration validated")
    except AssertionError as e:
        logger.critical(f"❌ Config error: {e}")
        sys.exit(1)

    # ── Shared queue (bounded for back-pressure) ─────────────────────────
    tick_queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)

    # ── Feature Engine ───────────────────────────────────────────────────
    feature_engine = FeatureEngine(
        rsi_period=settings.RSI_PERIOD,
        macd_fast=settings.MACD_FAST,
        macd_slow=settings.MACD_SLOW,
        macd_signal=settings.MACD_SIGNAL,
        bollinger_period=settings.BOLLINGER_PERIOD,
        bollinger_std_dev=settings.BOLLINGER_STD_DEV,
        atr_period=settings.ATR_PERIOD,
        vwap_period=settings.VWAP_PERIOD,
        streak_max=settings.STREAK_MAX,
    )

    # ── Inference Engine ─────────────────────────────────────────────────
    inference_engine = InferenceEngine(
        model_path=settings.MODEL_PATH,
        scaler_path=settings.SCALER_PATH,
        min_confidence=settings.MIN_CONFIDENCE,
        dry_run=settings.DRY_RUN,
    )

    # ── Execution Engine ─────────────────────────────────────────────────
    execution_engine = ExecutionEngine(
        api_base_url=settings.ORDER_API_BASE_URL,
        api_key=settings.BROKER_API_KEY,
        access_token=settings.BROKER_ACCESS_TOKEN,
        symbol=settings.TRADE_SYMBOL,
        exchange=settings.TRADE_EXCHANGE,
        max_position_size=settings.MAX_POSITION_SIZE,
        stop_loss_pct=settings.STOP_LOSS_PERCENT,
        take_profit_pct=settings.TAKE_PROFIT_PERCENT,
        max_open_positions=settings.MAX_OPEN_POSITIONS,
        max_daily_loss=settings.MAX_DAILY_LOSS,
        max_order_value=settings.MAX_ORDER_VALUE,
        order_timeout=settings.ORDER_TIMEOUT_SECONDS,
        trailing_stop_enabled=settings.TRAILING_STOP_ENABLED,
        trailing_stop_step=settings.TRAILING_STOP_STEP,
        signal_cooldown_sec=settings.SIGNAL_COOLDOWN_SECONDS,
        dry_run=settings.DRY_RUN,
    )
    await execution_engine.start()

    # ── WebSocket Client ─────────────────────────────────────────────────
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

    # ── Shutdown plumbing ────────────────────────────────────────────────
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    setup_signal_handlers(shutdown_event, loop)

    # ── Launch tasks ─────────────────────────────────────────────────────
    logger.info("🚀 Launching async tasks...")
    tasks = [
        asyncio.create_task(ws_client.start(),           name="websocket"),
        asyncio.create_task(
            process_ticks(
                tick_queue, feature_engine, inference_engine,
                execution_engine, shutdown_event,
            ),
            name="tick_processor",
        ),
        asyncio.create_task(
            status_monitor(
                ws_client, inference_engine, execution_engine,
                shutdown_event, interval=60.0,
            ),
            name="status_monitor",
        ),
    ]

    # ── Wait for shutdown ────────────────────────────────────────────────
    try:
        await shutdown_event.wait()
        logger.info("🔻 Shutdown initiated...")
    except Exception as e:
        logger.exception(f"Error awaiting shutdown: {e}")

    # ── Graceful teardown ────────────────────────────────────────────────
    await ws_client.stop()
    await execution_engine.stop()

    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("=" * 60)
    logger.info("  🏁 HFT TRADING BOT v2 — SHUTDOWN COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted. Goodbye! 👋")
    except Exception as e:
        print(f"\n[MAIN] Fatal error: {e}")
        sys.exit(1)
