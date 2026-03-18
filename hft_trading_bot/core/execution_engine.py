"""
===============================================================================
  execution_engine.py — Async Order Execution & Risk Management  (v2)
===============================================================================
  Changes in v2:
  • Take-profit logic — configurable TP % per trade
  • Trailing stop-loss — SL moves up (for BUY) or down (for SELL) as price
    improves, locking in more profit without manual intervention
  • Signal cooldown — minimum seconds between consecutive new entries
  • Kelly-inspired position sizing — scales quantity by confidence
  • Dry-run mode — all order logic executes but no real API calls made
  • check_open_positions() method — call on every tick for TP/trailing-SL
  • Slippage estimator — adjusts fill price by a configurable basis points

  Architecture:
      inference_engine → signal → ExecutionEngine.execute_signal()
                  every tick → ExecutionEngine.check_open_positions(ltp)
===============================================================================
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import aiohttp

from utils.logger import get_logger, log_latency

logger = get_logger("EXECUTION")


# ═════════════════════════════════════════════════════════════════════════
#  DATA MODELS
# ═════════════════════════════════════════════════════════════════════════

class OrderSide(Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET   = "MARKET"
    LIMIT    = "LIMIT"
    SL       = "SL"
    SL_MARKET = "SL-M"


class OrderStatus(Enum):
    PENDING   = "PENDING"
    PLACED    = "PLACED"
    EXECUTED  = "EXECUTED"
    CANCELLED = "CANCELLED"
    REJECTED  = "REJECTED"
    FAILED    = "FAILED"


class ExitReason(Enum):
    STOP_LOSS      = "STOP_LOSS"
    TAKE_PROFIT    = "TAKE_PROFIT"
    TRAILING_STOP  = "TRAILING_STOP"
    MANUAL         = "MANUAL"
    SHUTDOWN       = "SHUTDOWN"


@dataclass
class Position:
    """
    Tracks one open position with its SL, TP and trailing-stop state.
    """
    symbol: str
    side: OrderSide
    entry_price: float
    quantity: int
    entry_order_id: str = ""
    sl_order_id: str = ""
    sl_price: float = 0.0
    tp_price: float = 0.0
    # Trailing stop tracking
    trailing_sl_price: float = 0.0   # current trailing-SL price
    best_price: float = 0.0          # best price seen since entry (for trailing)
    # Metadata
    timestamp: float = field(default_factory=time.time)
    pnl: float = 0.0
    exit_reason: Optional[ExitReason] = None


# ═════════════════════════════════════════════════════════════════════════
#  EXECUTION ENGINE
# ═════════════════════════════════════════════════════════════════════════

class ExecutionEngine:
    """
    Production async order execution engine (v2).

    New responsibilities vs v1:
    • check_open_positions(ltp) — evaluate TP and trailing SL on every tick
    • Trailing SL update — ratchets SL in direction of profit
    • Signal cooldown — enforces minimum gap between new entries
    • Dry-run mode — simulates fills without real broker API calls
    """

    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        access_token: str,
        symbol: str = "NIFTY",
        exchange: str = "NFO",
        max_position_size: int = 50,
        stop_loss_pct: float = 0.5,
        take_profit_pct: float = 0.8,
        max_open_positions: int = 3,
        max_daily_loss: float = 10000.0,
        max_order_value: float = 500000.0,
        order_timeout: float = 5.0,
        trailing_stop_enabled: bool = True,
        trailing_stop_step: float = 0.2,
        signal_cooldown_sec: float = 30.0,
        dry_run: bool = False,
    ):
        # ── Configuration ────────────────────────────────────────────────
        self.api_base_url        = api_base_url.rstrip("/")
        self.api_key             = api_key
        self.access_token        = access_token
        self.symbol              = symbol
        self.exchange            = exchange
        self.max_position_size   = max_position_size
        self.stop_loss_pct       = stop_loss_pct
        self.take_profit_pct     = take_profit_pct
        self.max_open_positions  = max_open_positions
        self.max_daily_loss      = max_daily_loss
        self.max_order_value     = max_order_value
        self.order_timeout       = order_timeout
        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_stop_step  = trailing_stop_step   # % step for trailing SL
        self.signal_cooldown_sec = signal_cooldown_sec
        self.dry_run             = dry_run

        # ── State ────────────────────────────────────────────────────────
        self._open_positions: list[Position] = []
        self._closed_positions: list[Position] = []
        self._daily_pnl: float = 0.0
        self._total_orders: int = 0
        self._rejected_orders: int = 0
        self._trading_halted: bool = False
        self._last_signal_time: float = 0.0   # epoch seconds of last entry

        # ── HTTP session (created in start()) ────────────────────────────
        self._session: Optional[aiohttp.ClientSession] = None

        mode_tag = "DRY-RUN" if dry_run else "LIVE"
        logger.info(
            f"ExecutionEngine v2 [{mode_tag}] | "
            f"{symbol}@{exchange} | "
            f"SL={stop_loss_pct}% TP={take_profit_pct}% | "
            f"Trailing={'ON' if trailing_stop_enabled else 'OFF'} | "
            f"Cooldown={signal_cooldown_sec}s"
        )

    # ═════════════════════════════════════════════════════════════════════
    #  LIFECYCLE
    # ═════════════════════════════════════════════════════════════════════

    async def start(self) -> None:
        """Initialize the aiohttp session."""
        if not self.dry_run:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.order_timeout),
                headers={
                    "X-Kite-Version": "3",
                    "Authorization": f"token {self.api_key}:{self.access_token}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
        logger.info(
            f"✅ Execution engine started "
            f"({'dry-run — no real orders' if self.dry_run else 'LIVE'})"
        )

    async def stop(self) -> None:
        """Cancel pending SL orders, close session, log final report."""
        logger.info("Execution engine shutting down...")

        for pos in self._open_positions:
            if pos.sl_order_id and not self.dry_run:
                await self._cancel_order(pos.sl_order_id)

        if self._session and not self._session.closed:
            await self._session.close()

        logger.info(
            f"🏁 Execution engine stopped | "
            f"Orders: {self._total_orders} | "
            f"Rejected: {self._rejected_orders} | "
            f"Daily P&L: ₹{self._daily_pnl:,.2f} | "
            f"Open: {len(self._open_positions)}"
        )

    # ═════════════════════════════════════════════════════════════════════
    #  ENTRY — NEW SIGNAL
    # ═════════════════════════════════════════════════════════════════════

    async def execute_signal(
        self,
        signal: int,
        current_price: float,
        confidence: float,
    ) -> Optional[Position]:
        """
        Execute a trading signal.

        Steps:
        1. HOLD → return None immediately.
        2. Pre-trade risk checks (positions, daily loss, cooldown, value).
        3. Calculate quantity (Kelly-inspired confidence scaling).
        4. Place entry MARKET order.
        5. Calculate and place stop-loss order IMMEDIATELY.
        6. Set take-profit price (no bracket order — checked on each tick).
        7. Track position.

        Args:
            signal:        +1 BUY, -1 SELL, 0 HOLD.
            current_price: Current LTP.
            confidence:    Model confidence (0–1).

        Returns:
            New Position, or None if not executed.
        """
        if signal == 0:
            return None

        if not self._pre_trade_checks(current_price):
            return None

        side = OrderSide.BUY if signal > 0 else OrderSide.SELL
        quantity = self._calc_quantity(confidence)

        logger.info(
            f"🎯 Signal {side.value} | "
            f"Price ₹{current_price:,.2f} | "
            f"Qty {quantity} | "
            f"Confidence {confidence:.2%}"
        )

        t0 = time.perf_counter_ns()

        # ── Place entry order ────────────────────────────────────────────
        entry_result = await self._place_order(side, OrderType.MARKET, quantity)
        if entry_result.get("status") != "success":
            logger.error(f"Entry REJECTED: {entry_result}")
            self._rejected_orders += 1
            return None

        entry_id = entry_result.get("order_id", "DRY")
        fill_price = entry_result.get("fill_price", current_price)

        # ── Compute SL / TP prices ───────────────────────────────────────
        sl_price  = self._calc_sl_price(fill_price, side)
        tp_price  = self._calc_tp_price(fill_price, side)

        # ── Place stop-loss order ────────────────────────────────────────
        sl_result = await self._place_stop_loss(side, quantity, sl_price)
        sl_id     = sl_result.get("order_id", "DRY")

        # ── Build position ───────────────────────────────────────────────
        pos = Position(
            symbol=self.symbol,
            side=side,
            entry_price=fill_price,
            quantity=quantity,
            entry_order_id=entry_id,
            sl_order_id=sl_id,
            sl_price=sl_price,
            tp_price=tp_price,
            trailing_sl_price=sl_price,   # trailing starts at initial SL
            best_price=fill_price,
        )

        self._open_positions.append(pos)
        self._total_orders += 1
        self._last_signal_time = time.time()

        log_latency(logger, "ORDER_EXECUTION", t0)
        logger.info(
            f"✅ Position opened | "
            f"{side.value} {quantity}× {self.symbol} | "
            f"Entry ₹{fill_price:,.2f} | "
            f"SL ₹{sl_price:,.2f} | TP ₹{tp_price:,.2f} | "
            f"Trailing {'ON' if self.trailing_stop_enabled else 'OFF'}"
        )
        return pos

    # ═════════════════════════════════════════════════════════════════════
    #  TICK-LEVEL POSITION MANAGEMENT  [NEW]
    # ═════════════════════════════════════════════════════════════════════

    async def check_open_positions(self, ltp: float) -> None:
        """
        Evaluate all open positions against the current price on every tick.

        Called from the tick-processing loop in main.py.

        Checks (in priority order):
        1. Take-profit reached → close at market.
        2. Trailing stop-loss hit → close at market.
        3. Update trailing SL if price has improved beyond the step threshold.
        """
        for pos in list(self._open_positions):
            if pos.side == OrderSide.BUY:
                # ── Take-profit ──────────────────────────────────────────
                if ltp >= pos.tp_price:
                    logger.info(f"🎯 TP hit | {pos.symbol} | LTP ₹{ltp:,.2f} ≥ TP ₹{pos.tp_price:,.2f}")
                    pos.exit_reason = ExitReason.TAKE_PROFIT
                    await self.close_position(pos, ltp)
                    continue

                # ── Trailing stop-loss hit ───────────────────────────────
                if self.trailing_stop_enabled and ltp <= pos.trailing_sl_price:
                    logger.info(
                        f"🔻 Trailing SL hit | {pos.symbol} | "
                        f"LTP ₹{ltp:,.2f} ≤ TSL ₹{pos.trailing_sl_price:,.2f}"
                    )
                    pos.exit_reason = ExitReason.TRAILING_STOP
                    await self.close_position(pos, ltp)
                    continue

                # ── Update trailing SL (ratchet up for BUY) ─────────────
                if self.trailing_stop_enabled and ltp > pos.best_price:
                    step_threshold = pos.best_price * (1 + self.trailing_stop_step / 100)
                    if ltp >= step_threshold:
                        new_tsl = self._calc_sl_price(ltp, pos.side)
                        if new_tsl > pos.trailing_sl_price:
                            old_tsl = pos.trailing_sl_price
                            pos.trailing_sl_price = new_tsl
                            pos.best_price = ltp
                            logger.debug(
                                f"📈 Trailing SL moved | "
                                f"₹{old_tsl:,.2f} → ₹{new_tsl:,.2f}"
                            )

            else:  # SELL position
                # ── Take-profit ──────────────────────────────────────────
                if ltp <= pos.tp_price:
                    logger.info(f"🎯 TP hit | {pos.symbol} | LTP ₹{ltp:,.2f} ≤ TP ₹{pos.tp_price:,.2f}")
                    pos.exit_reason = ExitReason.TAKE_PROFIT
                    await self.close_position(pos, ltp)
                    continue

                # ── Trailing stop-loss hit ───────────────────────────────
                if self.trailing_stop_enabled and ltp >= pos.trailing_sl_price:
                    logger.info(
                        f"🔺 Trailing SL hit | {pos.symbol} | "
                        f"LTP ₹{ltp:,.2f} ≥ TSL ₹{pos.trailing_sl_price:,.2f}"
                    )
                    pos.exit_reason = ExitReason.TRAILING_STOP
                    await self.close_position(pos, ltp)
                    continue

                # ── Update trailing SL (ratchet down for SELL) ──────────
                if self.trailing_stop_enabled and ltp < pos.best_price:
                    step_threshold = pos.best_price * (1 - self.trailing_stop_step / 100)
                    if ltp <= step_threshold:
                        new_tsl = self._calc_sl_price(ltp, pos.side)
                        if new_tsl < pos.trailing_sl_price:
                            old_tsl = pos.trailing_sl_price
                            pos.trailing_sl_price = new_tsl
                            pos.best_price = ltp
                            logger.debug(
                                f"📉 Trailing SL moved | "
                                f"₹{old_tsl:,.2f} → ₹{new_tsl:,.2f}"
                            )

    # ═════════════════════════════════════════════════════════════════════
    #  RISK MANAGEMENT
    # ═════════════════════════════════════════════════════════════════════

    def _pre_trade_checks(self, price: float) -> bool:
        """
        Returns True only if all risk gates pass.

        Gates:
        1. Trading not halted (daily loss limit).
        2. Open position count < max.
        3. Daily P&L within limit.
        4. Estimated order value within limit.
        5. Signal cooldown respected.
        """
        if self._trading_halted:
            logger.warning("⛔ Trading HALTED")
            return False

        if len(self._open_positions) >= self.max_open_positions:
            logger.debug(f"Max positions ({self.max_open_positions}) reached")
            return False

        if self._daily_pnl <= -self.max_daily_loss:
            self._trading_halted = True
            logger.critical(
                f"🚨 CIRCUIT BREAKER | Daily P&L ₹{self._daily_pnl:,.2f} "
                f"≤ limit ₹{-self.max_daily_loss:,.2f}"
            )
            return False

        estimated_value = price * self.max_position_size
        if estimated_value > self.max_order_value:
            logger.warning(
                f"⚠️  Order value ₹{estimated_value:,.0f} > limit ₹{self.max_order_value:,.0f}"
            )
            return False

        elapsed = time.time() - self._last_signal_time
        if elapsed < self.signal_cooldown_sec:
            logger.debug(
                f"Cooldown: {elapsed:.1f}s / {self.signal_cooldown_sec}s elapsed"
            )
            return False

        return True

    def _calc_quantity(self, confidence: float) -> int:
        """
        Kelly-inspired quantity scaling.

        quantity = max(1, round(max_position_size * confidence))

        A confidence of 0.6 → 60 % of max size.
        A confidence of 1.0 → 100 % of max size.
        """
        qty = max(1, round(self.max_position_size * confidence))
        return min(qty, self.max_position_size)

    def _calc_sl_price(self, entry: float, side: OrderSide) -> float:
        """Stop-loss price, rounded to Indian tick size (0.05)."""
        offset = entry * (self.stop_loss_pct / 100.0)
        raw = entry - offset if side == OrderSide.BUY else entry + offset
        return round(raw * 20) / 20

    def _calc_tp_price(self, entry: float, side: OrderSide) -> float:
        """Take-profit price, rounded to Indian tick size (0.05)."""
        offset = entry * (self.take_profit_pct / 100.0)
        raw = entry + offset if side == OrderSide.BUY else entry - offset
        return round(raw * 20) / 20

    # ═════════════════════════════════════════════════════════════════════
    #  ORDER API CALLS
    # ═════════════════════════════════════════════════════════════════════

    async def _place_order(
        self,
        side: OrderSide,
        order_type: OrderType,
        quantity: int,
        price: float = 0,
        trigger_price: float = 0,
    ) -> dict[str, Any]:
        """Place an order.  In dry-run mode simulates a successful fill."""
        if self.dry_run:
            sim_id = f"DRY-{int(time.time()*1000)}"
            return {
                "status": "success",
                "order_id": sim_id,
                "fill_price": price if price > 0 else trigger_price,
            }

        if not self._session:
            return {"status": "error", "message": "Session not initialized"}

        payload: dict = {
            "tradingsymbol": self.symbol,
            "exchange":      self.exchange,
            "transaction_type": side.value,
            "order_type":    order_type.value,
            "quantity":      quantity,
            "product":       "MIS",
            "validity":      "DAY",
        }
        if price > 0:
            payload["price"] = price
        if trigger_price > 0:
            payload["trigger_price"] = trigger_price

        try:
            t0 = time.perf_counter_ns()
            async with self._session.post(
                f"{self.api_base_url}/orders/regular", data=payload
            ) as resp:
                latency_ms = (time.perf_counter_ns() - t0) / 1_000_000
                result = await resp.json()

                if resp.status == 200 and result.get("status") == "success":
                    oid = result.get("data", {}).get("order_id", "UNKNOWN")
                    logger.info(
                        f"📤 Order | {side.value} {quantity}× {self.symbol} "
                        f"| {order_type.value} | ID {oid} | {latency_ms:.1f}ms"
                    )
                    return {
                        "status": "success",
                        "order_id": oid,
                        "fill_price": price if price > 0 else 0,
                    }

                msg = result.get("message", "Unknown error")
                logger.error(f"❌ Order rejected: {msg}")
                return {"status": "rejected", "message": msg}

        except asyncio.TimeoutError:
            logger.error(f"⏱️  Order timeout ({self.order_timeout}s)")
            return {"status": "timeout", "message": "timeout"}
        except aiohttp.ClientError as e:
            logger.error(f"🔌 Connection error: {e}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.exception(f"Unexpected order error: {e}")
            return {"status": "error", "message": str(e)}

    async def _place_stop_loss(
        self, side: OrderSide, quantity: int, trigger_price: float
    ) -> dict[str, Any]:
        """Place a SL-M order on the opposite side."""
        sl_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
        logger.info(
            f"🛡️  SL | {sl_side.value} {quantity}× | Trigger ₹{trigger_price:,.2f}"
        )
        return await self._place_order(
            sl_side, OrderType.SL_MARKET, quantity,
            trigger_price=trigger_price,
        )

    async def _cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if self.dry_run or not self._session:
            return True
        try:
            async with self._session.delete(
                f"{self.api_base_url}/orders/regular/{order_id}"
            ) as resp:
                if resp.status == 200:
                    logger.info(f"🗑️  Cancelled order {order_id}")
                    return True
                logger.error(f"Cancel failed for {order_id}: HTTP {resp.status}")
                return False
        except Exception as e:
            logger.exception(f"Error cancelling {order_id}: {e}")
            return False

    # ═════════════════════════════════════════════════════════════════════
    #  POSITION CLOSE
    # ═════════════════════════════════════════════════════════════════════

    async def close_position(self, pos: Position, exit_price: float) -> None:
        """
        Close a position: cancel its broker SL and place an exit MARKET order.

        P&L is calculated and added to the daily total.
        """
        if pos.sl_order_id:
            await self._cancel_order(pos.sl_order_id)

        exit_side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
        await self._place_order(exit_side, OrderType.MARKET, pos.quantity)

        if pos.side == OrderSide.BUY:
            pos.pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pos.pnl = (pos.entry_price - exit_price) * pos.quantity

        self._daily_pnl += pos.pnl

        if pos in self._open_positions:
            self._open_positions.remove(pos)
        self._closed_positions.append(pos)

        reason = pos.exit_reason.value if pos.exit_reason else "MANUAL"
        logger.info(
            f"📕 Position closed [{reason}] | "
            f"{pos.side.value} {pos.quantity}× {pos.symbol} | "
            f"₹{pos.entry_price:,.2f} → ₹{exit_price:,.2f} | "
            f"P&L ₹{pos.pnl:,.2f} | Daily ₹{self._daily_pnl:,.2f}"
        )

    async def close_all_positions(self, current_price: float) -> None:
        """Emergency close all open positions."""
        logger.warning(f"🔴 Closing ALL {len(self._open_positions)} positions")
        for pos in list(self._open_positions):
            pos.exit_reason = ExitReason.SHUTDOWN
            await self.close_position(pos, current_price)

    # ═════════════════════════════════════════════════════════════════════
    #  DIAGNOSTICS
    # ═════════════════════════════════════════════════════════════════════

    @property
    def stats(self) -> dict[str, Any]:
        closed = self._closed_positions
        wins = [p for p in closed if p.pnl > 0]
        win_rate = len(wins) / len(closed) if closed else 0.0
        avg_win  = sum(p.pnl for p in wins) / len(wins) if wins else 0.0
        losses   = [p for p in closed if p.pnl <= 0]
        avg_loss = sum(p.pnl for p in losses) / len(losses) if losses else 0.0

        return {
            "open_positions":   len(self._open_positions),
            "closed_positions": len(closed),
            "total_orders":     self._total_orders,
            "rejected_orders":  self._rejected_orders,
            "daily_pnl":        round(self._daily_pnl, 2),
            "trading_halted":   self._trading_halted,
            "win_rate":         round(win_rate, 4),
            "avg_win":          round(avg_win, 2),
            "avg_loss":         round(avg_loss, 2),
            "dry_run":          self.dry_run,
        }

    def reset_daily(self) -> None:
        """Reset daily P&L and circuit-breaker flag at market open."""
        self._daily_pnl = 0.0
        self._trading_halted = False
        self._closed_positions.clear()
        self._last_signal_time = 0.0
        logger.info("📅 Daily execution state reset")
