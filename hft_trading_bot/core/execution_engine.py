"""
===============================================================================
  execution_engine.py — Async Order Execution & Risk Management
===============================================================================
  Handles order placement, simultaneous stop-loss enforcement, position
  tracking, and complete trade audit logging via async REST API calls.

  Key features:
  • Every entry order IMMEDIATELY fires a paired Stop-Loss order
  • Position tracking with configurable max-position enforcement
  • Daily P&L monitoring with circuit-breaker logic
  • Complete order lifecycle logging for audit trails
  • Async aiohttp for non-blocking HTTP calls

  Architecture:
      inference_engine → signal → ExecutionEngine.execute() → Broker API
===============================================================================
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

import aiohttp

from utils.logger import get_logger, log_latency

# ── Module logger ───────────────────────────────────────────────────────
logger = get_logger("EXECUTION")


# ═════════════════════════════════════════════════════════════════════════
#  DATA MODELS
# ═════════════════════════════════════════════════════════════════════════

class OrderSide(Enum):
    """Order direction."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type classification."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"              # Stop-Loss order
    SL_MARKET = "SL-M"     # Stop-Loss Market order


class OrderStatus(Enum):
    """Order lifecycle status."""
    PENDING = "PENDING"
    PLACED = "PLACED"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


@dataclass
class Position:
    """
    Tracks an open trading position with its associated stop-loss.

    Attributes:
        symbol:         Instrument symbol (e.g., "NIFTY25MARFUT").
        side:           BUY or SELL.
        entry_price:    Fill price of the entry order.
        quantity:       Number of lots/contracts.
        entry_order_id: Broker's order ID for the entry.
        sl_order_id:    Broker's order ID for the stop-loss.
        sl_price:       Trigger price for the stop-loss.
        timestamp:      Entry time (Unix epoch).
        pnl:            Realized P&L (updated on exit).
    """
    symbol: str
    side: OrderSide
    entry_price: float
    quantity: int
    entry_order_id: str = ""
    sl_order_id: str = ""
    sl_price: float = 0.0
    timestamp: float = field(default_factory=time.time)
    pnl: float = 0.0


# ═════════════════════════════════════════════════════════════════════════
#  EXECUTION ENGINE
# ═════════════════════════════════════════════════════════════════════════

class ExecutionEngine:
    """
    Production async order execution engine with built-in risk management.

    Responsibilities:
    1. Place entry orders based on inference signals.
    2. IMMEDIATELY place a paired stop-loss after entry confirmation.
    3. Track all open positions and enforce max-position limits.
    4. Monitor daily P&L and halt trading if max loss is breached.
    5. Log every order action for complete audit trail.

    Attributes:
        api_base_url:       Broker REST API base URL.
        api_key:            API key for authentication.
        access_token:       Session access token.
        symbol:             Trading instrument symbol.
        exchange:           Exchange code (e.g., "NFO").
        max_position_size:  Max lots per single position.
        stop_loss_pct:      Stop-loss percentage from entry price.
        max_open_positions: Maximum concurrent open positions.
        max_daily_loss:     Daily loss limit (circuit breaker).
        max_order_value:    Maximum single order value.
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
        max_open_positions: int = 3,
        max_daily_loss: float = 10000.0,
        max_order_value: float = 500000.0,
        order_timeout: float = 5.0,
    ):
        """
        Initialize the execution engine.

        Args:
            api_base_url:       Broker REST API endpoint.
            api_key:            API authentication key.
            access_token:       Session token.
            symbol:             Default trading symbol.
            exchange:           Exchange code.
            max_position_size:  Max quantity per position.
            stop_loss_pct:      SL distance as percentage.
            max_open_positions: Max simultaneous positions.
            max_daily_loss:     Daily loss limit (absolute).
            max_order_value:    Max single order value.
            order_timeout:      HTTP timeout for order API calls.
        """
        # ── Configuration ────────────────────────────────────────────────
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.access_token = access_token
        self.symbol = symbol
        self.exchange = exchange
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.max_open_positions = max_open_positions
        self.max_daily_loss = max_daily_loss
        self.max_order_value = max_order_value
        self.order_timeout = order_timeout

        # ── State tracking ───────────────────────────────────────────────
        self._open_positions: list[Position] = []
        self._closed_positions: list[Position] = []
        self._daily_pnl: float = 0.0
        self._total_orders: int = 0
        self._rejected_orders: int = 0
        self._trading_halted: bool = False

        # ── HTTP session (created in start()) ────────────────────────────
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info(
            f"ExecutionEngine initialized | "
            f"Symbol: {symbol} | Exchange: {exchange} | "
            f"MaxPos: {max_open_positions} | SL: {stop_loss_pct}% | "
            f"MaxDailyLoss: ₹{max_daily_loss:,.0f}"
        )

    # ═════════════════════════════════════════════════════════════════════
    #  LIFECYCLE
    # ═════════════════════════════════════════════════════════════════════

    async def start(self) -> None:
        """Initialize the HTTP session for order execution."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.order_timeout),
            headers={
                "X-Kite-Version": "3",
                "Authorization": f"token {self.api_key}:{self.access_token}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        logger.info("✅ Execution engine HTTP session initialized")

    async def stop(self) -> None:
        """
        Gracefully shut down: cancel all open SL orders and close session.
        """
        logger.info("Execution engine shutting down...")

        # ── Cancel pending stop-loss orders ──────────────────────────────
        for pos in self._open_positions:
            if pos.sl_order_id:
                await self._cancel_order(pos.sl_order_id)
                logger.info(f"Cancelled SL order {pos.sl_order_id} on shutdown")

        # ── Close HTTP session ───────────────────────────────────────────
        if self._session and not self._session.closed:
            await self._session.close()

        # ── Final report ─────────────────────────────────────────────────
        logger.info(
            f"🏁 Execution engine stopped | "
            f"Total orders: {self._total_orders} | "
            f"Rejected: {self._rejected_orders} | "
            f"Daily P&L: ₹{self._daily_pnl:,.2f} | "
            f"Open positions: {len(self._open_positions)}"
        )

    # ═════════════════════════════════════════════════════════════════════
    #  PUBLIC API — TRADE EXECUTION
    # ═════════════════════════════════════════════════════════════════════

    async def execute_signal(
        self,
        signal: int,
        current_price: float,
        confidence: float,
    ) -> Optional[Position]:
        """
        Execute a trading signal from the inference engine.

        Flow:
        1. Pre-trade risk checks (max positions, daily loss, order value).
        2. Place entry order (MARKET).
        3. On entry confirmation, IMMEDIATELY place a stop-loss order.
        4. Track the new position.

        Args:
            signal:         +1 (BUY), -1 (SELL), 0 (HOLD).
            current_price:  Current LTP for SL calculation.
            confidence:     Model confidence (0.0–1.0).

        Returns:
            Position object if order was placed, None otherwise.
        """
        # ── HOLD signal → do nothing ─────────────────────────────────────
        if signal == 0:
            return None

        # ── Pre-trade risk checks ────────────────────────────────────────
        if not self._pre_trade_checks(current_price):
            return None

        # ── Determine order side ─────────────────────────────────────────
        side = OrderSide.BUY if signal > 0 else OrderSide.SELL
        quantity = self._calculate_quantity(current_price, confidence)

        logger.info(
            f"🎯 Executing {side.value} | "
            f"Price: ₹{current_price:,.2f} | "
            f"Qty: {quantity} | "
            f"Confidence: {confidence:.2%}"
        )

        t_start = time.perf_counter_ns()

        # ── Step 1: Place entry order ────────────────────────────────────
        entry_result = await self._place_order(
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=0,  # Market order — no price needed
        )

        if entry_result.get("status") != "success":
            logger.error(f"Entry order REJECTED: {entry_result}")
            self._rejected_orders += 1
            return None

        entry_order_id = entry_result.get("order_id", "UNKNOWN")
        fill_price = entry_result.get("fill_price", current_price)

        # ── Step 2: Calculate  and place stop-loss IMMEDIATELY ───────────
        sl_price = self._calculate_stop_loss(fill_price, side)

        sl_result = await self._place_stop_loss(
            side=side,
            quantity=quantity,
            trigger_price=sl_price,
        )

        sl_order_id = sl_result.get("order_id", "UNKNOWN")

        # ── Step 3: Track the position ───────────────────────────────────
        position = Position(
            symbol=self.symbol,
            side=side,
            entry_price=fill_price,
            quantity=quantity,
            entry_order_id=entry_order_id,
            sl_order_id=sl_order_id,
            sl_price=sl_price,
        )

        self._open_positions.append(position)
        self._total_orders += 1

        elapsed_ms = log_latency(logger, "ORDER_EXECUTION", t_start)
        logger.info(
            f"✅ Position opened | "
            f"Side: {side.value} | "
            f"Entry: ₹{fill_price:,.2f} | "
            f"SL: ₹{sl_price:,.2f} | "
            f"Order ID: {entry_order_id} | "
            f"SL Order: {sl_order_id} | "
            f"Execution time: {elapsed_ms:.2f}ms"
        )

        return position

    # ═════════════════════════════════════════════════════════════════════
    #  RISK MANAGEMENT
    # ═════════════════════════════════════════════════════════════════════

    def _pre_trade_checks(self, current_price: float) -> bool:
        """
        Run all pre-trade risk checks before placing an order.

        Returns:
            True if all checks pass, False if trade should be blocked.
        """
        # ── Check 1: Trading halted? ─────────────────────────────────────
        if self._trading_halted:
            logger.warning("⛔ Trading HALTED — daily loss limit breached")
            return False

        # ── Check 2: Max open positions ──────────────────────────────────
        if len(self._open_positions) >= self.max_open_positions:
            logger.warning(
                f"⚠️  Max open positions reached ({self.max_open_positions})"
            )
            return False

        # ── Check 3: Daily loss limit ────────────────────────────────────
        if self._daily_pnl <= -self.max_daily_loss:
            self._trading_halted = True
            logger.critical(
                f"🚨 CIRCUIT BREAKER TRIGGERED | "
                f"Daily P&L: ₹{self._daily_pnl:,.2f} | "
                f"Limit: ₹{-self.max_daily_loss:,.2f}"
            )
            return False

        # ── Check 4: Max order value ─────────────────────────────────────
        estimated_value = current_price * self.max_position_size
        if estimated_value > self.max_order_value:
            logger.warning(
                f"⚠️  Order value ₹{estimated_value:,.0f} exceeds "
                f"limit ₹{self.max_order_value:,.0f}"
            )
            return False

        return True

    def _calculate_quantity(
        self, price: float, confidence: float
    ) -> int:
        """
        Calculate order quantity based on price and model confidence.

        Higher confidence → larger position (up to max_position_size).
        Lower confidence → smaller position (minimum 1 lot).
        """
        # Scale quantity by confidence: 50% confidence → 50% of max size
        scaled_qty = max(1, int(self.max_position_size * confidence))
        return min(scaled_qty, self.max_position_size)

    def _calculate_stop_loss(
        self, entry_price: float, side: OrderSide
    ) -> float:
        """
        Calculate stop-loss price based on entry price and direction.

        For BUY: SL is below entry price.
        For SELL: SL is above entry price.
        """
        sl_offset = entry_price * (self.stop_loss_pct / 100.0)

        if side == OrderSide.BUY:
            sl_price = entry_price - sl_offset
        else:
            sl_price = entry_price + sl_offset

        # Round to tick size (0.05 for most Indian instruments)
        sl_price = round(sl_price * 20) / 20

        return sl_price

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
        """
        Place an order via the broker REST API.

        Args:
            side:           BUY or SELL.
            order_type:     MARKET, LIMIT, SL, or SL-M.
            quantity:       Number of lots/contracts.
            price:          Limit price (0 for market orders).
            trigger_price:  Stop-loss trigger price (0 if not SL).

        Returns:
            Dict with "status", "order_id", and "fill_price" keys.
        """
        if not self._session:
            logger.error("HTTP session not initialized. Call start() first.")
            return {"status": "error", "message": "Session not initialized"}

        payload = {
            "tradingsymbol": self.symbol,
            "exchange": self.exchange,
            "transaction_type": side.value,
            "order_type": order_type.value,
            "quantity": quantity,
            "product": "MIS",  # Intraday
            "validity": "DAY",
        }

        if price > 0:
            payload["price"] = price
        if trigger_price > 0:
            payload["trigger_price"] = trigger_price

        try:
            t_start = time.perf_counter_ns()

            async with self._session.post(
                f"{self.api_base_url}/orders/regular",
                data=payload,
            ) as response:
                elapsed_ms = (time.perf_counter_ns() - t_start) / 1_000_000
                result = await response.json()

                if response.status == 200 and result.get("status") == "success":
                    order_id = result.get("data", {}).get("order_id", "UNKNOWN")
                    logger.info(
                        f"📤 Order placed | "
                        f"{side.value} {quantity}x {self.symbol} | "
                        f"Type: {order_type.value} | "
                        f"Order ID: {order_id} | "
                        f"API latency: {elapsed_ms:.2f}ms"
                    )
                    return {
                        "status": "success",
                        "order_id": order_id,
                        "fill_price": price if price > 0 else 0,
                    }
                else:
                    error_msg = result.get("message", "Unknown error")
                    logger.error(
                        f"❌ Order REJECTED | {side.value} {quantity}x {self.symbol} | "
                        f"Reason: {error_msg} | HTTP: {response.status}"
                    )
                    return {"status": "rejected", "message": error_msg}

        except asyncio.TimeoutError:
            logger.error(
                f"⏱️  Order TIMEOUT ({self.order_timeout}s) | "
                f"{side.value} {quantity}x {self.symbol}"
            )
            return {"status": "timeout", "message": "Request timed out"}

        except aiohttp.ClientError as e:
            logger.error(f"🔌 Connection error placing order: {e}")
            return {"status": "error", "message": str(e)}

        except Exception as e:
            logger.exception(f"Unexpected order error: {e}")
            return {"status": "error", "message": str(e)}

    async def _place_stop_loss(
        self,
        side: OrderSide,
        quantity: int,
        trigger_price: float,
    ) -> dict[str, Any]:
        """
        Place a stop-loss market order (the inverse of the entry side).

        A BUY entry → SELL stop-loss.
        A SELL entry → BUY stop-loss.
        """
        sl_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY

        logger.info(
            f"🛡️  Placing SL | {sl_side.value} {quantity}x {self.symbol} | "
            f"Trigger: ₹{trigger_price:,.2f}"
        )

        return await self._place_order(
            side=sl_side,
            order_type=OrderType.SL_MARKET,
            quantity=quantity,
            trigger_price=trigger_price,
        )

    async def _cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order by its ID.

        Returns:
            True if cancellation was successful.
        """
        if not self._session:
            return False

        try:
            async with self._session.delete(
                f"{self.api_base_url}/orders/regular/{order_id}"
            ) as response:
                result = await response.json()
                if response.status == 200:
                    logger.info(f"🗑️  Order {order_id} cancelled successfully")
                    return True
                else:
                    logger.error(f"Failed to cancel order {order_id}: {result}")
                    return False

        except Exception as e:
            logger.exception(f"Error cancelling order {order_id}: {e}")
            return False

    # ═════════════════════════════════════════════════════════════════════
    #  POSITION MANAGEMENT
    # ═════════════════════════════════════════════════════════════════════

    async def close_position(
        self, position: Position, exit_price: float
    ) -> None:
        """
        Close a position: cancel the SL order and place an exit order.

        Args:
            position:    The Position to close.
            exit_price:  Current market price for P&L calculation.
        """
        # ── Cancel the stop-loss ─────────────────────────────────────────
        if position.sl_order_id:
            await self._cancel_order(position.sl_order_id)

        # ── Place exit order (opposite side) ─────────────────────────────
        exit_side = (
            OrderSide.SELL if position.side == OrderSide.BUY
            else OrderSide.BUY
        )

        await self._place_order(
            side=exit_side,
            order_type=OrderType.MARKET,
            quantity=position.quantity,
        )

        # ── Calculate P&L ────────────────────────────────────────────────
        if position.side == OrderSide.BUY:
            position.pnl = (exit_price - position.entry_price) * position.quantity
        else:
            position.pnl = (position.entry_price - exit_price) * position.quantity

        self._daily_pnl += position.pnl

        # ── Move to closed positions ─────────────────────────────────────
        if position in self._open_positions:
            self._open_positions.remove(position)
        self._closed_positions.append(position)

        logger.info(
            f"📕 Position closed | "
            f"{position.side.value} {position.quantity}x {position.symbol} | "
            f"Entry: ₹{position.entry_price:,.2f} → "
            f"Exit: ₹{exit_price:,.2f} | "
            f"P&L: ₹{position.pnl:,.2f} | "
            f"Daily P&L: ₹{self._daily_pnl:,.2f}"
        )

    async def close_all_positions(self, current_price: float) -> None:
        """Emergency close of all open positions (e.g., on shutdown)."""
        logger.warning(f"🔴 Closing ALL {len(self._open_positions)} positions")
        for position in list(self._open_positions):
            await self.close_position(position, current_price)

    # ═════════════════════════════════════════════════════════════════════
    #  DIAGNOSTICS
    # ═════════════════════════════════════════════════════════════════════

    @property
    def stats(self) -> dict[str, Any]:
        """Return current execution statistics."""
        return {
            "open_positions": len(self._open_positions),
            "closed_positions": len(self._closed_positions),
            "total_orders": self._total_orders,
            "rejected_orders": self._rejected_orders,
            "daily_pnl": round(self._daily_pnl, 2),
            "trading_halted": self._trading_halted,
        }

    def reset_daily(self) -> None:
        """Reset daily P&L and halt flag (call at market open)."""
        self._daily_pnl = 0.0
        self._trading_halted = False
        self._closed_positions.clear()
        logger.info("📅 Daily execution state reset")
