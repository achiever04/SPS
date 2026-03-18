"""
===============================================================================
  websocket_client.py — Async WebSocket Feed Handler  (v2)
===============================================================================
  Connects to a live market data WebSocket, parses ticks, and publishes
  complete tick dicts to an asyncio.Queue for downstream consumption.

  Changes in v2:
  • Publishes full tick dicts (ltp, volume, oi, bid, ask) — v1 only had ltp.
    The enhanced FeatureEngine v2 requires all these fields.
  • _parse_tick is more defensive: handles missing depth gracefully.
  • Queue overflow warning when depth > 80 % of maxsize.
  • Separate _validate_tick() method for testability.

  Architecture:
      [Broker WS] → WebSocketClient → asyncio.Queue[dict] → feature_engine
===============================================================================
"""

import asyncio
import json
import random
import time
from typing import Any, Optional

import aiohttp

from utils.logger import get_logger, log_latency

logger = get_logger("WEBSOCKET")


class WebSocketClient:
    """
    Production-grade async WebSocket client (v2).

    The key change from v1: every tick published to `tick_queue` now
    carries the full snapshot needed by FeatureEngine v2:
        {
            "ltp":       float,   # Last traded price
            "volume":    float,   # Cumulative session volume
            "oi":        float,   # Open interest
            "bid":       float,   # Best bid price (0 if unavailable)
            "ask":       float,   # Best ask price (0 if unavailable)
            "timestamp": int,     # Exchange timestamp
            "token":     int,     # Instrument token
            "_recv_ns":  int,     # Local receive time (nanoseconds)
        }
    """

    def __init__(
        self,
        ws_url: str,
        tick_queue: asyncio.Queue,
        api_key: str,
        access_token: str,
        heartbeat_interval: float = 30.0,
        reconnect_base_delay: float = 1.0,
        reconnect_max_delay: float = 60.0,
        reconnect_multiplier: float = 2.0,
        queue_warn_pct: float = 0.8,    # warn when queue > 80 % full
    ):
        self.ws_url = ws_url
        self.tick_queue = tick_queue
        self.api_key = api_key
        self.access_token = access_token

        self._heartbeat_interval = heartbeat_interval
        self._reconnect_base_delay = reconnect_base_delay
        self._reconnect_max_delay = reconnect_max_delay
        self._reconnect_multiplier = reconnect_multiplier
        self._queue_warn_pct = queue_warn_pct

        # ── State ────────────────────────────────────────────────────────
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._running: bool = False
        self._reconnect_delay: float = reconnect_base_delay
        self._last_seq: int = -1
        self._total_ticks: int = 0
        self._dropped_packets: int = 0
        self._invalid_ticks: int = 0

    # ═════════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ═════════════════════════════════════════════════════════════════════

    async def start(self) -> None:
        """
        Start the WebSocket client with automatic reconnection.
        Designed to run as an asyncio.Task from main.py.
        """
        self._running = True
        logger.info(f"WebSocket client starting | URL: {self.ws_url}")
        self._session = aiohttp.ClientSession()

        try:
            while self._running:
                try:
                    await self._connect_and_stream()
                except aiohttp.WSServerHandshakeError as e:
                    logger.error(f"Handshake failed: {e}")
                except aiohttp.ClientConnectionError as e:
                    logger.error(f"Connection error: {e}")
                except asyncio.CancelledError:
                    logger.info("WebSocket task cancelled")
                    break
                except Exception as e:
                    logger.exception(f"Unexpected WS error: {e}")

                if self._running:
                    jitter = random.uniform(0, self._reconnect_delay * 0.1)
                    wait = self._reconnect_delay + jitter
                    logger.warning(f"Reconnecting in {wait:.2f}s")
                    await asyncio.sleep(wait)
                    self._reconnect_delay = min(
                        self._reconnect_delay * self._reconnect_multiplier,
                        self._reconnect_max_delay,
                    )
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """Signal graceful shutdown."""
        logger.info("WebSocket client stopping...")
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()

    # ═════════════════════════════════════════════════════════════════════
    #  PRIVATE
    # ═════════════════════════════════════════════════════════════════════

    async def _connect_and_stream(self) -> None:
        headers = {
            "X-Kite-Version": "3",
            "Authorization": f"token {self.api_key}:{self.access_token}",
        }
        logger.info("Attempting WebSocket connection...")

        async with self._session.ws_connect(
            self.ws_url,
            headers=headers,
            heartbeat=self._heartbeat_interval,
            timeout=aiohttp.ClientWSTimeout(ws_close=10.0),
        ) as ws:
            self._ws = ws
            self._reconnect_delay = self._reconnect_base_delay
            logger.info("✅ WebSocket connected")
            await self._subscribe(ws)

            async for msg in ws:
                if not self._running:
                    break

                t_recv = time.perf_counter_ns()

                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_text(msg.data, t_recv)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    await self._handle_binary(msg.data, t_recv)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WS error frame: {ws.exception()}")
                    break
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    logger.warning(f"WS closed by server: {msg.data}")
                    break

    async def _subscribe(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Send subscription message (Zerodha Kite format)."""
        await ws.send_str(json.dumps({
            "a": "subscribe",
            "v": [256265],   # NIFTY 50 instrument token — update per symbol
        }))
        await ws.send_str(json.dumps({
            "a": "mode",
            "v": ["full", [256265]],
        }))
        logger.info("📡 Subscribed — FULL quote mode")

    async def _handle_text(self, raw: str, recv_ns: int) -> None:
        """Parse JSON text frame and push tick to queue."""
        try:
            data: dict = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return

        # ── Sequence-gap detection ───────────────────────────────────────
        seq = data.get("sequence", -1)
        if self._last_seq > 0 and seq > 0 and seq > self._last_seq + 1:
            gap = seq - self._last_seq - 1
            self._dropped_packets += gap
            logger.warning(
                f"⚠️  {gap} packet(s) dropped | "
                f"seq {self._last_seq + 1}–{seq - 1} | "
                f"total dropped: {self._dropped_packets}"
            )
        if seq > 0:
            self._last_seq = seq

        tick = self._parse_tick(data)
        if tick is None:
            self._invalid_ticks += 1
            return

        if not self._validate_tick(tick):
            self._invalid_ticks += 1
            return

        tick["_recv_ns"] = recv_ns
        await self._enqueue(tick)

    async def _handle_binary(self, raw: bytes, recv_ns: int) -> None:
        """
        Handle binary tick data.
        Override with broker-specific binary protocol parser.
        Zerodha Kite v3 sends binary packets for full mode ticks.
        """
        # ── Placeholder — implement binary parsing for your broker ────────
        # Zerodha binary format: little-endian, 44 bytes per token
        # See: https://kite.trade/docs/connect/v3/websocket/#message-structure
        logger.debug(f"Binary frame: {len(raw)} bytes — implement broker parser")

    async def _enqueue(self, tick: dict) -> None:
        """Put tick on queue; warn if queue is filling up."""
        if self.tick_queue.maxsize > 0:
            fill_pct = self.tick_queue.qsize() / self.tick_queue.maxsize
            if fill_pct > self._queue_warn_pct:
                logger.warning(
                    f"⚠️  Tick queue {fill_pct:.0%} full "
                    f"({self.tick_queue.qsize()}/{self.tick_queue.maxsize}) "
                    f"— processing may be falling behind"
                )

        try:
            self.tick_queue.put_nowait(tick)
        except asyncio.QueueFull:
            logger.error("Tick queue FULL — dropping tick")
            return

        self._total_ticks += 1
        if self._total_ticks % 1000 == 0:
            logger.info(
                f"📈 Ticks: {self._total_ticks} | "
                f"Dropped: {self._dropped_packets} | "
                f"Invalid: {self._invalid_ticks} | "
                f"Queue: {self.tick_queue.qsize()}"
            )

    @staticmethod
    def _parse_tick(data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """
        Normalize raw broker tick into the standard internal format.

        Returns None on parse failure so the caller can increment
        an invalid-tick counter.
        """
        try:
            depth = data.get("depth", {})
            buy_depth  = depth.get("buy",  [{}])
            sell_depth = depth.get("sell", [{}])

            best_bid = float(buy_depth[0].get("price",  0)) if buy_depth  else 0.0
            best_ask = float(sell_depth[0].get("price", 0)) if sell_depth else 0.0

            return {
                "ltp":       float(data.get("last_price",     0)),
                "volume":    float(data.get("volume_traded",  0)),
                "oi":        float(data.get("oi",             0)),
                "bid":       best_bid,
                "ask":       best_ask,
                "timestamp": data.get("exchange_timestamp",  0),
                "token":     data.get("instrument_token",    0),
            }
        except (KeyError, ValueError, TypeError, IndexError) as e:
            logger.debug(f"Tick parse error: {e}")
            return None

    @staticmethod
    def _validate_tick(tick: dict) -> bool:
        """
        Basic sanity checks on a parsed tick.

        Rejects:
        • ltp ≤ 0
        • Negative volume or OI
        • ask < bid (crossed market — data error)
        """
        ltp = tick.get("ltp", 0)
        if ltp <= 0:
            return False
        if tick.get("volume", 0) < 0 or tick.get("oi", 0) < 0:
            return False
        bid, ask = tick.get("bid", 0), tick.get("ask", 0)
        if bid > 0 and ask > 0 and ask < bid:
            return False   # crossed market
        return True

    async def _cleanup(self) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info(
            f"🏁 WS stopped | Ticks: {self._total_ticks} | "
            f"Dropped: {self._dropped_packets} | Invalid: {self._invalid_ticks}"
        )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_ticks":     self._total_ticks,
            "dropped_packets": self._dropped_packets,
            "invalid_ticks":   self._invalid_ticks,
            "queue_size":      self.tick_queue.qsize(),
            "connected":       self._ws is not None and not self._ws.closed,
            "reconnect_delay": self._reconnect_delay,
        }
