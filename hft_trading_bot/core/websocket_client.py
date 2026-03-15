"""
===============================================================================
  websocket_client.py — Async WebSocket Feed Handler
===============================================================================
  Connects to a live market data WebSocket (e.g., Indian F&O broker feed),
  parses incoming tick data, and pushes it onto an asyncio.Queue for downstream
  consumption by the feature engine.

  Key features:
  • Exponential backoff with jitter for reconnection (1s → 60s cap)
  • Heartbeat/ping-pong monitoring for stale connections
  • Dropped packet detection via sequence number tracking
  • Graceful shutdown with proper resource cleanup

  Architecture:
      [Broker WS] → websocket_client → asyncio.Queue → feature_engine
===============================================================================
"""

import asyncio
import json
import time
import random
from typing import Any

import aiohttp

from utils.logger import get_logger, log_latency

# ── Module logger ───────────────────────────────────────────────────────
logger = get_logger("WEBSOCKET")


class WebSocketClient:
    """
    Production-grade async WebSocket client with fault-tolerant reconnection.

    Attributes:
        ws_url:           WebSocket endpoint URL.
        tick_queue:        asyncio.Queue to publish parsed tick dicts.
        api_key:          Broker API key for authentication.
        access_token:     Session access token.
        _session:         aiohttp.ClientSession (reused across reconnects).
        _ws:              Active WebSocket connection.
        _running:         Control flag for the run loop.
        _reconnect_delay: Current backoff delay in seconds.
        _last_seq:        Last received sequence number for gap detection.
        _total_ticks:     Total ticks received (lifetime counter).
        _dropped_packets: Count of detected sequence gaps.
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
    ):
        """
        Initialize the WebSocket client.

        Args:
            ws_url:               Broker WebSocket URL.
            tick_queue:            Queue for publishing parsed tick data.
            api_key:              Broker API key.
            access_token:         Session token for authentication.
            heartbeat_interval:   Seconds between heartbeat pings.
            reconnect_base_delay: Initial reconnection delay (seconds).
            reconnect_max_delay:  Maximum reconnection delay cap (seconds).
            reconnect_multiplier: Exponential backoff multiplier.
        """
        self.ws_url = ws_url
        self.tick_queue = tick_queue
        self.api_key = api_key
        self.access_token = access_token

        # ── Reconnection parameters ─────────────────────────────────────
        self._heartbeat_interval = heartbeat_interval
        self._reconnect_base_delay = reconnect_base_delay
        self._reconnect_max_delay = reconnect_max_delay
        self._reconnect_multiplier = reconnect_multiplier

        # ── Internal state ───────────────────────────────────────────────
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._running: bool = False
        self._reconnect_delay: float = reconnect_base_delay
        self._last_seq: int = -1
        self._total_ticks: int = 0
        self._dropped_packets: int = 0

    # ── Public API ───────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Start the WebSocket client. Enters a reconnection loop that
        persists until stop() is called.

        This is the main entry point — designed to be run as an
        asyncio.Task from main.py.
        """
        self._running = True
        logger.info(f"WebSocket client starting | URL: {self.ws_url}")

        # ── Create a persistent aiohttp session ─────────────────────────
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
                    logger.exception(f"Unexpected WebSocket error: {e}")

                # ── Reconnect with exponential backoff + jitter ──────────
                if self._running:
                    jitter = random.uniform(0, self._reconnect_delay * 0.1)
                    wait = self._reconnect_delay + jitter
                    logger.warning(
                        f"Reconnecting in {wait:.2f}s "
                        f"(backoff: {self._reconnect_delay:.1f}s)"
                    )
                    await asyncio.sleep(wait)

                    # Increase delay with exponential backoff, capped
                    self._reconnect_delay = min(
                        self._reconnect_delay * self._reconnect_multiplier,
                        self._reconnect_max_delay,
                    )
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """Signal the client to stop and close all connections gracefully."""
        logger.info("WebSocket client stopping...")
        self._running = False

        if self._ws and not self._ws.closed:
            await self._ws.close()

    # ── Private Methods ──────────────────────────────────────────────────

    async def _connect_and_stream(self) -> None:
        """
        Establish a WebSocket connection and stream tick data.
        Resets the backoff delay on successful connection.
        """
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

            # ── Reset backoff on successful connection ───────────────────
            self._reconnect_delay = self._reconnect_base_delay
            logger.info("✅ WebSocket connected successfully")

            # ── Subscribe to instruments ─────────────────────────────────
            await self._subscribe(ws)

            # ── Stream loop ──────────────────────────────────────────────
            async for msg in ws:
                if not self._running:
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    t_recv = time.perf_counter_ns()
                    await self._handle_text_message(msg.data, t_recv)

                elif msg.type == aiohttp.WSMsgType.BINARY:
                    t_recv = time.perf_counter_ns()
                    await self._handle_binary_message(msg.data, t_recv)

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    break

                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    logger.warning(f"WebSocket closed by server: {msg.data}")
                    break

    async def _subscribe(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """
        Send subscription message to the broker WebSocket.
        Customize this payload based on your broker's API specification.
        """
        subscribe_msg = json.dumps({
            "a": "subscribe",
            "v": [256265],  # Example: NIFTY 50 instrument token
        })
        await ws.send_str(subscribe_msg)
        logger.info("📡 Subscribed to instrument feed")

        # ── Request full mode (quote + depth) ────────────────────────────
        mode_msg = json.dumps({
            "a": "mode",
            "v": ["full", [256265]],
        })
        await ws.send_str(mode_msg)
        logger.info("📊 Set to FULL quote mode")

    async def _handle_text_message(
        self, raw_data: str, recv_time_ns: int
    ) -> None:
        """
        Parse a JSON text message from the WebSocket and push to queue.

        Args:
            raw_data:      Raw JSON string from the WebSocket.
            recv_time_ns:  Nanosecond timestamp when the message was received.
        """
        try:
            data: dict[str, Any] = json.loads(raw_data)

            # ── Sequence gap detection ───────────────────────────────────
            seq = data.get("sequence", -1)
            if self._last_seq > 0 and seq > self._last_seq + 1:
                gap = seq - self._last_seq - 1
                self._dropped_packets += gap
                logger.warning(
                    f"⚠️  Dropped {gap} packet(s) | "
                    f"Expected seq {self._last_seq + 1}, got {seq} | "
                    f"Total dropped: {self._dropped_packets}"
                )
            self._last_seq = seq

            # ── Extract tick data ────────────────────────────────────────
            tick = self._parse_tick(data)
            if tick:
                tick["_recv_ns"] = recv_time_ns  # Stamp for latency tracking
                await self.tick_queue.put(tick)
                self._total_ticks += 1

                if self._total_ticks % 1000 == 0:
                    logger.info(
                        f"📈 Ticks received: {self._total_ticks} | "
                        f"Dropped: {self._dropped_packets} | "
                        f"Queue size: {self.tick_queue.qsize()}"
                    )

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e} | Data: {raw_data[:200]}")
        except Exception as e:
            logger.exception(f"Error processing text message: {e}")

    async def _handle_binary_message(
        self, raw_data: bytes, recv_time_ns: int
    ) -> None:
        """
        Handle binary-format tick data (used by some brokers for speed).
        Override this for broker-specific binary protocol parsing.

        Args:
            raw_data:      Raw binary payload.
            recv_time_ns:  Nanosecond timestamp when the message was received.
        """
        # Binary parsing is broker-specific. This is a placeholder
        # that converts the binary data length into a log entry.
        logger.debug(
            f"Received binary message: {len(raw_data)} bytes "
            f"(implement broker-specific parser)"
        )

    @staticmethod
    def _parse_tick(data: dict[str, Any]) -> dict[str, Any] | None:
        """
        Normalize raw broker tick data into a standard internal format.

        Expected output format:
            {
                "ltp": 19450.25,       # Last Traded Price
                "volume": 1234567,     # Cumulative volume
                "oi": 9876543,         # Open Interest
                "bid": 19450.00,       # Best bid
                "ask": 19450.50,       # Best ask
                "timestamp": 1700000000,
            }

        Customize the field mappings based on your broker's data schema.
        """
        try:
            return {
                "ltp": float(data.get("last_price", 0)),
                "volume": int(data.get("volume_traded", 0)),
                "oi": int(data.get("oi", 0)),
                "bid": float(data.get("depth", {}).get("buy", [{}])[0].get("price", 0)),
                "ask": float(data.get("depth", {}).get("sell", [{}])[0].get("price", 0)),
                "timestamp": data.get("exchange_timestamp", 0),
                "token": data.get("instrument_token", 0),
            }
        except (KeyError, ValueError, IndexError, TypeError) as e:
            logger.warning(f"Tick parse error: {e}")
            return None

    async def _cleanup(self) -> None:
        """Release all resources (WebSocket + HTTP session)."""
        if self._ws and not self._ws.closed:
            await self._ws.close()
            logger.info("WebSocket connection closed")

        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("HTTP session closed")

        logger.info(
            f"🏁 WebSocket client stopped | "
            f"Total ticks: {self._total_ticks} | "
            f"Dropped packets: {self._dropped_packets}"
        )

    # ── Diagnostics ──────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        """Return current client statistics for monitoring."""
        return {
            "total_ticks": self._total_ticks,
            "dropped_packets": self._dropped_packets,
            "queue_size": self.tick_queue.qsize(),
            "connected": self._ws is not None and not self._ws.closed,
            "reconnect_delay": self._reconnect_delay,
        }
