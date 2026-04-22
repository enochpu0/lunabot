"""WebSocket client for nanobot gateway."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import aiohttp
from websockets.asyncio.client import ClientConnection, connect as _ws_connect


_logger = logging.getLogger("nanobot.gui.ws_client")


@dataclass
class WsEvent:
    """Inbound WebSocket event from nanobot gateway."""
    event: str
    chat_id: str | None = None
    text: str | None = None
    content: str | None = None
    detail: str | None = None


class WebSocketClient:
    """WebSocket client for nanobot gateway.

    Connects to the gateway's WebSocket endpoint, handles token bootstrapping,
    and delivers parsed events via an asyncio.Queue.
    """

    def __init__(
        self,
        gateway_url: str = "http://127.0.0.1:8765",
        ws_path: str = "/",
        client_id: str = "gui",
    ):
        self._gateway_url = gateway_url.rstrip("/")
        self._ws_path = ws_path
        self._client_id = client_id
        self._ws: ClientConnection | None = None
        self._receive_task: asyncio.Task | None = None
        self._running = False
        self._chat_id: str | None = None
        self._event_queue: asyncio.Queue[WsEvent | None] = asyncio.Queue()
        self._stream_buf: str = ""

    @property
    def chat_id(self) -> str | None:
        return self._chat_id

    async def _fetch_bootstrap_token(self) -> tuple[str, str]:
        """Fetch a short-lived token from the gateway bootstrap endpoint.

        Returns (token, ws_path).
        Raises aiohttp.ClientError on failure.
        """
        url = f"{self._gateway_url}/webui/bootstrap"
        _logger.debug("[bootstrap] GET %s", url)
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                _logger.debug("[bootstrap] status=%s", resp.status)
                if resp.status != 200:
                    text = await resp.text()
                    _logger.error("[bootstrap] failed status=%s body=%s", resp.status, text[:200])
                    raise aiohttp.ClientError(
                        f"Bootstrap failed with status {resp.status}: {text}"
                    )
                data = await resp.json()
                _logger.debug("[bootstrap] token received, ws_path=%s", data.get("ws_path"))
                return data["token"], data.get("ws_path", "/")

    def _make_ws_url(self, token: str) -> str:
        """Build the WebSocket URL with token query param."""
        scheme = "wss" if self._gateway_url.startswith("https") else "ws"
        host = self._gateway_url.split("://", 1)[1]
        return f"{scheme}://{host}{self._ws_path}?client_id={self._client_id}&token={token}"

    async def connect(self, token: str | None = None) -> str:
        """Connect to the gateway WebSocket endpoint.

        If token is None, fetches one from /webui/bootstrap.
        Returns the assigned chat_id.
        """
        if token is None:
            _logger.debug("[connect] No token, fetching bootstrap token")
            token, ws_path = await self._fetch_bootstrap_token()
            if ws_path:
                self._ws_path = ws_path
        else:
            _logger.debug("[connect] Using provided token")

        url = self._make_ws_url(token)
        _logger.debug("[connect] Connecting to WS %s", url)
        try:
            self._ws = await _ws_connect(url)
            _logger.debug("[connect] WS connection established")
        except Exception as e:
            _logger.error("[connect] WS connect failed: %s", e)
            raise

        self._running = True
        self._receive_task = asyncio.create_task(self._receive_loop())
        _logger.debug("[connect] Waiting for 'ready' event (timeout=5s)...")
        # Wait for the 'ready' event to confirm connection
        while self._chat_id is None and self._running:
            evt = await self.next_event(timeout=5.0)
            if evt is None:
                _logger.error("[connect] Timeout waiting for 'ready', connection dropped")
                raise RuntimeError("WebSocket connection dropped before 'ready'")
            _logger.debug("[connect] Received event: %s", evt.event)
        _logger.debug("[connect] Got chat_id=%s, connected", self._chat_id)
        return self._chat_id or ""

    async def _receive_loop(self) -> None:
        """Read frames from the WebSocket and enqueue parsed events."""
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                event = self._parse_frame(raw)
                if event is not None:
                    await self._event_queue.put(event)
        except Exception:
            pass
        finally:
            self._running = False
            await self._event_queue.put(None)  # sentinel

    async def wait_for_event(self) -> WsEvent | None:
        """Await the next WsEvent from the receive queue (blocking)."""
        return await self.next_event()

    def _parse_frame(self, raw: str) -> WsEvent | None:
        """Parse a raw JSON frame into a WsEvent."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        event = data.get("event", "")
        chat_id = data.get("chat_id")

        if event == "ready":
            self._chat_id = chat_id
            return WsEvent(event=event, chat_id=chat_id)

        if event == "message":
            return WsEvent(
                event=event,
                chat_id=chat_id,
                text=data.get("text"),
                content=data.get("content"),
            )

        if event == "delta":
            self._stream_buf += data.get("text", "")
            return WsEvent(
                event=event,
                chat_id=chat_id,
                text=data.get("text"),
            )

        if event == "stream_end":
            text = self._stream_buf
            self._stream_buf = ""
            return WsEvent(
                event=event,
                chat_id=chat_id,
                text=text,
            )

        if event == "attached":
            return WsEvent(event=event, chat_id=chat_id)

        if event == "error":
            return WsEvent(
                event=event,
                chat_id=chat_id,
                detail=data.get("detail"),
            )

        return WsEvent(event=event, chat_id=chat_id)

    async def send_message(self, content: str, chat_id: str | None = None) -> None:
        """Send a typed message envelope to the gateway (multiplex protocol)."""
        if self._ws is None:
            raise RuntimeError("Not connected")
        target = chat_id or self._chat_id
        if target is None:
            raise RuntimeError("No chat_id assigned")
        envelope = {"type": "message", "chat_id": target, "content": content}
        await self._ws.send(json.dumps(envelope, ensure_ascii=False))

    async def send_text(self, text: str) -> None:
        """Send a legacy plain-text frame routed to the connection's default chat_id.

        This mirrors what the test clients and websocat do, ensuring the server
        assigns the message to the correct chat_id without requiring an explicit
        ``{"type": "message", "chat_id": ..., "content": ...}`` envelope.
        """
        if self._ws is None:
            raise RuntimeError("Not connected")
        await self._ws.send(text)

    async def send_new_chat(self, content: str) -> None:
        """Start a new chat with the given message."""
        if self._ws is None:
            raise RuntimeError("Not connected")
        envelope = {"type": "new_chat", "content": content}
        await self._ws.send(json.dumps(envelope, ensure_ascii=False))

    async def next_event(self, timeout: float | None = None) -> WsEvent | None:
        """Wait for the next inbound event (or None on disconnect)."""
        try:
            return await asyncio.wait_for(self._event_queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    def is_connected(self) -> bool:
        return self._running and self._ws is not None and not self._ws.close_code

    async def close(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._running = False
        if self._receive_task:
            self._receive_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._receive_task
            self._receive_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
