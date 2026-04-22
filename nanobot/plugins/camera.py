"""Camera plugin using OpenCV VideoCapture (V4L2 backend)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import base64

import cv2
from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Base
from nanobot.plugins.base import BasePlugin


class CameraConfig(Base):
    """Camera plugin configuration."""

    enabled: bool = False
    device: str = "/dev/video0"
    width: int = 640
    height: int = 480
    fps: int = 30
    format: str = "MJPG"  # MJPG, YUYV, RGB24
    output_dir: str = ""  # Directory to save captured frames
    stream_to_bus: bool = False  # Publish frames to message bus
    inject_frame_to_llm: bool = False  # Inject latest frame into LLM messages
    capture_interval: float = 0.1  # Seconds between captures when streaming


class Cv2Camera:
    """Camera interface using OpenCV VideoCapture (cv2.CAP_V4L2)."""

    def __init__(self, device: int = 0, width: int = 640, height: int = 480, fmt: str = "MJPG"):
        self.device = device
        self.width = width
        self.height = height
        self.fmt = fmt.upper()
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        """Open the V4L2 device via OpenCV."""
        self._cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.device)
        return self._cap.isOpened()

    def configure(self, width: int, height: int, pixelformat: str) -> None:
        """Configure capture format and resolution."""
        if not self._cap:
            return
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        fourcc = cv2.VideoWriter_fourcc(*pixelformat)
        self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    def read(self) -> tuple[bool, bytes]:
        """Capture a frame. Returns (success, jpeg_bytes)."""
        if not self._cap or not self._cap.isOpened():
            return False, b""
        ret, frame = self._cap.read()
        if not ret:
            return False, b""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, buf = cv2.imencode('.jpg', frame, encode_param)
        return True, bytes(buf)

    def close(self) -> None:
        """Release the camera."""
        if self._cap:
            self._cap.release()
            self._cap = None


class CameraPlugin(BasePlugin, AgentHook):
    """
    Camera plugin using OpenCV VideoCapture.

    Publishes captured frames as inbound messages to the bus,
    and injects the latest frame into agent messages before each LLM call.
    """

    name = "camera"
    display_name = "Camera (V4L2)"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return CameraConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = CameraConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: CameraConfig = config

        self._camera: Cv2Camera | None = None
        self._capture_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._latest_frame: bytes | None = None
        self._frame_lock = threading.Lock()

        self._output_dir = Path(config.output_dir) if config.output_dir else None
        if self._output_dir:
            self._output_dir.mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        """Initialize and start the camera."""
        self._init_camera()
        if self.config.stream_to_bus:
            self._start_capture_thread()
        self._enabled = True

    async def stop(self) -> None:
        """Stop camera and release resources."""
        self._stop_event.set()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2)
        if self._camera:
            try:
                self._camera.close()
            except Exception:
                pass
            self._camera = None
        self._enabled = False

    def _init_camera(self) -> None:
        """Initialize the camera via OpenCV."""
        device_num = int(self.config.device.lstrip("/dev/video"))
        self._camera = Cv2Camera(
            device=device_num,
            width=self.config.width,
            height=self.config.height,
            fmt=self.config.format,
        )
        if not self._camera.open():
            raise RuntimeError(f"Cannot open camera device {self.config.device}")
        self._camera.configure(self.config.width, self.config.height, self.config.format)

    def _start_capture_thread(self) -> None:
        """Start the background capture thread."""
        self._stop_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        """Background loop to capture frames at configured interval."""
        import asyncio
        interval = self.config.capture_interval
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            pass

        while not self._stop_event.is_set():
            try:
                _, frame = self._camera.read()
                if frame:
                    with self._frame_lock:
                        self._latest_frame = frame

                    if loop and loop.is_running() and self.config.stream_to_bus:
                        asyncio.run_coroutine_threadsafe(self._publish_frame(frame), loop)
            except Exception:
                pass
            self._stop_event.wait(interval)

    def _capture_frame(self) -> bytes | None:
        """Capture a single frame from the camera."""
        if not self._camera:
            return None
        _, frame = self._camera.read()
        return frame if frame else None

    async def _publish_frame(self, data: bytes) -> None:
        """Publish captured frame to the message bus."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath_str = ""

        if self._output_dir:
            filepath = self._output_dir / f"frame_{timestamp}.jpg"
            try:
                with open(filepath, "wb") as f:
                    f.write(data)
                filepath_str = str(filepath)
            except Exception:
                pass

        encoded = base64.b64encode(data).decode("utf-8")

        msg = InboundMessage(
            channel="camera",
            sender_id="camera",
            chat_id="camera",
            content=f"[Camera frame captured at {timestamp}]",
            media=[filepath_str] if filepath_str else [],
            metadata={
                "frame_data": encoded,
                "timestamp": timestamp,
                "device": self.config.device,
                "format": self.config.format,
            },
        )
        await self.bus.publish_inbound(msg)

    def capture_frame(self) -> bytes | None:
        """Synchronously capture a single frame."""
        if not self._camera:
            return None
        with self._frame_lock:
            return self._latest_frame

    def get_latest_frame(self) -> bytes | None:
        """Get the most recently captured frame data."""
        with self._frame_lock:
            return self._latest_frame

    def capture_one_frame(self) -> bytes | None:
        """Capture a single frame on-demand (bypasses capture thread)."""
        if not self._camera:
            return None
        ok, frame = self._camera.read()
        return frame if ok and frame else None

    def save_frame(self, filepath: str | Path | None = None) -> str | None:
        """Save the latest frame to a file."""
        frame = self.get_latest_frame()
        if not frame:
            return None

        if filepath is None:
            if not self._output_dir:
                return None
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = self._output_dir / f"frame_{timestamp}.jpg"

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(filepath, "wb") as f:
                f.write(frame)
            return str(filepath)
        except Exception:
            return None

    async def before_model_call(
        self, context: AgentHookContext
    ) -> list[dict[str, Any]] | None:
        """Inject the latest camera frame into messages before the LLM call.

        If ``inject_frame_to_llm`` is disabled or no frame is available, returns None
        (no modification). Otherwise prepends a user message with the base64 frame
        as an image_url content block.
        """
        logger.info(
            "[camera] before_model_call called — inject={} iteration={} msg_count={}",
            self.config.inject_frame_to_llm,
            context.iteration,
            len(context.messages),
        )
        if not self.config.inject_frame_to_llm:
            return None

        frame = self.capture_one_frame()
        logger.debug("[camera] capture_one_frame result: {}", "ok" if frame else "none")
        if not frame:
            return None

        encoded = base64.b64encode(frame).decode("utf-8")
        logger.info(
            "[camera] injecting frame to LLM — frame_size={} bytes, base64_len={}",
            len(frame),
            len(encoded),
        )
        image_block: dict[str, Any] = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded}",
                "detail": "low",
            },
        }

        messages = list(context.messages)
        # Find the first user message and append image to its content array
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, list):
                    messages[i]["content"] = content + [image_block]
                elif isinstance(content, str):
                    messages[i]["content"] = [{"type": "text", "text": content}, image_block]
                else:
                    messages[i]["content"] = [image_block]
                break
        else:
            # No user message found, insert image as a new user message at front
            messages.insert(0, {
                "role": "user",
                "content": [image_block],
            })
        logger.info(
            "[camera] messages modified — before={} after={}",
            len(context.messages),
            len(messages),
        )
        return messages