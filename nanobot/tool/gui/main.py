"""Nanobot GUI Client - Tkinter-based main window."""

from __future__ import annotations

import asyncio
import logging
import queue
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

# Camera / image handling
from PIL import Image, ImageTk

# WebSocket client
from nanobot.tool.gui.ws_client import WebSocketClient


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("nanobot.gui")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Console handler — all DEBUG+ messages go to stderr (visible in terminal)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


_logger = _setup_logger()


# ---------------------------------------------------------------------------
# Camera capture (runs in background thread, puts frames in a queue)
# ---------------------------------------------------------------------------


class CameraCapture:
    """Background camera capture that puts frames into a queue."""

    def __init__(self, device: int = 0, width: int = 640, height: int = 480):
        self._device = device
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)

    def start(self) -> bool:
        if self._running:
            return True
        self._cap = cv2.VideoCapture(self._device)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None

    def read_frame(self) -> Any:
        """Get the latest frame, or None if no new frame is available."""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    def _capture_loop(self) -> None:
        while self._running:
            if self._cap is None:
                break
            ret, frame = self._cap.read()
            if not ret:
                continue
            # Drop old frames if we're behind
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break
            self._frame_queue.put(frame)

    def capture_one(self) -> Any:
        """Capture a single frame on demand (blocks up to 1s)."""
        if self._cap is None:
            return None
        try:
            return self._frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None


# ---------------------------------------------------------------------------
# Tkinter GUI
# ---------------------------------------------------------------------------

class _ChatMessage:
    """Simple dataclass for a chat message."""
    role: str
    content: str
    time: datetime = field(default_factory=datetime.now)


class NanobotGui:
    """Main GUI window using tkinter."""

    def __init__(self, gateway_url: str = "http://127.0.0.1:8765", camera_device: int = 0):
        self._gateway_url = gateway_url
        self._camera = CameraCapture(device=camera_device)
        self._camera_photo: ImageTk.PhotoImage | None = None
        self._ws_client: WebSocketClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws_thread: threading.Thread | None = None
        self._ws_queue: queue.Queue = queue.Queue()  # Tk → WS: send / close
        self._ws_events: queue.Queue = queue.Queue()  # WS → Tk: connected / message / delta / ...
        self._ws_close_done: threading.Event = threading.Event()
        self._connected = False
        self._streaming = False
        self._chat_id: str | None = None
        self._last_assistant_text = ""  # for streaming accumulation

        self._root = None
        self._chat_text: Any = None  # tkinter.Text
        self._input_entry: Any = None  # tkinter.Text
        self._send_btn: Any = None
        self._status_label: Any = None
        self._attach_var = None  # tkinter.BooleanVar
        self._input_lock = False  # guard against double-send

    # ------------------------------------------------------------------ UI setup
    def _setup_ui(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.title("Nanobot GUI")
        root.geometry("1000x650")
        self._root = root

        # Top-level horizontal paned window
        paned = ttk.PanedWindow(root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=4, pady=4)

        # ── Left panel: camera ──────────────────────────────────────
        left_frame = ttk.Frame(paned, width=360)
        paned.add(left_frame, weight=0)

        cam_label = ttk.Label(left_frame, text="Camera Preview", font=("Arial", 11, "bold"))
        cam_label.pack(pady=(0, 4))

        self._cam_canvas = tk.Canvas(
            left_frame, width=320, height=240,
            bg="#2b2b2b", highlightthickness=0,
        )
        self._cam_canvas.pack(pady=4)

        self._attach_var = tk.BooleanVar(value=False)
        attach_check = ttk.Checkbutton(
            left_frame, text="发送时附图",
            variable=self._attach_var,
        )
        attach_check.pack(pady=2)

        cam_status = ttk.Label(left_frame, text="", foreground="#888", font=("Arial", 9))
        cam_status.pack(pady=2)
        if self._camera.start():
            cam_status.config(text="✓ 摄像头已启动", foreground="#4c4")
        else:
            cam_status.config(text="✗ 摄像头未找到", foreground="#c44")

        ttk.Separator(left_frame, orient="horizontal").pack(fill="x", pady=8)

        # ── Right panel: chat ────────────────────────────────────────
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)

        # Status bar
        status_frame = ttk.Frame(right_frame)
        status_frame.pack(fill="x", pady=(0, 4))

        self._status_label = ttk.Label(
            status_frame, text="连接中...", foreground="#888",
        )
        self._status_label.pack(side="left")

        retry_btn = ttk.Button(status_frame, text="重试连接", command=self._connect_ws)
        retry_btn.pack(side="right")

        # Chat area
        chat_frame = ttk.Frame(right_frame)
        chat_frame.pack(fill="both", expand=True, pady=4)

        scrollbar = ttk.Scrollbar(chat_frame)
        scrollbar.pack(side="right", fill="y")

        self._chat_text = tk.Text(
            chat_frame, wrap="word", yscrollcommand=scrollbar.set,
            state="disabled", font=("Arial", 11),
            bg="#1e1e1e", fg="#e0e0e0",
            insertbackground="white",
            relief="flat", padx=6, pady=6,
        )
        self._chat_text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self._chat_text.yview)

        # Tag styles for chat bubbles
        self._chat_text.tag_config("user_bubble", background="#0078d4", foreground="white",
                                   lmargin1=8, lmargin2=8, rmargin=8, spacing3=4)
        self._chat_text.tag_config("user_label", foreground="#80b8e0",
                                   font=("Arial", 9, "bold"), lmargin1=8, lmargin2=8)
        self._chat_text.tag_config("assistant_bubble", background="#2b2b2b", foreground="#e0e0e0",
                                   lmargin1=8, lmargin2=8, rmargin=8, spacing3=4)
        self._chat_text.tag_config("assistant_label", foreground="#888",
                                   font=("Arial", 9, "bold"), lmargin1=8, lmargin2=8)
        self._chat_text.tag_config("system", foreground="#888", font=("Arial", 9, "italic"),
                                   lmargin1=8, lmargin2=8)
        self._chat_text.tag_config("debug", foreground="#666", font=("Arial", 8),
                                   lmargin1=8, lmargin2=8)

        # Input area
        input_frame = ttk.Frame(right_frame)
        input_frame.pack(fill="x", pady=(4, 0))

        self._input_entry = tk.Text(input_frame, height=3, font=("Arial", 11),
                                   bg="#2b2b2b", fg="#e0e0e0", insertbackground="white",
                                   relief="flat", padx=6, pady=4, wrap="word")
        self._input_entry.grid(row=0, column=0, sticky="ew")
        self._input_entry.bind("<Return>", self._on_input_return)
        self._input_entry.bind("<Shift-Return>", lambda e: None)  # allow newline

        self._send_btn = ttk.Button(input_frame, text="发送 →", command=self._on_send_clicked,
                                   width=8)
        self._send_btn.grid(row=0, column=1, padx=(4, 0))
        input_frame.columnconfigure(0, weight=1)
        self._set_input_enabled(False)

        # Close handler
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ Chat helpers
    def _append_chat(self, role: str, text: str, debug: dict | None = None) -> None:
        tag = "user" if role == "user" else "assistant"
        label_tag = f"{tag}_label"
        bubble_tag = f"{tag}_bubble"

        self._chat_text.config(state="normal")

        ts = datetime.now().strftime("%H:%M:%S")
        role_label = "YOU" if role == "user" else "BOT"
        self._chat_text.insert("end", f"{role_label}  {ts}\n", label_tag)
        self._chat_text.insert("end", f"{text}\n", bubble_tag)
        if debug:
            debug_str = "  ".join(f"{k}: {v}" for k, v in debug.items())
            self._chat_text.insert("end", f"  {debug_str}\n", "debug")
        self._chat_text.insert("end", "\n")
        self._chat_text.see("end")
        self._chat_text.config(state="disabled")

        # Log to file
        role_log = "USER" if role == "user" else "BOT"
        _logger.debug("[%s] %s: %s", ts, role_log, text)
        if debug:
            _logger.debug("[%s]   debug: %s", ts, debug)

    def _set_status(self, text: str, color: str = "#888") -> None:
        if self._status_label:
            self._status_label.config(text=text, foreground=color)

    def _set_input_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        if self._input_entry:
            self._input_entry.config(state=state)
        if self._send_btn:
            self._send_btn.config(state="normal" if enabled else "disabled")

    # ------------------------------------------------------------------ Event handlers
    def _on_input_return(self, event) -> None:
        # Return without Shift sends; Shift+Return allows newline
        self._on_send_clicked()
        return "break"

    def _on_send_clicked(self) -> None:
        if self._input_lock:
            return
        text = self._input_entry.get("1.0", "end").strip()
        if not text:
            return
        self._input_lock = True
        self._set_input_enabled(False)
        self._input_entry.delete("1.0", "end")

        attach = self._attach_var.get() and self._camera._running
        frame = None
        if attach:
            frame = self._camera.capture_one()

        debug: dict = {}
        if attach and frame is not None:
            debug["attach_photo"] = True
            debug["image_size"] = f"{frame.shape[1]}x{frame.shape[0]}"

        self._append_chat("user", text, debug)

        # Hand off to WS thread
        self._ws_queue.put(("send", text, frame))

    def _on_close(self) -> None:
        self._camera.stop()
        if self._ws_client:
            self._ws_queue.put(("close", None, None))
            # Wait for the WS thread to send close frame (max 2s)
            self._ws_close_done.wait(timeout=2.0)
        if self._root:
            self._root.destroy()

    # ------------------------------------------------------------------ WebSocket
    def _connect_ws(self) -> None:
        self._set_status("连接中...", "#888")
        self._set_input_enabled(False)
        self._ws_close_done.clear()
        t = threading.Thread(target=self._ws_worker, daemon=True)
        t.start()
        # Ensure thread doesn't silently die — log if it exits unexpectedly
        def check_thread():
            if t.is_alive():
                return
            _logger.warning("[WS] Worker thread exited unexpectedly (daemon=%s)", t.daemon)
        self._root.after(5000, check_thread)

    def _ws_worker(self) -> None:
        async def run() -> None:
            try:
                _logger.debug("[WS] Starting connection to %s", self._gateway_url)
                client = WebSocketClient(
                    gateway_url=self._gateway_url,
                    client_id="gui",
                )
                self._ws_client = client
                _logger.debug("[WS] Fetching bootstrap token from %s/webui/bootstrap", self._gateway_url)
                try:
                    chat_id = await asyncio.wait_for(client.connect(), timeout=15.0)
                    _logger.debug("[WS] Connected with chat_id=%s", chat_id)
                except asyncio.TimeoutError:
                    _logger.error("[WS] Connection timed out after 15s")
                    self._ws_events.put(("error", "连接超时 (15s)", None))
                    return
                except Exception as e:
                    _logger.error("[WS] Connection failed: %s", e)
                    self._ws_events.put(("error", str(e), None))
                    return
                # Signal connected; _process_ws_queue will set _connected=True
                self._ws_events.put(("connected", chat_id, None))
                _logger.debug("[_ws_worker] put connected in queue (qsize=%d)", self._ws_events.qsize())

                # Poll for inbound events and outbound commands
                while True:
                    # Non-blocking check for outbound commands from Tk thread.
                    # Only "send" and "close" flow through _ws_queue here; everything
                    # else (connected, message, delta, stream_end, error, attached) is
                    # produced by the WS client into _ws_events for _process_ws_queue.
                    try:
                        cmd = self._ws_queue.get_nowait()
                    except queue.Empty:
                        cmd = None

                    if cmd is not None:
                        kind, arg1, arg2 = cmd
                        if kind == "close":
                            await client.close()
                            self._ws_close_done.set()
                            break
                        if kind == "send":
                            text, _frame = arg1, arg2
                            await client.send_text(text)
                            continue  # check queue immediately before waiting for events
                    evt = await client.next_event(timeout=0.2)
                    if evt is None:
                        # Only break if connection is actually lost;
                        # a None from timeout just means no event yet — keep polling
                        if not client.is_connected():
                            break
                        continue
                    if evt.event == "message":
                        self._ws_events.put(("message", "assistant", evt.text or evt.content or ""))
                    elif evt.event == "delta":
                        self._ws_events.put(("delta", evt.text or "", None))
                    elif evt.event == "stream_end":
                        self._ws_events.put(("stream_end", None, None))
                    elif evt.event == "error":
                        self._ws_events.put(("error", evt.detail or "Unknown error", None))
                    elif evt.event == "attached":
                        if evt.chat_id:
                            self._ws_events.put(("attached", evt.chat_id, None))
            except Exception as e:
                self._ws_events.put(("error", str(e), None))

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run())
        finally:
            # Cancel any remaining tasks and close the loop
            pending = asyncio.all_tasks(loop)
            for t in pending:
                t.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    def _process_ws_queue(self) -> None:
        """Poll the WS events queue from the Tk main loop.

        All queue consumers are here: connected, message, delta, stream_end,
        error, attached.  _ws_worker only puts "send" and "close" into the
        _ws_queue and handles those directly.
        """
        try:
            msg = self._ws_events.get_nowait()
            kind, arg1, arg2 = msg

            if kind == "connected":
                self._chat_id = arg1
                self._connected = True
                self._set_status("✓ 已连接", "#4c4")
                self._set_input_enabled(True)
                self._append_chat("assistant", f"已连接到 nanobot gateway (chat_id={arg1})")
            elif kind == "message":
                self._streaming = False
                self._input_lock = False
                self._set_input_enabled(True)
                self._append_chat(arg1, arg2)
            elif kind == "delta":
                if not self._streaming:
                    self._streaming = True
                    self._last_assistant_text = ""
                    self._append_chat_streaming_start()
                self._last_assistant_text += arg1
                self._append_chat_streaming_delta(arg1)
            elif kind == "stream_end":
                self._streaming = False
                self._input_lock = False
                self._set_input_enabled(True)
            elif kind == "error":
                self._streaming = False
                self._input_lock = False
                self._set_input_enabled(True)
                self._set_status("✗ 连接失败", "#c44")
                self._append_chat("assistant", f"错误: {arg1}")
            elif kind == "attached":
                if arg1:
                    self._chat_id = arg1
        except queue.Empty:
            pass

        # Continue polling
        if self._root:
            self._root.after(50, self._process_ws_queue)

    def _append_chat_streaming_start(self) -> None:
        self._chat_text.config(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        self._chat_text.insert("end", f"BOT  {ts}\n", "assistant_label")
        self._chat_text.tag_add("assistant_bubble", "end-2l", "end")
        self._chat_text.see("end")
        self._chat_text.config(state="disabled")
        _logger.debug("[%s] BOT: [streaming start]", ts)

    def _append_chat_streaming_delta(self, delta: str) -> None:
        self._chat_text.config(state="normal")
        self._chat_text.insert("end", delta)
        self._chat_text.see("end")
        self._chat_text.config(state="disabled")
        _logger.debug("BOT: %s", delta)

    # ------------------------------------------------------------------ Camera preview
    def _update_camera(self) -> None:
        frame = self._camera.read_frame()
        if frame is not None and self._cam_canvas:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((320, 240), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._cam_canvas.create_image(160, 120, image=photo)
            self._camera_photo = photo  # keep reference
        if self._root:
            self._root.after(30, self._update_camera)

    # ------------------------------------------------------------------ Run
    def run(self) -> None:
        self._setup_ui()
        self._update_camera()
        self._connect_ws()  # start WS thread before polling queue
        self._process_ws_queue()
        self._root.mainloop()


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Nanobot GUI")
    parser.add_argument(
        "--url", default="http://127.0.0.1:8765",
        help="Gateway URL (default: http://127.0.0.1:8765)",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0, i.e. /dev/video0)",
    )
    args = parser.parse_args()
    try:
        gui = NanobotGui(gateway_url=args.url, camera_device=args.camera)
        gui.run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0
