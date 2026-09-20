"""Filter non-JSON-RPC lines from an MCP stdio server's stdout.

The upstream ``minimax-coding-plan-mcp`` server prints a startup banner to
stdout (which the MCP protocol reserves for JSON-RPC frames).  This wrapper
spawns the underlying server, then forwards only JSON-RPC frames on its own
stdout and diverts everything else to stderr.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from typing import Sequence


def _is_jsonrpc(line: bytes) -> bool:
    try:
        text = line.decode("utf-8", errors="replace").strip()
    except Exception:
        return False
    if not (text.startswith("{") and text.endswith("}")):
        return False
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(obj, dict) and "jsonrpc" in obj


def _pump_stdout(proc: subprocess.Popen[bytes]) -> None:
    """Read lines from the child and forward JSON-RPC frames only."""
    try:
        buf = b""
        stdout = sys.stdout.buffer
        stderr = sys.stderr.buffer
        while True:
            try:
                chunk = os.read(proc.stdout.fileno(), 4096)  # type: ignore[union-attr]
            except OSError:
                break
            if not chunk:
                if buf:
                    if _is_jsonrpc(buf):
                        stdout.write(buf)
                        stdout.flush()
                    else:
                        stderr.write(buf)
                        stderr.flush()
                break
            buf += chunk
            while b"\n" in buf:
                line, _, buf = buf.partition(b"\n")
                if _is_jsonrpc(line):
                    stdout.write(line + b"\n")
                    stdout.flush()
                else:
                    stderr.write(line + b"\n")
                    stderr.flush()
    except Exception:
        pass


def _pump_stderr(proc: subprocess.Popen[bytes]) -> None:
    try:
        stderr = sys.stderr.buffer
        while True:
            chunk = os.read(proc.stderr.fileno(), 4096)  # type: ignore[union-attr]
            if not chunk:
                return
            stderr.write(chunk)
            stderr.flush()
    except Exception:
        return


def _pump_stdin(proc: subprocess.Popen[bytes]) -> None:
    try:
        while True:
            chunk = os.read(sys.stdin.fileno(), 4096)
            if not chunk:
                proc.stdin.close()  # type: ignore[union-attr]
                return
            proc.stdin.write(chunk)  # type: ignore[union-attr]
            proc.stdin.flush()  # type: ignore[union-attr]
    except Exception:
        return


def main(argv: Sequence[str]) -> int:
    if len(argv) < 2:
        print("usage: mcp_filter_stdio.py <command> [args...]", file=sys.stderr)
        return 2
    proc = subprocess.Popen(
        list(argv[1:]),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    threading.Thread(target=_pump_stdin, args=(proc,), daemon=True).start()
    threading.Thread(target=_pump_stdout, args=(proc,), daemon=True).start()
    threading.Thread(target=_pump_stderr, args=(proc,), daemon=True).start()
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 130


if __name__ == "__main__":
    sys.exit(main(sys.argv))
