"""Small line-buffered socket reader shared by the twin server and client."""

from __future__ import annotations

import socket

__all__ = ["LineReader"]


class LineReader:
    """Split a TCP byte stream into text lines (LF or CRLF terminated)."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self._buffer = b""
        self.closed = False

    def _pop_line(self) -> str | None:
        if b"\n" not in self._buffer:
            return None
        raw, self._buffer = self._buffer.split(b"\n", 1)
        return raw.rstrip(b"\r").decode("ascii", errors="replace")

    def readline(self, timeout: float | None) -> str | None:
        """Block until a full line arrives; ``None`` on timeout or end of stream."""
        line = self._pop_line()
        if line is not None:
            return line
        self.sock.settimeout(timeout)
        while not self.closed:
            try:
                chunk = self.sock.recv(4096)
            except TimeoutError:
                return None
            except OSError:
                self.closed = True
                return None
            if not chunk:
                self.closed = True
                return None
            self._buffer += chunk
            line = self._pop_line()
            if line is not None:
                return line
        return None

    def poll(self) -> list[str]:
        """Return all complete lines that are already available (non-blocking)."""
        self.sock.settimeout(0.0)
        while not self.closed:
            try:
                chunk = self.sock.recv(4096)
            except (BlockingIOError, TimeoutError):
                break
            except OSError:
                self.closed = True
                break
            if not chunk:
                self.closed = True
                break
            self._buffer += chunk
        lines = []
        while (line := self._pop_line()) is not None:
            lines.append(line)
        return lines
