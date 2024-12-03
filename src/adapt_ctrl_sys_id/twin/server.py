"""Threaded TCP server that exposes :class:`PendulumTwin` to any client."""

from __future__ import annotations

import contextlib
import socket
import threading
import time

from ._io import LineReader
from .physics import PendulumTwin
from .protocol import DEFAULT_PORT, EXIT, START, TwinState, format_state, parse_command

__all__ = ["TwinServer"]


class TwinServer:
    """Serve the twin to one client at a time.

    Parameters
    ----------
    realtime:
        ``True`` paces the simulation with the wall clock: the physics advances
        by the *measured* elapsed time each cycle and commands are applied at
        the next tick, exactly like a real rig (timestamps jitter).
        ``False`` runs in lock-step: the twin advances one nominal sample per
        received command, which is deterministic and as fast as the client.
    timeout:
        Seconds to wait for the client before a session is abandoned.
    """

    def __init__(
        self,
        twin: PendulumTwin | None = None,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        realtime: bool = True,
        timeout: float = 5.0,
    ) -> None:
        self.twin = twin or PendulumTwin()
        self.host = host
        self.port = port
        self.realtime = realtime
        self.timeout = timeout
        self.sessions = 0
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # lifecycle ---------------------------------------------------------------
    def bind(self) -> TwinServer:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(1)
        sock.settimeout(0.2)
        self._sock = sock
        self.port = sock.getsockname()[1]
        return self

    def start(self) -> TwinServer:
        """Bind and serve in a background thread (use ``port=0`` for a free port)."""
        if self._sock is None:
            self.bind()
        self._thread = threading.Thread(target=self.serve_forever, name="twin-server", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> TwinServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def serve_forever(self) -> None:
        if self._sock is None:
            self.bind()
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._session(conn)

    # protocol ----------------------------------------------------------------
    def _session(self, conn: socket.socket) -> None:
        reader = LineReader(conn)
        while True:
            line = reader.readline(self.timeout)
            if line is None:
                return
            if line.strip().upper() == START:
                break
        self.sessions += 1
        self.twin.reset()
        try:
            if self.realtime:
                self._run_realtime(conn, reader)
            else:
                self._run_lockstep(conn, reader)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    @staticmethod
    def _send(conn: socket.socket, state: TwinState) -> None:
        conn.sendall(format_state(state.t, state.phi_deg, state.phid_deg).encode("ascii"))

    def _apply(self, line: str) -> bool:
        """Apply a command line; returns ``False`` when the client says goodbye."""
        if line.strip().upper() == EXIT:
            return False
        # malformed lines are ignored, like on a tolerant field bus
        with contextlib.suppress(ValueError):
            self.twin.command(*parse_command(line))
        return True

    def _run_lockstep(self, conn: socket.socket, reader: LineReader) -> None:
        self._send(conn, self.twin.measure())
        while not self._stop.is_set():
            line = reader.readline(self.timeout)
            if line is None or not self._apply(line):
                return
            self._send(conn, self.twin.advance())

    def _run_realtime(self, conn: socket.socket, reader: LineReader) -> None:
        ts = self.twin.config.ts
        last = time.perf_counter()
        next_tick = last
        self._send(conn, self.twin.measure())
        while not self._stop.is_set():
            next_tick += ts
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            for line in reader.poll():
                if not self._apply(line):
                    return
            if reader.closed:
                return
            now = time.perf_counter()
            state = self.twin.advance(min(now - last, 0.1))
            last = now
            self._send(conn, state)
