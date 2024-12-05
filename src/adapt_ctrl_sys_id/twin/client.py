"""Python client for the twin plus a generic closed-loop runner."""

from __future__ import annotations

import contextlib
import socket
from collections.abc import Callable

import numpy as np

from ..control import LoopResult
from ..signals import pendulum_reference_deg
from ._io import LineReader
from .protocol import DEFAULT_PORT, EXIT, START, TwinState, format_command, parse_state

__all__ = ["TwinClient", "run_closed_loop"]


class TwinClient:
    """Connect to a :class:`~adapt_ctrl_sys_id.twin.server.TwinServer` (or any compatible twin)."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._reader: LineReader | None = None

    def connect(self) -> TwinClient:
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        self._reader = LineReader(sock)
        self._write(START + "\n")
        return self

    def _write(self, text: str) -> None:
        if self._sock is None:
            raise RuntimeError("client is not connected")
        self._sock.sendall(text.encode("ascii"))

    def read_state(self) -> TwinState | None:
        """Next sample from the twin, or ``None`` when the stream ends."""
        if self._reader is None:
            raise RuntimeError("client is not connected")
        line = self._reader.readline(self.timeout)
        return None if line is None else parse_state(line)

    def send(self, u: float, deltax_mm: float = 0.0) -> None:
        self._write(format_command(u, deltax_mm))

    def close(self) -> None:
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._write(EXIT + "\n")
            self._sock.close()
            self._sock = None
            self._reader = None

    def __enter__(self) -> TwinClient:
        return self.connect()

    def __exit__(self, *exc: object) -> None:
        self.close()


def run_closed_loop(
    client: TwinClient,
    controller: Callable[[float, float, float], float],
    duration: float = 120.0,
    reference: Callable[[np.ndarray], np.ndarray] = pendulum_reference_deg,
    deltax_mm: float | Callable[[float], float] = 0.0,
) -> LoopResult:
    """Run ``controller(ref, phi, phid) -> u`` against a connected twin.

    ``deltax_mm`` is the slider command, either constant or a function of time.
    """
    slider = deltax_mm if callable(deltax_mm) else (lambda _t: float(deltax_mm))
    t, ref, phi, phid, u = [], [], [], [], []
    while True:
        state = client.read_state()
        if state is None or state.t > duration + 1e-9:
            break
        r = float(reference(np.asarray(state.t)))
        command = controller(r, state.phi_deg, state.phid_deg)
        client.send(command, slider(state.t))
        t.append(state.t)
        ref.append(r)
        phi.append(state.phi_deg)
        phid.append(state.phid_deg)
        u.append(command)
    return LoopResult(*(np.asarray(v, dtype=float) for v in (t, ref, phi, phid, u)))
