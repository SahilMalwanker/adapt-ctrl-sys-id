"""Message encoding for the twin's line-based TCP protocol."""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "DEFAULT_PORT",
    "EXIT",
    "START",
    "TwinState",
    "format_command",
    "format_state",
    "parse_command",
    "parse_state",
]

DEFAULT_PORT = 10815
START = "START"
EXIT = "EXIT"

_STATE = re.compile(r"<PP><t>\s*([-+0-9.,eE]+)\s*</t><PA>\s*([-+0-9.,eE]+)\s*</PA><PAD>\s*([-+0-9.,eE]+)\s*</PAD>")


@dataclass(frozen=True)
class TwinState:
    """One sample sent by the twin: time [s], angle [deg], angular rate [deg/s]."""

    t: float
    phi_deg: float
    phid_deg: float


def format_state(t: float, phi_deg: float, phid_deg: float) -> str:
    """Encode a sample; the blank before each closing tag matches ``sscanf`` patterns."""
    return f"<PP><t>{t:.4f} </t><PA>{phi_deg:.4f} </PA><PAD>{phid_deg:.4f} </PAD></PP>\n"


def parse_state(line: str) -> TwinState:
    """Decode a sample; decimal commas (locale-dependent senders) are accepted."""
    match = _STATE.search(line)
    if match is None:
        raise ValueError(f"not a twin state message: {line!r}")
    t, phi, phid = (float(v.replace(",", ".")) for v in match.groups())
    return TwinState(t, phi, phid)


def format_command(u: float, deltax_mm: float = 0.0) -> str:
    """Encode an actuator command (rotor [%], slider offset [mm])."""
    return f"{u:f};{deltax_mm:f}\n"


def parse_command(line: str) -> tuple[float, float]:
    """Decode ``"u;deltax"``; a missing slider value defaults to 0."""
    parts = line.strip().split(";")
    if not parts or not parts[0]:
        raise ValueError(f"empty command: {line!r}")
    u = float(parts[0])
    deltax = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
    return u, deltax
