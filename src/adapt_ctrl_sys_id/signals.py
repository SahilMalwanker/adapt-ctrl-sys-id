"""Excitation and reference signals shared by the experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "SERVO_EXCITATION",
    "pendulum_reference_deg",
    "piecewise_constant",
    "servo_excitation",
    "square_wave",
]

#: Piecewise-constant voltage profile (start time [s], level [V]) used to
#: excite the DC servo for identification. It mixes small and large steps of
#: both signs so that the experiment exposes the speed-dependent gain.
SERVO_EXCITATION: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (1.0, 1.0),
    (2.0, 4.0),
    (4.0, 7.0),
    (6.0, 1.0),
    (8.0, -5.0),
    (10.0, 0.5),
    (11.0, -0.5),
    (12.0, 3.0),
    (13.0, 1.0),
    (14.0, 0.0),
)


def piecewise_constant(t: ArrayLike, starts: ArrayLike, levels: ArrayLike) -> np.ndarray:
    """Evaluate a zero-order-hold profile; the signal is 0 before the first start."""
    t = np.asarray(t, dtype=float)
    starts = np.asarray(starts, dtype=float)
    levels = np.asarray(levels, dtype=float)
    if starts.shape != levels.shape or starts.ndim != 1:
        raise ValueError("starts and levels must be 1-D arrays of equal length")
    if np.any(np.diff(starts) <= 0):
        raise ValueError("start times must be strictly increasing")
    idx = np.searchsorted(starts, t, side="right") - 1
    out = np.where(idx >= 0, levels[np.clip(idx, 0, None)], 0.0)
    return out


def servo_excitation(t: ArrayLike) -> np.ndarray:
    """Voltage profile of the servo identification experiment (15 s long)."""
    starts, levels = zip(*SERVO_EXCITATION, strict=True)
    return piecewise_constant(t, starts, levels)


def square_wave(t: ArrayLike, low: float, high: float, period: float, t0: float = 0.0) -> np.ndarray:
    """Square wave that starts with its ``high`` half-period at ``t0``."""
    if period <= 0:
        raise ValueError("period must be positive")
    phase = np.mod(np.asarray(t, dtype=float) - t0, period)
    return np.where(phase < period / 2.0, high, low)


def pendulum_reference_deg(t: ArrayLike) -> np.ndarray:
    """Reference angle profile for the pendulum tracking task, in degrees.

    0 deg for 5 s, a 2 deg/s ramp up to 40 deg, then plateaus at 40, 30, 50 and
    40 deg. The profile combines a ramp (tracking) with up and down steps
    (gravity helps on the way down, the propeller must push on the way up).
    """
    t = np.asarray(t, dtype=float)
    conditions = [t < 5.0, t < 25.0, t < 40.0, t < 65.0, t < 90.0]
    choices = [np.zeros_like(t), (t - 5.0) * 2.0, np.full_like(t, 40.0), np.full_like(t, 30.0), np.full_like(t, 50.0)]
    return np.select(conditions, choices, default=40.0)
