"""Gyroscope helpers: rate integration and rotation counting."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import cumulative_trapezoid

__all__ = ["TurnCount", "count_turns", "integrate_rate"]


def integrate_rate(t: ArrayLike, rate: ArrayLike, initial: float = 0.0) -> np.ndarray:
    """Angle from an angular-rate signal (trapezoidal rule, irregular sampling allowed)."""
    return initial + cumulative_trapezoid(np.asarray(rate, dtype=float), np.asarray(t, dtype=float), initial=0.0)


@dataclass(frozen=True)
class TurnCount:
    """Rotation statistics of an angle record (all in turns = 2*pi rad)."""

    net: float
    """Signed end-to-start difference."""
    travelled: float
    """Total angle covered regardless of direction."""
    positive: float
    """Angle covered while turning in the positive direction."""
    negative: float
    """Angle covered while turning in the negative direction."""
    reversals: int
    """Number of direction changes (after hysteresis)."""


def count_turns(angle: ArrayLike, hysteresis: float = 0.2) -> TurnCount:
    """Count turns of an integrated angle; a reversal needs ``hysteresis`` rad of travel."""
    angle = np.asarray(angle, dtype=float)
    steps = np.diff(angle)
    positive = float(steps[steps > 0].sum())
    negative = float(-steps[steps < 0].sum())
    reversals = 0
    direction = 0
    extreme = start = angle[0]
    for value in angle[1:]:
        if direction == 0:
            if abs(value - start) > hysteresis:
                direction, extreme = (1 if value > start else -1), value
        elif direction == 1:
            if value > extreme:
                extreme = value
            elif extreme - value > hysteresis:
                direction, extreme, reversals = -1, value, reversals + 1
        elif value < extreme:
            extreme = value
        elif value - extreme > hysteresis:
            direction, extreme, reversals = 1, value, reversals + 1
    two_pi = 2.0 * math.pi
    return TurnCount(
        net=float((angle[-1] - angle[0]) / two_pi),
        travelled=(positive + negative) / two_pi,
        positive=positive / two_pi,
        negative=negative / two_pi,
        reversals=reversals,
    )
