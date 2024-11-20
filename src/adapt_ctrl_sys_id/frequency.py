"""Frequency-response tools: Bode helpers and a correlation (lock-in) analyser."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import trapezoid

__all__ = ["correlation_analysis", "to_db_deg"]


def to_db_deg(response: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude in dB and (unwrapped) phase in degrees of a complex response."""
    response = np.asarray(response, dtype=complex)
    return 20.0 * np.log10(np.abs(response)), np.degrees(np.unwrap(np.angle(response)))


def correlation_analysis(t: ArrayLike, y: ArrayLike, omega: float, amplitude: float, t_start: float = 0.0) -> complex:
    """Estimate ``G(j omega)`` from the response to ``u = amplitude * sin(omega t)``.

    The output is correlated with ``sin`` and ``cos`` over the largest integer
    number of periods after ``t_start`` (transient removal)::

        Re G = 2 / (A T_w) * integral(y sin(w t)),   Im G = 2 / (A T_w) * integral(y cos(w t))

    Correlating over whole periods rejects the DC offset and all harmonics, which
    makes the method robust to noise and to mild nonlinearities.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    period = 2.0 * np.pi / omega
    mask = t >= t_start
    t, y = t[mask], y[mask]
    n_periods = int(np.floor((t[-1] - t[0]) / period + 1e-9))
    if n_periods < 1:
        raise ValueError("record shorter than one period after t_start")
    mask = t <= t[0] + n_periods * period + 1e-12
    t, y = t[mask], y[mask]
    window = t[-1] - t[0]
    real = 2.0 / (amplitude * window) * trapezoid(y * np.sin(omega * t), t)
    imag = 2.0 / (amplitude * window) * trapezoid(y * np.cos(omega * t), t)
    return complex(real, imag)
