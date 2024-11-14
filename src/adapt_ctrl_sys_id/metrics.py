"""Goodness-of-fit and closed-loop performance metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "mean_square",
    "nrmse_fit",
    "overshoot_percent",
    "rmse",
    "settling_time",
    "tracking_costs",
]


def nrmse_fit(y: ArrayLike, y_hat: ArrayLike) -> float:
    """Model fit in percent, ``100 * (1 - ||y - y_hat|| / ||y - mean(y)||)``.

    This is the normalised-RMSE fit used by common identification tools:
    100 % is a perfect match, 0 % is as good as predicting the mean, and
    negative values are worse than the mean.
    """
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    if y.shape != y_hat.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {y_hat.shape}")
    spread = np.linalg.norm(y - y.mean())
    if spread == 0.0:
        raise ValueError("reference signal is constant; fit is undefined")
    return float(100.0 * (1.0 - np.linalg.norm(y - y_hat) / spread))


def rmse(y: ArrayLike, y_hat: ArrayLike) -> float:
    """Root-mean-square error."""
    diff = np.asarray(y, dtype=float) - np.asarray(y_hat, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def mean_square(t: ArrayLike, x: ArrayLike, t_start: float | None = None, t_end: float | None = None) -> float:
    """Mean of ``x**2`` over the samples with ``t_start <= t <= t_end``."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.ones_like(t, dtype=bool)
    if t_start is not None:
        mask &= t >= t_start
    if t_end is not None:
        mask &= t <= t_end
    if not mask.any():
        raise ValueError("evaluation window contains no samples")
    return float(np.mean(x[mask] ** 2))


def tracking_costs(
    t: ArrayLike,
    reference: ArrayLike,
    output: ArrayLike,
    effort: ArrayLike,
    window: tuple[float, float] = (30.0, 100.0),
) -> dict[str, float]:
    """Quadratic tracking error ``J_e`` and control effort ``J_u`` over a window.

    ``J_e = mean((r - y)^2)`` and ``J_u = mean(u^2)``, evaluated on the samples
    inside ``window`` (both bounds inclusive).
    """
    error = np.asarray(reference, dtype=float) - np.asarray(output, dtype=float)
    return {
        "J_e": mean_square(t, error, *window),
        "J_u": mean_square(t, effort, *window),
        "max_abs_error": float(np.max(np.abs(error[(np.asarray(t) >= window[0]) & (np.asarray(t) <= window[1])]))),
    }


def overshoot_percent(y: ArrayLike, y_start: float, y_final: float) -> float:
    """Peak overshoot of a step response in percent of the step height (>= 0)."""
    y = np.asarray(y, dtype=float)
    step = y_final - y_start
    if step == 0.0:
        raise ValueError("step height must be non-zero")
    beyond = (y - y_final) * np.sign(step)
    return float(max(0.0, beyond.max()) / abs(step) * 100.0)


def settling_time(
    t: ArrayLike,
    y: ArrayLike,
    y_final: float,
    band: float,
    t_step: float = 0.0,
) -> float:
    """Time after ``t_step`` from which ``|y - y_final| <= band`` holds until the end.

    Returns ``nan`` if the response never settles inside the band.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    after = t >= t_step
    t, y = t[after], y[after]
    outside = np.flatnonzero(np.abs(y - y_final) > band)
    if outside.size == 0:
        return 0.0
    last = outside[-1]
    if last == len(t) - 1:
        return float("nan")
    return float(t[last + 1] - t_step)
