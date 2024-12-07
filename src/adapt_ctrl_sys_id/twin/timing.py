"""Sampling-regularity analysis for real-time records."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["sampling_report"]


def sampling_report(t: ArrayLike, nominal: float | None = None, tolerance: float = 0.5e-3) -> dict[str, float]:
    """Summarise how evenly a record was sampled.

    ``nominal`` defaults to the median interval. An interval counts as a
    deviation when it differs from ``nominal`` by more than ``tolerance``.
    """
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    if dt.size == 0:
        raise ValueError("need at least two samples")
    nominal = float(np.median(dt)) if nominal is None else float(nominal)
    deviations = np.abs(dt - nominal) > tolerance
    return {
        "nominal": nominal,
        "intervals": int(dt.size),
        "deviations": int(deviations.sum()),
        "deviation_share": float(deviations.mean()),
        "mean_interval": float(dt.mean()),
        "std_interval": float(dt.std()),
        "p99_interval": float(np.percentile(dt, 99)),
        "max_interval": float(dt.max()),
        "duration": float(t[-1] - t[0]),
    }
