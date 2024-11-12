"""Loaders for the measurement records in ``data/``.

The folder is located next to ``src/``. Set the environment variable
``ADAPT_CTRL_SYS_ID_DATA`` to point somewhere else (e.g. for a non-editable install).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "TimeSeries",
    "data_dir",
    "gyro_chair_spin",
    "pendulum_free_fall",
    "pendulum_static_equilibrium",
    "pendulum_step_response",
    "read_csv",
    "servo_excitation_profile",
    "servo_frequency_response",
    "servo_static_map",
]

ENV_VAR = "ADAPT_CTRL_SYS_ID_DATA"


def data_dir() -> Path:
    """Directory that holds the CSV records."""
    env = os.environ.get(ENV_VAR)
    candidates = [Path(env)] if env else []
    candidates.append(Path(__file__).resolve().parents[2] / "data")
    for path in candidates:
        if (path / "pendulum_free_fall.csv").is_file():
            return path
    raise FileNotFoundError(
        f"measurement data not found; run from a source checkout or set {ENV_VAR} to the repository's data/ folder"
    )


def read_csv(name: str) -> dict[str, np.ndarray]:
    """Read a numeric CSV file with a single header row into column arrays."""
    path = data_dir() / name
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [[float(v) for v in row] for row in reader if row]
    values = np.asarray(rows, dtype=float)
    return {col: values[:, i] for i, col in enumerate(header)}


@dataclass(frozen=True)
class TimeSeries:
    """Pendulum record: time [s], angle [deg] and angular rate [deg/s].

    ``phid_deg`` follows the angle's sign convention. On the hardware the IMU
    axis points the other way (its raw rate correlates with the differentiated
    potentiometer angle at -0.99), so loaders negate the raw IMU column.
    """

    t: np.ndarray
    phi_deg: np.ndarray
    phid_deg: np.ndarray

    @property
    def phi(self) -> np.ndarray:
        """Angle in radians."""
        return np.deg2rad(self.phi_deg)

    @property
    def phid(self) -> np.ndarray:
        """Angular rate in rad/s."""
        return np.deg2rad(self.phid_deg)

    def resample(self, dt: float) -> TimeSeries:
        """Linear interpolation onto a uniform grid (the logger samples irregularly)."""
        grid = np.arange(self.t[0], self.t[-1] + 0.5 * dt, dt)
        grid = grid[grid <= self.t[-1]]
        return TimeSeries(grid, np.interp(grid, self.t, self.phi_deg), np.interp(grid, self.t, self.phid_deg))

    def window(self, t_start: float, t_end: float) -> TimeSeries:
        mask = (self.t >= t_start) & (self.t <= t_end)
        return TimeSeries(self.t[mask], self.phi_deg[mask], self.phid_deg[mask])


def servo_static_map() -> dict[str, np.ndarray]:
    """Steady-state speed of the servo for constant voltages (hardware)."""
    return read_csv("servo_static_map.csv")


def servo_frequency_response() -> dict[str, np.ndarray]:
    """Magnitude/phase measured with sine excitation at three frequencies (hardware)."""
    return read_csv("servo_frequency_response.csv")


def servo_excitation_profile() -> dict[str, np.ndarray]:
    """Piecewise-constant voltage profile used for identification."""
    return read_csv("servo_excitation_profile.csv")


def _series(name: str) -> TimeSeries:
    cols = read_csv(name)
    return TimeSeries(cols["t_s"], cols["phi_deg"], -cols["imu_rate_deg_s"])


def pendulum_free_fall() -> TimeSeries:
    """Release from about 90 deg with the rotor off (hardware, slider up)."""
    return _series("pendulum_free_fall.csv")


def pendulum_step_response() -> TimeSeries:
    """Thrust steps to about 20 deg and then about 40 deg (hardware, slider up)."""
    return _series("pendulum_step_response.csv")


def pendulum_static_equilibrium() -> dict[str, np.ndarray]:
    """Equilibrium angle for constant rotor commands at two slider positions (hardware)."""
    return read_csv("pendulum_static_equilibrium.csv")


def gyro_chair_spin() -> dict[str, np.ndarray]:
    """Integrated smartphone gyroscope yaw angle while spinning a swivel chair."""
    return read_csv("gyro_chair_spin.csv")
