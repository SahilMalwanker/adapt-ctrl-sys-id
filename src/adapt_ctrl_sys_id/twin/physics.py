"""Physics core of the digital twin (independent of networking)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..pendulum import CALIBRATED, PendulumParams, coefficients, rk4_step
from .protocol import TwinState

__all__ = ["CALIBRATED", "PendulumTwin", "TwinConfig"]


@dataclass(frozen=True)
class TwinConfig:
    """Sampling and sensor settings of the twin."""

    ts: float = 0.01
    """Nominal sample time [s]."""
    substeps: int = 5
    """RK4 steps per sample."""
    angle_noise_deg: float = 0.05
    """Standard deviation of the angle sensor noise [deg]."""
    rate_noise_deg: float = 0.5
    """Standard deviation of the rate sensor noise [deg/s]."""
    slider_speed: float = 0.1
    """Slider travel speed [m/s]."""
    seed: int | None = 0
    """Seed of the sensor-noise generator (``None`` = random)."""


class PendulumTwin:
    """Calibrated pendulum with rotor lag, dead band, moving slider and noisy sensors."""

    def __init__(self, params: PendulumParams = CALIBRATED, config: TwinConfig | None = None) -> None:
        self.base = params
        self.config = config or TwinConfig()
        self.reset()

    def reset(self, phi_deg: float = 0.0) -> None:
        self.time = 0.0
        self.state = (math.radians(phi_deg), 0.0, 0.0)
        self.u = 0.0
        self.x = self.base.x
        self.x_target = self.base.x
        self.params = self.base
        self._coeffs = coefficients(self.params)
        self._rng = np.random.default_rng(self.config.seed)

    def command(self, u: float, deltax_mm: float = 0.0) -> None:
        """Latch a new rotor command [%] and slider offset [mm] (held until replaced)."""
        self.u = float(u)
        self.x_target = self.base.geometry.slider_position(float(deltax_mm))

    def _move_slider(self, dt: float) -> None:
        if self.x == self.x_target:
            return
        step = self.config.slider_speed * dt
        delta = self.x_target - self.x
        self.x = self.x_target if abs(delta) <= step else self.x + math.copysign(step, delta)
        self.params = self.base.with_(x=self.x)
        self._coeffs = coefficients(self.params)

    def measure(self) -> TwinState:
        """Noisy sensor reading of the current state."""
        cfg = self.config
        phi = math.degrees(self.state[0]) + self._rng.normal(0.0, cfg.angle_noise_deg)
        phid = math.degrees(self.state[1]) + self._rng.normal(0.0, cfg.rate_noise_deg)
        return TwinState(self.time, phi, phid)

    def advance(self, dt: float | None = None) -> TwinState:
        """Integrate the physics over ``dt`` (default: one nominal sample) and measure."""
        dt = self.config.ts if dt is None else dt
        self._move_slider(dt)
        target = self.params.thrust_command(self.u)
        h = dt / self.config.substeps
        for _ in range(self.config.substeps):
            self.state = rk4_step(self._coeffs, self.state, target, h)
        self.time += dt
        return self.measure()
