"""Controller design and digital implementation for the propeller pendulum.

The pendulum is controlled with a two-degree-of-freedom PI-D law (derivative
acting on the measurement only)::

    u = Kp e + Ki/s e - Kd s y,        e = r - y

which is realised as a *cascade*: an outer PI loop produces a rate set-point
and an inner P loop on the measured rate drives the rotor::

    u_PI = (Kp/Kd) e + (Ki/Kd)/s e,    u = Kd (u_PI - s y)

Backward-Euler discretisation gives the incremental difference equations used
on the real-time target::

    u_PI[k] = u_PI[k-1] + q0_PI e[k] + q1_PI e[k-1]
    u[k]    = q0_P (u_PI[k] - ydot[k])

    q0_PI = Kp/Kd + Ki Ts/Kd,   q1_PI = -Kp/Kd,   q0_P = Kd
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .metrics import tracking_costs
from .pendulum import LinearPendulum, PendulumParams, coefficients, rk4_step
from .signals import pendulum_reference_deg

__all__ = [
    "DigitalPIP",
    "DiscretePID",
    "LoopResult",
    "PIDGains",
    "PIPCoefficients",
    "closed_loop_poles",
    "gravity_feedforward",
    "place_pid",
    "simulate_loop",
]

Controller = Callable[[float, float, float], float]


@dataclass(frozen=True)
class PIDGains:
    """Parallel PID gains (units follow the plant: % per deg, per deg*s, per deg/s)."""

    kp: float
    ki: float
    kd: float


def place_pid(lin: LinearPendulum, omega: float, zeta: float = 0.9, alpha: float = 1.0) -> PIDGains:
    """Pole placement for ``b / (a2 s^2 + a1 s + a0)`` under PI-D feedback.

    The closed-loop characteristic polynomial
    ``a2 s^3 + (a1 + b Kd) s^2 + (a0 + b Kp) s + b Ki`` is matched to
    ``a2 (s + alpha omega)(s^2 + 2 zeta omega s + omega^2)``.
    """
    a2, a1, a0, b = lin.a2, lin.a1, lin.a0, lin.b
    kd = (a2 * (2.0 * zeta + alpha) * omega - a1) / b
    kp = (a2 * omega**2 * (1.0 + 2.0 * zeta * alpha) - a0) / b
    ki = a2 * alpha * omega**3 / b
    return PIDGains(kp=kp, ki=ki, kd=kd)


def closed_loop_poles(lin: LinearPendulum, gains: PIDGains) -> np.ndarray:
    """Roots of the closed-loop characteristic polynomial for PI-D feedback."""
    return np.roots([lin.a2, lin.a1 + lin.b * gains.kd, lin.a0 + lin.b * gains.kp, lin.b * gains.ki])


@dataclass(frozen=True)
class PIPCoefficients:
    """Difference-equation coefficients of the digital PI-P cascade."""

    q0_pi: float
    q1_pi: float
    q0_p: float
    ts: float

    @classmethod
    def from_pid(cls, gains: PIDGains, ts: float) -> PIPCoefficients:
        if gains.kd <= 0.0:
            raise ValueError("the inner P loop needs Kd > 0")
        return cls(
            q0_pi=gains.kp / gains.kd + gains.ki * ts / gains.kd,
            q1_pi=-gains.kp / gains.kd,
            q0_p=gains.kd,
            ts=ts,
        )

    def to_pid(self) -> PIDGains:
        return PIDGains(
            kp=-self.q1_pi * self.q0_p,
            ki=(self.q0_pi + self.q1_pi) * self.q0_p / self.ts,
            kd=self.q0_p,
        )


class DigitalPIP:
    """Digital PI-P cascade with optional gravity feedforward and anti-windup.

    Anti-windup uses *conditioning*: when the output saturates, the PI state is
    reset to the value that would have produced the saturated output, so the
    integrator never charges beyond what the actuator can deliver.
    """

    def __init__(
        self,
        coeffs: PIPCoefficients,
        u_min: float = 0.0,
        u_max: float = 100.0,
        feedforward: Callable[[float], float] | None = None,
        anti_windup: bool = False,
    ) -> None:
        self.c = coeffs
        self.u_min = u_min
        self.u_max = u_max
        self.feedforward = feedforward
        self.anti_windup = anti_windup
        self.reset()

    def reset(self) -> None:
        self.u_pi = 0.0
        self.e_prev = 0.0

    def __call__(self, ref: float, y: float, ydot: float) -> float:
        c = self.c
        e = ref - y
        u_pi = self.u_pi + c.q0_pi * e + c.q1_pi * self.e_prev
        u_ff = self.feedforward(ref) if self.feedforward else 0.0
        u = c.q0_p * (u_pi - ydot) + u_ff
        u_sat = min(self.u_max, max(self.u_min, u))
        if self.anti_windup and u_sat != u:
            u_pi = (u_sat - u_ff) / c.q0_p + ydot
        self.u_pi = u_pi
        self.e_prev = e
        return u_sat


class DiscretePID:
    """One-degree-of-freedom parallel PID (derivative on the error, filtered).

    Kept for comparison: on reference steps the derivative of the error spikes
    ("derivative kick"), which the 2-DOF PI-P cascade avoids by construction.
    """

    def __init__(
        self,
        gains: PIDGains,
        ts: float,
        filter_n: float = 100.0,
        u_min: float = 0.0,
        u_max: float = 100.0,
    ) -> None:
        self.g = gains
        self.ts = ts
        self.tf = 1.0 / filter_n
        self.u_min = u_min
        self.u_max = u_max
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.deriv = 0.0
        self.e_prev: float | None = None

    def __call__(self, ref: float, y: float, ydot: float) -> float:
        e = ref - y
        e_prev = e if self.e_prev is None else self.e_prev
        self.integral += self.g.ki * self.ts * e
        self.deriv = (self.tf * self.deriv + self.g.kd * (e - e_prev)) / (self.tf + self.ts)
        self.e_prev = e
        u = self.g.kp * e + self.integral + self.deriv
        return min(self.u_max, max(self.u_min, u))


def gravity_feedforward(params: PendulumParams) -> Callable[[float], float]:
    """Static inverse ``u_ff(r) = a0 sin(r) / b0`` (reference in degrees)."""

    def feedforward(ref_deg: float) -> float:
        return max(0.0, params.equilibrium_input(math.radians(ref_deg)))

    return feedforward


@dataclass
class LoopResult:
    """Closed-loop record; angles in degrees, rotor command in %."""

    t: np.ndarray
    ref: np.ndarray
    phi: np.ndarray
    phid: np.ndarray
    u: np.ndarray

    def costs(self, window: tuple[float, float] = (30.0, 100.0)) -> dict[str, float]:
        return tracking_costs(self.t, self.ref, self.phi, self.u, window)


def simulate_loop(
    params: PendulumParams,
    controller: Controller,
    t_end: float = 120.0,
    ts: float = 0.01,
    substeps: int = 5,
    reference: Callable[[np.ndarray], np.ndarray] = pendulum_reference_deg,
    phi0_deg: float = 0.0,
    noise_deg: float = 0.0,
    noise_rate_deg: float = 0.0,
    seed: int | None = 0,
) -> LoopResult:
    """Sampled-data simulation: ZOH controller at ``ts``, RK4 plant between samples."""
    n = round(t_end / ts) + 1
    t = np.arange(n) * ts
    ref = np.asarray(reference(t), dtype=float)
    rng = np.random.default_rng(seed)
    noise_phi = rng.normal(0.0, noise_deg, n) if noise_deg > 0 else np.zeros(n)
    noise_phid = rng.normal(0.0, noise_rate_deg, n) if noise_rate_deg > 0 else np.zeros(n)
    c = coefficients(params)
    h = ts / substeps
    phi = np.empty(n)
    phid = np.empty(n)
    u_log = np.empty(n)
    state = (math.radians(phi0_deg), 0.0, 0.0)
    for k in range(n):
        phi[k] = math.degrees(state[0])
        phid[k] = math.degrees(state[1])
        u = controller(float(ref[k]), phi[k] + noise_phi[k], phid[k] + noise_phid[k])
        u_log[k] = u
        target = params.thrust_command(u)
        for _ in range(substeps):
            state = rk4_step(c, state, target, h)
    return LoopResult(t, ref, phi, phid, u_log)
