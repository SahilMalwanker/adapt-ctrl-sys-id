"""DC servo with inertia disk: virtual plant, cascade position control, adaptive control.

The virtual servo is a *calibrated* plant, not a textbook PT1:

* its steady state follows the quadratic static map fitted to the hardware
  readings (gain falls from ~28 to ~24 rad/s/V towards full speed),
* its time constant is calibrated so that a PT1 identified on the standard
  excitation reproduces the hardware estimate ``T = 0.319 s``,
* the angle is read through a 2048 counts/rev encoder and the speed through a
  filtered finite difference, which creates realistic quantisation noise.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .identification import PT1, QuadraticStaticMap, RecursiveLeastSquares

__all__ = [
    "HARDWARE_PT1",
    "AdaptiveCascadeController",
    "CascadeController",
    "ServoParams",
    "ServoRun",
    "VirtualServo",
    "actuator_limited_pole",
    "cascade_gains",
    "position_loop_poles",
    "rule_of_thumb_pole",
    "run_open_loop",
    "run_position_control",
    "sine_test",
]

#: First-order model identified on the physical servo (step experiments).
HARDWARE_PT1 = PT1(K=25.172, T=0.31919)


@dataclass(frozen=True)
class ServoParams:
    """Parameters of the calibrated virtual servo (defaults: calibrated values)."""

    K0: float = 28.63
    """Small-signal gain at low speed [rad/s/V] (static-map fit)."""
    beta: float = 3.25e-5
    """Quadratic loss coefficient [V s^2/rad^2] (static-map fit)."""
    tau: float = 0.353
    """Mechanical time constant [s] (calibrated against the hardware PT1)."""
    u_max: float = 10.0
    """Amplifier limit [V]."""
    counts_per_rev: int = 2048
    """Encoder resolution (quadrature decoded)."""
    velocity_filter_tc: float = 0.004
    """Time constant of the speed-estimation low-pass [s]."""
    ts: float = 0.002
    """Sample time of the real-time loop [s]."""

    @property
    def static_map(self) -> QuadraticStaticMap:
        return QuadraticStaticMap(self.K0, self.beta)


class VirtualServo:
    """Discrete-time simulation of the servo including the sensor chain."""

    def __init__(self, params: ServoParams | None = None, inertia_scale: float = 1.0) -> None:
        self.params = params or ServoParams()
        self.inertia_scale = inertia_scale
        self._quantum = 2.0 * math.pi / self.params.counts_per_rev
        self.reset()

    def reset(self, theta: float = 0.0, omega: float = 0.0) -> None:
        self.theta = theta
        self.omega = omega
        self._theta_meas = self._quantize(theta)
        self._omega_meas = omega

    def _quantize(self, value: float) -> float:
        return round(value / self._quantum) * self._quantum

    def _accel(self, omega: float, u: float) -> float:
        p = self.params
        drive = p.K0 * (u - p.beta * omega * abs(omega))
        return (drive - omega) / (p.tau * self.inertia_scale)

    @property
    def measurement(self) -> tuple[float, float]:
        """Latest encoder angle [rad] and estimated speed [rad/s]."""
        return self._theta_meas, self._omega_meas

    def step(self, u: float) -> tuple[float, float]:
        """Hold ``u`` for one sample (amplifier-limited) and return the new measurement."""
        p = self.params
        h = p.ts
        u = min(p.u_max, max(-p.u_max, u))
        w = self.omega
        k1 = self._accel(w, u)
        k2 = self._accel(w + 0.5 * h * k1, u)
        k3 = self._accel(w + 0.5 * h * k2, u)
        k4 = self._accel(w + h * k3, u)
        self.theta += h / 6.0 * (w + 2.0 * (w + 0.5 * h * k1) + 2.0 * (w + 0.5 * h * k2) + (w + h * k3))
        self.omega = w + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        theta_meas = self._quantize(self.theta)
        raw_rate = (theta_meas - self._theta_meas) / h
        self._theta_meas = theta_meas
        alpha = h / (p.velocity_filter_tc + h)
        self._omega_meas += alpha * (raw_rate - self._omega_meas)
        return self.measurement


def run_open_loop(u: ArrayLike, params: ServoParams | None = None, inertia_scale: float = 1.0) -> dict[str, np.ndarray]:
    """Apply a sampled voltage sequence; sample ``k`` is measured *before* ``u[k]`` acts."""
    plant = VirtualServo(params, inertia_scale)
    u = np.asarray(u, dtype=float)
    theta = np.empty_like(u)
    omega = np.empty_like(u)
    omega_true = np.empty_like(u)
    for k, uk in enumerate(u):
        theta[k], omega[k] = plant.measurement
        omega_true[k] = plant.omega
        plant.step(uk)
    return {"theta": theta, "omega": omega, "omega_true": omega_true}


def sine_test(
    omega: float,
    amplitude: float = 1.0,
    params: ServoParams | None = None,
    periods: int = 10,
    settle_periods: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sine experiment on the virtual servo; returns ``t``, input and measured speed.

    The input is evaluated at mid-sample so that the zero-order hold adds no
    half-sample phase lag to the estimate.
    """
    p = params or ServoParams()
    t_end = (periods + settle_periods) * 2.0 * math.pi / omega
    t = np.arange(0.0, t_end, p.ts)
    u = amplitude * np.sin(omega * (t + 0.5 * p.ts))
    return t, u, run_open_loop(u, p)["omega"]


# --------------------------------------------------------------------------- design
def cascade_gains(K: float, T: float, s_vel: float) -> tuple[float, float]:
    """Gains of the P-velocity / P-position cascade for a PT1 speed plant.

    The inner loop ``Kd`` places the velocity pole at ``s_vel``; the outer loop
    ``Kp`` then gives the position loop a damping ratio of ``1/sqrt(2)``
    (about 4 % overshoot) with poles at ``s_vel/2 * (1 +/- j)``::

        Kd = (-s_vel T - 1) / K,     Kp = (1 + Kd K)^2 / (2 T Kd K)
    """
    if not s_vel < -1.0 / T:
        raise ValueError("velocity pole must be faster than the open-loop pole -1/T")
    kd = (-s_vel * T - 1.0) / K
    kp = (1.0 + kd * K) ** 2 / (2.0 * T * kd * K)
    return kd, kp


def actuator_limited_pole(K: float, T: float, u_max: float, step: float) -> float:
    """Fastest velocity pole for which a position step keeps ``|u(0+)| <= u_max``.

    Right after a step of height ``r0`` the cascade outputs
    ``u(0+) = Kd Kp r0 = r0 T s_vel^2 / (2 K)``; solving for ``s_vel`` gives
    ``s_vel = -sqrt(2 K u_max / (r0 T))``.
    """
    return -math.sqrt(2.0 * K * u_max / (step * T))


def rule_of_thumb_pole(T: float) -> float:
    """Integer linear approximation ``round(-44.3 + 43 T)`` of the actuator limit."""
    return float(round(-44.3 + 43.0 * T))


def position_loop_poles(K: float, T: float, kd: float, kp: float) -> np.ndarray:
    """Closed-loop poles of ``T s^2 + (1 + Kd K) s + Kp Kd K``."""
    return np.roots([T, 1.0 + kd * K, kp * kd * K])


# --------------------------------------------------------------------------- controllers
class CascadeController:
    """Fixed-gain cascade ``u = Kd (Kp (r - theta) - omega)`` with saturation."""

    def __init__(self, kd: float, kp: float, u_max: float = 10.0) -> None:
        self.kd = kd
        self.kp = kp
        self.u_max = u_max
        self.model: PT1 | None = None

    def __call__(self, ref: float, theta: float, omega: float) -> float:
        u = self.kd * (self.kp * (ref - theta) - omega)
        return min(self.u_max, max(-self.u_max, u))


class AdaptiveCascadeController(CascadeController):
    """Indirect adaptive cascade (certainty equivalence).

    Every ``decimation`` samples an RLS estimator updates a model of the
    *encoder increments* ``d[m] = theta[m] - theta[m-1]``. For a zero-order-held
    ``K / (s (T s + 1))`` they obey exactly::

        d[m] = a d[m-1] + b1 u[m-1] + b2 u[m-2],     a = exp(-h/T)
        b1 + b2 = K h (1 - a)

    (``u`` = mean voltage over a window ``h``). Using increments instead of a
    differentiated, filtered speed keeps encoder quantisation noise from
    biasing the estimate; dividing the increments by ``h`` gives all
    regressors a similar scale. The estimate is mapped to ``(K, T)``, projected
    into plausible bounds and pushed through the actuator-aware design rule,
    so the loop keeps ``zeta = 0.707`` and respects the amplifier limit when
    the inertia changes.
    """

    def __init__(
        self,
        initial: PT1,
        ts: float,
        u_max: float = 10.0,
        design_step: float = math.pi / 2.0,
        margin: float = 0.95,
        forgetting: float = 0.98,
        decimation: int = 10,
        p0: float = 1e-2,
        k_bounds: tuple[float, float] = (5.0, 80.0),
        t_bounds: tuple[float, float] = (0.05, 3.0),
    ) -> None:
        self.h = ts * decimation
        self.decimation = decimation
        self.design_step = design_step
        self.margin = margin
        self.k_bounds = k_bounds
        self.t_bounds = t_bounds
        self.rls = RecursiveLeastSquares(
            3, forgetting=forgetting, p0=p0, theta0=self.increment_model(initial, self.h), max_trace=1.0
        )
        self._k = 0
        self._u_sum = 0.0
        self._theta_prev: float | None = None
        self._rate_prev: float | None = None
        self._u_prev: float | None = None
        super().__init__(0.0, 0.0, u_max)
        self._set_model(initial)

    @staticmethod
    def increment_model(model: PT1, h: float) -> list[float]:
        """Parameters ``[a, b1/h, b2/h]`` of the scaled increment model."""
        a = math.exp(-h / model.T)
        lag = model.T * (1.0 - a)
        return [a, model.K * (h - lag) / h, model.K * (lag - a * h) / h]

    def _set_model(self, model: PT1) -> None:
        self.model = model
        s_vel = self.margin * actuator_limited_pole(model.K, model.T, self.u_max, self.design_step)
        self.kd, self.kp = cascade_gains(model.K, model.T, s_vel)

    def _identify(self) -> None:
        a, c1, c2 = self.rls.theta
        if not (0.0 < a < 1.0 and c1 + c2 > 0.0):
            return  # keep the last valid model (parameter projection)
        K = min(max((c1 + c2) / (1.0 - a), self.k_bounds[0]), self.k_bounds[1])
        T = min(max(-self.h / math.log(a), self.t_bounds[0]), self.t_bounds[1])
        self._set_model(PT1(K, T))

    def __call__(self, ref: float, theta: float, omega: float) -> float:
        if self._k % self.decimation == 0:
            if self._theta_prev is not None:
                rate = (theta - self._theta_prev) / self.h
                u_mean = self._u_sum / self.decimation
                if self._rate_prev is not None and self._u_prev is not None:
                    self.rls.update([self._rate_prev, u_mean, self._u_prev], rate)
                    self._identify()
                self._rate_prev, self._u_prev = rate, u_mean
            self._theta_prev = theta
            self._u_sum = 0.0
        u = super().__call__(ref, theta, omega)
        self._u_sum += u
        self._k += 1
        return u


# --------------------------------------------------------------------------- closed loop
@dataclass
class ServoRun:
    """Logged signals of a position-control run."""

    t: np.ndarray
    ref: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    u: np.ndarray
    kd: np.ndarray
    kp: np.ndarray
    K_hat: np.ndarray
    T_hat: np.ndarray
    inertia_scale: np.ndarray


def run_position_control(
    controller: CascadeController,
    reference: Callable[[np.ndarray], np.ndarray],
    t_end: float,
    params: ServoParams | None = None,
    inertia_schedule: Callable[[float], float] | None = None,
) -> ServoRun:
    """Closed-loop simulation; ``inertia_schedule(t)`` scales the disk inertia over time."""
    p = params or ServoParams()
    plant = VirtualServo(p)
    t = np.arange(0.0, t_end + 0.5 * p.ts, p.ts)
    ref = np.asarray(reference(t), dtype=float)
    n = len(t)
    log = {name: np.empty(n) for name in ("theta", "omega", "u", "kd", "kp", "K", "T", "J")}
    for k in range(n):
        if inertia_schedule is not None:
            plant.inertia_scale = inertia_schedule(t[k])
        theta_m, omega_m = plant.measurement
        u = controller(ref[k], theta_m, omega_m)
        model = controller.model
        log["theta"][k] = theta_m
        log["omega"][k] = omega_m
        log["u"][k] = u
        log["kd"][k] = controller.kd
        log["kp"][k] = controller.kp
        log["K"][k] = model.K if model else np.nan
        log["T"][k] = model.T if model else np.nan
        log["J"][k] = plant.inertia_scale
        plant.step(u)
    return ServoRun(t, ref, log["theta"], log["omega"], log["u"], log["kd"], log["kp"], log["K"], log["T"], log["J"])
