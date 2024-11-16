"""Propeller pendulum (beam + rotor + slider): physics, estimation and linearisation.

Equation of motion about the pivot (angles in rad, thrust command ``u`` in %)::

    J phi'' = b0 w - a0 sin(phi) - (d + d_u w) phi' - c phi' |phi'|
    tau_r w' = sat(u) - w                     (rotor spin-up, optional)

with ``a0 = m g l(x)``, ``b0 = k l_r`` and the slider position ``x`` changing
both the centre of mass ``l(x)`` and the inertia ``J(x)``. The lab-level model
is the special case ``d_u = c = tau_r = 0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import odeint
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from .metrics import nrmse_fit

__all__ = [
    "CALIBRATED",
    "LAB_MODEL",
    "FreeFallFit",
    "Geometry",
    "LinearPendulum",
    "PendulumParams",
    "coefficients",
    "com_from_two_positions",
    "decay_rate",
    "detect_release",
    "energy_inertia_bound",
    "fit_free_fall",
    "linearization_error",
    "rk4_step",
    "simulate",
    "simulate_free_fall",
    "simulate_linear",
    "step_comparison",
    "thrust_from_angle",
]


@dataclass(frozen=True)
class Geometry:
    """Masses [kg] and lengths [m] of beam, rotor and slider (measured on the rig)."""

    m_beam: float = 0.300
    m_rotor: float = 0.087
    m_slider: float = 0.430
    l_beam: float = 0.473
    l_rotor: float = 0.500
    g: float = 9.81
    x_up: float = 0.09
    x_down: float = 0.34

    @property
    def mass(self) -> float:
        return self.m_beam + self.m_rotor + self.m_slider

    def first_moment(self, x: float) -> float:
        """``m * l(x)``: beam as a rod, rotor and slider as point masses."""
        return x * self.m_slider + 0.5 * self.l_beam * self.m_beam + self.l_rotor * self.m_rotor

    def com(self, x: float) -> float:
        """Distance pivot -> centre of gravity ``l(x)``."""
        return self.first_moment(x) / self.mass

    def inertia(self, x: float) -> float:
        """Moment of inertia about the pivot ``J(x)`` (Steiner for the rod end)."""
        return self.m_beam * self.l_beam**2 / 3.0 + self.m_rotor * self.l_rotor**2 + self.m_slider * x**2

    def slider_position(self, deltax_mm: float) -> float:
        """Slider command in mm (0 = top) to the position ``x`` in m."""
        travel = 1000.0 * (self.x_down - self.x_up)
        return self.x_up + min(max(deltax_mm, 0.0), travel) / 1000.0


@dataclass(frozen=True)
class LinearPendulum:
    """Linearisation ``a2 dphi'' + a1 dphi' + a0 dphi = b du`` around ``phi_op``."""

    a2: float
    a1: float
    a0: float
    b: float
    phi_op: float
    u_op: float

    @property
    def natural_frequency(self) -> float:
        return math.sqrt(self.a0 / self.a2)

    @property
    def damping_ratio(self) -> float:
        return self.a1 / (2.0 * math.sqrt(self.a0 * self.a2))

    def in_degrees(self) -> LinearPendulum:
        """Same model with the angle expressed in degrees (input still in %)."""
        return replace(self, b=self.b * 180.0 / math.pi)


@dataclass(frozen=True)
class PendulumParams:
    """Model parameters; defaults are the *lab-level* (geometry-only) model."""

    x: float = 0.09
    viscous: float = 0.0045
    drag: float = 0.0
    thrust_damping: float = 0.0
    thrust_gain: float = 0.05
    inertia_offset: float = 0.0
    rotor_tau: float = 0.0
    deadband: float = 0.0
    u_max: float = 100.0
    geometry: Geometry = field(default_factory=Geometry)

    # lab notation ------------------------------------------------------------
    @property
    def J(self) -> float:
        return self.geometry.inertia(self.x) + self.inertia_offset

    @property
    def a2(self) -> float:
        return self.J

    @property
    def a1(self) -> float:
        return self.viscous

    @property
    def a0(self) -> float:
        return self.geometry.g * self.geometry.first_moment(self.x)

    @property
    def b0(self) -> float:
        return self.thrust_gain * self.geometry.l_rotor

    def with_(self, **changes: float) -> PendulumParams:
        return replace(self, **changes)

    # physics -----------------------------------------------------------------
    def thrust_command(self, u: float) -> float:
        """Unidirectional, saturated rotor command with a start-up dead band."""
        u = min(max(u, 0.0), self.u_max)
        return 0.0 if u < self.deadband else u

    def equilibrium_input(self, phi: float) -> float:
        return self.a0 * math.sin(phi) / self.b0

    def equilibrium_angle(self, u: float) -> float:
        ratio = self.b0 * self.thrust_command(u) / self.a0
        return math.asin(min(max(ratio, -1.0), 1.0))

    def linearize(self, phi_op: float) -> LinearPendulum:
        u_op = self.equilibrium_input(phi_op)
        return LinearPendulum(
            a2=self.J,
            a1=self.viscous + self.thrust_damping * u_op,
            a0=self.a0 * math.cos(phi_op),
            b=self.b0,
            phi_op=phi_op,
            u_op=u_op,
        )


#: Lab-level model: inertia from geometry, hand-tuned damping, round thrust gain.
LAB_MODEL = PendulumParams()

#: Model calibrated on the hardware records (free fall, static equilibria and
#: step response); regenerate the numbers with ``adapt-ctrl-sys-id reproduce``.
CALIBRATED = PendulumParams(
    viscous=0.00211,
    drag=5.1e-4,
    thrust_damping=7.6e-4,
    thrust_gain=0.0526,
    inertia_offset=0.00611,
    rotor_tau=0.044,
    deadband=6.0,
)


def _accel(c: tuple[float, ...], phi: float, phid: float, w: float) -> float:
    J, a0, b0, d, d_u, drag = c[:6]
    return (b0 * w - a0 * math.sin(phi) - (d + d_u * w) * phid - drag * phid * abs(phid)) / J


def coefficients(p: PendulumParams) -> tuple[float, ...]:
    """Constants used by the integrator: ``(J, a0, b0, d, d_u, c, 1/tau_r)``."""
    inv_tau = 1.0 / p.rotor_tau if p.rotor_tau > 0.0 else 0.0
    return (p.J, p.a0, p.b0, p.viscous, p.thrust_damping, p.drag, inv_tau)


def rk4_step(
    c: tuple[float, ...], state: tuple[float, float, float], target: float, h: float
) -> tuple[float, float, float]:
    """One RK4 step of ``(phi, phid, w)`` towards the rotor command ``target`` [%].

    ``c`` comes from :func:`coefficients`; without rotor lag (``1/tau_r = 0``)
    the thrust state jumps straight to ``target``.
    """
    phi, phid, w = state
    inv_tau = c[6]
    if inv_tau == 0.0:
        w = target
    k1p, k1v, k1w = phid, _accel(c, phi, phid, w), (target - w) * inv_tau
    p2, v2, w2 = phi + 0.5 * h * k1p, phid + 0.5 * h * k1v, w + 0.5 * h * k1w
    k2p, k2v, k2w = v2, _accel(c, p2, v2, w2), (target - w2) * inv_tau
    p3, v3, w3 = phi + 0.5 * h * k2p, phid + 0.5 * h * k2v, w + 0.5 * h * k2w
    k3p, k3v, k3w = v3, _accel(c, p3, v3, w3), (target - w3) * inv_tau
    p4, v4, w4 = phi + h * k3p, phid + h * k3v, w + h * k3w
    k4p, k4v, k4w = v4, _accel(c, p4, v4, w4), (target - w4) * inv_tau
    return (
        phi + h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p),
        phid + h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
        w + h / 6.0 * (k1w + 2.0 * k2w + 2.0 * k3w + k4w),
    )


def simulate(
    p: PendulumParams,
    t: ArrayLike,
    u: ArrayLike | float,
    phi0: float = 0.0,
    phid0: float = 0.0,
    substeps: int = 1,
) -> dict[str, np.ndarray]:
    """Fixed-step simulation on the grid ``t`` with ``u[k]`` held on ``[t_k, t_k+1)``.

    The rotor starts in steady state for ``u[0]``. Angles are in rad.
    """
    t = np.asarray(t, dtype=float)
    u = np.broadcast_to(np.asarray(u, dtype=float), t.shape)
    c = coefficients(p)
    phi = np.empty_like(t)
    phid = np.empty_like(t)
    thrust = np.empty_like(t)
    state = (phi0, phid0, p.thrust_command(float(u[0])))
    for k in range(len(t)):
        phi[k], phid[k], thrust[k] = state
        if k + 1 < len(t):
            h = (t[k + 1] - t[k]) / substeps
            target = p.thrust_command(float(u[k]))
            for _ in range(substeps):
                state = rk4_step(c, state, target, h)
    return {"phi": phi, "phid": phid, "thrust": thrust}


def simulate_linear(lin: LinearPendulum, t: ArrayLike, du: ArrayLike | float, dphi0: float = 0.0) -> np.ndarray:
    """Deviation response of the linearised model (RK4, input held per sample)."""
    t = np.asarray(t, dtype=float)
    du = np.broadcast_to(np.asarray(du, dtype=float), t.shape)
    out = np.empty_like(t)
    x, v = dphi0, 0.0

    def acc(x_: float, v_: float, u_: float) -> float:
        return (lin.b * u_ - lin.a1 * v_ - lin.a0 * x_) / lin.a2

    for k in range(len(t)):
        out[k] = x
        if k + 1 < len(t):
            h, uk = t[k + 1] - t[k], float(du[k])
            k1x, k1v = v, acc(x, v, uk)
            k2x, k2v = v + 0.5 * h * k1v, acc(x + 0.5 * h * k1x, v + 0.5 * h * k1v, uk)
            k3x, k3v = v + 0.5 * h * k2v, acc(x + 0.5 * h * k2x, v + 0.5 * h * k2v, uk)
            k4x, k4v = v + h * k3v, acc(x + h * k3x, v + h * k3v, uk)
            x += h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            v += h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return out


# --------------------------------------------------------------------------- free fall
def detect_release(t: ArrayLike, phi: ArrayLike, threshold: float = math.radians(1.0)) -> tuple[float, float]:
    """Return ``(t_release, phi0)`` of a pendulum that is held and then dropped."""
    t = np.asarray(t, dtype=float)
    phi = np.asarray(phi, dtype=float)
    phi0 = float(np.median(phi[: max(10, len(phi) // 200)]))
    moved = np.flatnonzero(np.abs(phi - phi0) > threshold)
    if moved.size == 0:
        raise ValueError("no release detected")
    return float(t[moved[0]]), phi0


def energy_inertia_bound(geometry: Geometry, x: float, phid_max: float, phi0: float = math.pi / 2.0) -> float:
    """Upper bound on ``J`` from energy conservation, ``J <= 2 m g l (1 - cos phi0) / phid_max^2``.

    Friction removes energy before the pendulum passes the bottom, so the
    measured peak speed is slightly too small and the bound is conservative.
    """
    return 2.0 * geometry.g * geometry.first_moment(x) * (1.0 - math.cos(phi0)) / phid_max**2


def decay_rate(
    t: ArrayLike, phi: ArrayLike, min_amplitude: float = math.radians(3.0)
) -> tuple[float, np.ndarray, np.ndarray]:
    """Exponential decay rate ``sigma`` of the positive peaks (log-linear fit)."""
    t = np.asarray(t, dtype=float)
    phi = np.asarray(phi, dtype=float)
    dt = float(np.median(np.diff(t)))
    peaks, _ = find_peaks(phi, distance=max(1, int(0.5 / dt)), prominence=math.radians(1.0))
    tp, ap = t[peaks], phi[peaks]
    keep = ap > min_amplitude
    tp, ap = tp[keep], ap[keep]
    if len(tp) < 3:
        raise ValueError("not enough oscillation peaks to estimate the decay")
    slope, _ = np.polyfit(tp, np.log(ap), 1)
    return float(-slope), tp, ap


def _free_fall_rhs(state: np.ndarray, _t: float, a0: float, J: float, d: float, c: float) -> list[float]:
    phi, phid = state
    return [phid, (-a0 * math.sin(phi) - d * phid - c * phid * abs(phid)) / J]


def simulate_free_fall(p: PendulumParams, t: ArrayLike, phi0: float, t_release: float) -> np.ndarray:
    """Angle of the unpowered pendulum held at ``phi0`` until ``t_release``."""
    t = np.asarray(t, dtype=float)
    out = np.full_like(t, phi0)
    after = t >= t_release
    if after.any():
        grid = np.r_[t_release, t[after]]
        sol = odeint(_free_fall_rhs, [phi0, 0.0], grid, args=(p.a0, p.J, p.viscous, p.drag), rtol=1e-8, atol=1e-10)
        out[after] = sol[1:, 0]
    return out


@dataclass(frozen=True)
class FreeFallFit:
    """Result of fitting the free-fall record."""

    params: PendulumParams
    t_release: float
    phi0: float
    fit_percent: float


def fit_free_fall(t: ArrayLike, phi: ArrayLike, base: PendulumParams, estimate_drag: bool = True) -> FreeFallFit:
    """Estimate ``J``, viscous damping ``d`` and (optionally) quadratic drag ``c``.

    ``a0 = m g l(x)`` is kept from geometry: the free fall only constrains the
    ratios ``a0/J`` and ``d/J``, so one absolute quantity has to be fixed.
    """
    t = np.asarray(t, dtype=float)
    phi = np.asarray(phi, dtype=float)
    t_rel0, phi0 = detect_release(t, phi)
    j_geom = base.geometry.inertia(base.x)

    def unpack(z: np.ndarray) -> tuple[PendulumParams, float]:
        drag = z[3] if estimate_drag else 0.0
        params = base.with_(inertia_offset=z[0] - j_geom, viscous=z[1], drag=drag)
        return params, z[2]

    def residual(z: np.ndarray) -> np.ndarray:
        params, t_rel = unpack(z)
        return simulate_free_fall(params, t, phi0, t_rel) - phi

    x0 = [j_geom, 0.004, t_rel0 - 0.05] + ([5e-4] if estimate_drag else [])
    lower = [0.5 * j_geom, 0.0, t_rel0 - 0.5] + ([0.0] if estimate_drag else [])
    upper = [2.0 * j_geom, 0.05, t_rel0 + 0.2] + ([0.05] if estimate_drag else [])
    scale = [0.01, 0.001, 0.1] + ([1e-4] if estimate_drag else [])
    sol = least_squares(residual, x0=x0, bounds=(lower, upper), x_scale=scale)
    params, t_rel = unpack(sol.x)
    fit = nrmse_fit(phi, simulate_free_fall(params, t, phi0, t_rel))
    return FreeFallFit(params=params, t_release=float(t_rel), phi0=phi0, fit_percent=fit)


# --------------------------------------------------------------------------- statics
def thrust_from_angle(phi: ArrayLike, x: float, geometry: Geometry) -> np.ndarray:
    """Rotor force from the equilibrium ``F l_r = m g l(x) sin(phi)``."""
    return np.sin(np.asarray(phi, dtype=float)) * geometry.g * geometry.first_moment(x) / geometry.l_rotor


def com_from_two_positions(phi: float, phi_prime: float, x: float, x_prime: float, geometry: Geometry) -> float:
    """Centre of gravity ``l(x)`` from two equilibria with the *same* thrust.

    Moving the slider from ``x`` to ``x'`` shifts the centre of gravity by
    ``(m_s/m)(x' - x)``; equating the gravity torques of both equilibria gives
    ``l(x) = (m_s/m) (x - x') / (1 - sin(phi)/sin(phi'))``.
    """
    ratio = math.sin(phi) / math.sin(phi_prime)
    return geometry.m_slider / geometry.mass * (x - x_prime) / (1.0 - ratio)


# --------------------------------------------------------------------------- linearisation
def step_comparison(
    p: PendulumParams, phi_op: float, half_step: float, t_end: float = 15.0, dt: float = 0.01
) -> dict[str, np.ndarray]:
    """Nonlinear and linearised responses [rad] to the step ``phi_op - half_step -> phi_op + half_step``.

    Both models receive the *same* physical input (the nonlinear equilibrium
    inputs before/after the step), so static and dynamic errors both show up.
    """
    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    phi_a, phi_b = phi_op - half_step, phi_op + half_step
    u_a, u_b = p.equilibrium_input(phi_a), p.equilibrium_input(phi_b)
    u = np.where(t < dt, u_a, u_b)
    nonlinear = simulate(p, t, u, phi0=phi_a)["phi"]
    lin = p.linearize(phi_op)
    dphi_start = lin.b * (u_a - lin.u_op) / lin.a0
    linear = phi_op + simulate_linear(lin, t, u - lin.u_op, dphi0=dphi_start)
    return {"t": t, "nonlinear": nonlinear, "linear": linear}


def linearization_error(
    p: PendulumParams, phi_op: float, half_step: float, t_end: float = 15.0, dt: float = 0.01
) -> float:
    """Worst-case angle error [rad] of the linear model, see :func:`step_comparison`."""
    run = step_comparison(p, phi_op, half_step, t_end, dt)
    return float(np.max(np.abs(run["nonlinear"] - run["linear"])))
