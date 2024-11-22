"""System identification: least squares, recursive least squares and model conversions.

Sign conventions follow the usual ARX notation::

    A(q) y[k] = B(q) u[k]
    A(q) = 1 + a1 q^-1 + ... + a_na q^-na
    B(q) = q^-nk (b1 + b2 q^-1 + ... + b_nb q^-(nb-1))

so a first-order lag sampled with a zero-order hold reads
``y[k] = -a1 y[k-1] + b1 u[k-1]`` with ``a1 = -exp(-Ts/T)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares
from scipy.signal import lfilter

__all__ = [
    "PT1",
    "ArxModel",
    "QuadraticStaticMap",
    "RecursiveLeastSquares",
    "SecondOrderFit",
    "arx_ls",
    "arx_regressors",
    "continuous_second_order",
    "fit_gain_through_origin",
    "fit_pt1_output_error",
    "fit_quadratic_static_map",
    "physical_from_second_order",
    "rls_identify",
]


# --------------------------------------------------------------------------- ARX
@dataclass(frozen=True)
class ArxModel:
    """Discrete-time ARX model ``A(q) y = B(q) u`` (see module docstring)."""

    a: tuple[float, ...]
    b: tuple[float, ...]
    nk: int = 1

    @property
    def na(self) -> int:
        return len(self.a)

    @property
    def nb(self) -> int:
        return len(self.b)

    def simulate(self, u: ArrayLike) -> np.ndarray:
        """Free-run simulation from zero initial conditions."""
        num = np.r_[np.zeros(self.nk), self.b]
        den = np.r_[1.0, self.a]
        return lfilter(num, den, np.asarray(u, dtype=float))

    def dc_gain(self) -> float:
        return float(sum(self.b) / (1.0 + sum(self.a)))

    def poles(self) -> np.ndarray:
        return np.roots(np.r_[1.0, self.a])


def arx_regressors(u: ArrayLike, y: ArrayLike, na: int, nb: int, nk: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Regression matrix ``Phi`` and target ``Y`` so that ``Y ~ Phi @ [a, b]``."""
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    if u.shape != y.shape or u.ndim != 1:
        raise ValueError("u and y must be 1-D arrays of equal length")
    if na < 0 or nb < 1 or nk < 0:
        raise ValueError("require na >= 0, nb >= 1, nk >= 0")
    start = max(na, nb + nk - 1)
    n = len(y)
    if n <= start:
        raise ValueError("not enough samples for the requested model orders")
    phi = np.empty((n - start, na + nb))
    for i in range(1, na + 1):
        phi[:, i - 1] = -y[start - i : n - i]
    for j in range(nb):
        delay = nk + j
        phi[:, na + j] = u[start - delay : n - delay]
    return phi, y[start:]


def arx_ls(u: ArrayLike, y: ArrayLike, na: int = 1, nb: int = 1, nk: int = 1) -> ArxModel:
    """Batch least-squares (equation-error) ARX estimate."""
    phi, target = arx_regressors(u, y, na, nb, nk)
    theta, *_ = np.linalg.lstsq(phi, target, rcond=None)
    return ArxModel(tuple(float(v) for v in theta[:na]), tuple(float(v) for v in theta[na:]), nk)


# --------------------------------------------------------------------------- PT1
@dataclass(frozen=True)
class PT1:
    """First-order lag ``G(s) = K / (T s + 1)``."""

    K: float
    T: float

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError("time constant must be positive")

    def to_arx(self, ts: float) -> ArxModel:
        """Exact zero-order-hold discretisation."""
        pole = math.exp(-ts / self.T)
        return ArxModel((-pole,), (self.K * (1.0 - pole),), nk=1)

    @classmethod
    def from_arx(cls, model: ArxModel, ts: float) -> PT1:
        """Invert :meth:`to_arx`; the discrete pole must lie in ``(0, 1)``."""
        if model.na != 1 or model.nb != 1 or model.nk != 1:
            raise ValueError("expected an ARX(1,1,1) model")
        a1, b1 = model.a[0], model.b[0]
        if not -1.0 < a1 < 0.0:
            raise ValueError(f"discrete pole {-a1:.6f} has no first-order lag equivalent")
        return cls(K=b1 / (1.0 + a1), T=-ts / math.log(-a1))

    def simulate(self, u: ArrayLike, ts: float) -> np.ndarray:
        return self.to_arx(ts).simulate(u)

    def frequency_response(self, omega: ArrayLike) -> np.ndarray:
        return self.K / (1.0 + 1j * np.asarray(omega, dtype=float) * self.T)

    @property
    def cutoff(self) -> float:
        """-3 dB corner frequency in rad/s."""
        return 1.0 / self.T


def fit_pt1_output_error(u: ArrayLike, y: ArrayLike, ts: float, initial: PT1 | None = None) -> PT1:
    """Output-error fit: minimise the simulation error ``||y - G u||``.

    Unlike the equation-error ARX estimate this criterion is not biased by
    coloured measurement noise, at the price of a nonlinear optimisation.
    """
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    if initial is None:
        initial = PT1.from_arx(arx_ls(u, y), ts)

    def residual(p: np.ndarray) -> np.ndarray:
        return PT1(p[0], math.exp(p[1])).simulate(u, ts) - y

    sol = least_squares(residual, x0=[initial.K, math.log(initial.T)], method="lm")
    return PT1(float(sol.x[0]), float(math.exp(sol.x[1])))


# --------------------------------------------------------------------------- RLS
class RecursiveLeastSquares:
    """Recursive least squares with exponential forgetting.

    Parameters
    ----------
    n_params:
        Number of parameters.
    forgetting:
        Forgetting factor ``lambda`` in ``(0, 1]``; the effective memory is
        about ``1 / (1 - lambda)`` samples.
    p0:
        Initial covariance ``P0 = p0 * I`` (large = little trust in ``theta0``).
    theta0:
        Initial estimate (defaults to zeros).
    max_trace:
        Optional upper bound on ``trace(P)``; prevents covariance wind-up when
        the excitation fades while forgetting is active.
    """

    def __init__(
        self,
        n_params: int,
        forgetting: float = 1.0,
        p0: float = 1e4,
        theta0: ArrayLike | None = None,
        max_trace: float | None = None,
    ) -> None:
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting factor must lie in (0, 1]")
        self.forgetting = float(forgetting)
        self.max_trace = max_trace
        self.theta = np.zeros(n_params) if theta0 is None else np.array(theta0, dtype=float)
        if self.theta.shape != (n_params,):
            raise ValueError("theta0 has the wrong shape")
        self.P = np.eye(n_params) * float(p0)

    def update(self, phi: ArrayLike, y: float) -> float:
        """Process one sample; returns the a-priori prediction error."""
        phi = np.asarray(phi, dtype=float)
        p_phi = self.P @ phi
        gain = p_phi / (self.forgetting + phi @ p_phi)
        error = float(y - phi @ self.theta)
        self.theta = self.theta + gain * error
        self.P = (self.P - np.outer(gain, p_phi)) / self.forgetting
        self.P = 0.5 * (self.P + self.P.T)
        if self.max_trace is not None:
            trace = float(np.trace(self.P))
            if trace > self.max_trace:
                self.P *= self.max_trace / trace
        return error


def rls_identify(
    u: ArrayLike,
    y: ArrayLike,
    na: int = 1,
    nb: int = 1,
    nk: int = 1,
    forgetting: float = 1.0,
    p0: float = 1e4,
    theta0: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run RLS over a whole record.

    Returns the parameter history (one row per regression row, i.e. starting
    at sample ``max(na, nb + nk - 1)``) and the a-priori prediction errors.
    """
    phi, target = arx_regressors(u, y, na, nb, nk)
    rls = RecursiveLeastSquares(na + nb, forgetting=forgetting, p0=p0, theta0=theta0)
    history = np.empty_like(phi)
    errors = np.empty(len(target))
    for k in range(len(target)):
        errors[k] = rls.update(phi[k], target[k])
        history[k] = rls.theta
    return history, errors


# --------------------------------------------------------------------------- static maps
def fit_gain_through_origin(u: ArrayLike, y: ArrayLike) -> float:
    """Least-squares slope of ``y = K u`` (line through the origin)."""
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(u @ y / (u @ u))


@dataclass(frozen=True)
class QuadraticStaticMap:
    """Steady-state map ``u = w / K0 + beta * w * |w|``.

    ``K0`` is the small-signal gain at low speed and ``beta`` lumps
    speed-dependent losses (aerodynamic drag of the disk, viscous effects that
    grow with speed) expressed as an equivalent voltage.
    """

    K0: float
    beta: float

    def voltage(self, w: ArrayLike) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        return w / self.K0 + self.beta * w * np.abs(w)

    def speed(self, u: ArrayLike) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        if self.beta == 0.0:
            return self.K0 * u
        inv = 1.0 / self.K0
        mag = (-inv + np.sqrt(inv**2 + 4.0 * self.beta * np.abs(u))) / (2.0 * self.beta)
        return np.sign(u) * mag

    def incremental_gain(self, w: ArrayLike) -> np.ndarray:
        """Small-signal gain ``dw/du`` around the operating speed ``w``."""
        return 1.0 / (1.0 / self.K0 + 2.0 * self.beta * np.abs(np.asarray(w, dtype=float)))


def fit_quadratic_static_map(u: ArrayLike, w: ArrayLike) -> QuadraticStaticMap:
    """Linear least squares for ``u = alpha * w + beta * w|w|``."""
    u = np.asarray(u, dtype=float)
    w = np.asarray(w, dtype=float)
    design = np.column_stack([w, w * np.abs(w)])
    (alpha, beta), *_ = np.linalg.lstsq(design, u, rcond=None)
    return QuadraticStaticMap(K0=float(1.0 / alpha), beta=float(beta))


# --------------------------------------------------------------------------- 2nd order
@dataclass(frozen=True)
class SecondOrderFit:
    """Continuous-time equivalent of a discrete second-order model."""

    wn: float
    zeta: float
    dc_gain: float
    poles_s: tuple[complex, complex]


def continuous_second_order(model: ArxModel, ts: float) -> SecondOrderFit:
    """Map the poles of an ARX(2, nb) model to ``s = ln(z) / Ts``.

    The natural frequency and damping follow from the pole product and sum,
    which also covers the over-damped (two real poles) case.
    """
    if model.na != 2:
        raise ValueError("expected a model with two poles")
    z = model.poles().astype(complex)
    if np.any(np.abs(z) >= 1.0):
        raise ValueError("model is unstable")
    s = np.log(z) / ts
    wn = math.sqrt(float((s[0] * s[1]).real))
    zeta = float(-(s[0] + s[1]).real / (2.0 * wn))
    return SecondOrderFit(wn=wn, zeta=zeta, dc_gain=model.dc_gain(), poles_s=(complex(s[0]), complex(s[1])))


def physical_from_second_order(fit: SecondOrderFit, inertia: float) -> dict[str, float]:
    """Physical coefficients of ``J y'' + a1 y' + a0 y = b u`` given the inertia ``J``.

    A transfer function only fixes the coefficient *ratios*, so one physical
    quantity (here the moment of inertia) must come from elsewhere.
    """
    a0 = inertia * fit.wn**2
    return {"a2": inertia, "a1": 2.0 * fit.zeta * fit.wn * inertia, "a0": a0, "b": fit.dc_gain * a0}
