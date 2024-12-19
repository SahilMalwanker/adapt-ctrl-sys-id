import math

import numpy as np
import pytest

from adapt_ctrl_sys_id.control import (
    DigitalPIP,
    DiscretePID,
    PIDGains,
    PIPCoefficients,
    closed_loop_poles,
    gravity_feedforward,
    place_pid,
    simulate_loop,
)
from adapt_ctrl_sys_id.pendulum import CALIBRATED


@pytest.fixture
def lin():
    return CALIBRATED.linearize(math.radians(40)).in_degrees()


def test_pole_placement_hits_the_target_polynomial(lin):
    omega, zeta, alpha = 6.0, 0.8, 1.5
    gains = place_pid(lin, omega, zeta, alpha)
    poles = np.sort_complex(closed_loop_poles(lin, gains))
    target = np.sort_complex(np.roots(np.polymul([1, alpha * omega], [1, 2 * zeta * omega, omega**2])))
    np.testing.assert_allclose(poles, target, rtol=1e-9)


def test_pip_coefficients_round_trip():
    gains = PIDGains(kp=5.6, ki=18.2, kd=0.78)
    q = PIPCoefficients.from_pid(gains, 0.01)
    assert q.q0_pi == pytest.approx(5.6 / 0.78 + 18.2 * 0.01 / 0.78)
    assert q.q1_pi == pytest.approx(-5.6 / 0.78)
    back = q.to_pid()
    assert (back.kp, back.ki, back.kd) == pytest.approx((gains.kp, gains.ki, gains.kd))


def test_pip_requires_positive_inner_gain():
    with pytest.raises(ValueError):
        PIPCoefficients.from_pid(PIDGains(1.0, 1.0, 0.0), 0.01)


def test_digital_pip_equals_parallel_pi_d_law():
    gains = PIDGains(kp=2.0, ki=5.0, kd=0.5)
    ts = 0.01
    ctrl = DigitalPIP(PIPCoefficients.from_pid(gains, ts), u_min=-1e9, u_max=1e9)
    rng = np.random.default_rng(4)
    integral = 0.0
    for _ in range(50):
        ref, y, ydot = rng.normal(size=3)
        e = ref - y
        integral += e * ts
        expected = gains.kp * e + gains.ki * integral - gains.kd * ydot
        assert ctrl(ref, y, ydot) == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_anti_windup_limits_the_integrator():
    q = PIPCoefficients.from_pid(PIDGains(kp=1.0, ki=10.0, kd=0.5), 0.01)
    plain, guarded = DigitalPIP(q), DigitalPIP(q, anti_windup=True)
    for _ in range(500):  # unreachable reference -> output pinned at 100 %
        plain(200.0, 0.0, 0.0)
        guarded(200.0, 0.0, 0.0)
    assert guarded.u_pi == pytest.approx(100.0 / q.q0_p)
    assert plain.u_pi > 10 * guarded.u_pi


def test_feedforward_is_the_static_inverse():
    ff = gravity_feedforward(CALIBRATED)
    assert ff(40.0) == pytest.approx(CALIBRATED.equilibrium_input(math.radians(40.0)))
    assert ff(-10.0) == 0.0


def test_designed_controller_tracks_the_profile(lin):
    coeffs = PIPCoefficients.from_pid(place_pid(lin, 8.0, 0.9, 1.0), 0.01)
    ctrl = DigitalPIP(coeffs, feedforward=gravity_feedforward(CALIBRATED), anti_windup=True)
    run = simulate_loop(CALIBRATED, ctrl, t_end=60.0)
    assert run.costs((30.0, 60.0))["J_e"] < 1.0
    assert np.all((run.u >= 0.0) & (run.u <= 100.0))
    assert abs(run.phi[-1] - run.ref[-1]) < 0.2


def test_one_dof_pid_has_derivative_kick(lin):
    gains = place_pid(lin, 8.0, 0.9, 1.0)
    pid = DiscretePID(gains, 0.001, u_min=-1e9, u_max=1e9)
    pip = DigitalPIP(PIPCoefficients.from_pid(gains, 0.001), u_min=-1e9, u_max=1e9)
    pid(0.0, 0.0, 0.0)
    pip(0.0, 0.0, 0.0)
    assert pid(10.0, 0.0, 0.0) > 10 * pip(10.0, 0.0, 0.0) > 0.0
