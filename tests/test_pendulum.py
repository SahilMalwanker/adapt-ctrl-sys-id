import math

import numpy as np
import pytest

from adapt_ctrl_sys_id import data
from adapt_ctrl_sys_id.pendulum import (
    CALIBRATED,
    LAB_MODEL,
    Geometry,
    PendulumParams,
    com_from_two_positions,
    decay_rate,
    detect_release,
    energy_inertia_bound,
    fit_free_fall,
    linearization_error,
    simulate,
    simulate_free_fall,
    simulate_linear,
    step_comparison,
    thrust_from_angle,
)


def test_geometry_matches_hand_calculation():
    g = Geometry()
    assert g.mass == pytest.approx(0.817)
    assert g.com(0.09) == pytest.approx((0.09 * 0.43 + 0.2365 * 0.3 + 0.5 * 0.087) / 0.817)
    assert g.inertia(0.09) == pytest.approx(0.3 * 0.473**2 / 3 + 0.087 * 0.25 + 0.43 * 0.09**2)
    assert LAB_MODEL.a0 == pytest.approx(1.5024, abs=1e-4)
    assert g.slider_position(500.0) == pytest.approx(g.x_down)


def test_equilibrium_input_and_angle_are_inverse():
    p = CALIBRATED.with_(deadband=0.0)
    for phi in np.radians([5, 20, 40, 60]):
        assert p.equilibrium_angle(p.equilibrium_input(phi)) == pytest.approx(phi, abs=1e-12)


def test_thrust_is_unidirectional_with_dead_band():
    p = CALIBRATED
    assert p.thrust_command(-20.0) == 0.0
    assert p.thrust_command(p.deadband - 0.1) == 0.0
    assert p.thrust_command(150.0) == p.u_max


def test_linearisation_matches_numerical_jacobian():
    p = PendulumParams(viscous=0.004)
    phi_op = math.radians(35)
    lin = p.linearize(phi_op)
    u_op = lin.u_op

    def accel(phi, phid, u):
        return (p.b0 * u - p.a0 * math.sin(phi) - p.viscous * phid) / p.J

    eps = 1e-6
    d_phi = (accel(phi_op + eps, 0, u_op) - accel(phi_op - eps, 0, u_op)) / (2 * eps)
    d_phid = (accel(phi_op, eps, u_op) - accel(phi_op, -eps, u_op)) / (2 * eps)
    d_u = (accel(phi_op, 0, u_op + eps) - accel(phi_op, 0, u_op - eps)) / (2 * eps)
    assert -lin.a0 / lin.a2 == pytest.approx(d_phi, rel=1e-6)
    assert -lin.a1 / lin.a2 == pytest.approx(d_phid, rel=1e-6)
    assert lin.b / lin.a2 == pytest.approx(d_u, rel=1e-6)


def test_undamped_simulation_conserves_energy():
    p = PendulumParams(viscous=0.0)
    t = np.arange(0, 10, 0.002)
    out = simulate(p, t, 0.0, phi0=math.radians(80))
    energy = 0.5 * p.J * out["phid"] ** 2 + p.a0 * (1 - np.cos(out["phi"]))
    assert np.ptp(energy) / energy[0] < 1e-6


def test_linear_and_nonlinear_agree_for_small_steps():
    p = CALIBRATED.with_(rotor_tau=0.0, deadband=0.0)
    assert math.degrees(linearization_error(p, math.radians(40), math.radians(1))) < 0.2
    assert math.degrees(linearization_error(p, math.radians(40), math.radians(15))) > 2.0


def test_step_comparison_starts_and_settles_at_the_step_ends():
    p = CALIBRATED.with_(rotor_tau=0.0, deadband=0.0)
    op, half = math.radians(40), math.radians(2)
    run = step_comparison(p, op, half, t_end=60.0)
    assert run["nonlinear"][0] == pytest.approx(op - half)
    assert run["nonlinear"][-1] == pytest.approx(op + half, abs=1e-4)
    # the linear model starts on its own equilibrium for the same input, close to the true one
    assert run["linear"][0] == pytest.approx(op - half, abs=math.radians(0.05))
    error = np.max(np.abs(run["nonlinear"] - run["linear"]))
    assert error == pytest.approx(linearization_error(p, op, half, t_end=60.0))
    # a +/-20 deg step around 55 deg drives the nonlinear model past horizontal
    big = step_comparison(p, math.radians(55), math.radians(20))
    assert np.max(big["nonlinear"]) > math.pi / 2


def test_simulate_linear_step_reaches_static_gain():
    lin = CALIBRATED.linearize(math.radians(30))
    t = np.arange(0, 120, 0.01)
    y = simulate_linear(lin, t, 1.0)
    assert y[-1] == pytest.approx(lin.b / lin.a0, rel=1e-3)


def test_free_fall_fit_recovers_synthetic_parameters():
    truth = LAB_MODEL.with_(inertia_offset=0.006, viscous=0.002, drag=5e-4)
    t = np.arange(0, 20, 0.01)
    phi = simulate_free_fall(truth, t, math.radians(90), t_release=0.5)
    fit = fit_free_fall(t, phi + np.random.default_rng(0).normal(0, 0.002, t.size), LAB_MODEL)
    assert pytest.approx(truth.J, rel=0.01) == fit.params.J
    assert fit.params.drag == pytest.approx(5e-4, rel=0.1)
    assert fit.fit_percent > 98


def test_energy_bound_and_decay_rate_on_synthetic_data():
    truth = LAB_MODEL.with_(viscous=0.004)
    t = np.arange(0, 30, 0.005)
    phi = simulate_free_fall(truth, t, math.radians(30), t_release=0.0)
    phid = np.gradient(phi, t)
    bound = energy_inertia_bound(truth.geometry, truth.x, np.max(np.abs(phid)), math.radians(30))
    assert truth.J <= bound <= 1.1 * truth.J
    sigma, *_ = decay_rate(t, phi)
    assert sigma == pytest.approx(truth.viscous / (2 * truth.J), rel=0.05)


def test_detect_release():
    t = np.linspace(0, 2, 2001)
    phi = np.where(t < 0.7, 1.5, 1.5 - 3.0 * (t - 0.7) ** 2)
    t_rel, phi0 = detect_release(t, phi, threshold=0.01)
    assert phi0 == pytest.approx(1.5)
    assert 0.7 < t_rel < 0.8


def test_two_position_centre_of_gravity_on_model_data():
    g = Geometry()
    force = 1.2
    phi = math.asin(force * g.l_rotor / (g.g * g.first_moment(g.x_up)))
    phi_p = math.asin(force * g.l_rotor / (g.g * g.first_moment(g.x_down)))
    assert com_from_two_positions(phi, phi_p, g.x_up, g.x_down, g) == pytest.approx(g.com(g.x_up), rel=1e-9)
    assert float(thrust_from_angle(phi, g.x_up, g)) == pytest.approx(force)


def test_calibrated_constants_match_the_hardware_records():
    ff = data.pendulum_free_fall().resample(0.02)
    fit = fit_free_fall(ff.t, ff.phi, LAB_MODEL)
    assert pytest.approx(CALIBRATED.J, rel=0.02) == fit.params.J
    assert fit.params.viscous == pytest.approx(CALIBRATED.viscous, rel=0.15)
    assert fit.params.drag == pytest.approx(CALIBRATED.drag, rel=0.15)
    assert fit.fit_percent > 85
