import math

import numpy as np
import pytest
from scipy.signal import cont2discrete

from adapt_ctrl_sys_id.identification import (
    PT1,
    ArxModel,
    QuadraticStaticMap,
    RecursiveLeastSquares,
    arx_ls,
    continuous_second_order,
    fit_gain_through_origin,
    fit_pt1_output_error,
    fit_quadratic_static_map,
    physical_from_second_order,
    rls_identify,
)
from adapt_ctrl_sys_id.metrics import nrmse_fit


def test_pt1_arx_round_trip():
    model = PT1(K=25.172, T=0.31919)
    back = PT1.from_arx(model.to_arx(0.002), 0.002)
    assert pytest.approx(model.K, rel=1e-12) == back.K
    assert pytest.approx(model.T, rel=1e-12) == back.T


def test_pt1_simulation_matches_analytic_step():
    model = PT1(K=2.0, T=0.5)
    ts = 0.01
    t = np.arange(0, 3, ts)
    y = model.simulate(np.ones_like(t), ts)
    expected = 2.0 * (1.0 - np.exp(-t / 0.5))
    np.testing.assert_allclose(y, expected, atol=1e-12)


def test_pt1_rejects_invalid_pole():
    with pytest.raises(ValueError):
        PT1.from_arx(ArxModel((0.5,), (1.0,)), 0.01)


def test_arx_ls_recovers_second_order_system():
    rng = np.random.default_rng(1)
    truth = ArxModel(a=(-1.6, 0.68), b=(0.05, 0.03), nk=1)
    u = rng.normal(size=4000)
    y = truth.simulate(u)
    est = arx_ls(u, y, na=2, nb=2, nk=1)
    np.testing.assert_allclose(est.a, truth.a, atol=1e-10)
    np.testing.assert_allclose(est.b, truth.b, atol=1e-10)


def test_rls_converges_to_batch_least_squares():
    rng = np.random.default_rng(2)
    truth = PT1(3.0, 0.2).to_arx(0.01)
    u = np.sign(np.sin(np.arange(3000) * 0.013)) + 0.1 * rng.normal(size=3000)
    y = truth.simulate(u) + 0.01 * rng.normal(size=3000)
    history, _ = rls_identify(u, y, forgetting=1.0, p0=1e6)
    batch = arx_ls(u, y)
    np.testing.assert_allclose(history[-1], [*batch.a, *batch.b], rtol=1e-4)


def test_rls_forgetting_tracks_a_parameter_jump():
    rls = RecursiveLeastSquares(1, forgetting=0.95, p0=10.0)
    for k in range(400):
        gain = 1.0 if k < 200 else 3.0
        phi = np.array([math.sin(0.3 * k) + 1.5])
        rls.update(phi, gain * phi[0])
    assert rls.theta[0] == pytest.approx(3.0, rel=1e-3)


def test_rls_rejects_bad_forgetting_factor():
    with pytest.raises(ValueError):
        RecursiveLeastSquares(2, forgetting=1.2)


def test_output_error_fit_is_robust_to_measurement_noise():
    rng = np.random.default_rng(3)
    truth = PT1(25.0, 0.32)
    ts = 0.002
    u = np.repeat([0.0, 1.0, 4.0, 7.0, 1.0, -5.0, 0.5, 3.0], 1000)
    y = truth.simulate(u, ts) + rng.normal(0, 1.0, u.size)
    est = fit_pt1_output_error(u, y, ts)
    assert pytest.approx(truth.K, rel=0.01) == est.K
    assert pytest.approx(truth.T, rel=0.02) == est.T


def test_quadratic_static_map_fit_and_inverse():
    truth = QuadraticStaticMap(K0=28.0, beta=3e-5)
    w = np.linspace(-200, 200, 41)
    est = fit_quadratic_static_map(truth.voltage(w), w)
    assert pytest.approx(28.0, rel=1e-9) == est.K0
    assert est.beta == pytest.approx(3e-5, rel=1e-9)
    u = np.linspace(-9, 9, 19)
    np.testing.assert_allclose(est.voltage(est.speed(u)), u, atol=1e-9)


def test_gain_through_origin():
    u = np.array([-2.0, -1.0, 1.0, 2.0])
    assert fit_gain_through_origin(u, 3.0 * u) == pytest.approx(3.0)


def test_second_order_mapping_recovers_continuous_parameters():
    wn, zeta, gain, ts = 4.6, 0.12, 1.3, 0.05
    num, den, _ = cont2discrete(([gain * wn**2], [1.0, 2 * zeta * wn, wn**2]), ts, method="zoh")
    num = np.ravel(num)
    model = ArxModel(a=tuple(den[1:]), b=tuple(num[1:]), nk=1)
    fit = continuous_second_order(model, ts)
    assert fit.wn == pytest.approx(wn, rel=1e-9)
    assert fit.zeta == pytest.approx(zeta, rel=1e-9)
    assert fit.dc_gain == pytest.approx(gain, rel=1e-9)
    phys = physical_from_second_order(fit, inertia=0.05)
    assert phys["a0"] == pytest.approx(0.05 * wn**2)
    assert phys["b"] / phys["a0"] == pytest.approx(gain)


def test_nrmse_fit_reference_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert nrmse_fit(y, y) == pytest.approx(100.0)
    assert nrmse_fit(y, np.full(4, y.mean())) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        nrmse_fit(np.ones(3), np.ones(3))
