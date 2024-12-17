import math

import numpy as np
import pytest

from adapt_ctrl_sys_id.frequency import correlation_analysis, to_db_deg
from adapt_ctrl_sys_id.identification import PT1, arx_ls, fit_pt1_output_error
from adapt_ctrl_sys_id.metrics import overshoot_percent
from adapt_ctrl_sys_id.servo import (
    HARDWARE_PT1,
    AdaptiveCascadeController,
    CascadeController,
    ServoParams,
    VirtualServo,
    actuator_limited_pole,
    cascade_gains,
    position_loop_poles,
    rule_of_thumb_pole,
    run_open_loop,
    run_position_control,
)
from adapt_ctrl_sys_id.signals import servo_excitation, square_wave


def test_cascade_gains_give_damping_of_one_over_sqrt_two():
    K, T, s_vel = 25.172, 0.31919, -31.0
    kd, kp = cascade_gains(K, T, s_vel)
    assert kd == pytest.approx(0.35337, abs=1e-5)
    poles = position_loop_poles(K, T, kd, kp)
    np.testing.assert_allclose(sorted(poles.imag), [s_vel / 2, -s_vel / 2], atol=1e-9)
    np.testing.assert_allclose(poles.real, [s_vel / 2, s_vel / 2], atol=1e-9)


def test_cascade_gains_reject_slow_inner_pole():
    with pytest.raises(ValueError):
        cascade_gains(25.0, 0.3, -2.0)


def test_actuator_limit_saturates_exactly_at_u_max():
    K, T, u_max, step = 25.172, 0.31919, 10.0, math.pi / 2
    s = actuator_limited_pole(K, T, u_max, step)
    kd, kp = cascade_gains(K, T, s)
    assert kd * kp * step == pytest.approx(u_max, rel=1e-12)
    assert rule_of_thumb_pole(T) == -31.0
    assert rule_of_thumb_pole(T) > s  # the rule of thumb stays on the safe side


def test_virtual_servo_reproduces_the_hardware_pt1():
    params = ServoParams()
    t = np.arange(0, 15.0 + params.ts / 2, params.ts)
    u = servo_excitation(t)
    y = run_open_loop(u, params)["omega"]
    est = fit_pt1_output_error(u, y, params.ts)
    assert pytest.approx(HARDWARE_PT1.T, rel=0.01) == est.T
    assert pytest.approx(HARDWARE_PT1.K, rel=0.03) == est.K
    assert pytest.approx(HARDWARE_PT1.T, rel=0.05) == PT1.from_arx(arx_ls(u, y), params.ts).T


def test_encoder_quantisation():
    servo = VirtualServo()
    servo.step(5.0)
    theta, _ = servo.measurement
    quantum = 2 * math.pi / servo.params.counts_per_rev
    assert theta / quantum == pytest.approx(round(theta / quantum), abs=1e-9)


def test_amplifier_limit_is_enforced():
    a, b = VirtualServo(), VirtualServo()
    for _ in range(100):
        a.step(10.0)
        b.step(1000.0)
    assert a.omega == pytest.approx(b.omega)


def test_fixed_cascade_step_response():
    K, T = HARDWARE_PT1.K, HARDWARE_PT1.T
    kd, kp = cascade_gains(K, T, 0.95 * actuator_limited_pole(K, T, 10.0, math.pi / 2))
    run = run_position_control(CascadeController(kd, kp), lambda t: np.full_like(t, math.pi / 2), 2.0)
    assert overshoot_percent(run.theta, 0.0, math.pi / 2) < 6.0
    assert np.max(np.abs(run.u)) <= 10.0
    assert run.theta[-1] == pytest.approx(math.pi / 2, abs=0.01)


def test_adaptive_cascade_beats_fixed_gains_after_inertia_change():
    K, T = HARDWARE_PT1.K, HARDWARE_PT1.T
    kd, kp = cascade_gains(K, T, 0.95 * actuator_limited_pole(K, T, 10.0, math.pi / 2))

    def ref(t):
        return square_wave(t, 0.0, math.pi / 2, 2.0)

    def schedule(t):
        return 1.85 if t >= 4.9 else 1.0

    fixed = run_position_control(CascadeController(kd, kp), ref, 10.0, inertia_schedule=schedule)
    adaptive = run_position_control(
        AdaptiveCascadeController(HARDWARE_PT1, 0.002), ref, 10.0, inertia_schedule=schedule
    )

    def overshoot(run):
        mask = (run.t >= 8.0) & (run.t < 8.95)
        return overshoot_percent(run.theta[mask], 0.0, math.pi / 2)

    assert overshoot(adaptive) < 0.5 * overshoot(fixed)
    tau = ServoParams().tau * 1.85
    assert adaptive.T_hat[-1] == pytest.approx(tau, rel=0.15)
    assert np.max(np.abs(adaptive.u)) <= 10.0


def test_increment_model_matches_exact_discretisation():
    model = PT1(25.0, 0.3)
    h = 0.02
    a, c1, c2 = AdaptiveCascadeController.increment_model(model, h)
    t_fine = np.arange(0, 2.0, 1e-4)
    u = np.where(t_fine < 0.5, 1.0, -2.0)
    omega = model.simulate(u, 1e-4)
    theta = np.concatenate([[0.0], np.cumsum(omega[:-1]) * 1e-4])
    idx = np.arange(0, len(t_fine), round(h / 1e-4))
    rate = np.diff(theta[idx]) / h  # mean speed over window j
    u_win = u[idx[:-1]]  # input held during window j
    pred = a * rate[1:-1] + c1 * u_win[2:] + c2 * u_win[1:-1]
    np.testing.assert_allclose(pred, rate[2:], atol=0.05)


def test_correlation_analysis_recovers_gain_and_phase():
    omega, gain, phase = 3.0, 2.5, -0.7
    t = np.arange(0, 30, 0.001)
    y = 1.5 * gain * np.sin(omega * t + phase) + 0.3
    g = correlation_analysis(t, y, omega, amplitude=1.5)
    assert abs(g) == pytest.approx(gain, rel=1e-3)
    assert np.angle(g) == pytest.approx(phase, abs=1e-3)


def test_bode_helpers_for_pt1():
    model = PT1(10.0, 0.5)
    mag, phase = to_db_deg(model.frequency_response([model.cutoff]))
    assert mag[0] == pytest.approx(20 * math.log10(10 / math.sqrt(2)))
    assert phase[0] == pytest.approx(-45.0)
