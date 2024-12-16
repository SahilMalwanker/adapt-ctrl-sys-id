import math

import numpy as np
import pytest
from scipy.signal import savgol_filter

from adapt_ctrl_sys_id import data
from adapt_ctrl_sys_id.gyro import count_turns, integrate_rate
from adapt_ctrl_sys_id.signals import pendulum_reference_deg, piecewise_constant, servo_excitation, square_wave


def test_dataset_shapes_and_columns():
    assert len(data.servo_static_map()["u_V"]) == 22
    assert set(data.servo_frequency_response()) == {"omega_rad_s", "magnitude_db", "phase_deg"}
    st = data.pendulum_static_equilibrium()
    assert set(np.unique(st["x_m"])) == {0.09, 0.34}
    ff = data.pendulum_free_fall()
    assert ff.t[0] == 0.0
    assert ff.phi_deg[0] == pytest.approx(91.1, abs=0.5)
    assert len(data.gyro_chair_spin()["t_s"]) == 1993


def test_imu_rate_follows_the_angle_sign_convention():
    ff = data.pendulum_free_fall().resample(0.005)
    pot_rate = savgol_filter(ff.phi_deg, 31, 3, deriv=1, delta=0.005)
    assert np.corrcoef(pot_rate, ff.phid_deg)[0, 1] > 0.98


def test_resample_is_uniform():
    ts = data.pendulum_step_response().resample(0.01)
    np.testing.assert_allclose(np.diff(ts.t), 0.01, atol=1e-9)


def test_servo_excitation_profile_matches_dataset():
    profile = data.servo_excitation_profile()
    np.testing.assert_allclose(servo_excitation(profile["t_start_s"] + 1e-6), profile["u_V"])


def test_signal_helpers():
    assert piecewise_constant([-1.0, 0.5, 2.0], [0.0, 1.0], [3.0, 4.0]).tolist() == [0.0, 3.0, 4.0]
    assert square_wave([0.1, 1.1], 0.0, 1.0, 2.0).tolist() == [1.0, 0.0]
    ref = pendulum_reference_deg(np.array([0.0, 15.0, 30.0, 50.0, 70.0, 95.0]))
    assert ref.tolist() == [0.0, 20.0, 40.0, 30.0, 50.0, 40.0]


def test_rate_integration_and_turn_counting():
    t = np.linspace(0, 10, 10001)
    rate = np.where(t < 6, 2 * math.pi, -2 * math.pi)  # 6 turns forward, 4 back
    angle = integrate_rate(t, rate)
    turns = count_turns(angle)
    assert turns.net == pytest.approx(2.0, abs=1e-3)
    assert turns.travelled == pytest.approx(10.0, abs=1e-3)
    assert turns.reversals == 1


def test_chair_spin_record():
    chair = data.gyro_chair_spin()
    turns = count_turns(chair["yaw_rad"])
    assert turns.net == pytest.approx(-3.43, abs=0.01)
    assert turns.reversals == 1
