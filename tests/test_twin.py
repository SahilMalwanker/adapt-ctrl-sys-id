import re

import numpy as np
import pytest

from adapt_ctrl_sys_id.control import DigitalPIP, PIDGains, PIPCoefficients
from adapt_ctrl_sys_id.twin import (
    CALIBRATED,
    PendulumTwin,
    TwinClient,
    TwinConfig,
    TwinServer,
    format_command,
    format_state,
    parse_command,
    parse_state,
    run_closed_loop,
    sampling_report,
)

# pattern used by MATLAB clients: sscanf(line, '<PP><t>%f </t><PA>%f </PA><PAD>%f </PAD>')
MATLAB_SSCANF = re.compile(r"^<PP><t>[-0-9.]+ </t><PA>[-0-9.]+ </PA><PAD>[-0-9.]+ </PAD>")


def test_state_message_round_trip_and_matlab_layout():
    line = format_state(1.23456, -12.5, 300.25)
    assert line.endswith("\n")
    assert MATLAB_SSCANF.match(line)
    state = parse_state(line)
    assert (state.t, state.phi_deg, state.phid_deg) == pytest.approx((1.2346, -12.5, 300.25))


def test_state_parser_accepts_decimal_commas():
    state = parse_state("<PP><t>0,5 </t><PA>10,25 </PA><PAD>-3,5 </PAD></PP>")
    assert (state.t, state.phi_deg, state.phid_deg) == pytest.approx((0.5, 10.25, -3.5))
    with pytest.raises(ValueError):
        parse_state("hello")


def test_command_round_trip():
    assert parse_command(format_command(37.5, 120.0)) == pytest.approx((37.5, 120.0))
    assert parse_command("42") == (42.0, 0.0)
    with pytest.raises(ValueError):
        parse_command(" ")


def test_twin_slider_moves_with_finite_speed():
    twin = PendulumTwin(CALIBRATED, TwinConfig(angle_noise_deg=0.0, rate_noise_deg=0.0))
    twin.command(0.0, 250.0)
    for _ in range(100):  # 1 s at 0.1 m/s
        twin.advance()
    assert twin.x == pytest.approx(CALIBRATED.geometry.x_up + 0.1, abs=1e-9)


def test_twin_rests_at_equilibrium_for_constant_thrust():
    twin = PendulumTwin(CALIBRATED, TwinConfig(angle_noise_deg=0.0, rate_noise_deg=0.0))
    u = CALIBRATED.equilibrium_input(np.radians(30.0))
    twin.command(u)
    for _ in range(6000):
        state = twin.advance()
    assert state.phi_deg == pytest.approx(30.0, abs=0.05)


def _controller():
    gains = PIDGains(kp=5.6, ki=18.2, kd=0.78)
    return DigitalPIP(PIPCoefficients.from_pid(gains, 0.01))


def test_lockstep_session_is_deterministic():
    runs = []
    for _ in range(2):
        twin = PendulumTwin(CALIBRATED, TwinConfig(seed=11))
        with TwinServer(twin, port=0, realtime=False) as server, TwinClient(port=server.port) as client:
            runs.append(run_closed_loop(client, _controller(), duration=8.0))
    a, b = runs
    assert len(a.t) == 801
    np.testing.assert_allclose(a.phi, b.phi)
    np.testing.assert_allclose(np.diff(a.t), 0.01, atol=1e-9)
    assert a.phi[-1] == pytest.approx(a.ref[-1], abs=1.0)


def test_realtime_session_smoke():
    with TwinServer(PendulumTwin(CALIBRATED), port=0, realtime=True) as server, TwinClient(port=server.port) as client:
        run = run_closed_loop(client, _controller(), duration=0.5)
    assert len(run.t) > 20
    assert np.all(np.diff(run.t) > 0)


def test_sampling_report():
    t = np.cumsum(np.r_[0.0, np.full(99, 0.01), 0.05, np.full(50, 0.01)])
    report = sampling_report(t, nominal=0.01)
    assert report["deviations"] == 1
    assert report["max_interval"] == pytest.approx(0.05)
    assert report["intervals"] == 150
