"""End-to-end analysis pipeline.

Every number and curve shown in the README is produced by the three study
functions below; ``adapt-ctrl-sys-id reproduce`` runs them, renders the figures and
writes ``results.json``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from . import data
from .control import (
    DigitalPIP,
    DiscretePID,
    LoopResult,
    PIDGains,
    PIPCoefficients,
    closed_loop_poles,
    gravity_feedforward,
    place_pid,
    simulate_loop,
)
from .frequency import correlation_analysis
from .gyro import TurnCount, count_turns, integrate_rate
from .identification import (
    PT1,
    ArxModel,
    QuadraticStaticMap,
    SecondOrderFit,
    arx_ls,
    continuous_second_order,
    fit_gain_through_origin,
    fit_pt1_output_error,
    fit_quadratic_static_map,
    physical_from_second_order,
    rls_identify,
)
from .metrics import nrmse_fit, overshoot_percent, rmse
from .pendulum import (
    CALIBRATED,
    LAB_MODEL,
    FreeFallFit,
    PendulumParams,
    com_from_two_positions,
    decay_rate,
    detect_release,
    energy_inertia_bound,
    fit_free_fall,
    simulate,
    simulate_free_fall,
    step_comparison,
    thrust_from_angle,
)
from .servo import (
    HARDWARE_PT1,
    AdaptiveCascadeController,
    CascadeController,
    ServoParams,
    ServoRun,
    actuator_limited_pole,
    cascade_gains,
    position_loop_poles,
    rule_of_thumb_pole,
    run_open_loop,
    run_position_control,
    sine_test,
)
from .signals import servo_excitation, square_wave
from .twin import PendulumTwin, TwinClient, TwinConfig, TwinServer, run_closed_loop, sampling_report

__all__ = ["ControlStudy", "PendulumStudy", "ServoStudy", "control_study", "pendulum_study", "servo_study"]

DESIGN_STEP = math.pi / 2.0
INERTIA_RATIO = 3.0 * math.sqrt(0.7 - HARDWARE_PT1.T)  # heavier disk: T2/T1 = 1.85
T_SWITCH = 4.9
CONTROL_TS = 0.01
DESIGN_OMEGA = 8.0
DESIGN_ZETA = 0.9
DESIGN_ALPHA = 1.0
OPERATING_POINT_DEG = 40.0
HERO_SLIDER_TIME = 100.0
HERO_SLIDER_MM = 250.0


def hero_slider_mm(t: float) -> float:
    """Slider command of the showcase run: move the slider down at 100 s."""
    return HERO_SLIDER_MM if t >= HERO_SLIDER_TIME else 0.0


def _r(value: float, digits: int = 4) -> float:
    return float(round(float(value), digits))


# ============================================================================ servo
@dataclass
class ServoStudy:
    """DC servo: static map, identification, validation, cascade and adaptive control."""

    static_u: np.ndarray
    static_w: np.ndarray
    k_linear: float
    static_map: QuadraticStaticMap
    rms_linear: float
    rms_quadratic: float
    ts: float
    t_id: np.ndarray
    u_id: np.ndarray
    y_id: np.ndarray
    pt1_arx: PT1
    pt1_oe: PT1
    fit_arx: float
    fit_oe: float
    fit_hardware: float
    rls_t: np.ndarray
    rls_K: np.ndarray
    rls_T: np.ndarray
    w_hw: np.ndarray
    mag_hw: np.ndarray
    phase_hw: np.ndarray
    w_grid: np.ndarray
    g_linear: np.ndarray
    w_lockin: np.ndarray
    g_lockin: np.ndarray
    g_lockin_hw: np.ndarray
    s_limit: float
    s_rule: float
    s_design: float
    kd: float
    kp: float
    poles: np.ndarray
    step_run: ServoRun
    sine_run: ServoRun
    fixed_run: ServoRun
    adaptive_run: ServoRun
    forgetting_runs: dict[float, ServoRun]

    def adaptive_metrics(self) -> dict[str, float]:
        """Overshoot of the rising edges and IAE after the inertia change."""
        out: dict[str, float] = {}
        for name, run in (("fixed", self.fixed_run), ("adaptive", self.adaptive_run)):
            shoots = []
            for edge in (6.0, 8.0, 10.0):
                mask = (run.t >= edge) & (run.t < edge + 0.95)
                shoots.append(overshoot_percent(run.theta[mask], 0.0, DESIGN_STEP))
            after = run.t >= T_SWITCH
            out[f"{name}_overshoot_pct"] = float(np.mean(shoots))
            out[f"{name}_iae_after_change"] = float(np.sum(np.abs(run.ref[after] - run.theta[after])) * self.ts)
        return out

    def summary(self) -> dict[str, object]:
        mag_lin = 20 * np.log10(np.abs(HARDWARE_PT1.frequency_response(self.w_hw)))
        mag_nl = 20 * np.log10(np.abs(self.g_lockin_hw))
        return {
            "static_map": {
                "K_linear_rad_s_per_V": _r(self.k_linear),
                "K0_low_speed_rad_s_per_V": _r(self.static_map.K0),
                "beta_V_s2_per_rad2": float(f"{self.static_map.beta:.4g}"),
                "rms_error_linear_rad_s": _r(self.rms_linear, 3),
                "rms_error_quadratic_rad_s": _r(self.rms_quadratic, 3),
            },
            "identification": {
                "hardware_K": HARDWARE_PT1.K,
                "hardware_T": HARDWARE_PT1.T,
                "virtual_arx_K": _r(self.pt1_arx.K),
                "virtual_arx_T": _r(self.pt1_arx.T),
                "virtual_oe_K": _r(self.pt1_oe.K),
                "virtual_oe_T": _r(self.pt1_oe.T),
                "fit_oe_pct": _r(self.fit_oe, 2),
                "fit_arx_pct": _r(self.fit_arx, 2),
                "fit_hardware_model_pct": _r(self.fit_hardware, 2),
                "rls_final_K": _r(self.rls_K[-1]),
                "rls_final_T": _r(self.rls_T[-1]),
            },
            "frequency_validation": {
                "omega_rad_s": [float(w) for w in self.w_hw],
                "hardware_db": [float(m) for m in self.mag_hw],
                "linear_pt1_db": [_r(m, 2) for m in mag_lin],
                "calibrated_virtual_db": [_r(m, 2) for m in mag_nl],
                "mean_abs_error_linear_db": _r(np.mean(np.abs(mag_lin - self.mag_hw)), 2),
                "mean_abs_error_calibrated_db": _r(np.mean(np.abs(mag_nl - self.mag_hw)), 2),
            },
            "cascade": {
                "s_vel_actuator_limit": _r(self.s_limit, 3),
                "s_vel_rule_of_thumb": self.s_rule,
                "s_vel_design": _r(self.s_design, 3),
                "Kd": _r(self.kd),
                "Kp": _r(self.kp),
                "poles": [str(complex(round(p.real, 2), round(p.imag, 2))) for p in self.poles],
                "step_overshoot_pct": _r(
                    overshoot_percent(self.step_run.theta[self.step_run.t >= 0.5], 0.0, DESIGN_STEP), 2
                ),
                "step_peak_voltage": _r(np.max(np.abs(self.step_run.u)), 2),
            },
            "adaptive": {k: _r(v, 3) for k, v in self.adaptive_metrics().items()}
            | {"inertia_ratio": _r(INERTIA_RATIO, 3), "t_switch_s": T_SWITCH},
        }


def _rls_trajectories(u: np.ndarray, y: np.ndarray, ts: float) -> tuple[np.ndarray, np.ndarray]:
    hist, _ = rls_identify(u, y, forgetting=1.0, p0=1e4)
    a1, b1 = hist[:, 0], hist[:, 1]
    valid = (a1 > -1.0) & (a1 < 0.0)
    K = np.full(len(a1), np.nan)
    T = np.full(len(a1), np.nan)
    K[valid] = b1[valid] / (1.0 + a1[valid])
    T[valid] = -ts / np.log(-a1[valid])
    return K, T


def servo_study(fast: bool = False) -> ServoStudy:
    sm = data.servo_static_map()
    u_s, w_s = sm["u_V"], sm["omega_rad_s"]
    k_lin = fit_gain_through_origin(u_s, w_s)
    smap = fit_quadratic_static_map(u_s, w_s)

    params = ServoParams()
    ts = params.ts
    t = np.arange(0.0, 15.0 + 0.5 * ts, ts)
    u = servo_excitation(t)
    y = run_open_loop(u, params)["omega"]
    pt1_arx = PT1.from_arx(arx_ls(u, y), ts)
    pt1_oe = fit_pt1_output_error(u, y, ts, pt1_arx)
    rls_K, rls_T = _rls_trajectories(u, y, ts)

    fr = data.servo_frequency_response()
    w_hw = fr["omega_rad_s"]
    w_grid = np.logspace(-1, 2, 400)
    w_lockin = np.geomspace(0.4, 40.0, 7 if fast else 13)
    periods = 4 if fast else 10

    def lockin(w: float) -> complex:
        tt, _, yy = sine_test(w, 1.0, params, periods=periods)
        return correlation_analysis(tt, yy, w, 1.0, t_start=tt[-1] - periods * 2 * math.pi / w)

    g_lockin = np.array([lockin(w) for w in w_lockin])
    g_lockin_hw = np.array([lockin(w) for w in w_hw])

    K, T = HARDWARE_PT1.K, HARDWARE_PT1.T
    s_limit = actuator_limited_pole(K, T, params.u_max, DESIGN_STEP)
    s_design = 0.95 * s_limit
    kd, kp = cascade_gains(K, T, s_design)

    def step_ref(tt: np.ndarray) -> np.ndarray:
        return np.where(tt >= 0.5, DESIGN_STEP, 0.0)

    def sine_ref(tt: np.ndarray) -> np.ndarray:
        return math.pi * np.sin(2.0 * math.pi / 6.0 * tt)

    def square_ref(tt: np.ndarray) -> np.ndarray:
        return square_wave(tt, 0.0, DESIGN_STEP, 2.0)

    def schedule(tt: float) -> float:
        return INERTIA_RATIO if tt >= T_SWITCH else 1.0

    t_end = 12.0
    step_run = run_position_control(CascadeController(kd, kp), step_ref, 3.0, params)
    sine_run = run_position_control(CascadeController(kd, kp), sine_ref, 12.0, params)
    fixed = run_position_control(CascadeController(kd, kp), square_ref, t_end, params, schedule)
    lambdas = (1.0, 0.995, 0.98) if not fast else (1.0, 0.98)
    forgetting_runs = {
        lam: run_position_control(
            AdaptiveCascadeController(HARDWARE_PT1, ts, forgetting=lam), square_ref, t_end, params, schedule
        )
        for lam in lambdas
    }
    return ServoStudy(
        static_u=u_s,
        static_w=w_s,
        k_linear=k_lin,
        static_map=smap,
        rms_linear=rmse(w_s, k_lin * u_s),
        rms_quadratic=rmse(w_s, smap.speed(u_s)),
        ts=ts,
        t_id=t,
        u_id=u,
        y_id=y,
        pt1_arx=pt1_arx,
        pt1_oe=pt1_oe,
        fit_arx=nrmse_fit(y, pt1_arx.simulate(u, ts)),
        fit_oe=nrmse_fit(y, pt1_oe.simulate(u, ts)),
        fit_hardware=nrmse_fit(y, HARDWARE_PT1.simulate(u, ts)),
        rls_t=t[1:],
        rls_K=rls_K,
        rls_T=rls_T,
        w_hw=w_hw,
        mag_hw=fr["magnitude_db"],
        phase_hw=fr["phase_deg"],
        w_grid=w_grid,
        g_linear=HARDWARE_PT1.frequency_response(w_grid),
        w_lockin=w_lockin,
        g_lockin=g_lockin,
        g_lockin_hw=g_lockin_hw,
        s_limit=s_limit,
        s_rule=rule_of_thumb_pole(T),
        s_design=s_design,
        kd=kd,
        kp=kp,
        poles=position_loop_poles(K, T, kd, kp),
        step_run=step_run,
        sine_run=sine_run,
        fixed_run=fixed,
        adaptive_run=forgetting_runs[0.98],
        forgetting_runs=forgetting_runs,
    )


# ============================================================================ pendulum
@dataclass
class PendulumStudy:
    """Propeller pendulum: estimation from hardware data, linearisation, identification."""

    lab: PendulumParams
    calibrated: PendulumParams
    free_fall: data.TimeSeries
    t_release: float
    phi0: float
    inertia_bound: float
    decay_sigma: float
    peaks_t: np.ndarray
    peaks_phi: np.ndarray
    fit_lab: float
    fit_viscous: FreeFallFit
    fit_drag: FreeFallFit
    sim_lab: np.ndarray
    sim_viscous: np.ndarray
    sim_drag: np.ndarray
    gyro_angle_deg: np.ndarray
    gyro_drift_deg_s: float
    imu_scale: float
    imu_bias_deg_s: float
    imu_corr: float
    raw_free_fall: data.TimeSeries
    static: dict[str, np.ndarray]
    thrust: np.ndarray
    thrust_gain: float
    com_experiment: float
    com_band: tuple[float, float]
    step: data.TimeSeries
    phi1: float
    phi2: float
    u1: float
    u2: float
    t_step: float
    step_overshoot: float
    fit_step_lab: float
    fit_step_no_thrust_damping: float
    fit_step_calibrated: float
    step_sim_lab: np.ndarray
    step_sim_calibrated: np.ndarray
    step_window: np.ndarray
    arx_ts: float
    arx_t: np.ndarray
    arx_y: np.ndarray
    arx_sim: np.ndarray
    arx: ArxModel
    arx_fit: float
    second_order: SecondOrderFit
    physical: dict[str, float]
    lin_ops_deg: np.ndarray
    lin_steps_deg: np.ndarray
    lin_error_deg: np.ndarray
    lin_peak_deg: np.ndarray
    chair_t: np.ndarray
    chair_angle: np.ndarray
    chair_turns: TurnCount

    def summary(self) -> dict[str, object]:
        g = self.lab.geometry
        cal = self.calibrated
        return {
            "geometry": {
                "l_up_m": _r(g.com(g.x_up)),
                "l_down_m": _r(g.com(g.x_down)),
                "J_up_kgm2": _r(g.inertia(g.x_up), 5),
                "J_down_kgm2": _r(g.inertia(g.x_down), 5),
                "a0_up_Nm": _r(self.lab.a0),
            },
            "free_fall": {
                "J_energy_bound": _r(self.inertia_bound, 5),
                "J_geometry": _r(self.lab.J, 5),
                "J_fit_viscous": _r(self.fit_viscous.params.J, 5),
                "J_fit_drag": _r(self.fit_drag.params.J, 5),
                "d_envelope": _r(2.0 * self.fit_drag.params.J * self.decay_sigma, 5),
                "d_fit_viscous": _r(self.fit_viscous.params.viscous, 5),
                "d_fit_drag": _r(self.fit_drag.params.viscous, 5),
                "c_fit_drag": float(f"{self.fit_drag.params.drag:.3g}"),
                "fit_lab_pct": _r(self.fit_lab, 1),
                "fit_viscous_pct": _r(self.fit_viscous.fit_percent, 1),
                "fit_drag_pct": _r(self.fit_drag.fit_percent, 1),
                "gyro_drift_deg_per_s": _r(self.gyro_drift_deg_s, 4),
                "imu_scale_vs_potentiometer": _r(self.imu_scale, 4),
                "imu_bias_deg_per_s": _r(self.imu_bias_deg_s, 3),
                "imu_correlation": _r(self.imu_corr, 4),
            },
            "statics": {
                "thrust_gain_N_per_pct": _r(self.thrust_gain, 5),
                "b0_Nm_per_pct": _r(self.thrust_gain * g.l_rotor, 5),
                "com_geometry_m": _r(g.com(g.x_up)),
                "com_two_position_m": _r(self.com_experiment),
                "com_two_position_band_m": [_r(v) for v in self.com_band],
            },
            "step_response": {
                "phi_start_deg": _r(math.degrees(self.phi1), 2),
                "phi_end_deg": _r(math.degrees(self.phi2), 2),
                "u_start_pct": _r(self.u1, 2),
                "u_end_pct": _r(self.u2, 2),
                "overshoot_pct": _r(self.step_overshoot, 1),
                "fit_lab_pct": _r(self.fit_step_lab, 1),
                "fit_without_thrust_damping_pct": _r(self.fit_step_no_thrust_damping, 1),
                "fit_calibrated_pct": _r(self.fit_step_calibrated, 1),
                "thrust_damping": float(f"{cal.thrust_damping:.3g}"),
                "rotor_tau_s": _r(cal.rotor_tau, 4),
                "arx2_fit_pct": _r(self.arx_fit, 1),
                "arx2_wn_rad_s": _r(self.second_order.wn, 3),
                "arx2_zeta": _r(self.second_order.zeta, 4),
                "arx2_dc_gain_deg_per_pct": _r(self.second_order.dc_gain, 4),
                "arx2_physical": {k: _r(v, 5) for k, v in self.physical.items()},
            },
            "linearisation": self._linearisation_summary(),
            "calibrated_model": {
                "J": _r(cal.J, 5),
                "viscous": cal.viscous,
                "drag": cal.drag,
                "thrust_damping": cal.thrust_damping,
                "thrust_gain": cal.thrust_gain,
                "rotor_tau": cal.rotor_tau,
                "deadband_pct": cal.deadband,
            },
            "gyro_chair": {
                "net_turns": _r(self.chair_turns.net, 3),
                "travelled_turns": _r(self.chair_turns.travelled, 3),
                "reversals": self.chair_turns.reversals,
            },
        }

    def _linearisation_summary(self) -> dict[str, object]:
        err, steps = self.lin_error_deg, self.lin_steps_deg
        row = int(np.argmin(np.abs(self.lin_ops_deg - OPERATING_POINT_DEG)))
        small = steps <= 2.0
        valid = np.isfinite(err)
        return {
            "max_error_deg_steps_up_to_2deg": _r(np.nanmax(err[:, small]), 3),
            "error_deg_at_operating_point": {
                f"+-{s:g}": _r(e, 3) for s, e in zip(steps, err[row], strict=True) if np.isfinite(e)
            },
            "cells_evaluated": int(valid.sum()),
            "cells_passing_horizontal": int(np.sum(valid & (self.lin_peak_deg > 90.0))),
        }


def _plateau(ts: data.TimeSeries, t0: float, t1: float) -> float:
    mask = (ts.t >= t0) & (ts.t <= t1)
    return float(np.mean(ts.phi[mask]))


def pendulum_study(fast: bool = False) -> PendulumStudy:
    lab = LAB_MODEL
    g = lab.geometry

    # free fall ------------------------------------------------------------------
    raw_ff = data.pendulum_free_fall()
    ff = raw_ff.resample(0.02 if fast else 0.005)
    t_rel, phi0 = detect_release(ff.t, ff.phi)
    after = ff.t > t_rel
    bound = energy_inertia_bound(g, lab.x, float(np.max(np.abs(ff.phid[after]))), phi0)
    sigma, peaks_t, peaks_phi = decay_rate(ff.t[after], ff.phi[after])
    sim_lab = simulate_free_fall(lab, ff.t, phi0, t_rel)
    fit_v = fit_free_fall(ff.t, ff.phi, lab, estimate_drag=False)
    fit_d = fit_free_fall(ff.t, ff.phi, lab, estimate_drag=True)

    gyro_angle = integrate_rate(raw_ff.t, raw_ff.phid_deg, initial=raw_ff.phi_deg[0])
    drift = float(np.polyfit(raw_ff.t, gyro_angle - raw_ff.phi_deg, 1)[0])
    fine_ff = raw_ff.resample(0.005)
    pot_rate = savgol_filter(fine_ff.phi_deg, 31, 3, deriv=1, delta=0.005)
    imu_scale, imu_bias = (float(v) for v in np.polyfit(pot_rate, fine_ff.phid_deg, 1))
    imu_corr = float(np.corrcoef(pot_rate, fine_ff.phid_deg)[0, 1])

    # statics --------------------------------------------------------------------
    st = data.pendulum_static_equilibrium()
    thrust = np.array(
        [float(thrust_from_angle(math.radians(p), x, g)) for x, p in zip(st["x_m"], st["phi_deg"], strict=True)]
    )
    active = st["u_pct"] > 5.0
    k = fit_gain_through_origin(st["u_pct"][active], thrust[active])
    up, down = st["x_m"] == g.x_up, st["x_m"] == g.x_down
    phi_up = math.radians(float(st["phi_deg"][up & (st["u_pct"] == 30.0)][0]))
    phi_down = math.radians(float(st["phi_deg"][down & (st["u_pct"] == 30.0)][0]))
    com_exp = com_from_two_positions(phi_up, phi_down, g.x_up, g.x_down, g)
    one = math.radians(1.0)
    band = sorted(
        com_from_two_positions(phi_up + a, phi_down + b, g.x_up, g.x_down, g) for a in (-one, one) for b in (-one, one)
    )

    calibrated = fit_d.params.with_(thrust_gain=k)

    # step response -------------------------------------------------------------
    step = data.pendulum_step_response().resample(0.02 if fast else 0.005)
    phi1 = _plateau(step, 25.0, 35.5)
    phi2 = _plateau(step, 55.0, 65.0)
    u1, u2 = calibrated.equilibrium_input(phi1), calibrated.equilibrium_input(phi2)
    win = (step.t >= 33.0) & (step.t <= 65.0)
    tt, yy = step.t[win], step.phi[win]

    def step_sim(p: PendulumParams, t_step: float) -> np.ndarray:
        return simulate(p, tt, np.where(tt < t_step, u1, u2), phi0=phi1)["phi"]

    sol = least_squares(
        lambda z: step_sim(calibrated.with_(thrust_damping=z[0], rotor_tau=z[1]), z[2]) - yy,
        x0=[4e-4, 0.1, 35.7],
        bounds=([0.0, 0.01, 35.0], [0.01, 1.0, 36.5]),
        x_scale=[1e-4, 0.05, 0.1],
    )
    d_u, tau_r, t_step = (float(v) for v in sol.x)
    calibrated = calibrated.with_(thrust_damping=d_u, rotor_tau=tau_r, deadband=CALIBRATED.deadband)
    lab_k = lab.with_(thrust_gain=k)
    sim_step_lab = step_sim(lab_k, t_step)
    sim_step_cal = step_sim(calibrated, t_step)

    # discrete second-order identification of the step ------------------------------
    # 50 ms sampling with a matching moving-average anti-alias filter: at faster
    # rates the equation-error estimate is dominated by sensor noise and biased.
    arx_ts = 0.05
    fine = data.pendulum_step_response().resample(0.005)
    n_avg = round(arx_ts / 0.005)
    smooth = np.convolve(fine.phi_deg, np.ones(n_avg) / n_avg, mode="same")
    grid = np.arange(33.0, 65.0, arx_ts)
    dy = np.interp(grid, fine.t, smooth) - math.degrees(phi1)
    du = np.where(grid < t_step, 0.0, u2 - u1)
    arx = arx_ls(du, dy, na=2, nb=2, nk=1)
    arx_sim = arx.simulate(du)
    so2 = continuous_second_order(arx, arx_ts)
    phys = physical_from_second_order(so2, calibrated.J)
    physical = {
        "J": phys["a2"],
        "a1": phys["a1"],
        "a0_lin": phys["a0"],
        "b0_Nm_per_pct": phys["b"] * math.pi / 180.0,
        "model_a0_lin": calibrated.a0 * math.cos(phi2),
        "model_b0_Nm_per_pct": calibrated.b0,
    }

    # linearisation validity map -----------------------------------------------------
    lin_model = calibrated.with_(rotor_tau=0.0, deadband=0.0)
    ops = np.arange(10.0, 71.0, 10.0) if fast else np.arange(5.0, 71.0, 5.0)
    steps = np.array([1.0, 2.0, 5.0, 10.0, 20.0] if fast else [1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0])
    err = np.full((len(ops), len(steps)), np.nan)
    peak = np.full_like(err, np.nan)
    for i, op in enumerate(ops):
        for j, half in enumerate(steps):
            if op - half >= 0.0 and op + half <= 75.0:
                run = step_comparison(lin_model, math.radians(op), math.radians(half))
                err[i, j] = math.degrees(float(np.max(np.abs(run["nonlinear"] - run["linear"]))))
                peak[i, j] = math.degrees(float(np.max(run["nonlinear"])))

    chair = data.gyro_chair_spin()

    return PendulumStudy(
        lab=lab,
        calibrated=calibrated,
        free_fall=ff,
        t_release=t_rel,
        phi0=phi0,
        inertia_bound=bound,
        decay_sigma=sigma,
        peaks_t=peaks_t,
        peaks_phi=peaks_phi,
        fit_lab=nrmse_fit(ff.phi, sim_lab),
        fit_viscous=fit_v,
        fit_drag=fit_d,
        sim_lab=sim_lab,
        sim_viscous=simulate_free_fall(fit_v.params, ff.t, phi0, fit_v.t_release),
        sim_drag=simulate_free_fall(fit_d.params, ff.t, phi0, fit_d.t_release),
        gyro_angle_deg=gyro_angle,
        gyro_drift_deg_s=drift,
        imu_scale=imu_scale,
        imu_bias_deg_s=imu_bias,
        imu_corr=imu_corr,
        raw_free_fall=raw_ff,
        static=st,
        thrust=thrust,
        thrust_gain=k,
        com_experiment=com_exp,
        com_band=(band[0], band[-1]),
        step=step,
        phi1=phi1,
        phi2=phi2,
        u1=u1,
        u2=u2,
        t_step=t_step,
        step_overshoot=overshoot_percent(yy[tt >= t_step], phi1, phi2),
        fit_step_lab=nrmse_fit(yy, sim_step_lab),
        fit_step_no_thrust_damping=nrmse_fit(yy, step_sim(fit_d.params.with_(thrust_gain=k), t_step)),
        fit_step_calibrated=nrmse_fit(yy, sim_step_cal),
        step_sim_lab=sim_step_lab,
        step_sim_calibrated=sim_step_cal,
        step_window=tt,
        arx_ts=arx_ts,
        arx_t=grid,
        arx_y=dy,
        arx_sim=arx_sim,
        arx=arx,
        arx_fit=nrmse_fit(dy, arx_sim),
        second_order=so2,
        physical=physical,
        lin_ops_deg=ops,
        lin_steps_deg=steps,
        lin_error_deg=err,
        lin_peak_deg=peak,
        chair_t=chair["t_s"],
        chair_angle=chair["yaw_rad"],
        chair_turns=count_turns(chair["yaw_rad"]),
    )


# ============================================================================ control
@dataclass
class ControlStudy:
    """Pendulum tracking control: design, comparison, bandwidth sweep and twin runs."""

    params: PendulumParams
    gains: PIDGains
    coeffs: PIPCoefficients
    poles: np.ndarray
    runs: dict[str, LoopResult]
    sweep_omega: np.ndarray
    sweep_cost: dict[str, np.ndarray]
    sweep_peak: dict[str, np.ndarray]
    twin_run: LoopResult
    hero_run: LoopResult
    twin_openloop: dict[str, np.ndarray]
    timing: dict[str, float] | None
    timing_t: np.ndarray | None

    def summary(self) -> dict[str, object]:
        costs = {name: run.costs() for name, run in self.runs.items()}
        twin = self.twin_run.costs()
        out: dict[str, object] = {
            "design": {
                "operating_point_deg": OPERATING_POINT_DEG,
                "omega_rad_s": DESIGN_OMEGA,
                "zeta": DESIGN_ZETA,
                "alpha": DESIGN_ALPHA,
                "Kp": _r(self.gains.kp),
                "Ki": _r(self.gains.ki),
                "Kd": _r(self.gains.kd),
                "q0_PI": _r(self.coeffs.q0_pi),
                "q1_PI": _r(self.coeffs.q1_pi),
                "q0_P": _r(self.coeffs.q0_p),
                "Ts_s": self.coeffs.ts,
                "poles": [str(complex(round(p.real, 2), round(p.imag, 2))) for p in self.poles],
            },
            "model_runs": {
                name: {
                    "J_e_deg2": _r(c["J_e"], 3),
                    "J_u_pct2": _r(c["J_u"], 1),
                    "u_peak_pct": _r(float(np.max(self.runs[name].u)), 1),
                }
                for name, c in costs.items()
            },
            "twin_run": {"J_e_deg2": _r(twin["J_e"], 3), "J_u_pct2": _r(twin["J_u"], 1)},
            "bandwidth_sweep": {
                "omega_rad_s": [float(w) for w in self.sweep_omega],
                **{f"J_e_deg2_{name}": [float(f"{v:.3g}") for v in cost] for name, cost in self.sweep_cost.items()},
            },
        }
        if self.timing is not None:
            out["realtime_timing"] = {k: _r(v, 5) for k, v in self.timing.items()}
        return out


def _controllers(params: PendulumParams, gains: PIDGains, coeffs: PIPCoefficients) -> dict[str, object]:
    return {
        "pip": DigitalPIP(coeffs),
        "pip_ff_aw": DigitalPIP(coeffs, feedforward=gravity_feedforward(params), anti_windup=True),
        "pid_1dof": DiscretePID(gains, 0.001),
    }


def twin_staircase(twin: PendulumTwin, duration: float = 120.0) -> dict[str, np.ndarray]:
    """Open-loop exploration of the twin: rotor and slider staircases."""
    levels = [
        (10.0, 5.0, 50.0),
        (20.0, 10.0, 100.0),
        (40.0, 20.0, 150.0),
        (60.0, 30.0, 200.0),
        (80.0, 40.0, 250.0),
        (100.0, 50.0, 250.0),
        (math.inf, 0.0, 0.0),
    ]
    twin.reset()
    log = {k: [] for k in ("t", "phi", "u", "x")}
    state = twin.measure()
    while state.t <= duration + 1e-9:
        u, dx = next((u, dx) for t_end, u, dx in levels if state.t < t_end)
        twin.command(u, dx)
        log["t"].append(state.t)
        log["phi"].append(state.phi_deg)
        log["u"].append(u)
        log["x"].append(twin.x)
        state = twin.advance()
    return {k: np.asarray(v) for k, v in log.items()}


def _twin_session(controller: object, realtime: bool, duration: float, slider: float | object = 0.0) -> LoopResult:
    twin = PendulumTwin(CALIBRATED, TwinConfig(seed=7))
    with TwinServer(twin, port=0, realtime=realtime) as server, TwinClient(port=server.port) as client:
        return run_closed_loop(client, controller, duration=duration, deltax_mm=slider)


def control_study(fast: bool = False, realtime_seconds: float = 0.0) -> ControlStudy:
    params = CALIBRATED
    lin = params.linearize(math.radians(OPERATING_POINT_DEG)).in_degrees()
    gains = place_pid(lin, DESIGN_OMEGA, DESIGN_ZETA, DESIGN_ALPHA)
    coeffs = PIPCoefficients.from_pid(gains, CONTROL_TS)
    t_end = 60.0 if fast else 120.0

    runs = {}
    for name, ctrl in _controllers(params, gains, coeffs).items():
        ts = 0.001 if name == "pid_1dof" else CONTROL_TS
        runs[name] = simulate_loop(params, ctrl, t_end=t_end, ts=ts, substeps=1 if ts < CONTROL_TS else 5)

    omegas = np.array([3.0, 5.0, 8.0, 12.0] if fast else [2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0])
    sweep_cost = {"pip": np.empty(len(omegas)), "pip_ff_aw": np.empty(len(omegas))}
    sweep_peak = {"pip": np.empty(len(omegas)), "pip_ff_aw": np.empty(len(omegas))}
    window = (30.0, min(100.0, t_end))
    for i, w in enumerate(omegas):
        q = PIPCoefficients.from_pid(place_pid(lin, w, DESIGN_ZETA, DESIGN_ALPHA), CONTROL_TS)
        for name, ctrl in (
            ("pip", DigitalPIP(q)),
            ("pip_ff_aw", DigitalPIP(q, feedforward=gravity_feedforward(params), anti_windup=True)),
        ):
            run = simulate_loop(params, ctrl, t_end=t_end)
            sweep_cost[name][i] = run.costs(window)["J_e"]
            sweep_peak[name][i] = float(np.max(run.u))

    best = DigitalPIP(coeffs, feedforward=gravity_feedforward(params), anti_windup=True)
    twin_run = _twin_session(best, realtime=False, duration=t_end)
    hero = DigitalPIP(coeffs, feedforward=gravity_feedforward(params), anti_windup=True)
    hero_run = _twin_session(hero, realtime=False, duration=120.0, slider=hero_slider_mm)
    openloop = twin_staircase(PendulumTwin(CALIBRATED, TwinConfig(seed=3)), duration=t_end)

    timing = timing_t = None
    if realtime_seconds > 0:
        live = DigitalPIP(coeffs, feedforward=gravity_feedforward(params), anti_windup=True)
        rt = _twin_session(live, realtime=True, duration=realtime_seconds)
        timing, timing_t = sampling_report(rt.t, nominal=CONTROL_TS), rt.t

    return ControlStudy(
        params=params,
        gains=gains,
        coeffs=coeffs,
        poles=closed_loop_poles(lin, gains),
        runs=runs,
        sweep_omega=omegas,
        sweep_cost=sweep_cost,
        sweep_peak=sweep_peak,
        twin_run=twin_run,
        hero_run=hero_run,
        twin_openloop=openloop,
        timing=timing,
        timing_t=timing_t,
    )
