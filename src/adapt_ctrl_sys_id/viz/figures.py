"""Static result figures (PNG) rendered from the pipeline studies."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

from ..frequency import to_db_deg
from ..pendulum import step_comparison
from ..pipeline import (
    DESIGN_OMEGA,
    DESIGN_STEP,
    INERTIA_RATIO,
    T_SWITCH,
    ControlStudy,
    PendulumStudy,
    ServoStudy,
)
from ..servo import HARDWARE_PT1, ServoParams, actuator_limited_pole, rule_of_thumb_pole
from .drawing import PendulumArtist
from .style import COLORS, apply_style, save

__all__ = ["render_all"]

C = COLORS


def _note(ax, text: str, xy=(0.98, 0.04), ha="right", va="bottom", size=9) -> None:
    ax.text(
        *xy,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=size,
        color=C["ink"],
        zorder=10,
        bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": C["grid"], "lw": 1.0, "alpha": 0.95},
    )


def _shade_change(ax, t0: float, t1: float, label: str | None = None) -> None:
    ax.axvspan(t0, t1, color="#fef3c7", alpha=0.55, lw=0, zorder=0)
    if label:
        ax.text(
            t0 + 0.08,
            0.04,
            label,
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="#92400e",
        )


def _twin_axis(ax, color: str, label: str):
    other = ax.twinx()
    other.set_ylabel(label, color=color)
    other.grid(False)
    other.spines["right"].set_visible(True)
    return other


# ============================================================================ servo
def fig_servo_static(s: ServoStudy, out: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.22})
    u = np.linspace(-9.5, 9.5, 400)
    ax1.plot(u, s.k_linear * u, color=C["lab"], ls="--", lw=1.6, label=f"line through origin, K = {s.k_linear:.2f}")
    ax1.plot(
        u,
        s.static_map.speed(u),
        color=C["model"],
        lw=2.2,
        label=rf"$u = \omega/K_0 + \beta\,\omega|\omega|$,  $K_0$ = {s.static_map.K0:.1f}",
    )
    ax1.scatter(
        s.static_u,
        s.static_w,
        s=34,
        color=C["measured"],
        zorder=4,
        edgecolor="white",
        lw=0.8,
        label="hardware readings",
    )
    ax1.set(
        xlabel="voltage u [V]", ylabel=r"steady-state speed $\omega$ [rad/s]", title="Measured static characteristic"
    )
    ax1.legend(loc="upper left")
    _note(ax1, f"RMS error\nstraight line   {s.rms_linear:.2f} rad/s\nquadratic map  {s.rms_quadratic:.2f} rad/s")

    mask = np.abs(s.static_u) >= 0.25
    gain = s.static_w[mask] / s.static_u[mask]
    uu = np.linspace(0.25, 9.5, 300)
    ax2.plot(uu, s.static_map.speed(uu) / uu, color=C["model"], lw=2.2, label="quadratic map")
    ax2.axhline(
        HARDWARE_PT1.K, color=C["ink"], lw=1.2, ls=":", label=f"PT1 gain from step tests ({HARDWARE_PT1.K:.1f})"
    )
    ax2.axhline(s.k_linear, color=C["lab"], lw=1.4, ls="--", label=f"straight-line fit ({s.k_linear:.1f})")
    ax2.scatter(
        np.abs(s.static_u[mask]),
        gain,
        s=34,
        color=C["measured"],
        zorder=4,
        edgecolor="white",
        lw=0.8,
        label=r"$\omega/u$ per reading",
    )
    ax2.set(xlabel="|u| [V]", ylabel=r"gain $\omega/u$ [rad/s/V]", title="The gain falls with speed", ylim=(17, 33))
    ax2.legend(loc="lower left")
    ax2.annotate(
        "small-signal tests\nsee a higher gain",
        xy=(1.05, 29.6),
        xytext=(2.7, 30.6),
        fontsize=9,
        color=C["muted"],
        arrowprops={"arrowstyle": "->", "color": C["muted"]},
    )
    return save(
        fig,
        out / "servo_static_map.png",
        "Servo static map: almost linear, but not quite",
        "22 steady-state hardware readings. A quadratic loss term halves the error of the straight-line fit.",
    )


def fig_servo_identification(s: ServoStudy, out: Path) -> Path:
    fig = plt.figure(figsize=(12, 7.2))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.8, 1.6, 1.1], hspace=0.55, wspace=0.22)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.step(s.t_id, s.u_id, where="post", color=C["effort"], lw=1.8)
    ax0.set(ylabel="u [V]", title="Excitation: small and large steps of both signs", xlim=(0, 15))
    ax1 = fig.add_subplot(gs[1, :], sharex=ax0)
    ax1.plot(s.t_id, s.y_id, color=C["measured"], lw=1.2, label="measured speed")
    ax1.plot(
        s.t_id,
        s.pt1_oe.simulate(s.u_id, s.ts),
        color=C["model"],
        lw=1.8,
        ls="--",
        label=f"PT1 output-error fit: K = {s.pt1_oe.K:.2f}, T = {s.pt1_oe.T:.3f} s  ({s.fit_oe:.1f} %)",
    )
    ax1.set(ylabel=r"$\omega$ [rad/s]", title="Model vs. measurement")
    ax1.legend(loc="lower left")
    _note(
        ax1,
        f"hardware PT1: K = {HARDWARE_PT1.K:.2f}, T = {HARDWARE_PT1.T:.3f} s\n"
        f"its fit on this record: {s.fit_hardware:.1f} %",
        xy=(0.99, 0.95),
        va="top",
    )
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(s.rls_t, s.rls_K, color=C["measured"], lw=1.6)
    ax2.axhline(s.pt1_arx.K, color=C["ink"], ls=":", lw=1.2)
    ax2.set(ylim=(15, 35), xlabel="time [s]", ylabel=r"$\hat K$ [rad/s/V]", title="RLS gain estimate", xlim=(0, 15))
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.plot(s.rls_t, s.rls_T, color=C["model"], lw=1.6)
    ax3.axhline(s.pt1_arx.T, color=C["ink"], ls=":", lw=1.2)
    ax3.set(
        ylim=(0.0, 0.6), xlabel="time [s]", ylabel=r"$\hat T$ [s]", title="RLS time-constant estimate", xlim=(0, 15)
    )
    _note(ax3, f"converges to the batch LS solution\nK = {s.pt1_arx.K:.2f}, T = {s.pt1_arx.T:.3f} s")
    return save(
        fig,
        out / "servo_identification.png",
        "Identifying the servo: batch, output-error and recursive least squares",
        "Standard 15 s excitation on the calibrated virtual servo (2048-count encoder, filtered speed estimate).",
    )


def fig_servo_bode(s: ServoStudy, out: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.8), sharex=True, gridspec_kw={"hspace": 0.22})
    mag_lin, ph_lin = to_db_deg(s.g_linear)
    mag_v, _ = to_db_deg(s.g_lockin)
    ax1.semilogx(s.w_grid, mag_lin, color=C["lab"], lw=2.0, ls="--", label="linear PT1 (hardware K, T)")
    ax1.semilogx(
        s.w_lockin,
        mag_v,
        "o-",
        color=C["model"],
        lw=1.8,
        ms=5,
        mfc="white",
        label="calibrated virtual servo, 1 V lock-in",
    )
    ax1.semilogx(s.w_hw, s.mag_hw, "D", color=C["measured"], ms=9, mec="white", label="hardware sine tests", zorder=5)
    ax1.axvline(HARDWARE_PT1.cutoff, color=C["faint"], lw=1, ls=":")
    ax1.text(HARDWARE_PT1.cutoff * 1.06, 2, r"$\omega_c = 1/T$", color=C["muted"], fontsize=9)
    ax1.set(ylabel="|G| [dB]", ylim=(0, 32), title="Magnitude")
    ax1.legend(loc="lower left")
    ax2.semilogx(s.w_grid, ph_lin, color=C["lab"], lw=2.0, ls="--")
    ax2.semilogx(s.w_lockin, np.degrees(np.angle(s.g_lockin)), "o-", color=C["model"], lw=1.8, ms=5, mfc="white")
    ax2.semilogx(s.w_hw, s.phase_hw, "D", color=C["measured"], ms=9, mec="white", zorder=5)
    ax2.axvline(HARDWARE_PT1.cutoff, color=C["faint"], lw=1, ls=":")
    ax2.set(
        xlabel=r"$\omega$ [rad/s]", ylabel="phase [deg]", ylim=(-100, 5), yticks=[-90, -60, -45, -30, 0], title="Phase"
    )
    summ = s.summary()["frequency_validation"]
    _note(
        ax1,
        f"mean |error| at the hardware points\nlinear PT1   {summ['mean_abs_error_linear_db']:.2f} dB\n"
        f"calibrated   {summ['mean_abs_error_calibrated_db']:.2f} dB",
        xy=(0.99, 0.95),
        va="top",
    )
    return save(
        fig,
        out / "servo_bode.png",
        "Frequency-response validation against independent hardware sine tests",
        "The virtual servo was calibrated only on the static map and the PT1 time constant; "
        "the sine tests are unseen data.",
    )


def fig_servo_cascade(s: ServoStudy, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), gridspec_kw={"width_ratios": [1, 1.05, 1.05], "wspace": 0.55})
    ax = axes[0]
    T = np.linspace(0.08, 1.0, 400)
    limit = np.array([actuator_limited_pole(HARDWARE_PT1.K, t, 10.0, DESIGN_STEP) for t in T])
    ax.fill_between(T, limit, -80, color="#fee2e2", alpha=0.8, lw=0)
    ax.plot(T, limit, color=C["bad"], lw=2.2, label=r"limit $-\sqrt{2 K u_{max} / (r_0 T)}$")
    ax.step(
        T,
        [rule_of_thumb_pole(t) for t in T],
        color=C["muted"],
        lw=1.3,
        where="mid",
        label="rule of thumb round(-44.3 + 43 T)",
    )
    for t_, lbl, off in (
        (HARDWARE_PT1.T, "light disk", (14, -26)),
        (HARDWARE_PT1.T * INERTIA_RATIO, "heavy disk", (14, -30)),
    ):
        sv = 0.95 * actuator_limited_pole(HARDWARE_PT1.K, t_, 10.0, DESIGN_STEP)
        ax.plot(t_, sv, "o", color=C["good"], ms=9, mec="white", zorder=5)
        ax.annotate(
            f"{lbl}: {sv:.1f}",
            (t_, sv),
            xytext=off,
            textcoords="offset points",
            fontsize=8.5,
            arrowprops={"arrowstyle": "-", "color": C["muted"], "lw": 0.8},
        )
    ax.text(0.42, -62, "|u| would exceed 10 V", color=C["bad"], fontsize=8.5)
    ax.set(
        xlabel="time constant T [s]",
        ylabel=r"velocity pole $s_{vel}$ [1/s]",
        ylim=(-80, 0),
        xlim=(0.08, 1.0),
        title="Pole selection",
    )
    ax.legend(loc="upper left", fontsize=8)

    run = s.step_run
    ax = axes[1]
    ax.plot(run.t, run.ref, color=C["reference"], lw=1.3, ls="--", label="reference")
    ax.plot(run.t, run.theta, color=C["measured"], lw=2.0, label="angle")
    ax.set(xlabel="time [s]", ylabel="angle [rad]", title=r"Step of $\pi/2$", xlim=(0.3, 2.2))
    axu = _twin_axis(ax, C["effort"], "voltage [V]")
    axu.plot(run.t, run.u, color=C["effort"], lw=1.2, alpha=0.85)
    axu.axhline(10, color=C["bad"], lw=0.8, ls=":")
    axu.set_ylim(-11, 11)
    ax.legend(loc="center right")
    summ = s.summary()["cascade"]
    _note(
        ax,
        f"Kd = {s.kd:.3f}, Kp = {s.kp:.2f}\npoles {summ['poles'][0].strip('()')}\n"
        f"overshoot {summ['step_overshoot_pct']:.1f} %, peak {summ['step_peak_voltage']:.1f} V",
        xy=(0.97, 0.05),
    )

    run = s.sine_run
    ax = axes[2]
    ax.plot(run.t, run.ref, color=C["reference"], lw=1.3, ls="--", label="reference")
    ax.plot(run.t, run.theta, color=C["measured"], lw=2.0, label="angle")
    ax.plot(run.t, 10 * (run.ref - run.theta), color=C["bad"], lw=1.2, label="error x 10")
    ax.set(xlabel="time [s]", ylabel="angle [rad]", title="Sine tracking", xlim=(0, 12))
    ax.legend(loc="lower left", fontsize=8)
    return save(
        fig,
        out / "servo_cascade.png",
        "Cascade position control with an actuator-aware pole choice",
        r"Inner P loop on speed, outer P loop on angle: $\zeta = 1/\sqrt{2}$ by construction, "
        r"$|u(0^+)| \leq 10$ V by the pole limit.",
    )


def fig_servo_adaptive(s: ServoStudy, out: Path) -> Path:
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 8.4), sharex=True, gridspec_kw={"height_ratios": [1.5, 1, 1], "hspace": 0.32}
    )
    fixed, adapt = s.fixed_run, s.adaptive_run
    metrics = s.adaptive_metrics()
    ax = axes[0]
    _shade_change(ax, T_SWITCH, 12.0, "heavier disk")
    ax.plot(fixed.t, fixed.ref, color=C["reference"], lw=1.2, ls="--", label="reference")
    ax.plot(
        fixed.t,
        fixed.theta,
        color=C["bad"],
        lw=1.8,
        label=f"fixed gains (overshoot {metrics['fixed_overshoot_pct']:.0f} %)",
    )
    ax.plot(
        adapt.t,
        adapt.theta,
        color=C["good"],
        lw=2.0,
        label=f"adaptive (overshoot {metrics['adaptive_overshoot_pct']:.1f} %)",
    )
    ax.set(ylabel="angle [rad]", title="Position tracking", ylim=(-0.3, 2.1))
    ax.legend(loc="upper left", ncols=3, fontsize=8.5)
    ax = axes[1]
    _shade_change(ax, T_SWITCH, 12.0)
    tau = ServoParams().tau
    ax.plot(adapt.t, tau * adapt.inertia_scale, color=C["reference"], lw=1.3, ls="--", label=r"true plant $\tau$")
    palette = {1.0: C["faint"], 0.995: C["accent"], 0.98: C["good"]}
    for lam, run in s.forgetting_runs.items():
        ax.plot(run.t, run.T_hat, color=palette.get(lam, C["alt"]), lw=1.8, label=rf"$\lambda$ = {lam:g}")
    ax.set(
        ylabel=r"$\hat T$ [s]",
        title="Estimated time constant: forgetting trades noise for tracking speed",
        ylim=(0.2, 0.8),
    )
    ax.legend(loc="upper left", ncols=4, fontsize=8.5)
    ax = axes[2]
    _shade_change(ax, T_SWITCH, 12.0)
    ax.plot(adapt.t, adapt.kp, color=C["measured"], lw=1.8)
    ax.set_ylabel("Kp [1/s]", color=C["measured"])
    axd = _twin_axis(ax, C["model"], "Kd [V s/rad]")
    axd.plot(adapt.t, adapt.kd, color=C["model"], lw=1.8)
    ax.set(xlabel="time [s]", title="Controller gains follow the plant", xlim=(0, 12))
    lines = [Line2D([], [], color=C["measured"], lw=1.8), Line2D([], [], color=C["model"], lw=1.8)]
    ax.legend(lines, ["Kp (outer loop)", "Kd (inner loop)"], loc="center left", fontsize=8.5)
    return save(
        fig,
        out / "servo_adaptive.png",
        "Adaptive cascade: the disk gets 1.85x heavier at t = 4.9 s",
        "RLS on encoder increments (50 Hz, forgetting 0.98) re-identifies K and T; "
        "the design rule recomputes Kd and Kp at every update.",
    )


# ============================================================================ pendulum
def fig_pendulum_schematic(p: PendulumStudy, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    g = p.lab.geometry
    phi = math.radians(32)
    x = g.x_up + 0.08
    art = PendulumArtist(ax, g, show_angle=True, arc_radius=0.25)
    art.update(phi, x=x, u=65.0)
    d = np.array([math.sin(phi), -math.cos(phi)])
    n = np.array([math.cos(phi), math.sin(phi)])
    com = d * g.com(x)
    ax.annotate(
        "",
        xy=com + np.array([0.0, -0.17]),
        xytext=com,
        arrowprops={"arrowstyle": "-|>", "color": C["measured"], "lw": 2.2},
    )
    ax.plot(*com, "o", color=C["measured"], ms=7, mec="white", zorder=8)
    ax.text(com[0] + 0.02, com[1] - 0.15, r"$m\,g$", color=C["measured"], fontsize=12)
    ax.text(com[0] + 0.035, com[1] + 0.012, "CoG", color=C["measured"], fontsize=9)
    hub = d * g.l_rotor + 0.33 * n
    ax.text(hub[0] - 0.03, hub[1] + 0.035, r"$F_r = k\,u$", color=C["model"], fontsize=12)
    mid = math.radians(-90 + 16)
    ax.text(
        0.31 * math.cos(mid),
        0.31 * math.sin(mid),
        r"$\varphi$",
        color=C["model"],
        fontsize=15,
        ha="center",
        va="center",
    )

    def dim(r1: float, offset: float, label: str) -> None:
        a, b = -offset * n, d * r1 - offset * n
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "<->", "color": C["muted"], "lw": 1.1})
        mid = 0.5 * (a + b) - 0.045 * n
        ax.text(*mid, label, color=C["muted"], fontsize=10.5, ha="center", va="center")

    dim(x, 0.07, r"$x$")
    dim(g.com(x), 0.15, r"$l(x)$")
    dim(g.l_rotor, 0.23, r"$l_r$")
    ax.text(0.03, 0.03, "pivot", color=C["muted"], fontsize=9)
    ax.set(xlim=(-0.42, 0.62), ylim=(-0.66, 0.16), aspect="equal")
    ax.axis("off")
    ax.text(
        -0.41,
        -0.64,
        r"$J(x)\,\ddot\varphi = k\,l_r\,u - m\,g\,l(x)\sin\varphi - d\,\dot\varphi$",
        fontsize=12,
        color=C["ink"],
    )
    return save(fig, out / "pendulum_schematic.png", dpi=170)


def fig_pendulum_free_fall(p: PendulumStudy, out: Path) -> Path:
    fig = plt.figure(figsize=(13, 5.4))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.35, 1.35, 1], hspace=0.5, wspace=0.3)
    ff = p.free_fall
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(ff.t, ff.phi_deg, color=C["measured"], lw=1.1, label="measured")
    ax.plot(
        ff.t, np.degrees(p.sim_lab), color=C["lab"], lw=1.3, ls="--", label=f"geometry model ({p.fit_lab:.0f} % fit)"
    )
    ax.plot(
        ff.t,
        np.degrees(p.sim_drag),
        color=C["model"],
        lw=1.3,
        label=f"calibrated model ({p.fit_drag.fit_percent:.0f} % fit)",
    )
    after = ff.t > p.t_release
    env = math.degrees(p.phi0) * np.exp(-p.decay_sigma * (ff.t[after] - p.t_release))
    ax.plot(ff.t[after], env, color=C["muted"], lw=1, ls=":", label="exponential envelope")
    ax.set(ylabel=r"$\varphi$ [deg]", xlim=(0, 60), ylim=(-100, 118), title="Full record")
    ax.legend(loc="upper right", ncols=4, fontsize=8.2)
    ax = fig.add_subplot(gs[1, :2])
    zoom = (ff.t > 38) & (ff.t < 46)
    ax.plot(ff.t[zoom], ff.phi_deg[zoom], color=C["measured"], lw=2)
    ax.plot(ff.t[zoom], np.degrees(p.sim_lab[zoom]), color=C["lab"], lw=1.6, ls="--")
    ax.plot(ff.t[zoom], np.degrees(p.sim_drag[zoom]), color=C["model"], lw=1.8)
    ax.set(xlabel="time [s]", ylabel=r"$\varphi$ [deg]", title="Zoom 38-46 s: the geometry model is now out of phase")
    ax = fig.add_subplot(gs[:, 2])
    ax.plot(ff.phi_deg[after], ff.phid_deg[after], color=C["measured"], lw=0.7, alpha=0.9, label="measured")
    phid_model = np.gradient(np.degrees(p.sim_drag), ff.t)
    ax.plot(np.degrees(p.sim_drag[after]), phid_model[after], color=C["model"], lw=0.7, alpha=0.8, label="calibrated")
    ax.set(xlabel=r"$\varphi$ [deg]", ylabel=r"$\dot\varphi$ [deg/s]", title="Phase portrait", ylim=(-760, 640))
    ax.legend(loc="upper right")
    fd = p.fit_drag.params
    _note(
        ax,
        f"J = {fd.J:.4f} kg m²  (geometry {p.lab.J:.4f})\nenergy bound  J ≤ {p.inertia_bound:.4f}\n"
        f"d = {fd.viscous:.4f} N m s,  c = {fd.drag:.1e}",
        xy=(0.03, 0.03),
        ha="left",
        size=8.3,
    )
    return save(
        fig,
        out / "pendulum_free_fall.png",
        "Free fall from 91 deg: geometry alone drifts out of phase",
        "Hardware record (potentiometer). Fitted: inertia J, viscous damping d and quadratic air drag c.",
    )


def fig_pendulum_statics(p: PendulumStudy, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), gridspec_kw={"width_ratios": [1.1, 1.1, 0.9], "wspace": 0.32})
    st = p.static
    g = p.lab.geometry
    up, down = st["x_m"] == g.x_up, st["x_m"] == g.x_down
    cal = p.calibrated
    uu = np.linspace(0, 70, 300)
    ax = axes[0]
    for mask, x, col, lbl in ((up, g.x_up, C["measured"], "slider up"), (down, g.x_down, C["model"], "slider down")):
        model = cal.with_(x=x, deadband=0.0)
        reach = model.b0 * uu / model.a0 < 0.995
        ax.plot(uu[reach], [math.degrees(model.equilibrium_angle(v)) for v in uu[reach]], color=col, lw=1.8, alpha=0.85)
        ax.scatter(st["u_pct"][mask], st["phi_deg"][mask], s=40, color=col, edgecolor="white", zorder=4, label=lbl)
    ax.set(xlabel="rotor command u [%]", ylabel=r"equilibrium $\varphi$ [deg]", title="Equilibrium angle")
    ax.legend(loc="upper left")
    _note(ax, r"$\sin\varphi = k\,l_r\,u\,/\,(m\,g\,l(x))$", xy=(0.97, 0.05))
    ax = axes[1]
    ax.scatter(st["u_pct"], p.thrust, s=40, color=C["measured"], edgecolor="white", zorder=4, label="from equilibria")
    ax.plot(uu, p.thrust_gain * uu, color=C["model"], lw=2, label=f"k = {p.thrust_gain:.4f} N/%")
    ax.axvspan(0, 6, color="#fee2e2", alpha=0.7, lw=0)
    ax.text(0.7, 2.9, "dead\nband", fontsize=8, color=C["bad"])
    ax.set(xlabel="rotor command u [%]", ylabel=r"rotor force $F_r$ [N]", title="Thrust is linear in the command")
    ax.legend(loc="lower right")
    ax = axes[2]
    lo, hi = p.com_band
    ax.set_axisbelow(True)
    ax.bar([0, 1], [g.com(g.x_up), p.com_experiment], color=[C["lab"], C["model"]], width=0.55)
    ax.errorbar(
        [1],
        [p.com_experiment],
        yerr=[[p.com_experiment - lo], [hi - p.com_experiment]],
        color=C["ink"],
        capsize=8,
        lw=1.6,
    )
    ax.set_xticks([0, 1], ["geometry", "two-position\nexperiment"])
    ax.set(ylabel="l(x) [m]", title="Centre of gravity", ylim=(0, 0.38))
    ax.text(1, hi + 0.01, "±1 deg reading error", ha="center", fontsize=8.5, color=C["muted"])
    return save(
        fig,
        out / "pendulum_statics.png",
        "Static equilibria: thrust map and centre of gravity",
        "Hardware equilibrium angles for constant rotor commands at two slider positions.",
    )


def fig_pendulum_step(p: PendulumStudy, out: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.0), gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.4})
    st = p.step
    win = (st.t >= 33) & (st.t <= 65)
    ax = axes[0]
    ax.plot(st.t[win], st.phi_deg[win], color=C["measured"], lw=1.2, label="measured")
    ax.plot(
        p.step_window,
        np.degrees(p.step_sim_lab),
        color=C["lab"],
        lw=1.3,
        ls="--",
        label=f"geometry baseline ({p.fit_step_lab:.0f} % fit)",
    )
    ax.plot(
        p.step_window,
        np.degrees(p.step_sim_calibrated),
        color=C["model"],
        lw=1.6,
        label=f"calibrated + thrust damping + rotor lag ({p.fit_step_calibrated:.0f} % fit)",
    )
    ax.set(ylabel=r"$\varphi$ [deg]", title="Nonlinear models", xlim=(33, 65), ylim=(15, 68))
    ax.legend(loc="upper right", fontsize=8.5)
    cal = p.calibrated
    _note(
        ax,
        f"fitted: d_u = {cal.thrust_damping:.1e} N m s/%,  rotor lag {cal.rotor_tau * 1000:.0f} ms\n"
        f"without d_u the fit drops to {p.fit_step_no_thrust_damping:.0f} %\n"
        "the ±5 deg ripple after 40 s is not modelled",
        xy=(0.99, 0.05),
    )
    ax = axes[1]
    ax.plot(p.arx_t, p.arx_y, color=C["measured"], lw=1.4, label=r"measured $\Delta\varphi$ (50 ms, anti-aliased)")
    ax.plot(p.arx_t, p.arx_sim, color=C["alt"], lw=1.8, ls="--", label=f"discrete ARX(2,2) ({p.arx_fit:.0f} % fit)")
    ax.set(
        xlabel="time [s]", ylabel=r"$\Delta\varphi$ [deg]", title="Discrete second-order identification", xlim=(33, 65)
    )
    ax.legend(loc="upper right", fontsize=8.5)
    ph = p.physical
    _note(
        ax,
        rf"$\omega_n$ = {p.second_order.wn:.2f} rad/s,  $\zeta$ = {p.second_order.zeta:.3f}"
        "\n"
        f"a0,lin = {ph['a0_lin']:.3f} N m   (model {ph['model_a0_lin']:.3f})\n"
        f"b0 = {ph['b0_Nm_per_pct']:.4f} N m/%   (model {ph['model_b0_Nm_per_pct']:.4f})",
        xy=(0.99, 0.05),
        size=8.5,
    )
    return save(
        fig,
        out / "pendulum_step.png",
        "Powered step 22 → 42 deg: propeller inflow adds damping",
        "Hardware record. Without a thrust-dependent damping term no model reproduces the fast initial decay.",
    )


def fig_pendulum_linearization(p: PendulumStudy, out: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.25})
    err = np.ma.masked_invalid(p.lin_error_deg.T)
    mesh = ax1.pcolormesh(
        p.lin_ops_deg, p.lin_steps_deg, err, cmap="magma_r", shading="nearest", norm=LogNorm(vmin=0.05, vmax=20.0)
    )
    cs = ax1.contour(
        p.lin_ops_deg, p.lin_steps_deg, err.filled(np.nan), levels=[0.5, 1, 2, 5], colors="white", linewidths=1.2
    )
    ax1.clabel(cs, fmt="%g°", fontsize=8.5)
    ops, steps = np.meshgrid(p.lin_ops_deg, p.lin_steps_deg)
    over = np.nan_to_num(p.lin_peak_deg.T) > 90.0
    if over.any():
        ax1.scatter(
            ops[over],
            steps[over],
            marker="x",
            s=46,
            color=C["accent"],
            lw=2.0,
            label="nonlinear response passes 90 deg",
        )
        ax1.legend(loc="upper left", fontsize=8.5, labelcolor=C["ink"], facecolor="white", frameon=True)
    cb = fig.colorbar(mesh, ax=ax1, pad=0.02, extend="max")
    cb.set_ticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    cb.set_label("max |error| [deg]")
    ax1.set(xlabel="operating point [deg]", ylabel="half step height [deg]", title="Validity map")
    ax1.grid(False)
    ax1.set_facecolor("#f1f5f9")
    cal = p.calibrated.with_(rotor_tau=0.0, deadband=0.0)
    for half, col in ((2.0, C["good"]), (15.0, C["bad"])):
        run = step_comparison(cal, math.radians(40.0), math.radians(half), t_end=12.0)
        ax2.plot(run["t"], np.degrees(run["nonlinear"]) - 40, color=col, lw=2, label=f"nonlinear, ±{half:g} deg")
        ax2.plot(run["t"], np.degrees(run["linear"]) - 40, color=col, lw=1.6, ls="--", label=f"linear, ±{half:g} deg")
    ax2.set(xlabel="time [s]", ylabel=r"$\varphi - 40$ [deg]", title="Steps around 40 deg")
    ax2.legend(loc="lower right", ncols=2, fontsize=8.5)
    return save(
        fig,
        out / "pendulum_linearization.png",
        "Where is the linear model good enough?",
        "Worst-case angle error of the linearised model for the same physical input step (calibrated model).",
    )


def fig_sensors(p: PendulumStudy, out: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"wspace": 0.25})
    raw = p.raw_free_fall
    ax1.plot(raw.t, raw.phi_deg, color=C["measured"], lw=0.8, label="potentiometer angle")
    ax1.plot(raw.t, p.gyro_angle_deg, color=C["model"], lw=0.8, alpha=0.85, label="integrated IMU rate")
    ax1.plot(raw.t, p.gyro_angle_deg - raw.phi_deg, color=C["bad"], lw=1.8, label="difference")
    ax1.set(
        xlabel="time [s]", ylabel="angle [deg]", title="Free fall: potentiometer vs. integrated IMU rate", xlim=(0, 60)
    )
    ax1.legend(loc="lower left", ncols=3, fontsize=8.5)
    _note(
        ax1,
        f"IMU axis inverted (corr {p.imu_corr:+.3f} after sign fix)\nscale {p.imu_scale:.3f}, "
        f"bias {p.imu_bias_deg_s:+.2f} deg/s\nintegrated drift {p.gyro_drift_deg_s:+.2f} deg/s",
        xy=(0.98, 0.95),
        va="top",
    )
    turns = p.chair_angle / (2 * math.pi)
    ax2.plot(p.chair_t, turns, color=C["measured"], lw=2)
    ax2.axhline(0, color=C["faint"], lw=1)
    tc = p.chair_turns
    ax2.set(xlabel="time [s]", ylabel="turns", title="Smartphone gyro on a spinning chair")
    _note(
        ax2,
        f"net {tc.net:+.2f} turns\ntravelled {tc.travelled:.2f} turns\nreversals {tc.reversals}",
        xy=(0.97, 0.95),
        va="top",
    )
    return save(
        fig,
        out / "sensors.png",
        "Sensor sanity checks",
        "Cross-checking two sensors exposes an inverted IMU axis and a rate bias; "
        "counting turns needs the sign of every rotation.",
    )


# ============================================================================ control
LABELS = {
    "pip": ("PI-P cascade", C["measured"]),
    "pip_ff_aw": ("PI-P + feedforward + anti-windup", C["good"]),
    "pid_1dof": ("1-DOF PID, same gains", C["bad"]),
}


def fig_control_comparison(c: ControlStudy, out: Path) -> Path:
    fig, axes = plt.subplots(
        2, 1, figsize=(12.5, 7.0), sharex=True, gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.1}
    )
    ax = axes[0]
    base = c.runs["pip"]
    ax.axvspan(30, min(100, base.t[-1]), color="#f1f5f9", zorder=0)
    ax.text(31, -1.5, "evaluation window for the costs", fontsize=8.5, color=C["muted"], va="bottom")
    ax.plot(base.t, base.ref, color=C["reference"], lw=1.4, ls="--", label="reference")
    for name in ("pid_1dof", "pip", "pip_ff_aw"):
        lbl, col = LABELS[name]
        run = c.runs[name]
        ax.plot(run.t, run.phi, color=col, lw=1.6, label=f"{lbl}   J_e = {run.costs()['J_e']:.2f} deg²")
    ax.set(ylabel=r"$\varphi$ [deg]", ylim=(-3, 62))
    ax.legend(loc="upper left", fontsize=8.5)
    if base.t[-1] > 69:
        axin = ax.inset_axes([0.64, 0.07, 0.34, 0.42])
        for name in ("pid_1dof", "pip", "pip_ff_aw"):
            run = c.runs[name]
            m = (run.t > 64.5) & (run.t < 68.5)
            axin.plot(run.t[m], run.phi[m], color=LABELS[name][1], lw=1.6)
        m = (base.t > 64.5) & (base.t < 68.5)
        axin.plot(base.t[m], base.ref[m], color=C["reference"], ls="--", lw=1.2)
        axin.set_title("30 → 50 deg step", fontsize=8.5, loc="left")
        axin.tick_params(labelsize=7.5)
    ax = axes[1]
    for name in ("pid_1dof", "pip", "pip_ff_aw"):
        run = c.runs[name]
        ax.plot(run.t, run.u, color=LABELS[name][1], lw=1.1, alpha=0.9)
    ax.set(xlabel="time [s]", ylabel="rotor command u [%]", ylim=(-3, 105), xlim=(0, base.t[-1]))
    return save(
        fig,
        out / "control_comparison.png",
        "Tracking the reference profile on the calibrated nonlinear model",
        rf"Digital controllers at 100 Hz, rotor limited to 0-100 %. Pole placement at $\omega$ = {DESIGN_OMEGA:g} "
        "rad/s around 40 deg.",
    )


def fig_control_sweep(c: ControlStudy, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    w = np.asarray(c.sweep_omega, dtype=float)
    diverged = 1e3  # J_e above this means the loop lost the pendulum
    band_y = 20.0
    diverged_at: list[float] = []
    for name in ("pip", "pip_ff_aw"):
        lbl, col = LABELS[name]
        cost = np.asarray(c.sweep_cost[name], dtype=float)
        ok = np.isfinite(cost) & (cost < diverged)
        ax.plot(w[ok], cost[ok], "o-", color=col, lw=2, ms=6, mec="white", label=f"{lbl}: tracking cost")
        if not ok.all():
            ax.plot(w[~ok], np.full(int((~ok).sum()), band_y), "X", color=col, ms=11, mec="white", zorder=5)
            diverged_at.extend(w[~ok])
    ax.set_yscale("log")
    ax.set_ylim(0.7, 30.0)
    ax.set_yticks([1, 2, 5, 10])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    if diverged_at:
        ax.axhspan(14.0, 30.0, color="#fee2e2", lw=0, zorder=0)
        ax.text(
            min(diverged_at) - 0.3,
            band_y,
            "diverges: integrator winds up\nagainst the 0-100 % rotor limits",
            ha="right",
            va="center",
            fontsize=8.5,
            color=C["bad"],
        )
    ax.set(xlabel=r"design bandwidth $\omega$ [rad/s]", ylabel=r"$J_e$ [deg²]  (log scale)")
    ax.axvline(DESIGN_OMEGA, color=C["muted"], ls=":", lw=1.2)
    ax.text(
        DESIGN_OMEGA + 0.15,
        0.03,
        "chosen design",
        color=C["muted"],
        fontsize=9,
        transform=ax.get_xaxis_transform(),
        va="bottom",
    )
    ax2 = _twin_axis(ax, C["muted"], "peak rotor command [%]  (dotted)")
    for name in ("pip", "pip_ff_aw"):
        ax2.plot(w, c.sweep_peak[name], "s:", color=LABELS[name][1], lw=1.2, ms=4, alpha=0.7)
    ax2.set_ylim(0, 160)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2, fontsize=8.8)
    protected = np.asarray(c.sweep_cost["pip_ff_aw"], dtype=float)[w >= DESIGN_OMEGA]
    subtitle = f"With feedforward + anti-windup J_e ≤ {np.max(protected):.1f} deg² up to ω = {w.max():g} rad/s."
    if diverged_at:
        subtitle = f"Plain PI-P diverges from ω = {min(diverged_at):g} rad/s. " + subtitle
    return save(fig, out / "control_sweep.png", "Bandwidth sweep", subtitle)


def fig_twin(c: ControlStudy, out: Path) -> Path:
    fig, axes = plt.subplots(
        2, 2, figsize=(13.5, 6.4), gridspec_kw={"width_ratios": [1.6, 1], "hspace": 0.42, "wspace": 0.3}
    )
    run = c.twin_run
    costs = run.costs()
    ax = axes[0, 0]
    ax.plot(run.t, run.ref, color=C["reference"], lw=1.3, ls="--", label="reference")
    ax.plot(run.t, run.phi, color=C["good"], lw=1.2, label="twin angle (noisy sensor)")
    ax.set(
        ylabel=r"$\varphi$ [deg]", title="PI-P + feedforward + anti-windup through the TCP client", xlim=(0, run.t[-1])
    )
    ax.legend(loc="upper left", fontsize=8.5)
    _note(ax, f"J_e = {costs['J_e']:.2f} deg²    J_u = {costs['J_u']:.0f} %²", xy=(0.98, 0.05))
    ax = axes[1, 0]
    ax.plot(run.t, run.u, color=C["effort"], lw=0.9)
    ax.set(xlabel="time [s]", ylabel="u [%]", xlim=(0, run.t[-1]), ylim=(-3, 105))
    ol = c.twin_openloop
    ax = axes[0, 1]
    ax.plot(ol["t"], ol["phi"], color=C["measured"], lw=1.0)
    ax.set(ylabel=r"$\varphi$ [deg]", title="Open loop: rotor and slider staircases", xlim=(0, ol["t"][-1]))
    ax = axes[1, 1]
    ax.step(ol["t"], ol["u"], color=C["effort"], lw=1.6, where="post")
    ax.set(xlabel="time [s]", ylabel="u [%]", xlim=(0, ol["t"][-1]))
    axx = _twin_axis(ax, C["gold"], "slider x [mm]")
    axx.plot(ol["t"], 1000 * ol["x"], color=C["gold"], lw=1.8)
    return save(
        fig,
        out / "twin.png",
        "The digital twin: closed loop over TCP and open-loop exploration",
        "Calibrated physics + rotor lag + dead band + sensor noise, served on localhost:10815.",
    )


def fig_timing(c: ControlStudy, out: Path) -> Path | None:
    if c.timing is None or c.timing_t is None:
        return None
    dt_ms = np.diff(c.timing_t) * 1000
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.25})
    ax1.plot(c.timing_t[1:], dt_ms, color=C["measured"], lw=0.8)
    ax1.axhline(c.timing["nominal"] * 1000, color=C["model"], lw=1.2, ls="--")
    ax1.set(xlabel="time [s]", ylabel="interval [ms]", title="Interval per sample")
    counts, _, _ = ax2.hist(dt_ms, bins=60, color=C["measured"], alpha=0.85)
    ax2.set(xlabel="interval [ms]", ylabel="count", title="Distribution", yscale="log")
    ax2.set_ylim(0.7, max(float(counts.max()), 1.0) * 200)  # headroom for the statistics box
    tm = c.timing
    _note(
        ax2,
        f"nominal {tm['nominal'] * 1000:.1f} ms\nmean {tm['mean_interval'] * 1000:.2f} ms\n"
        f"p99 {tm['p99_interval'] * 1000:.2f} ms, max {tm['max_interval'] * 1000:.1f} ms\n"
        f"off by > 0.5 ms: {100 * tm['deviation_share']:.1f} %",
        xy=(0.97, 0.95),
        va="top",
    )
    return save(
        fig,
        out / "twin_timing.png",
        "Real-time behaviour of the twin",
        "Sample intervals of a live session (wall-clock paced server, Python client).",
    )


def render_all(servo: ServoStudy, pend: PendulumStudy, ctrl: ControlStudy, out: Path) -> list[Path]:
    apply_style()
    paths = [
        fig_servo_static(servo, out),
        fig_servo_identification(servo, out),
        fig_servo_bode(servo, out),
        fig_servo_cascade(servo, out),
        fig_servo_adaptive(servo, out),
        fig_pendulum_schematic(pend, out),
        fig_pendulum_free_fall(pend, out),
        fig_pendulum_statics(pend, out),
        fig_pendulum_step(pend, out),
        fig_pendulum_linearization(pend, out),
        fig_sensors(pend, out),
        fig_control_comparison(ctrl, out),
        fig_control_sweep(ctrl, out),
        fig_twin(ctrl, out),
    ]
    timing = fig_timing(ctrl, out)
    if timing is not None:
        paths.append(timing)
    return paths
