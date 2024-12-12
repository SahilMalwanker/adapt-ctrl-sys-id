"""Animated GIFs for the README (matplotlib frames, Pillow encoder with one shared palette)."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

from ..pendulum import CALIBRATED
from ..pipeline import (
    HERO_SLIDER_TIME,
    T_SWITCH,
    ControlStudy,
    PendulumStudy,
    ServoStudy,
    hero_slider_mm,
)
from ..servo import ServoParams
from ..twin import TwinConfig
from .drawing import DiskArtist, PendulumArtist
from .style import COLORS, apply_style

__all__ = ["adaptive_gif", "free_fall_gif", "hero_gif", "render_animations", "write_gif"]

C = COLORS


def _grab(fig) -> Image.Image:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(np.ascontiguousarray(rgba[..., :3]))


def write_gif(frames: list[Image.Image], path: Path, fps: float, colors: int = 96) -> Path:
    """Encode frames with a single median-cut palette (no dithering = small, crisp files)."""
    picks = np.linspace(0, len(frames) - 1, min(10, len(frames))).astype(int)
    width, height = frames[0].size
    montage = Image.new("RGB", (width, height * len(picks)))
    for i, k in enumerate(picks):
        montage.paste(frames[k], (0, i * height))
    palette_img = montage.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    palette = palette_img.getpalette()[: 3 * colors]
    for i in range(colors):
        if min(palette[3 * i : 3 * i + 3]) > 244:
            palette[3 * i : 3 * i + 3] = [255, 255, 255]
    palette_img.putpalette(palette)
    quantized = [f.quantize(palette=palette_img, dither=Image.Dither.NONE) for f in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
        disposal=1,
    )
    return path


def _brand(fig, title: str, subtitle: str) -> None:
    fig.text(0.018, 0.955, title, fontsize=15, fontweight="bold", color=C["ink"], va="top")
    fig.text(0.018, 0.885, subtitle, fontsize=10, color=C["muted"], va="top")


# ============================================================================ hero
def hero_gif(c: ControlStudy, out: Path, fps: float = 16.0, speed: float = 8.0) -> Path:
    """Propeller pendulum twin tracking the reference profile (slider moved at 100 s)."""
    apply_style()
    run = c.hero_run
    g = CALIBRATED.geometry
    speed_mps = TwinConfig().slider_speed
    fig = plt.figure(figsize=(9.6, 4.8), dpi=100)
    _brand(
        fig,
        "adapt-ctrl-sys-id",
        "Digital PI-P cascade + gravity feedforward driving the propeller-pendulum twin over TCP",
    )
    ax_m = fig.add_axes([0.0, 0.0, 0.43, 0.84])
    ax_m.set(xlim=(-0.24, 0.74), ylim=(-0.64, 0.12), aspect="equal")
    ax_m.axis("off")
    art = PendulumArtist(ax_m, g, show_reference=True, arc_radius=0.12)
    readout = ax_m.text(
        0.72,
        -0.62,
        "",
        fontsize=9.5,
        color=C["muted"],
        family="DejaVu Sans Mono",
        ha="right",
        va="bottom",
        multialignment="left",
    )
    ax_m.legend(
        handles=[Line2D([], [], color=C["good"], ls="--", lw=2.0), Line2D([], [], color=C["model"], lw=2.4)],
        labels=["reference angle", "rotor thrust"],
        loc="upper right",
        fontsize=8.5,
    )

    ax_p = fig.add_axes([0.5, 0.44, 0.475, 0.38])
    ax_u = fig.add_axes([0.5, 0.11, 0.475, 0.24], sharex=ax_p)
    ax_p.plot(run.t, run.ref, color=C["reference"], lw=1.2, ls="--", alpha=0.55)
    (line_phi,) = ax_p.plot([], [], color=C["good"], lw=2.0)
    (line_u,) = ax_u.plot([], [], color=C["effort"], lw=1.2)
    cursor_p = ax_p.axvline(0, color=C["faint"], lw=1)
    cursor_u = ax_u.axvline(0, color=C["faint"], lw=1)
    ax_p.set(xlim=(0, run.t[-1]), ylim=(-3, 60), ylabel=r"$\varphi$ [deg]")
    ax_p.tick_params(labelbottom=False)
    ax_u.set(ylim=(-5, 105), ylabel="u [%]", xlabel="time [s]")
    ax_p.axvline(HERO_SLIDER_TIME, color=C["gold"], lw=1.2, ls=":")
    ax_p.text(HERO_SLIDER_TIME - 1.5, 55, "slider moves down", fontsize=8, color="#b45309", ha="right", va="top")

    def slider_x(t: float) -> float:
        target = g.slider_position(hero_slider_mm(t))
        moved = max(0.0, t - HERO_SLIDER_TIME) * speed_mps
        return min(target, g.x_up + moved) if t >= HERO_SLIDER_TIME else g.x_up

    frames = []
    for i, t in enumerate(np.arange(0.0, run.t[-1] + 1e-9, speed / fps)):
        k = min(int(np.searchsorted(run.t, t)), len(run.t) - 1)
        art.update(math.radians(run.phi[k]), x=slider_x(t), u=run.u[k], phi_ref=math.radians(run.ref[k]), spin=1.7 * i)
        line_phi.set_data(run.t[: k + 1], run.phi[: k + 1])
        line_u.set_data(run.t[: k + 1], run.u[: k + 1])
        cursor_p.set_xdata([t, t])
        cursor_u.set_xdata([t, t])
        readout.set_text(f"t = {t:5.1f} s  ({speed:g}x)\nphi = {run.phi[k]:5.1f} deg\nu   = {run.u[k]:5.1f} %")
        frames.append(_grab(fig))
    plt.close(fig)
    return write_gif(frames, out / "hero.gif", fps)


# ============================================================================ adaptive
def adaptive_gif(s: ServoStudy, out: Path, fps: float = 20.0, speed: float = 1.0) -> Path:
    """Fixed-gain vs adaptive cascade when the disk inertia jumps."""
    apply_style()
    fixed, adapt = s.fixed_run, s.adaptive_run
    fig = plt.figure(figsize=(9.6, 4.5), dpi=100)
    _brand(
        fig,
        "Adaptive position control",
        "The disk becomes 1.85x heavier at t = 4.9 s; RLS + design rule re-tune the cascade",
    )
    ax_d = fig.add_axes([0.0, 0.03, 0.43, 0.76])
    ax_d.set(xlim=(-2.55, 2.55), ylim=(-1.55, 1.65), aspect="equal")
    ax_d.axis("off")
    disks = {
        "fixed": DiskArtist(ax_d, (-1.25, -0.05), 0.78, C["bad"]),
        "adaptive": DiskArtist(ax_d, (1.25, -0.05), 0.78, C["good"]),
    }
    ax_d.text(-1.25, 1.35, "fixed gains", ha="center", fontsize=11, fontweight="bold", color=C["bad"])
    ax_d.text(1.25, 1.35, "adaptive", ha="center", fontsize=11, fontweight="bold", color=C["good"])
    banner = ax_d.text(0, -1.45, "", ha="center", fontsize=9.5, color="#92400e")
    clock = ax_d.text(-2.45, -1.45, "", fontsize=9, color=C["muted"], family="DejaVu Sans Mono")

    ax_t = fig.add_axes([0.5, 0.42, 0.475, 0.38])
    ax_k = fig.add_axes([0.5, 0.11, 0.475, 0.22], sharex=ax_t)
    for ax in (ax_t, ax_k):
        ax.axvspan(T_SWITCH, fixed.t[-1], color="#fef3c7", alpha=0.55, lw=0, zorder=0)
    ax_t.plot(fixed.t, fixed.ref, color=C["reference"], lw=1.1, ls="--", alpha=0.6)
    (l_fixed,) = ax_t.plot([], [], color=C["bad"], lw=1.8, label="fixed")
    (l_adapt,) = ax_t.plot([], [], color=C["good"], lw=2.0, label="adaptive")
    ax_t.set(xlim=(0, fixed.t[-1]), ylim=(-0.3, 2.1), ylabel="angle [rad]")
    ax_t.tick_params(labelbottom=False)
    ax_t.legend(loc="upper right", ncols=2, fontsize=8.5)
    ax_k.plot(adapt.t, ServoParams().tau * adapt.inertia_scale, color=C["reference"], lw=1.1, ls="--", alpha=0.6)
    (l_T,) = ax_k.plot([], [], color=C["good"], lw=1.8)
    ax_k.set(ylim=(0.2, 0.8), ylabel=r"$\hat T$ [s]", xlabel="time [s]")
    cursors = [ax.axvline(0, color=C["faint"], lw=1) for ax in (ax_t, ax_k)]

    frames = []
    heavy = False
    for t in np.arange(0.0, fixed.t[-1] + 1e-9, speed / fps):
        k = min(int(np.searchsorted(fixed.t, t)), len(fixed.t) - 1)
        if t >= T_SWITCH and not heavy:
            disks["fixed"].set_style("#b91c1c", 0.95)
            disks["adaptive"].set_style("#047857", 0.95)
            banner.set_text("      heavier disk: inertia x1.85")
            heavy = True
        disks["fixed"].update(fixed.theta[k], fixed.ref[k])
        disks["adaptive"].update(adapt.theta[k], adapt.ref[k])
        l_fixed.set_data(fixed.t[: k + 1], fixed.theta[: k + 1])
        l_adapt.set_data(adapt.t[: k + 1], adapt.theta[: k + 1])
        l_T.set_data(adapt.t[: k + 1], adapt.T_hat[: k + 1])
        for cur in cursors:
            cur.set_xdata([t, t])
        clock.set_text(f"t = {t:4.1f} s")
        frames.append(_grab(fig))
    plt.close(fig)
    return write_gif(frames, out / "adaptive_servo.gif", fps)


# ============================================================================ free fall
def free_fall_gif(
    p: PendulumStudy,
    out: Path,
    fps: float = 15.0,
    segments: tuple[tuple[float, float], ...] = ((0.0, 9.0), (36.0, 45.0)),
) -> Path:
    """Measured free fall vs geometry-only and calibrated models (two time windows)."""
    apply_style()
    ff = p.free_fall
    series = {
        "measured": (ff.phi, C["ink"], 1.0),
        "geometry": (p.sim_lab, C["lab"], 0.55),
        "calibrated": (p.sim_drag, C["model"], 0.75),
    }
    fig = plt.figure(figsize=(9.6, 4.5), dpi=100)
    _brand(
        fig,
        "Calibrating the pendulum from a free fall",
        "Hardware record vs. the geometry-only model and the model fitted to the data",
    )
    ax_m = fig.add_axes([0.0, 0.0, 0.42, 0.86])
    ax_m.set(xlim=(-0.58, 0.58), ylim=(-0.6, 0.12), aspect="equal")
    ax_m.axis("off")
    arts = {
        "geometry": PendulumArtist(
            ax_m,
            color=C["lab"],
            accent=C["lab"],
            alpha=0.55,
            show_stand=False,
            show_thrust=False,
            show_angle=False,
            zorder=3,
        ),
        "calibrated": PendulumArtist(
            ax_m,
            color=C["model"],
            accent=C["model"],
            alpha=0.8,
            show_stand=False,
            show_thrust=False,
            show_angle=False,
            zorder=5,
        ),
        "measured": PendulumArtist(ax_m, show_thrust=False, show_angle=False, zorder=7),
    }
    clock = ax_m.text(-0.56, -0.58, "", fontsize=10, color=C["muted"], family="DejaVu Sans Mono")

    ax_p = fig.add_axes([0.48, 0.12, 0.495, 0.66])
    lines = {
        name: ax_p.plot(
            [], [], color=col, lw=2.2 if name == "measured" else 1.7, ls="--" if name == "geometry" else "-", label=name
        )[0]
        for name, (_, col, _) in series.items()
    }
    ax_p.set(ylim=(-100, 118), ylabel=r"$\varphi$ [deg]", xlabel="time [s]")
    ax_p.legend(loc="upper left", ncols=3, fontsize=8.5)
    fits = ax_p.text(
        0.99,
        0.04,
        f"fit: geometry {p.fit_lab:.0f} %   calibrated {p.fit_drag.fit_percent:.0f} %",
        transform=ax_p.transAxes,
        ha="right",
        fontsize=9,
        color=C["muted"],
    )
    fits.set_zorder(10)
    banner = ax_p.text(0.5, 0.8, "", transform=ax_p.transAxes, ha="center", fontsize=10, color="#92400e")

    frames = []
    for seg_index, (t0, t1) in enumerate(segments):
        ax_p.set_xlim(t0, t1)
        banner.set_text("" if seg_index == 0 else f"... {t0:.0f} s after release: the geometry model is out of phase")
        for t in np.arange(t0, t1 + 1e-9, 1.0 / fps):
            k = min(int(np.searchsorted(ff.t, t)), len(ff.t) - 1)
            start = int(np.searchsorted(ff.t, t0))
            for name, (phi, _, _) in series.items():
                arts[name].update(float(phi[k]))
                lines[name].set_data(ff.t[start : k + 1], np.degrees(phi[start : k + 1]))
            clock.set_text(f"t = {t:5.2f} s")
            frames.append(_grab(fig))
    plt.close(fig)
    return write_gif(frames, out / "free_fall.gif", fps)


def render_animations(servo: ServoStudy, pend: PendulumStudy, ctrl: ControlStudy, out: Path) -> list[Path]:
    from .banner import banner_gif

    return [banner_gif(servo, ctrl, out), hero_gif(ctrl, out), adaptive_gif(servo, out), free_fall_gif(pend, out)]
