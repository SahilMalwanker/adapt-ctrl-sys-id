"""Reusable mechanism drawings (pendulum side view, servo disk top view)."""

from __future__ import annotations

import math

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, Polygon, Wedge

from ..pendulum import Geometry
from .style import COLORS

__all__ = ["DiskArtist", "PendulumArtist"]


def _rotated_rect(center: np.ndarray, along: np.ndarray, length: float, width: float) -> np.ndarray:
    normal = np.array([-along[1], along[0]])
    half_l, half_w = 0.5 * length * along, 0.5 * width * normal
    return np.array(
        [center - half_l - half_w, center + half_l - half_w, center + half_l + half_w, center - half_l + half_w]
    )


class PendulumArtist:
    """Side view of the propeller pendulum: pivot, beam, slider, rotor and thrust."""

    def __init__(
        self,
        ax: Axes,
        geometry: Geometry | None = None,
        color: str = COLORS["ink"],
        accent: str = COLORS["model"],
        alpha: float = 1.0,
        show_stand: bool = True,
        show_thrust: bool = True,
        show_reference: bool = False,
        show_angle: bool = True,
        arc_radius: float = 0.15,
        zorder: int = 3,
    ) -> None:
        self.ax = ax
        self.g = geometry or Geometry()
        self.alpha = alpha
        self.show_thrust = show_thrust
        self.show_angle = show_angle
        self.arc_radius = arc_radius
        self.color = color
        self.accent = accent
        z = zorder
        if show_stand:
            ax.add_patch(
                Polygon(
                    [[-0.09, 0.075], [0.09, 0.075], [0.025, 0.0], [-0.025, 0.0]],
                    closed=True,
                    fc="#cbd5e1",
                    ec="none",
                    zorder=1,
                )
            )
            ax.plot([-0.16, 0.16], [0.075, 0.075], color="#94a3b8", lw=3, solid_capstyle="round", zorder=1)
            ax.plot([0.0, 0.0], [0.0, -0.62], color="#cbd5e1", lw=1.0, ls=(0, (4, 4)), zorder=0)
        self.ref_line = Line2D([], [], color=COLORS["good"], lw=2.2, ls=(0, (5, 3)), alpha=0.9, zorder=z - 1)
        self.ref_line.set_visible(show_reference)
        ax.add_line(self.ref_line)
        self.beam = Line2D([], [], color=color, lw=6, solid_capstyle="round", alpha=alpha, zorder=z)
        ax.add_line(self.beam)
        self.slider = Polygon(np.zeros((4, 2)), closed=True, fc="#475569", ec=color, lw=1.2, alpha=alpha, zorder=z + 1)
        ax.add_patch(self.slider)
        self.motor = Polygon(np.zeros((4, 2)), closed=True, fc="#334155", ec=color, lw=1.0, alpha=alpha, zorder=z + 1)
        ax.add_patch(self.motor)
        self.blur = Ellipse((0, 0), 0.2, 0.03, fc=accent, ec="none", alpha=0.18 * alpha, zorder=z + 1)
        ax.add_patch(self.blur)
        self.prop = Line2D([], [], color=color, lw=2.2, alpha=alpha, zorder=z + 2)
        ax.add_line(self.prop)
        self.pivot = Circle((0, 0), 0.016, fc="white", ec=color, lw=2, zorder=z + 3, alpha=alpha)
        ax.add_patch(self.pivot)
        self.thrust = FancyArrowPatch(
            (0, 0),
            (0, 0),
            arrowstyle="-|>,head_length=6,head_width=3.5",
            mutation_scale=2.2,
            color=accent,
            lw=2.4,
            zorder=z + 2,
            alpha=alpha,
        )
        self.thrust.set_visible(show_thrust)
        ax.add_patch(self.thrust)
        self.arc: Arc | None = None

    def update(
        self, phi: float, x: float | None = None, u: float = 0.0, phi_ref: float | None = None, spin: float = 0.0
    ) -> None:
        """Redraw for angle ``phi`` [rad], slider ``x`` [m], rotor command ``u`` [%]."""
        g = self.g
        x = g.x_up if x is None else x
        d = np.array([math.sin(phi), -math.cos(phi)])
        n = np.array([math.cos(phi), math.sin(phi)])
        tip = d * g.l_beam
        self.beam.set_data([0.0, tip[0]], [0.0, tip[1]])
        self.slider.set_xy(_rotated_rect(d * x, d, 0.055, 0.075))
        rotor = d * g.l_rotor
        self.motor.set_xy(_rotated_rect(rotor, d, 0.06, 0.05))
        hub = rotor + 0.045 * n
        half = 0.5 * (0.08 + 0.12 * abs(math.cos(spin)))
        self.prop.set_data([hub[0] - half * d[0], hub[0] + half * d[0]], [hub[1] - half * d[1], hub[1] + half * d[1]])
        self.blur.set_center(hub)
        self.blur.set_angle(math.degrees(math.atan2(d[1], d[0])))
        self.blur.set_alpha(min(0.35, 0.05 + 0.004 * u) * self.alpha)
        if self.show_thrust:
            length = 0.28 * min(max(u, 0.0), 100.0) / 100.0
            self.thrust.set_positions(tuple(hub + 0.02 * n), tuple(hub + (0.02 + length) * n))
            self.thrust.set_visible(length > 0.004)
        if phi_ref is not None:
            dr = np.array([math.sin(phi_ref), -math.cos(phi_ref)])
            self.ref_line.set_data([0.0, dr[0] * g.l_rotor], [0.0, dr[1] * g.l_rotor])
        if self.show_angle:
            if self.arc is not None:
                self.arc.remove()
            lo, hi = sorted((-90.0, -90.0 + math.degrees(phi)))
            size = 2.0 * self.arc_radius
            self.arc = Arc(
                (0, 0), size, size, theta1=lo, theta2=hi, color=COLORS["model"], lw=1.6, zorder=2, alpha=self.alpha
            )
            self.ax.add_patch(self.arc)


class DiskArtist:
    """Top view of a servo inertia disk with angle pointer and reference marker."""

    def __init__(
        self, ax: Axes, center: tuple[float, float] = (0.0, 0.0), radius: float = 1.0, color: str = COLORS["measured"]
    ) -> None:
        self.ax = ax
        self.center = np.asarray(center, dtype=float)
        self.radius = radius
        self.disk = Circle(center, radius, fc=color, ec=COLORS["ink"], lw=1.5, alpha=0.18, zorder=2)
        ax.add_patch(self.disk)
        self.rim = Circle(center, radius, fc="none", ec=color, lw=4, zorder=3)
        ax.add_patch(self.rim)
        for k in range(12):
            a = k * math.pi / 6.0
            r0 = 1.08 if k % 3 == 0 else 1.12
            p0 = self.center + radius * r0 * np.array([math.cos(a), math.sin(a)])
            p1 = self.center + radius * 1.2 * np.array([math.cos(a), math.sin(a)])
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=COLORS["faint"], lw=1.4 if k % 3 == 0 else 0.8, zorder=1)
        self.ref = Wedge(center, radius * 1.32, 0, 0, width=radius * 0.1, fc=COLORS["good"], ec="none", zorder=1)
        ax.add_patch(self.ref)
        self.pointer = Line2D([], [], color=COLORS["ink"], lw=3.2, solid_capstyle="round", zorder=4)
        ax.add_line(self.pointer)
        self.hub = Circle(center, radius * 0.1, fc="white", ec=COLORS["ink"], lw=2, zorder=5)
        ax.add_patch(self.hub)

    def set_style(self, color: str, radius: float) -> None:
        self.radius = radius
        for patch in (self.disk, self.rim):
            patch.set_radius(radius)
        self.disk.set_facecolor(color)
        self.rim.set_edgecolor(color)
        self.hub.set_radius(radius * 0.1)

    def update(self, theta: float, theta_ref: float) -> None:
        """Angles in rad, measured counter-clockwise from the +x axis (0 = 'east')."""
        tip = self.center + 0.92 * self.radius * np.array([math.cos(theta), math.sin(theta)])
        self.pointer.set_data([self.center[0], tip[0]], [self.center[1], tip[1]])
        ref_deg = math.degrees(theta_ref)
        self.ref.set_theta1(ref_deg - 6.0)
        self.ref.set_theta2(ref_deg + 6.0)
        self.ref.set_radius(self.radius * 1.32)
        self.ref.set_width(self.radius * 0.1)
