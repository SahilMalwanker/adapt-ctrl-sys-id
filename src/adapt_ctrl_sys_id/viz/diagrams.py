"""Block diagrams (SVG) drawn with matplotlib primitives."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from .style import COLORS, apply_style

__all__ = ["render_diagrams"]

C = COLORS
FILL = {
    "plant": "#e0f2fe",
    "ctrl": "#dcfce7",
    "adapt": "#fef3c7",
    "io": "#f1f5f9",
    "sensor": "#ede9fe",
    "data": "#fee2e2",
}
EDGE = {
    "plant": "#0284c7",
    "ctrl": "#059669",
    "adapt": "#d97706",
    "io": "#64748b",
    "sensor": "#7c3aed",
    "data": "#dc2626",
}


class Canvas:
    """Tiny helper for control-loop block diagrams in data coordinates."""

    def __init__(self, width: float, height: float, scale: float = 0.62, ymin: float = 0.0) -> None:
        apply_style()
        self.fig, self.ax = plt.subplots(figsize=(width * scale, (height - ymin) * scale))
        self.ax.set(xlim=(0, width), ylim=(ymin, height), aspect="equal")
        self.ax.axis("off")
        self.fig.subplots_adjust(0, 0, 1, 1)

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        kind: str = "ctrl",
        size: float = 10.5,
        sub: str | None = None,
    ) -> dict[str, tuple[float, float]]:
        self.ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.18",
                fc=FILL[kind],
                ec=EDGE[kind],
                lw=1.6,
                zorder=3,
            )
        )
        if sub:
            self.ax.text(x, y + 0.17 * h, text, ha="center", va="center", fontsize=size, color=C["ink"], zorder=4)
            self.ax.text(
                x, y - 0.24 * h, sub, ha="center", va="center", fontsize=size - 2.5, color=C["muted"], zorder=4
            )
        else:
            self.ax.text(x, y, text, ha="center", va="center", fontsize=size, color=C["ink"], zorder=4)
        return {"w": (x - w / 2, y), "e": (x + w / 2, y), "n": (x, y + h / 2), "s": (x, y - h / 2)}

    def junction(self, x: float, y: float, signs: dict[str, str], r: float = 0.28) -> dict[str, tuple[float, float]]:
        self.ax.add_patch(Circle((x, y), r, fc="white", ec=C["ink"], lw=1.5, zorder=3))
        offsets = {"w": (-1.35, 0.55), "s": (0.55, -1.35), "n": (0.55, 1.35), "e": (1.35, 0.55)}
        for side, sign in signs.items():
            dx, dy = offsets[side]
            self.ax.text(x + dx * r, y + dy * r, sign, ha="center", va="center", fontsize=11, color=C["ink"], zorder=4)
        return {"w": (x - r, y), "e": (x + r, y), "n": (x, y + r), "s": (x, y - r), "c": (x, y)}

    def wire(
        self,
        *pts: tuple[float, float],
        label: str | None = None,
        at: int = 0,
        dashed: bool = False,
        color: str | None = None,
        arrow: bool = True,
        lpos: str = "above",
    ) -> None:
        col = color or C["ink"]
        style = (0, (4, 3)) if dashed else "-"
        for a, b in pairwise(pts[:-1]):
            self.ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=1.5, ls=style, zorder=2, solid_capstyle="butt")
        self.ax.add_patch(
            FancyArrowPatch(
                pts[-2],
                pts[-1],
                arrowstyle="-|>,head_length=7,head_width=3.5" if arrow else "-",
                color=col,
                lw=1.5,
                linestyle=style,
                zorder=2,
                shrinkA=0,
                shrinkB=0,
            )
        )
        if label:
            a, b = pts[at], pts[at + 1]
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            dy = 0.22 if lpos == "above" else -0.3
            self.ax.text(
                mx,
                my + dy,
                label,
                ha="center",
                va="bottom" if lpos == "above" else "top",
                fontsize=9.5,
                color=col if color else C["ink"],
            )

    def dot(self, x: float, y: float) -> None:
        self.ax.plot(x, y, "o", color=C["ink"], ms=4.5, zorder=4)

    def text(self, x: float, y: float, s: str, **kw: object) -> None:
        kw.setdefault("fontsize", 10)
        kw.setdefault("color", C["muted"])
        self.ax.text(x, y, s, **kw)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.12)
        plt.close(self.fig)
        return path


def cascade_adaptive(out: Path) -> Path:
    cv = Canvas(19.5, 7.4, ymin=1.1)
    y = 4.2
    cv.text(
        0.2, 7.0, "Adaptive cascade position control of the DC servo", fontsize=13, color=C["ink"], fontweight="bold"
    )
    cv.wire((0.3, y), (1.2, y), label=r"$\theta_{ref}$")
    s1 = cv.junction(1.5, y, {"w": "+", "s": "−"})
    kp = cv.box(3.5, y, 1.6, 1.1, r"$K_p$", "ctrl", 13, sub="outer P")
    cv.wire(s1["e"], kp["w"])
    s2 = cv.junction(5.6, y, {"w": "+", "s": "−"})
    cv.wire(kp["e"], s2["w"], label=r"$\omega_{ref}$")
    kd = cv.box(7.6, y, 1.6, 1.1, r"$K_d$", "ctrl", 13, sub="inner P")
    cv.wire(s2["e"], kd["w"])
    sat = cv.box(9.7, y, 1.5, 1.1, "±10 V", "io", 10.5, sub="amplifier")
    cv.wire(kd["e"], sat["w"])
    plant = cv.box(12.6, y, 2.6, 1.3, r"$\dfrac{K}{T s + 1}$", "plant", 13, sub="motor + disk")
    cv.wire(sat["e"], plant["w"], label="u")
    integ = cv.box(15.4, y, 1.3, 1.1, r"$\dfrac{1}{s}$", "plant", 13)
    cv.wire(plant["e"], integ["w"], label=r"$\omega$")
    enc = cv.box(17.6, y, 1.6, 1.1, "encoder", "sensor", 10.5, sub="2048 cnt/rev")
    cv.wire(integ["e"], enc["w"], label=r"$\theta$")
    cv.wire(enc["e"], (19.3, y))
    cv.text(19.35, y + 0.12, r"$\theta$", fontsize=12, color=C["ink"])
    # speed estimate feedback
    cv.dot(18.7, y)
    cv.wire((18.7, y), (18.7, 2.6), (13.6, 2.6))
    filt = cv.box(12.6, 2.6, 2.0, 0.9, "d/dt + low-pass", "sensor", 9.5)
    cv.wire(filt["w"], (5.6, 2.6), s2["s"], label=r"$\hat\omega$", at=0)
    # position feedback
    cv.wire((18.7, 2.6), (18.7, 1.5), (1.5, 1.5), s1["s"])
    # adaptation layer
    rls = cv.box(13.5, 6.2, 3.4, 1.1, "RLS on encoder increments", "adapt", 10, sub=r"50 Hz,  $\lambda$ = 0.98")
    design = cv.box(
        7.6,
        6.2,
        4.4,
        1.1,
        "actuator-aware design rule",
        "adapt",
        10,
        sub=r"$s_{vel}=-0.95\sqrt{2\hat K u_{max}/(r_0 \hat T)}$",
    )
    cv.wire(rls["w"], design["e"], label=r"$\hat K,\ \hat T$", color=EDGE["adapt"])
    cv.wire((10.6, y), (10.6, 5.4), (12.4, 5.4), (12.4, 5.65), color=EDGE["adapt"], dashed=True)
    cv.wire((18.2, y + 0.55), (18.2, 6.2), (15.2, 6.2), color=EDGE["adapt"], dashed=True)
    cv.wire((7.0, 5.65), (3.9, 5.65), (3.9, kp["n"][1]), color=EDGE["adapt"], dashed=True)
    cv.wire((7.6, 5.65), (7.6, kd["n"][1]), color=EDGE["adapt"], dashed=True)
    cv.text(4.0, 5.05, r"$K_p$", fontsize=10, color=EDGE["adapt"])
    cv.text(7.7, 5.05, r"$K_d$", fontsize=10, color=EDGE["adapt"])
    return cv.save(out / "loop_servo_adaptive.svg")


def pip_cascade(out: Path) -> Path:
    cv = Canvas(20.5, 8.0, ymin=0.9)
    y = 4.6
    cv.text(
        0.2,
        7.6,
        "Digital 2-DOF PI-P cascade for the propeller pendulum (Ts = 10 ms)",
        fontsize=13,
        color=C["ink"],
        fontweight="bold",
    )
    cv.wire((0.2, y), (1.1, y), label=r"$\varphi_{ref}$")
    cv.dot(0.7, y)
    s1 = cv.junction(1.4, y, {"w": "+", "s": "−"})
    pi = cv.box(3.7, y, 2.5, 1.3, r"$\dfrac{q_0 + q_1 z^{-1}}{1 - z^{-1}}$", "ctrl", 12, sub="outer PI")
    cv.wire(s1["e"], pi["w"], label="e")
    s2 = cv.junction(6.1, y, {"w": "+", "s": "−"})
    cv.wire(pi["e"], s2["w"], label=r"$u_{PI}$")
    p = cv.box(7.8, y, 1.5, 1.1, r"$q_{0,P}$", "ctrl", 12, sub="inner P")
    cv.wire(s2["e"], p["w"])
    s3 = cv.junction(9.5, y, {"w": "+", "n": "+"})
    cv.wire(p["e"], s3["w"])
    sat = cv.box(11.2, y, 1.6, 1.1, "0 … 100 %", "io", 10, sub="rotor limit")
    cv.wire(s3["e"], sat["w"])
    zoh = cv.box(13.3, y, 1.3, 1.1, "ZOH", "io", 10.5)
    cv.wire(sat["e"], zoh["w"], label="u[k]")
    plant = cv.box(
        16.2, y, 3.1, 1.5, "pendulum", "plant", 11.5, sub=r"$J\ddot\varphi = b_0 u - a_0\sin\varphi - d\dot\varphi$"
    )
    cv.wire(zoh["e"], plant["w"])
    cv.wire(plant["e"], (20.3, y))
    cv.text(19.7, y + 0.25, r"$\varphi$", fontsize=12, color=C["ink"])
    # gravity feedforward
    ff = cv.box(7.8, 6.6, 3.0, 0.95, r"$u_{ff} = a_0\sin\varphi_{ref} / b_0$", "adapt", 10)
    cv.text(7.8, 7.22, "gravity feedforward", ha="center", fontsize=9.5, color=EDGE["adapt"])
    cv.wire((0.7, y), (0.7, 6.6), ff["w"], color=EDGE["adapt"])
    cv.wire(ff["e"], (9.5, 6.6), s3["n"], color=EDGE["adapt"])
    # sensors and samplers
    sensor = cv.box(16.2, 2.6, 3.1, 0.95, "angle + rate sensors", "sensor", 10)
    cv.wire(plant["s"], sensor["n"])
    cv.wire((14.65, 2.82), (6.1, 2.82), s2["s"], label=r"$\dot\varphi[k]$")
    cv.wire((14.65, 2.38), (13.4, 2.38), (13.4, 1.4), (1.4, 1.4), s1["s"], label=r"$\varphi[k]$", at=2)
    # anti-windup
    cv.wire(
        (11.2, y - 0.55),
        (11.2, 3.5),
        (3.7, 3.5),
        (3.7, y - 0.65),
        dashed=True,
        color=EDGE["data"],
        label="anti-windup: reset PI state on saturation",
        at=1,
        lpos="below",
    )
    return cv.save(out / "loop_pendulum_pip.svg")


def overview(out: Path) -> Path:
    cv = Canvas(21.0, 9.6, scale=0.6)
    cv.text(0.2, 9.15, "adapt-ctrl-sys-id at a glance", fontsize=14, color=C["ink"], fontweight="bold")
    cols = [
        (1.8, "Measure"),
        (6.0, "Model & identify"),
        (10.6, "Validate"),
        (15.0, "Design & control"),
        (19.0, "Deploy"),
    ]
    for x, title in cols:
        cv.ax.text(x, 8.35, title.upper(), ha="center", fontsize=10, color=C["muted"], fontweight="bold")
    lanes = [
        (
            6.6,
            "plant",
            "DC servo + disk",
            [
                "static map\n22 hardware points",
                "PT1 + quadratic map\nLS · OE · RLS",
                "sine tests\nlock-in, Bode",
                "P-P cascade\nactuator-aware poles",
                "adaptive loop\nRLS + design rule",
            ],
        ),
        (
            3.9,
            "sensor",
            "Propeller pendulum",
            [
                "free fall · statics · step\nhardware records",
                "nonlinear model\nJ, d, c, k, d_u fitted",
                "linearisation map\nARX(2,2) cross-check",
                "2-DOF PI-P cascade\n+ feedforward + AW",
                "TCP digital twin\nreal-time / lock-step",
            ],
        ),
        (
            1.3,
            "data",
            "Smartphone gyro",
            ["chair spin\n100 Hz gyro", "rate integration", "turn counting\nIMU sign & bias check", "", ""],
        ),
    ]
    for y, kind, name, cells in lanes:
        cv.ax.text(-0.1, y + 0.95, name, ha="left", fontsize=10.5, color=EDGE[kind], fontweight="bold")
        prev = None
        for (x, _), text in zip(cols, cells, strict=True):
            if not text:
                continue
            b = cv.box(x, y, 3.6, 1.35, text, kind, 9.2)
            if prev is not None:
                cv.wire(prev["e"], b["w"], color=EDGE[kind])
            prev = b
    cv.wire((6.0, 3.9 + 0.675), (6.0, 6.6 - 0.675), color=C["muted"], dashed=True, arrow=False)
    cv.ax.text(6.15, 5.25, "shared LS / OE / RLS\nidentification toolbox", fontsize=8.5, color=C["muted"], va="center")
    cv.wire((15.0, 3.9 + 0.675), (15.0, 6.6 - 0.675), color=C["muted"], dashed=True, arrow=False)
    cv.ax.text(15.15, 5.25, "cascade idea: fast inner\nloop on the rate", fontsize=8.5, color=C["muted"], va="center")
    cv.wire((10.6, 1.3 + 0.675), (10.6, 3.9 - 0.675), color=C["muted"], dashed=True, arrow=False)
    cv.ax.text(10.75, 2.6, "same sensor checks on\nthe pendulum's IMU", fontsize=8.5, color=C["muted"], va="center")
    return cv.save(out / "overview.svg")


def render_diagrams(out: Path) -> list[Path]:
    return [overview(out), cascade_adaptive(out), pip_cascade(out)]
