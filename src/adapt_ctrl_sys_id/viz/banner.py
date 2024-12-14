"""Branded bitmap masthead with accelerated playback of the two simulated plants."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from ..pipeline import ControlStudy, ServoStudy
from .animations import write_gif

WIDTH, HEIGHT = 1440, 660
TITLE = "adapt-ctrl-sys-id"
INK = "#202c2b"
MUTED = "#647674"
TEAL = "#087f75"
BLUE = "#2864cf"
ORANGE = "#d66337"
PALE = "#dce8e4"


def banner_gif(servo: ServoStudy, ctrl: ControlStudy, out: Path, fps: int = 20) -> Path:
    """Render actual simulation states as a 12-second, crossfaded project masthead.

    The plants are independent examples, not mechanically coupled. Routing
    pulses illustrate the shared workflow, not network packet measurements.
    """
    regular = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
    bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    fonts = {
        "subtitle": ImageFont.truetype(regular, 27),
        "label": ImageFont.truetype(bold, 21),
        "small": ImageFont.truetype(regular, 20),
        "number": ImageFont.truetype(bold, 19),
    }
    base = Image.new("RGB", (WIDTH, HEIGHT), "#ffffff")
    draw = ImageDraw.Draw(base)
    draw.rectangle((0, 0, WIDTH, 9), fill=INK)
    draw.rectangle((0, 0, 480, 9), fill=TEAL)
    draw.rectangle((480, 0, 960, 9), fill=BLUE)
    _title(draw, ImageFont.truetype(bold, _fit_size(bold, TITLE, max_width=1008, start=132)))
    draw.text((70, 207), "System identification. Adaptive control. Digital twins.", font=fonts["subtitle"], fill=MUTED)
    draw.text((1110, 61), "MEASURE", font=fonts["label"], fill=TEAL)
    draw.text((1110, 95), "UNDERSTAND", font=fonts["label"], fill=BLUE)
    draw.text((1110, 129), "CONTROL", font=fonts["label"], fill=ORANGE)
    draw.line((70, 270, 1370, 270), fill=PALE, width=2)
    for grid_x in range(82, 1370, 24):
        for grid_y in range(304, 576, 24):
            draw.point((grid_x, grid_y), fill="#d6e2de")
    draw.text((230, 296), "DC SERVO", font=fonts["label"], fill=INK, anchor="mt")
    draw.text((1190, 296), "PROPELLER PENDULUM", font=fonts["label"], fill=INK, anchor="mt")
    draw.line((70, 597, 1370, 597), fill=PALE, width=2)
    draw.text((70, 621), "TWO PLANTS. ONE CONTROL TOOLCHAIN.", font=fonts["label"], fill=INK)
    draw.text((1370, 621), "PYTHON  +  MATLAB / SIMULINK", font=fonts["small"], fill=MUTED, anchor="rt")

    route = np.array([[475, 393], [925, 393], [925, 501], [475, 501], [475, 393]], dtype=float)
    segment_lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    draw.rounded_rectangle((475, 393, 925, 501), radius=17, outline=PALE, width=3)
    draw.line((375, 445, 475, 445), fill=PALE, width=3)
    draw.line((925, 445, 1010, 445), fill=PALE, width=3)
    for node_x, node_y, label, color in (
        (555, 393, "IDENTIFY", TEAL),
        (845, 393, "DESIGN", BLUE),
        (845, 501, "SIMULATE", ORANGE),
        (555, 501, "VALIDATE", TEAL),
    ):
        draw.ellipse((node_x - 6, node_y - 6, node_x + 6, node_y + 6), fill=color)
        draw.text(
            (node_x, node_y - 35 if node_y < 450 else node_y + 23), label, font=fonts["label"], fill=color, anchor="mt"
        )
    draw.text((700, 433), "CLOSE THE LOOP", font=fonts["label"], fill=INK, anchor="mt")

    def route_point(progress: float) -> tuple[float, float]:
        distance = (progress % 1.0) * cumulative[-1]
        segment = min(int(np.searchsorted(cumulative, distance, side="right")) - 1, len(segment_lengths) - 1)
        fraction = (distance - cumulative[segment]) / segment_lengths[segment]
        point = route[segment] + fraction * (route[segment + 1] - route[segment])
        return float(point[0]), float(point[1])

    def frame_at(progress: float) -> Image.Image:
        frame = base.copy()
        canvas = ImageDraw.Draw(frame)
        servo_time = 6.0 + 6.0 * progress
        pendulum_time = 30.0 + 65.0 * progress
        theta = float(np.interp(servo_time, servo.sine_run.t, servo.sine_run.theta))
        reference = float(np.interp(servo_time, servo.sine_run.t, servo.sine_run.ref))
        phi = math.radians(float(np.interp(pendulum_time, ctrl.hero_run.t, ctrl.hero_run.phi)))
        phi_ref = math.radians(float(np.interp(pendulum_time, ctrl.hero_run.t, ctrl.hero_run.ref)))
        _disk(canvas, theta, reference)
        _pendulum(canvas, phi, phi_ref, progress * math.tau * 16)
        for packet in range(4):
            packet_progress = progress * 2 + packet / 4
            trail = [route_point(packet_progress - offset) for offset in np.linspace(0.026, 0, 12)]
            canvas.line(trail, fill=TEAL if packet % 2 == 0 else BLUE, width=4)
            point_x, point_y = trail[-1]
            canvas.ellipse((point_x - 5, point_y - 5, point_x + 5, point_y + 5), fill=INK)
        canvas.text((230, 559), "ENCODER FEEDBACK", font=fonts["small"], fill=MUTED, anchor="mt")
        canvas.text((1190, 559), "VIRTUAL PLANT", font=fonts["small"], fill=MUTED, anchor="mt")
        return frame

    frames = [frame_at(frame_index / (12 * fps)) for frame_index in range(12 * fps)]
    transition_frames = fps
    for frame_index in range(transition_frames):
        blend = (frame_index + 1) / transition_frames
        blend = blend * blend * (3 - 2 * blend)
        frames[-transition_frames + frame_index] = Image.blend(
            frames[-transition_frames + frame_index], frames[0], blend
        )
    out.mkdir(parents=True, exist_ok=True)
    frames[0].save(out / "banner.png")
    return write_gif(frames, out / "banner.gif", fps=fps, colors=128)


def _fit_size(font_path: str, text: str, max_width: float, start: int) -> int:
    size = start
    while size > 40 and ImageFont.truetype(font_path, size).getlength(text) > max_width:
        size -= 2
    return size


def _title(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont) -> None:
    # baseline chosen so descenders (p, y) end just above the subtitle at y = 207
    baseline = 196 - round(0.236 * font.size)
    x = 62.0
    words = TITLE.split("-")
    for index, word in enumerate(words):
        draw.text((x, baseline), word, font=font, fill=INK, anchor="ls")
        x += font.getlength(word)
        if index < len(words) - 1:
            draw.text((x, baseline), "-", font=font, fill=(TEAL, BLUE, ORANGE)[index % 3], anchor="ls")
            x += font.getlength("-")


def _disk(draw: ImageDraw.ImageDraw, theta: float, reference: float) -> None:
    center_x, center_y, radius = 230, 439, 99
    draw.ellipse((center_x - radius, center_y - radius + 7, center_x + radius, center_y + radius + 7), fill="#e8eeeb")
    draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        fill="#f1f6f4",
        outline=INK,
        width=3,
    )
    draw.ellipse((center_x - 83, center_y - 83, center_x + 83, center_y + 83), outline="#bdceca", width=1)
    for tick in range(48):
        angle = tick * math.tau / 48
        outer = radius + 15
        inner = outer - (9 if tick % 4 == 0 else 4)
        draw.line(
            (
                center_x + inner * math.cos(angle),
                center_y + inner * math.sin(angle),
                center_x + outer * math.cos(angle),
                center_y + outer * math.sin(angle),
            ),
            fill=MUTED,
            width=2,
        )
    for spoke in range(6):
        angle = theta + spoke * math.tau / 6
        draw.line(
            (
                center_x + 40 * math.cos(angle),
                center_y - 40 * math.sin(angle),
                center_x + 72 * math.cos(angle),
                center_y - 72 * math.sin(angle),
            ),
            fill="#bdceca",
            width=10,
        )
    reference_x = center_x + 120 * math.cos(reference)
    reference_y = center_y - 120 * math.sin(reference)
    draw.ellipse((reference_x - 6, reference_y - 6, reference_x + 6, reference_y + 6), fill=ORANGE)
    draw.line(
        (center_x, center_y, center_x + 87 * math.cos(theta), center_y - 87 * math.sin(theta)), fill=TEAL, width=8
    )
    draw.ellipse((center_x - 24, center_y - 24, center_x + 24, center_y + 24), fill="white", outline=INK, width=3)
    draw.ellipse((center_x - 8, center_y - 8, center_x + 8, center_y + 8), fill=INK)


def _pendulum(draw: ImageDraw.ImageDraw, phi: float, reference: float, spin: float) -> None:
    pivot = np.array([1125.0, 353.0])
    along = np.array([math.sin(phi), math.cos(phi)])
    normal = np.array([math.cos(phi), -math.sin(phi)])
    tip = pivot + 185 * along
    draw.line((1084, 338, 1166, 338), fill=INK, width=4)
    draw.polygon([(1095, 338), (1155, 338), (1137, 353), (1113, 353)], fill="#dce8e4")
    reference_tip = pivot + 185 * np.array([math.sin(reference), math.cos(reference)])
    for dash in range(0, 12, 2):
        start = pivot + dash / 12 * (reference_tip - pivot)
        stop = pivot + (dash + 1) / 12 * (reference_tip - pivot)
        draw.line([tuple(start), tuple(stop)], fill=ORANGE, width=2)
    draw.line([tuple(pivot), tuple(tip)], fill=INK, width=12)
    draw.line([tuple(pivot + 3 * normal), tuple(tip + 3 * normal)], fill="#899e97", width=3)
    for distance, half_length, half_width, color in ((44, 15, 20, TEAL), (185, 15, 12, INK)):
        center = pivot + distance * along
        corners = [
            tuple(center + length_sign * half_length * along + width_sign * half_width * normal)
            for length_sign, width_sign in ((-1, -1), (-1, 1), (1, 1), (1, -1))
        ]
        draw.polygon(corners, fill=color)
    hub = tip + 23 * normal
    blade = (17 + 21 * abs(math.cos(spin))) * along
    draw.line([tuple(tip), tuple(hub)], fill=INK, width=5)
    draw.line([tuple(hub - blade), tuple(hub + blade)], fill=BLUE, width=6)
    force_start = hub + 13 * normal
    force_end = hub + 51 * normal
    draw.line([tuple(force_start), tuple(force_end)], fill=TEAL, width=3)
    draw.polygon(
        [tuple(force_end), tuple(force_end - 12 * normal + 5 * along), tuple(force_end - 12 * normal - 5 * along)],
        fill=TEAL,
    )
    draw.ellipse((1117, 345, 1133, 361), fill="white", outline=INK, width=3)
