import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageChops

from adapt_ctrl_sys_id.cli import main


def test_info_prints_design(capsys):
    assert main(["info"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["pip_controller"]["Ts"] == 0.01
    assert info["pendulum"]["calibrated"]["J"] > info["pendulum"]["lab_model"]["J"]


def test_banner_is_animated_seamless_and_has_static_fallback(tmp_path):
    from adapt_ctrl_sys_id.viz.banner import banner_gif

    servo_time = np.linspace(0, 12, 100)
    twin_time = np.linspace(0, 120, 100)
    servo_angle = np.sin(servo_time)
    twin_angle = 40 + 10 * np.sin(twin_time / 10)
    servo = SimpleNamespace(sine_run=SimpleNamespace(t=servo_time, theta=servo_angle, ref=servo_angle))
    control = SimpleNamespace(hero_run=SimpleNamespace(t=twin_time, phi=twin_angle, ref=twin_angle))
    path = banner_gif(servo, control, tmp_path, fps=2)
    with Image.open(path) as image:
        assert image.size == (1440, 660)
        assert image.n_frames >= 20
        assert image.info["loop"] == 0
        first = image.convert("RGB")
        image.seek(image.n_frames // 2)
        middle = image.convert("RGB")
        assert ImageChops.difference(first, middle).getbbox() is not None
        assert ImageChops.difference(first.crop((0, 0, 1440, 270)), middle.crop((0, 0, 1440, 270))).getbbox() is None
        image.seek(image.n_frames - 1)
        assert ImageChops.difference(first, image.convert("RGB")).getbbox() is None
    with Image.open(tmp_path / "banner.png") as fallback:
        assert fallback.size == (1440, 660)
    assert path.stat().st_size < 5_000_000


@pytest.mark.slow
def test_reproduce_fast_end_to_end(tmp_path):
    assert main(["reproduce", "--fast", "--no-animations", "--out", str(tmp_path)]) == 0
    results = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    servo, pend, ctrl = results["servo"], results["pendulum"], results["control"]
    assert (
        servo["frequency_validation"]["mean_abs_error_calibrated_db"]
        < servo["frequency_validation"]["mean_abs_error_linear_db"]
    )
    assert servo["adaptive"]["adaptive_overshoot_pct"] < servo["adaptive"]["fixed_overshoot_pct"]
    assert pend["free_fall"]["fit_drag_pct"] > pend["free_fall"]["fit_viscous_pct"] > pend["free_fall"]["fit_lab_pct"]
    assert pend["step_response"]["fit_calibrated_pct"] > pend["step_response"]["fit_lab_pct"]
    assert ctrl["twin_run"]["J_e_deg2"] < 1.0
    assert len(list((tmp_path / "figures").glob("*.png"))) >= 14
    assert len(list((tmp_path / "diagrams").glob("*.svg"))) == 3
