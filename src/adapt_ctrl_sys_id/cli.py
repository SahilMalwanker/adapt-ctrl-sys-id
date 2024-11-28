"""Command-line interface.

Examples::

    adapt-ctrl-sys-id reproduce                 # all studies -> docs/assets (figures, diagrams, GIFs, results.json)
    adapt-ctrl-sys-id reproduce --fast --no-animations
    adapt-ctrl-sys-id twin serve                # real-time twin on localhost:10815 (Ctrl+C to stop)
    adapt-ctrl-sys-id twin run --duration 120   # PI-P cascade client against a running twin
    adapt-ctrl-sys-id info                      # calibrated parameters and controller design
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

__all__ = ["main"]


def _say(text: str) -> None:
    print(text, flush=True)


def _cmd_reproduce(args: argparse.Namespace) -> int:
    from . import pipeline
    from .viz.diagrams import render_diagrams
    from .viz.figures import render_all

    out = Path(args.out)
    started = time.perf_counter()
    _say("[1/3] servo study ...")
    servo = pipeline.servo_study(fast=args.fast)
    _say("[2/3] pendulum study ...")
    pend = pipeline.pendulum_study(fast=args.fast)
    _say("[3/3] control + twin study ...")
    ctrl = pipeline.control_study(fast=args.fast, realtime_seconds=args.realtime)
    results = {"servo": servo.summary(), "pendulum": pend.summary(), "control": ctrl.summary()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    paths = render_all(servo, pend, ctrl, out / "figures")
    paths += render_diagrams(out / "diagrams")
    if not args.no_animations:
        from .viz.animations import render_animations

        _say("rendering animations ...")
        paths += render_animations(servo, pend, ctrl, out / "anim")
    for path in paths:
        _say(f"  wrote {path}")
    _say(f"done in {time.perf_counter() - started:.0f} s -> {out / 'results.json'}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .twin import CALIBRATED, PendulumTwin, TwinConfig, TwinServer

    twin = PendulumTwin(CALIBRATED, TwinConfig(seed=args.seed))
    server = TwinServer(twin, host=args.host, port=args.port, realtime=not args.lockstep).bind()
    mode = "lock-step" if args.lockstep else "real-time"
    _say(f"pendulum twin ({mode}) listening on {args.host}:{server.port} - Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _say("stopping")
    finally:
        server.stop()
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .control import DigitalPIP, PIPCoefficients, gravity_feedforward, place_pid
    from .pendulum import CALIBRATED
    from .pipeline import DESIGN_ALPHA, DESIGN_OMEGA, DESIGN_ZETA, OPERATING_POINT_DEG
    from .twin import TwinClient, run_closed_loop

    lin = CALIBRATED.linearize(math.radians(OPERATING_POINT_DEG)).in_degrees()
    coeffs = PIPCoefficients.from_pid(place_pid(lin, args.omega or DESIGN_OMEGA, DESIGN_ZETA, DESIGN_ALPHA), 0.01)
    ff = None if args.no_feedforward else gravity_feedforward(CALIBRATED)
    controller = DigitalPIP(coeffs, feedforward=ff, anti_windup=not args.no_feedforward)
    with TwinClient(args.host, args.port) as client:
        result = run_closed_loop(client, controller, duration=args.duration)
    window = (30.0, min(100.0, float(result.t[-1])))
    costs = result.costs(window)
    _say(
        f"samples {len(result.t)}  J_e {costs['J_e']:.3f} deg^2  J_u {costs['J_u']:.1f} %^2  "
        f"(window {window[0]:.0f}-{window[1]:.0f} s)"
    )
    if args.save:
        import numpy as np

        np.savetxt(
            args.save,
            np.column_stack([result.t, result.ref, result.phi, result.phid, result.u]),
            delimiter=",",
            header="t_s,ref_deg,phi_deg,phid_deg_s,u_pct",
            comments="",
            fmt="%.5f",
        )
        _say(f"saved {args.save}")
    return 0


def _cmd_info(_args: argparse.Namespace) -> int:
    from .control import PIPCoefficients, place_pid
    from .pendulum import CALIBRATED, LAB_MODEL
    from .pipeline import DESIGN_ALPHA, DESIGN_OMEGA, DESIGN_ZETA, OPERATING_POINT_DEG
    from .servo import HARDWARE_PT1, ServoParams

    lin = CALIBRATED.linearize(math.radians(OPERATING_POINT_DEG)).in_degrees()
    gains = place_pid(lin, DESIGN_OMEGA, DESIGN_ZETA, DESIGN_ALPHA)
    q = PIPCoefficients.from_pid(gains, 0.01)
    info = {
        "servo": {"hardware_pt1": {"K": HARDWARE_PT1.K, "T": HARDWARE_PT1.T}, "virtual": ServoParams().__dict__},
        "pendulum": {
            "lab_model": {"J": LAB_MODEL.J, "a0": LAB_MODEL.a0, "b0": LAB_MODEL.b0, "d": LAB_MODEL.viscous},
            "calibrated": {k: v for k, v in CALIBRATED.__dict__.items() if k != "geometry"} | {"J": CALIBRATED.J},
        },
        "pip_controller": {
            "Kp": gains.kp,
            "Ki": gains.ki,
            "Kd": gains.kd,
            "q0_PI": q.q0_pi,
            "q1_PI": q.q1_pi,
            "q0_P": q.q0_p,
            "Ts": q.ts,
        },
    }
    _say(json.dumps(info, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adapt-ctrl-sys-id", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    rep = sub.add_parser("reproduce", help="run every study and regenerate figures, GIFs and results.json")
    rep.add_argument("--out", default="docs/assets", help="output folder (default: docs/assets)")
    rep.add_argument("--fast", action="store_true", help="coarser grids for a quick check")
    rep.add_argument("--no-animations", action="store_true", help="skip the GIF rendering")
    rep.add_argument(
        "--realtime",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="also record a wall-clock-paced twin session for the timing figure",
    )
    rep.set_defaults(func=_cmd_reproduce)

    twin = sub.add_parser("twin", help="digital twin server and client")
    twin_sub = twin.add_subparsers(dest="twin_command", required=True)
    serve = twin_sub.add_parser("serve", help="serve the pendulum twin over TCP")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=10815)
    serve.add_argument("--lockstep", action="store_true", help="advance one sample per command (deterministic)")
    serve.add_argument("--seed", type=int, default=None, help="sensor-noise seed")
    serve.set_defaults(func=_cmd_serve)
    run = twin_sub.add_parser("run", help="run the PI-P cascade against a twin")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=10815)
    run.add_argument("--duration", type=float, default=120.0)
    run.add_argument("--omega", type=float, default=None, help="design bandwidth [rad/s]")
    run.add_argument("--no-feedforward", action="store_true", help="plain PI-P without feedforward/anti-windup")
    run.add_argument("--save", metavar="CSV", help="write the logged signals to a CSV file")
    run.set_defaults(func=_cmd_run)

    info = sub.add_parser("info", help="print calibrated parameters and the controller design")
    info.set_defaults(func=_cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
