# Control and the digital twin

Code: [`control.py`](../src/adapt_ctrl_sys_id/control.py) and
[`twin/`](../src/adapt_ctrl_sys_id/twin). The numbers come from
`pipeline.control_study`.

## 1. Design model

The calibrated pendulum is linearised at φ<sub>op</sub> = 40°. That is the
middle of the reference profile, which moves between 30° and 50°. The angle is
in degrees and the command in %:

```math
G(s) = \frac{b}{a_2 s^2 + a_1 s + a_0},\qquad a_2 = J,\quad a_1 = d + d_u u_\text{op},\quad a_0 = m g l \cos\varphi_\text{op},\quad b = b_0\,\tfrac{180}{\pi}
```

The thrust-dependent damping makes a<sub>1</sub> about 14 times larger than
the free-fall friction alone. Even so, the open loop is lightly damped
(ζ ≈ 0.06).

## 2. Pole placement for a PI-D law

The control law is a PI-D: the derivative acts on the measurement only, so a
reference step does not kick the output:

```math
U(s) = \Bigl(K_p + \frac{K_i}{s}\Bigr)E(s) - K_d\,s\,Y(s)
```

The closed-loop polynomial
a<sub>2</sub>s³ + (a<sub>1</sub> + bK<sub>d</sub>)s² + (a<sub>0</sub> + bK<sub>p</sub>)s + bK<sub>i</sub>
is matched to a<sub>2</sub>(s + αω)(s² + 2ζωs + ω²):

```math
K_d = \frac{a_2(2\zeta + \alpha)\,\omega - a_1}{b},\qquad
K_p = \frac{a_2\,\omega^2 (1 + 2\zeta\alpha) - a_0}{b},\qquad
K_i = \frac{a_2\,\alpha\,\omega^3}{b}
```

With ω = 8 rad/s, ζ = 0.9 and α = 1, this gives K<sub>p</sub> = 5.62 %/deg,
K<sub>i</sub> = 18.25 %/(deg s) and K<sub>d</sub> = 0.779 %/(deg/s). The poles
are −8 and −7.2 ± 3.49j.

## 3. Digital PI-P cascade

Dividing by K<sub>d</sub> turns the PI-D law into a cascade. An outer PI loop
produces a rate set-point, and an inner P loop on the measured rate drives
the rotor. Backward Euler at T<sub>s</sub> = 10 ms gives the incremental
difference equations that run on the target:

```math
\begin{aligned}
u_{PI}[k] &= u_{PI}[k-1] + q_{0,PI}\,e[k] + q_{1,PI}\,e[k-1] \\
u[k] &= q_{0,P}\,\bigl(u_{PI}[k] - \dot\varphi[k]\bigr) + u_{ff}[k] \\
q_{0,PI} &= \tfrac{K_p}{K_d} + \tfrac{K_i T_s}{K_d} = 7.458,\quad q_{1,PI} = -\tfrac{K_p}{K_d} = -7.224,\quad q_{0,P} = K_d = 0.779
\end{aligned}
```

Two additions make the controller robust on the real actuator, which only
pushes and saturates at 0 and 100 %:

- **Gravity feedforward** u<sub>ff</sub> = a<sub>0</sub> sin r / b<sub>0</sub>
  supplies the static torque for the reference. The integrator then only
  corrects model errors.
- **Anti-windup by conditioning.** When the output saturates, the PI state is
  reset to the value that would have produced the saturated output. The
  integrator therefore never charges beyond what the rotor can deliver.

## 4. Results on the nonlinear model

Costs are evaluated between 30 and 100 s:
J<sub>e</sub> = mean((r − φ)²) and J<sub>u</sub> = mean(u²).

| Controller | J<sub>e</sub> [deg²] | J<sub>u</sub> [%²] |
|---|---|---|
| PI-P cascade | 1.150 | 1376 |
| 1-DOF PID with the same gains (1 kHz, filtered derivative) | 1.102 | 1379 |
| PI-P + feedforward + anti-windup | **1.070** | 1369 |

The differences at the design bandwidth are small. The difference shows up in
the **bandwidth sweep** over ω from 2.5 to 14 rad/s:

- Slow designs: the feedforward and anti-windup combination cuts J<sub>e</sub>
  from 9.5 to 3.1 deg² at ω = 2.5 rad/s.
- Fast designs: from ω = 12 rad/s the plain PI-P **diverges**. The output
  saturates, the integrator winds up, and the pendulum swings over the top
  (J<sub>e</sub> ≈ 10⁶ deg²). At ω = 12 and 14 rad/s, anti-windup alone keeps
  J<sub>e</sub> at 1.09 and 1.07 deg²; feedforward alone still diverges. With
  both, J<sub>e</sub> stays at or below 1.1 deg² for every design from 7 to
  14 rad/s.

## 5. The twin

[`PendulumTwin`](../src/adapt_ctrl_sys_id/twin/physics.py) contains:

- the calibrated nonlinear model with rotor lag, dead band and
  thrust-dependent damping, integrated with RK4 (5 substeps per 10 ms sample)
- a slider actuator that travels at 0.1 m/s towards the commanded offset, so
  J(x) and l(x) change continuously
- sensor noise: 0.05° on the angle and 0.5 °/s on the rate (seeded)

[`TwinServer`](../src/adapt_ctrl_sys_id/twin/server.py) serves one client at a time on
`127.0.0.1:10815`:

| Mode | Behaviour | Use |
|---|---|---|
| Real time (default) | Advances by the *measured* elapsed time every 10 ms tick and applies commands at the next tick. A late tick is followed by catch-up ticks | Realistic timing, interactive sessions |
| Lock-step (`--lockstep`) | Advances exactly one sample per received command | Deterministic runs, tests, CI |

### Protocol

The protocol is line based, ASCII, LF terminated. It is simple enough for
`sscanf` in MATLAB or a microcontroller client.

| Direction | Message | Meaning |
|---|---|---|
| Client → twin | `START` | Reset the twin and start streaming |
| Twin → client | `<PP><t>12.3400 </t><PA>40.1000 </PA><PAD>-0.5000 </PAD></PP>` | Time [s], angle [deg], rate [deg/s] |
| Client → twin | `37.250000;0.000000` | Rotor command [%]; slider offset [mm] (optional, default 0) |
| Client → twin | `EXIT` | End the session |

The state decoder accepts decimal commas from locale-dependent senders.
Actuator commands use decimal points. Malformed command lines are ignored.
This is a local simulation protocol, without authentication or encryption;
keep the default loopback binding.

### Timing

On a desktop OS the 10 ms tick is a request, not a guarantee. The timing
figure comes from a 20 s wall-clock-paced session. Most intervals are within
±0.5 ms. Scheduler hiccups produce isolated late ticks, and the catch-up
schedule follows each one with short intervals. The share of intervals more
than 0.5 ms from nominal includes both early and late ticks. Across the runs behind this
repository it ranged from 5 % to 37 %, with worst cases between 16 and 37 ms.
The published run has 36.9 % off-nominal intervals and a 31 ms maximum.
The physics uses the measured elapsed time, capped at 100 ms per tick.
The PI-P controller still uses fixed 10 ms coefficients. Neither this timing
test nor the simulated tracking result certifies real-time hardware control.

## Reproduce

```bash
adapt-ctrl-sys-id twin serve                 # terminal 1
adapt-ctrl-sys-id twin run --save run.csv    # terminal 2
python -m pytest tests/test_control.py tests/test_twin.py
```

MATLAB versions: [`matlab/control`](../matlab/control) (design, simulation,
TCP clients) and [`matlab/simulink`](../matlab/simulink) (model builders).
