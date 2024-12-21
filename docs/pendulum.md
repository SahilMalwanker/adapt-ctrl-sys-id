# Propeller pendulum: from geometry to a calibrated nonlinear model

A beam swings about a pivot. A propeller at the far end (lever arm
l<sub>r</sub> = 0.5 m) pushes it up. A slider mass changes the centre of
gravity and the inertia. The pendulum angle is measured by a potentiometer and
the rate by an IMU. Code: [`pendulum.py`](../src/adapt_ctrl_sys_id/pendulum.py). The
numbers come from `pipeline.pendulum_study`.

## 1. Model

```math
J(x)\,\ddot\varphi = b_0\,w - a_0(x)\sin\varphi - (d + d_u w)\,\dot\varphi - c\,\dot\varphi\,|\dot\varphi|,
\qquad \tau_r\,\dot w = u_\text{eff} - w
```

| Symbol | Meaning | Geometry baseline | Calibrated |
|---|---|---|---|
| J | Inertia about the pivot | 0.04761 kg m² (geometry) | 0.05372 kg m² |
| a<sub>0</sub> = m g l(x) | Gravity torque | 1.5024 N m | Same (fixed) |
| b<sub>0</sub> = k l<sub>r</sub> | Torque per % command | 0.025 N m/% | 0.0263 N m/% |
| d | Viscous friction | 0.0045 N m s | 0.00211 N m s |
| c | Quadratic air drag | — | 5.1·10⁻⁴ N m s² |
| d<sub>u</sub> | Thrust-dependent damping (propeller inflow) | — | 7.6·10⁻⁴ N m s/% |
| τ<sub>r</sub> | Rotor lag | — | 44 ms |
| dead band | No thrust below | — | 6 % |

u<sub>eff</sub> is the command limited to 0–100 %. The propeller only pushes,
and inside the dead band u<sub>eff</sub> is 0. The geometry treats the beam as
a rod and the rotor and slider as point masses:

```math
l(x) = \frac{x\,m_s + \tfrac{l_b}{2}\,m_b + l_r\,m_r}{m},\qquad J(x) = \tfrac13 m_b l_b^2 + m_r l_r^2 + m_s x^2
```

## 2. Free fall: inertia, damping and drag

The pendulum is held at about 91° and released with the rotor off.

- **Energy bound.** Friction removes energy before the pendulum passes the
  bottom, so the measured peak rate is a slight underestimate and
  J ≤ 2 m g l (1 − cos φ₀)/φ̇²<sub>max</sub> = 0.0582 kg m² is an upper bound.
  Both the geometry value (0.0476) and the fitted value (0.0537) respect it.
- **Envelope.** A log-linear fit of the swing peaks gives the decay rate. For
  a viscous model it corresponds to d = 2Jσ.
- **Output-error fit.** J, d, c and the release instant are fitted to the
  whole 60 s record. a<sub>0</sub> stays fixed from geometry, because a free
  fall only constrains the ratios a<sub>0</sub>/J and d/J. A viscous-only
  model leaves systematic errors. Air drag removes more energy on large swings
  and closes most of the gap:

| Model | NRMSE fit |
|---|---|
| Geometry baseline | −16.2 % |
| Fitted J and viscous d | 84.7 % |
| Fitted J, d and quadratic drag c | 93.0 % |

## 3. Static equilibria: thrust map and centre of gravity

At an equilibrium, F l<sub>r</sub> = m g l(x) sin φ. The measured angles for
constant commands therefore give the rotor force. It is linear in the command
above a start-up dead band: k = 0.0526 N/%.

The same command at two slider positions x and x′ gives the centre of gravity
without a scale. Moving the slider shifts l by (m<sub>s</sub>/m)(x′ − x), and
the gravity torques must match the same thrust:

```math
l(x) = \frac{m_s}{m}\,\frac{x - x'}{1 - \sin\varphi / \sin\varphi'}
```

With 30 % at both positions the experiment gives 0.253 m against 0.188 m from
geometry. The formula is sensitive: a ±1° reading error moves the result
between 0.207 and 0.317 m. The geometry value lies outside even this band. The
model keeps the geometry value for a<sub>0</sub> = m g l. The dynamic
experiments only constrain ratios such as a<sub>0</sub>/J, so the records
alone cannot settle the discrepancy.

## 4. Powered step: thrust damping and rotor lag

The rotor command steps so that the pendulum moves from 22.3° to 42.4°. The
first swing overshoots by 100 %, yet it decays far faster than the free fall
suggests. Calibrated free-fall damping alone fits only −37 %. Two physical
effects close the gap:

- **Thrust-dependent damping d<sub>u</sub>·u·φ̇.** A spinning propeller that
  moves sideways through its own inflow feels a force against the motion. This
  damping grows with the thrust.
- **Rotor lag τ<sub>r</sub> = 44 ms.** The propeller needs time to spin up.

With both, the nonlinear model fits 62.4 %. A ±5° ripple that persists after
about 40 s is not modelled.

**Discrete cross-check.** An ARX(2,2) model on the same record gives
ω<sub>n</sub> = 4.54 rad/s and ζ = 0.126. It is sampled at 50 ms with a
moving-average anti-alias filter, because at 2–20 ms the equation-error
estimate is dominated by sensor noise. A transfer function only fixes
coefficient ratios, so J is taken from the free fall. The physical
coefficients then follow:

| | ARX(2,2) | Calibrated model |
|---|---|---|
| a<sub>0,lin</sub> = a<sub>0</sub> cos φ<sub>2</sub> | 1.106 N m | 1.110 N m |
| b<sub>0</sub> | 0.0241 N m/% | 0.0263 N m/% |

## 5. Linearisation

Around an operating point φ<sub>op</sub>, with
u<sub>op</sub> = a<sub>0</sub> sin φ<sub>op</sub> / b<sub>0</sub>:

```math
J\,\Delta\ddot\varphi + (d + d_u u_\text{op})\,\Delta\dot\varphi + a_0\cos\varphi_\text{op}\,\Delta\varphi = b_0\,\Delta u
```

The validity map applies the *same physical input step* to both models: the
nonlinear equilibrium inputs before and after the step. It records the
worst-case angle error over 15 s. Steps of up to ±2° stay within 0.71° at
every operating point. Around 40°, ±5° costs 2.7°, ±10° costs 11.6° and ±15°
costs 28°. In 12 of the 108 evaluated cells, the nonlinear response passes
the horizontal, which the linear model cannot represent.

## 6. Sensors

- **IMU vs. potentiometer.** The integrated IMU rate drifts away from the
  potentiometer. Regressing the IMU rate on a Savitzky–Golay derivative of the
  angle shows that the IMU axis is **inverted** (the loaders flip the sign), a
  scale of 0.966 and a bias of 0.73 °/s. After the sign fix the correlation is
  0.991.
- **Smartphone gyro on a chair.** Integrating the yaw rate and counting with a
  0.2 rad hysteresis gives a net −3.43 turns, a travelled 6.53 turns and one
  reversal. Counting only the net angle would miss almost half of the motion.

## Reproduce

```bash
python -m pytest tests/test_pendulum.py
adapt-ctrl-sys-id reproduce --fast --no-animations
```

MATLAB versions: [`matlab/pendulum`](../matlab/pendulum).
