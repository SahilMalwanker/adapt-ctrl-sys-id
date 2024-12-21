# DC servo: identification, cascade design and adaptive control

The plant is a DC motor with an inertia disk. The amplifier accepts ±10 V,
and the position is read through a 2048 counts/rev encoder in a 2 ms
real-time loop. Code: [`servo.py`](../src/adapt_ctrl_sys_id/servo.py) and
[`identification.py`](../src/adapt_ctrl_sys_id/identification.py). The numbers come
from `pipeline.servo_study`.

## 1. Static map

The hardware readings are steady-state speeds for constant voltages between
±0.1 and ±9 V. A line through the origin fits them with K = 24.56 rad/s/V and
an RMS error of 4.09 rad/s. The residuals are systematic: the effective gain
ω/u falls from about 28 rad/s/V at low speed to 24 at full speed.

A loss term that grows with the speed explains this:

```math
u = \frac{\omega}{K_0} + \beta\,\omega\,|\omega|
```

The model is linear in the unknowns (1/K<sub>0</sub>, β), so ordinary least
squares fits it: K<sub>0</sub> = 28.63 rad/s/V and β = 3.25·10⁻⁵ V s²/rad².
The RMS error drops to 1.88 rad/s. The incremental gain
dω/du = 1/(1/K<sub>0</sub> + 2β|ω|) is what a small-signal model sees around
an operating speed.

## 2. The calibrated virtual servo

The exported servo session is not treated as verified raw hardware data;
its response appears consistent with a re-simulation of the identified PT1.
The time-domain experiments here explicitly run on a **virtual servo** that
is calibrated against the reported hardware constants:

| Element | Model |
|---|---|
| Mechanics | τ J<sub>scale</sub> ω̇ = K<sub>0</sub>(u − βω\|ω\|) − ω, RK4 at 2 ms |
| Amplifier | Saturation at ±10 V, zero-order hold |
| Encoder | 2048 counts/rev quantisation |
| Speed | Finite difference of the encoder angle + first-order low-pass (4 ms) |

τ = 0.353 s is calibrated so that a PT1 identified on the standard excitation
profile returns the hardware estimate T = 0.319 s. The excitation mixes small
and large steps of both signs, so the identified PT1 is an average over the
operating range. That is why τ differs from T.

## 3. Identification

`servo_excitation_profile.csv` defines the 15 s excitation profile. Three
estimators run on the same record:

| Estimator | K [rad/s/V] | T [s] | Fit |
|---|---|---|---|
| Equation error, ARX(1,1) least squares | 25.51 | 0.326 | 96.9 % |
| Output error, `least_squares` on the simulation error | 25.49 | 0.319 | 96.9 % |
| Hardware estimate (reference) | 25.17 | 0.319 | 96.7 % |

The ARX estimate is slightly biased because the speed noise enters the
regressors. Output error does not have that problem. Recursive least squares
(λ = 1) converges to the batch ARX solution. With λ < 1 it keeps tracking
changes, at the cost of noisier estimates.

The fit is the NRMSE fit, 100·(1 − ‖y − ŷ‖/‖y − ȳ‖): 100 % is perfect and
0 % is as good as the mean.

## 4. Frequency-domain validation

The hardware sine tests at 1.03, 3.13 and 9.40 rad/s are the reference. The
middle frequency is the corner 1/T. The same experiments run on the virtual
servo, and its response is measured with a **correlation (lock-in)
analyser**. The output is correlated with sin and cos over an integer number
of periods after the transient, which rejects offsets, harmonics and
quantisation noise:

```math
\mathrm{Re}\,G = \frac{2}{A\,T_w}\int y\,\sin\omega t\;dt,\qquad \mathrm{Im}\,G = \frac{2}{A\,T_w}\int y\,\cos\omega t\;dt
```

The linear PT1 misses the magnitude by 1.60 dB on average and the virtual
servo by 1.01 dB. The quadratic map raises the gain for the 1 V test
amplitude, as the hardware shows.

## 5. Cascade position control

The controller is u = K<sub>d</sub>(K<sub>p</sub>(r − θ) − ω), a P velocity
loop inside a P position loop. With the PT1 speed plant, the closed position
loop is

```math
T s^2 + (1 + K_d K)\,s + K_p K_d K = 0 .
```

K<sub>d</sub> places the velocity pole at s<sub>vel</sub>. K<sub>p</sub> then
gives the position loop a damping ratio of 1/√2, with poles at
s<sub>vel</sub>/2·(1 ± j):

```math
K_d = \frac{-s_\text{vel}T - 1}{K}, \qquad K_p = \frac{(1 + K_d K)^2}{2\,T\,K_d K}
```

**Where to put s<sub>vel</sub>.** Right after a step of height r₀ the output
is u(0⁺) = K<sub>d</sub>K<sub>p</sub>r₀ = r₀Ts²<sub>vel</sub>/(2K). Keeping it
within the amplifier limit gives

```math
s_\text{vel} \ge -\sqrt{\frac{2\,K\,u_\max}{r_0\,T}} = -31.7\ \text{s}^{-1}\quad(r_0 = \pi/2).
```

The integer rule `round(-44.3 + 43 T)` gives −31 for the identified T, close
to this bound. The design uses 95 % of the bound: s<sub>vel</sub> = −30.1,
K<sub>d</sub> = 0.342 V s/rad, K<sub>p</sub> = 16.8 1/s and position poles
−15.05 ± 15.05j. On the virtual servo a 90° step overshoots by 2.3 %, and the
voltage peaks at 9.0 V.

## 6. Adaptive cascade

At t = 4.9 s the disk inertia increases by the factor 3√(0.7 − T) = 1.85.
With fixed gains, the overshoot on the following edges rises to 13.7 %.

The adaptive controller is indirect, using certainty equivalence:

1. Every 10 samples (h = 20 ms) it forms the encoder increment
   d[m] = θ[m] − θ[m−1]. For a zero-order-held K/(s(Ts+1)) the increments obey
   exactly

   ```math
   d[m] = a\,d[m-1] + b_1\,u[m-1] + b_2\,u[m-2],\qquad a = e^{-h/T},\quad b_1 + b_2 = K h (1 - a)
   ```

   Here u is the mean voltage over the window.
2. RLS with forgetting λ = 0.98 estimates [a, b<sub>1</sub>/h,
   b<sub>2</sub>/h]. Dividing by h gives all regressors a similar scale. A
   trace limit on P prevents covariance wind-up when the excitation fades.
3. The estimate maps back to (K, T) and is projected into plausible bounds.
   The same actuator-aware rule as above then recomputes K<sub>d</sub> and
   K<sub>p</sub>.

The estimated time constant rises from 0.33 s to 0.62 s. It covers half of
the change within 1.3 s and 90 % within about 4 s. The overshoot on the
following edges stays at 3.8 %.
An earlier variant estimated the PT1 from the filtered speed. The encoder
quantisation biased it badly, which is why the increment model is used.

## Reproduce

```bash
adapt-ctrl-sys-id reproduce --fast --no-animations   # quick check
python -m pytest tests/test_servo.py tests/test_identification.py
```

MATLAB versions: [`matlab/servo`](../matlab/servo).
