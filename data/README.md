# Data

These are hardware records from the two rigs and a smartphone, converted to
plain CSV with SI-style units in the column names. Loaders live in
[`src/adapt_ctrl_sys_id/data.py`](../src/adapt_ctrl_sys_id/data.py). Set `ADAPT_CTRL_SYS_ID_DATA` to read
the files from another folder.

| File | Rows | Columns | Content |
|---|---|---|---|
| `servo_static_map.csv` | 22 | `u_V`, `omega_rad_s` | Steady-state disk speed for constant voltages, ±0.1 to ±9 V |
| `servo_frequency_response.csv` | 3 | `omega_rad_s`, `magnitude_db`, `phase_deg` | Sine tests on the servo (speed / voltage) |
| `servo_excitation_profile.csv` | 11 | `t_start_s`, `u_V` | Breakpoints of the piecewise-constant identification input (0 V before the first breakpoint, 15 s long) |
| `pendulum_free_fall.csv` | 38 814 | `t_s`, `phi_deg`, `imu_rate_deg_s` | Release from about 91° with the rotor off, 60 s |
| `pendulum_static_equilibrium.csv` | 13 | `x_m`, `u_pct`, `phi_deg` | Equilibrium angles for constant rotor commands at two slider positions |
| `pendulum_step_response.csv` | 44 771 | `t_s`, `phi_deg`, `imu_rate_deg_s` | Rotor command step that moves the pendulum from about 22° to 42°, 70 s |
| `gyro_chair_spin.csv` | 1 993 | `t_s`, `yaw_rad` | Integrated yaw rate of a phone on a spinning office chair, 20 s at 100 Hz |

## Conventions and caveats

- **Pendulum angle.** `phi_deg` is the potentiometer angle: 0° hanging, and
  positive in the direction the propeller pushes.
- **IMU sign.** `imu_rate_deg_s` is stored **as recorded**. The IMU axis
  points the opposite way to the potentiometer: after the sign flip the two
  agree with a correlation of 0.991. The loaders negate the column, so
  `TimeSeries.phid_deg` follows the angle convention. The rate also carries a
  bias of about 0.73 °/s and a 3.4 % scale error (see the
  [sensor figure](../docs/assets/figures/sensors.png)).
- **Irregular sampling.** The pendulum logger wrote samples about every
  1–2 ms with jitter (median 1 ms). `TimeSeries.resample` interpolates
  linearly onto a uniform grid (5 ms for the fits).
- **Reading resolution.** The static-equilibrium angles were read to the
  nearest degree. The centre-of-gravity estimate is sensitive to this, as
  `docs/pendulum.md` explains.
- **Servo time series.** The exported session is not used as verified raw
  hardware data. The time-domain experiments (identification, cascade,
  adaptive control) run on the calibrated virtual servo in
  [servo.py](../src/adapt_ctrl_sys_id/servo.py), which is tuned to
  reproduce the identified hardware constants K = 25.172 rad/s/V and
  T = 0.31919 s.
- **Slider positions.** `x_m` is 0.09 m (up) or 0.34 m (down), measured from
  the pivot.
