function s = acsi_servo_step(s, u, p)
%ACSI_SERVO_STEP Advance the calibrated virtual servo by one sample with the voltage u held.
%   s = ACSI_SERVO_STEP(s, u, p) with p = ACSI_PARAMS().servo and s from ACSI_SERVO_INIT.
%
%   Plant:  tau * J_scale * w' = K0 * (u - beta w|w|) - w   (quadratic static map)
%   Sensor: 2048-count encoder, speed = low-pass filtered finite difference,
%   so a controller only sees s.theta_meas and s.omega_meas.

h = p.ts;
u = min(p.u_max, max(-p.u_max, u));
w = s.omega;
k1 = accel(w, u, p, s.inertia_scale);
k2 = accel(w + 0.5 * h * k1, u, p, s.inertia_scale);
k3 = accel(w + 0.5 * h * k2, u, p, s.inertia_scale);
k4 = accel(w + h * k3, u, p, s.inertia_scale);
s.theta = s.theta + h / 6 * (w + 2 * (w + 0.5 * h * k1) + 2 * (w + 0.5 * h * k2) + (w + h * k3));
s.omega = w + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);

theta_meas = round(s.theta / s.quantum) * s.quantum;
raw_rate = (theta_meas - s.theta_meas) / h;
s.theta_meas = theta_meas;
alpha = h / (p.vel_filter_tc + h);
s.omega_meas = s.omega_meas + alpha * (raw_rate - s.omega_meas);
end

function a = accel(w, u, p, inertia_scale)
a = (p.K0 * (u - p.beta * w * abs(w)) - w) / (p.tau * inertia_scale);
end
