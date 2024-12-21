function d = acsi_place_pid(lin, omega, zeta, alpha, ts)
%ACSI_PLACE_PID Pole placement for b / (a2 s^2 + a1 s + a0) under PI-D feedback, plus PI-P coefficients.
%   The closed-loop polynomial  a2 s^3 + (a1 + b Kd) s^2 + (a0 + b Kp) s + b Ki
%   is matched to  a2 (s + alpha omega) (s^2 + 2 zeta omega s + omega^2).
%   Pass lin.b * 180/pi to obtain gains for angles in degrees.
%
%   Digital PI-P cascade (backward Euler, sample time ts):
%       u_PI(k) = u_PI(k-1) + q0_PI e(k) + q1_PI e(k-1)
%       u(k)    = q0_P (u_PI(k) - phid(k))
%   q0_PI = Kp/Kd + Ki ts/Kd,  q1_PI = -Kp/Kd,  q0_P = Kd

a2 = lin.a2;
a1 = lin.a1;
a0 = lin.a0;
b = lin.b;
d.Kd = (a2 * (2 * zeta + alpha) * omega - a1) / b;
d.Kp = (a2 * omega^2 * (1 + 2 * zeta * alpha) - a0) / b;
d.Ki = a2 * alpha * omega^3 / b;
d.poles = roots([a2, a1 + b * d.Kd, a0 + b * d.Kp, b * d.Ki]);
d.ts = ts;
d.q0_pi = d.Kp / d.Kd + d.Ki * ts / d.Kd;
d.q1_pi = -d.Kp / d.Kd;
d.q0_p = d.Kd;
end
