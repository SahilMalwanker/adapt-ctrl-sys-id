%% PI-P cascade design for the propeller pendulum
% Plant (linearised at 40 deg, angle in degrees, rotor command in %):
%     G(s) = b / (a2 s^2 + a1 s + a0)
% PI-D law (derivative on the measurement only), realised as a cascade of an
% outer PI loop that produces a rate set-point and an inner P loop on the rate:
%     u_PI = (Kp/Kd) e + (Ki/Kd)/s e,      u = Kd (u_PI - phid)
% Pole placement: closed loop = a2 (s + alpha w)(s^2 + 2 zeta w s + w^2).
% Python counterpart: adapt_ctrl_sys_id.control.place_pid / PIPCoefficients

if exist('acsi_params', 'file') ~= 2
    error('Run setup_adapt_ctrl_sys_id from the matlab folder first.');
end
p = acsi_params();
dz = p.design;
lin = acsi_pendulum_linearize(p.calibrated, p.geometry, deg2rad(dz.phi_op_deg));
lin_deg = lin;
lin_deg.b = lin.b * 180 / pi;           % angle in degrees, input in %

d = acsi_place_pid(lin_deg, dz.omega, dz.zeta, dz.alpha, dz.ts);
fprintf('operating point %g deg: u_op = %.2f %%, wn = %.3f rad/s, zeta = %.4f (open loop)\n', ...
        dz.phi_op_deg, lin.u_op, lin.wn, lin.zeta);
fprintf('PID gains: Kp = %.4f %%/deg, Ki = %.4f %%/(deg s), Kd = %.4f %%/(deg/s)\n', d.Kp, d.Ki, d.Kd);
fprintf('closed-loop poles: %s\n', mat2str(d.poles.', 4));
fprintf('PI-P coefficients (Ts = %g s): q0_PI = %.4f, q1_PI = %.4f, q0_P = %.4f\n', d.ts, d.q0_pi, d.q1_pi, d.q0_p);
fprintf('(Python reference: Kp 5.6242, Ki 18.2513, Kd 0.7786, poles -8, -7.2 +/- 3.49j)\n');

%% Optional: compare with an automatic PI-D tuning (Control System Toolbox)
if exist('pidtune', 'file') == 2
    G = tf(lin_deg.b, [lin.a2, lin.a1, lin.a0]);
    C = pidtune(G, 'PI-D');
    fprintf('pidtune PI-D for comparison: Kp = %.4f, Ki = %.4f, Kd = %.4f\n', C.Kp, C.Ki, C.Kd);
end
