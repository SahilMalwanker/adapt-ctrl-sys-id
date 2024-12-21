%% Cascade position control of the servo: design from the actuator limit
% Plant: theta = K / (s (T s + 1)) u with the hardware PT1 (K, T).
% Controller: u = Kd (Kp (r - theta) - omega), saturated at +/- 10 V.
%
% Right after a reference step r0 the controller outputs u(0+) = Kd Kp r0.
% With Kd Kp = T s_vel^2 / (2 K) the step stays inside the amplifier limit if
%     s_vel >= -sqrt(2 K u_max / (r0 T))
% The integer rule round(-44.3 + 43 T) is a linear fit of that limit.
% The design uses 95 % of the limit and gives the position loop a damping
% ratio of 1/sqrt(2). Python counterpart: adapt_ctrl_sys_id.servo.cascade_gains

if exist('acsi_params', 'file') ~= 2
    error('Run setup_adapt_ctrl_sys_id from the matlab folder first.');
end
p = acsi_params();
sp = p.servo;
K = sp.K;
T = sp.T;
r0 = pi / 2;

s_limit = -sqrt(2 * K * sp.u_max / (r0 * T));
s_rule = round(-44.3 + 43 * T);
s_design = 0.95 * s_limit;
[kd, kp] = acsi_cascade_gains(K, T, s_design);
poles = roots([T, 1 + kd * K, kp * kd * K]);
fprintf('velocity pole: actuator limit %.3f, rule of thumb %d, design %.3f 1/s\n', s_limit, s_rule, s_design);
fprintf('Kd = %.4f V s/rad, Kp = %.4f 1/s, position poles %s\n', kd, kp, mat2str(poles.', 4));
fprintf('initial voltage for a %.3f rad step: %.2f V\n', r0, kd * kp * r0);

%% Step and sine tracking on the virtual servo (2 ms loop, encoder feedback)
runs = {struct('name', 'step', 't_end', 3, 'ref', @(t) r0 * (t >= 0.5)), ...
        struct('name', 'sine', 't_end', 12, 'ref', @(t) pi * sin(2 * pi / 6 * t))};
figure('Name', 'Servo cascade');
tiledlayout(2, 2, 'TileSpacing', 'compact');
for i = 1:numel(runs)
    cfg = runs{i};
    t = (0:sp.ts:cfg.t_end)';
    r = cfg.ref(t);
    theta = zeros(size(t));
    u = zeros(size(t));
    s = acsi_servo_init(sp);
    for k = 1:numel(t)
        theta(k) = s.theta_meas;
        u(k) = min(sp.u_max, max(-sp.u_max, kd * (kp * (r(k) - theta(k)) - s.omega_meas)));
        s = acsi_servo_step(s, u(k), sp);
    end
    if strcmp(cfg.name, 'step')
        fprintf('step: overshoot %.2f %%, peak voltage %.2f V\n', 100 * (max(theta) - r0) / r0, max(abs(u)));
    end
    nexttile(i);
    plot(t, r, 'k--', t, theta, 'LineWidth', 1.4);
    grid on;
    ylabel('\theta [rad]');
    title([cfg.name ' response']);
    nexttile(i + 2);
    plot(t, u, 'LineWidth', 1.2);
    yline([-sp.u_max, sp.u_max], ':');
    grid on;
    xlabel('t [s]');
    ylabel('u [V]');
end
