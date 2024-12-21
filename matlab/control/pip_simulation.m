%% Digital PI-P cascade on the calibrated nonlinear pendulum
% Sampled-data simulation of three controllers with the same PID gains:
%   * PI-P cascade (2-DOF: no derivative kick on reference steps)
%   * PI-P + gravity feedforward u_ff = a0 sin(r) / b0 + anti-windup (conditioning)
%   * 1-DOF parallel PID with a filtered derivative on the error, run at 1 kHz
% The rotor accepts 0..100 %; costs are evaluated between 30 and 100 s:
%   J_e = mean((r - phi)^2) [deg^2],  J_u = mean(u^2) [%^2]
% Python counterpart: adapt_ctrl_sys_id.control.simulate_loop

if exist('acsi_params', 'file') ~= 2
    error('Run setup_adapt_ctrl_sys_id from the matlab folder first.');
end
p = acsi_params();
g = p.geometry;
cal = p.calibrated;
dz = p.design;
c = acsi_pendulum_coeffs(cal, g);
lin = acsi_pendulum_linearize(cal, g, deg2rad(dz.phi_op_deg));
lin.b = lin.b * 180 / pi;
d = acsi_place_pid(lin, dz.omega, dz.zeta, dz.alpha, dz.ts);
feedforward = @(r) max(0, c.a0 * sind(r) / c.b0);

configs = {struct('name', 'PI-P cascade', 'type', 'pip', 'ts', 0.01, 'ff', false, 'aw', false), ...
           struct('name', 'PI-P + feedforward + anti-windup', 'type', 'pip', 'ts', 0.01, 'ff', true, 'aw', true), ...
           struct('name', '1-DOF PID, same gains', 'type', 'pid', 'ts', 0.001, 'ff', false, 'aw', false)};
t_end = 120;
figure('Name', 'PI-P simulation');
tiledlayout(2, 1, 'TileSpacing', 'compact');
ax1 = nexttile;
hold on;
grid on;
ax2 = nexttile;
hold on;
grid on;
for i = 1:numel(configs)
    cfg = configs{i};
    substeps = 1 + 4 * (cfg.ts >= 0.01);        % 2 ms integration steps at 100 Hz, 1 ms at 1 kHz
    n = round(t_end / cfg.ts) + 1;
    t = (0:n - 1)' * cfg.ts;
    r = acsi_reference(t);
    phi = zeros(n, 1);
    u = zeros(n, 1);
    x = [0; 0; 0];
    st = struct('u_pi', 0, 'e_prev', NaN, 'integral', 0, 'deriv', 0);
    for k = 1:n
        phi(k) = rad2deg(x(1));
        phid = rad2deg(x(2));
        if strcmp(cfg.type, 'pip')
            u_ff = 0;
            if cfg.ff
                u_ff = feedforward(r(k));
            end
            [u(k), st] = pip_step(st, d, r(k), phi(k), phid, u_ff, cfg.aw);
        else
            [u(k), st] = pid_step(st, d, cfg.ts, r(k), phi(k));
        end
        for s = 1:substeps
            x = acsi_pendulum_step(x, u(k), cfg.ts / substeps, cal, c);
        end
    end
    window = t >= 30 & t <= 100;
    fprintf('%-34s J_e = %.3f deg^2, J_u = %.1f %%^2\n', cfg.name, mean((r(window) - phi(window)).^2), ...
            mean(u(window).^2));
    plot(ax1, t, phi, 'LineWidth', 1.3, 'DisplayName', cfg.name);
    plot(ax2, t, u, 'LineWidth', 1.0);
end
plot(ax1, t, r, 'k--', 'DisplayName', 'reference');
ylabel(ax1, '\phi [deg]');
legend(ax1, 'Location', 'northwest');
xlabel(ax2, 't [s]');
ylabel(ax2, 'u [%]');
fprintf('(Python reference: J_e 1.15 / 1.07 / 1.10 deg^2)\n');

%% Local functions
function [u_sat, st] = pip_step(st, d, r, y, ydot, u_ff, anti_windup)
    % u_PI(k) = u_PI(k-1) + q0_PI e(k) + q1_PI e(k-1),   u(k) = q0_P (u_PI(k) - ydot(k)) + u_ff
    e = r - y;
    e_prev = st.e_prev;
    if isnan(e_prev)
        e_prev = 0;
    end
    u_pi = st.u_pi + d.q0_pi * e + d.q1_pi * e_prev;
    u = d.q0_p * (u_pi - ydot) + u_ff;
    u_sat = min(100, max(0, u));
    if anti_windup && u_sat ~= u
        u_pi = (u_sat - u_ff) / d.q0_p + ydot;   % conditioning: the state that reproduces the limit
    end
    st.u_pi = u_pi;
    st.e_prev = e;
end

function [u, st] = pid_step(st, d, ts, r, y)
    % Parallel PID, derivative of the error with a first-order filter (N = 100).
    tau_d = 1 / 100;
    e = r - y;
    e_prev = st.e_prev;
    if isnan(e_prev)
        e_prev = e;
    end
    st.integral = st.integral + d.Ki * ts * e;
    st.deriv = (tau_d * st.deriv + d.Kd * (e - e_prev)) / (tau_d + ts);
    st.e_prev = e;
    u = min(100, max(0, d.Kp * e + st.integral + st.deriv));
end
