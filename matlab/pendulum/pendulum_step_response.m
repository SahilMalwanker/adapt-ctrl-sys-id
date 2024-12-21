%% Powered step response: propeller inflow adds damping
% The rotor command steps so that the pendulum moves from about 22 to 42 deg.
%   1. nonlinear models: lab model vs calibrated model; the fast decay of the
%      first swings needs a thrust-dependent damping d_u * u and a short rotor lag
%   2. discrete second-order ARX(2,2) identification of the same record
%      (50 ms sampling with a matching moving-average anti-alias filter)
% Python counterpart: adapt_ctrl_sys_id.pipeline.pendulum_study

if exist('acsi_params', 'file') ~= 2
    error('Run setup_adapt_ctrl_sys_id from the matlab folder first.');
end
p = acsi_params();
g = p.geometry;
cal = p.calibrated;
c = acsi_pendulum_coeffs(cal, g);

d = acsi_read('pendulum_step_response');
[t, phi_deg] = acsi_resample(d.t_s, 0.01, d.phi_deg);
phi = deg2rad(phi_deg);
phi1 = mean(phi(t >= 25 & t <= 35.5));
phi2 = mean(phi(t >= 55 & t <= 65));
u1 = c.a0 * sin(phi1) / c.b0;
u2 = c.a0 * sin(phi2) / c.b0;
fprintf('plateaus %.2f -> %.2f deg need u = %.2f -> %.2f %%\n', rad2deg(phi1), rad2deg(phi2), u1, u2);

win = t >= 33 & t <= 65;
tt = t(win);
yy = phi(win);
step_sim = @(model, t_step) acsi_pendulum_simulate(model, g, tt, u1 + (u2 - u1) * (tt >= t_step), phi1);

%% Fit thrust damping, rotor lag and the step instant (start from the stored values)
scale = [1e-4, 0.01, 0.1];
model = @(z) with_rotor(cal, max(z(1), 0) * scale(1), max(z(2), 0.1) * scale(2));
z = fminsearch(@(z) sum((step_sim(model(z), z(3) * scale(3)) - yy).^2), ...
               [cal.thrust_damping, cal.rotor_tau, 35.7] ./ scale, optimset('TolX', 1e-4, 'TolFun', 1e-8));
fitted = model(z);
t_step = z(3) * scale(3);
fprintf('fitted: d_u = %.2e N m s/%%, rotor lag %.0f ms, step at %.2f s\n', fitted.thrust_damping, ...
        1000 * fitted.rotor_tau, t_step);

lab = p.lab;
lab.thrust_gain = cal.thrust_gain;
no_du = with_rotor(cal, 0, 0);
y_lab = step_sim(lab, t_step);
y_cal = step_sim(fitted, t_step);
fprintf('NRMSE fit: lab model %.1f %%, without thrust damping %.1f %%, calibrated %.1f %%\n', ...
        acsi_fit_percent(yy, y_lab), acsi_fit_percent(yy, step_sim(no_du, t_step)), acsi_fit_percent(yy, y_cal));
after = tt >= t_step;
fprintf('measured overshoot %.0f %%\n', 100 * (max(yy(after)) - phi2) / (phi2 - phi1));

%% ARX(2,2) at 50 ms
ts = 0.05;
[tf_, fine_deg] = acsi_resample(d.t_s, 0.005, d.phi_deg);
smooth = movmean(fine_deg, round(ts / 0.005));
grid_t = (33:ts:65 - ts)';
dy = interp1(tf_, smooth, grid_t) - rad2deg(phi1);
du = (u2 - u1) * (grid_t >= t_step);
n = numel(grid_t);
Phi = [-dy(2:n - 1), -dy(1:n - 2), du(2:n - 1), du(1:n - 2)];
theta = Phi \ dy(3:n);
A = [1, theta(1), theta(2)];
B = [0, theta(3), theta(4)];
dy_sim = filter(B, A, du);
s = log(roots(A)) / ts;
wn = sqrt(real(s(1) * s(2)));
zeta = -real(s(1) + s(2)) / (2 * wn);
dc = sum(B) / sum(A);              % deg per %
a0_lin = c.J * wn^2;               % a transfer function fixes ratios only: take J from the free fall
fprintf('ARX(2,2): fit %.1f %%, wn = %.3f rad/s, zeta = %.4f\n', acsi_fit_percent(dy, dy_sim), wn, zeta);
fprintf('physical: a0,lin = %.3f N m (model %.3f), b0 = %.4f N m/%% (model %.4f)\n', a0_lin, ...
        c.a0 * cos(phi2), dc * a0_lin * pi / 180, c.b0);

%% Plot
figure('Name', 'Step response');
tiledlayout(2, 1, 'TileSpacing', 'compact');
nexttile;
plot(tt, rad2deg(yy), 'k', tt, rad2deg(y_lab), '--', tt, rad2deg(y_cal), 'LineWidth', 1.2);
grid on;
ylabel('\phi [deg]');
legend('measured', 'lab model', 'calibrated model', 'Location', 'southeast');
title('Nonlinear models');
nexttile;
plot(grid_t, dy, 'k', grid_t, dy_sim, '--', 'LineWidth', 1.3);
grid on;
xlabel('t [s]');
ylabel('\Delta\phi [deg]');
legend('measured (50 ms, anti-aliased)', 'ARX(2,2)', 'Location', 'southeast');
title('Discrete second-order identification');

%% Local functions
function m = with_rotor(m, thrust_damping, rotor_tau)
    m.thrust_damping = thrust_damping;
    m.rotor_tau = rotor_tau;
end
