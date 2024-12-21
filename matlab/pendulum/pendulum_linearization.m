%% Where is the linearised pendulum model good enough?
% Linearising  J phi'' + a1 phi' + a0 sin(phi) = b0 u  around phi_op gives
%     J dphi'' + (d + d_u u_op) dphi' + a0 cos(phi_op) dphi = b0 du
% Both models receive the same physical input step (the nonlinear equilibrium
% inputs before and after), so static and dynamic errors both show up.
% Python counterpart: adapt_ctrl_sys_id.pendulum.step_comparison

if exist('acsi_params', 'file') ~= 2
    error('Run setup_adapt_ctrl_sys_id from the matlab folder first.');
end
p = acsi_params();
g = p.geometry;
model = p.calibrated;
model.rotor_tau = 0;      % rotor lag and dead band are not part of the linear model
model.deadband = 0;

%% Steps of +/-2 and +/-15 deg around 40 deg
t = (0:0.01:12)';
figure('Name', 'Linearisation');
hold on;
grid on;
colors = lines(2);
halves = [2, 15];
for i = 1:2
    [nl, li] = step_pair(model, g, deg2rad(40), deg2rad(halves(i)), t);
    plot(t, rad2deg(nl) - 40, '-', 'Color', colors(i, :), 'LineWidth', 1.8);
    plot(t, rad2deg(li) - 40, '--', 'Color', colors(i, :), 'LineWidth', 1.4);
    fprintf('step 40 +/- %g deg: max |error| %.2f deg\n', halves(i), rad2deg(max(abs(nl - li))));
end
xlabel('t [s]');
ylabel('\phi - 40 [deg]');
legend('nonlinear \pm2', 'linear \pm2', 'nonlinear \pm15', 'linear \pm15', 'Location', 'southeast');
title('Steps around 40 deg');

%% Coarse validity map: worst-case error [deg]; NaN = outside 0..75 deg
ops = 10:10:70;
steps = [1, 2, 5, 10, 20];
err = nan(numel(ops), numel(steps));
t = (0:0.01:15)';
for i = 1:numel(ops)
    for j = 1:numel(steps)
        if ops(i) - steps(j) >= 0 && ops(i) + steps(j) <= 75
            [nl, li] = step_pair(model, g, deg2rad(ops(i)), deg2rad(steps(j)), t);
            err(i, j) = rad2deg(max(abs(nl - li)));
        end
    end
end
disp(array2table(round(err, 2), 'VariableNames', compose('pm%g', steps), 'RowNames', compose('%g deg', ops)));

%% Local functions
function [nl, li] = step_pair(model, g, phi_op, half, t)
    % Nonlinear and linearised responses to the step phi_op - half -> phi_op + half.
    c = acsi_pendulum_coeffs(model, g);
    ua = c.a0 * sin(phi_op - half) / c.b0;
    ub = c.a0 * sin(phi_op + half) / c.b0;
    u = ub * ones(size(t));
    u(1) = ua;
    nl = acsi_pendulum_simulate(model, g, t, u, phi_op - half);
    lin = acsi_pendulum_linearize(model, g, phi_op);
    li = phi_op + linear_response(lin, t, u - lin.u_op, lin.b * (ua - lin.u_op) / lin.a0);
end

function x_out = linear_response(lin, t, du, x0)
    % RK4 with the input held per sample (same scheme as the nonlinear model).
    x_out = zeros(size(t));
    x = x0;
    v = 0;
    acc = @(x, v, u) (lin.b * u - lin.a1 * v - lin.a0 * x) / lin.a2;
    for k = 1:numel(t)
        x_out(k) = x;
        if k < numel(t)
            h = t(k + 1) - t(k);
            uk = du(k);
            k1x = v;
            k1v = acc(x, v, uk);
            k2x = v + 0.5 * h * k1v;
            k2v = acc(x + 0.5 * h * k1x, v + 0.5 * h * k1v, uk);
            k3x = v + 0.5 * h * k2v;
            k3v = acc(x + 0.5 * h * k2x, v + 0.5 * h * k2v, uk);
            k4x = v + h * k3v;
            k4v = acc(x + h * k3x, v + h * k3v, uk);
            x = x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x);
            v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v);
        end
    end
end
