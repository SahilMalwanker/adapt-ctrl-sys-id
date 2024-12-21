%% Static characteristic of the DC servo
% Steady-state speed for constant voltages in both directions. A line through
% the origin misses the falling gain at high speed; a quadratic loss term
%
%     u = w / K0 + beta * w * |w|
%
% captures it and is still linear in the unknowns (1/K0, beta).
% Python counterpart: adapt_ctrl_sys_id.identification.fit_quadratic_static_map

if exist('acsi_params', 'file') ~= 2
    error('Run setup_adapt_ctrl_sys_id from the matlab folder first.');
end

d = acsi_read('servo_static_map');
u = d.u_V;
w = d.omega_rad_s;

%% Fits
K_lin = (u' * w) / (u' * u);            % w = K u, least squares through the origin
theta = [w, w .* abs(w)] \ u;           % u = w/K0 + beta w|w|
K0 = 1 / theta(1);
beta = theta(2);
speed = @(v) sign(v) .* (-1 / K0 + sqrt(1 / K0^2 + 4 * beta * abs(v))) / (2 * beta);

rms_lin = sqrt(mean((w - K_lin * u).^2));
rms_quad = sqrt(mean((w - speed(u)).^2));
fprintf('linear map:    K  = %.4f rad/s/V              rms error %.2f rad/s\n', K_lin, rms_lin);
fprintf('quadratic map: K0 = %.4f rad/s/V, beta = %.3g  rms error %.2f rad/s\n', K0, beta, rms_quad);
fprintf('(Python reference: K = 24.5646, K0 = 28.6252, beta = 3.249e-05, rms 4.09 vs 1.88)\n');

%% Plot
uu = linspace(-max(abs(u)), max(abs(u)), 401)';
uu(uu == 0) = [];
figure('Name', 'Servo static map');
tiledlayout(1, 2, 'TileSpacing', 'compact');
nexttile;
plot(u, w, 'ko', 'MarkerFaceColor', 'k');
hold on;
plot(uu, K_lin * uu, '--', 'LineWidth', 1.4);
plot(uu, speed(uu), '-', 'LineWidth', 1.8);
grid on;
xlabel('u [V]');
ylabel('\omega [rad/s]');
title('Steady-state speed');
legend('measured', sprintf('line, K = %.2f', K_lin), 'quadratic loss', 'Location', 'northwest');
nexttile;
plot(u, w ./ u, 'ko', 'MarkerFaceColor', 'k');
hold on;
plot(uu, speed(uu) ./ uu, '-', 'LineWidth', 1.8);
yline(K_lin, '--');
grid on;
xlabel('u [V]');
ylabel('\omega / u [rad/s/V]');
title('Effective gain falls with speed');
