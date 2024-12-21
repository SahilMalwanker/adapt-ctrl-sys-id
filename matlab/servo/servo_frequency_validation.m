%% Frequency-response validation of the servo model
% Three sine experiments on the physical servo (data/servo_frequency_response.csv)
% are compared with
%   * the linear hardware PT1  K / (T s + 1)
%   * sine experiments on the calibrated virtual servo, evaluated with a
%     correlation (lock-in) analyser over whole periods.
% The virtual servo carries the quadratic static map, which lifts the gain at
% small amplitudes - exactly where the linear PT1 under-predicts the hardware.
% Python counterpart: adapt_ctrl_sys_id.frequency.correlation_analysis

if exist('acsi_params', 'file') ~= 2
    error('Run setup_adapt_ctrl_sys_id from the matlab folder first.');
end
p = acsi_params();
sp = p.servo;
hw = acsi_read('servo_frequency_response');

amplitude = 1.0;      % [V]
periods = 10;         % analysed periods
settle = 4;           % discarded periods (transient)
n = height(hw);
G_lin = sp.K ./ (1 + 1j * hw.omega_rad_s * sp.T);
G_virtual = zeros(n, 1);
for i = 1:n
    w0 = hw.omega_rad_s(i);
    t = (0:sp.ts:(periods + settle) * 2 * pi / w0)';
    u = amplitude * sin(w0 * (t + 0.5 * sp.ts));   % mid-sample: no half-sample ZOH lag in the estimate
    y = zeros(size(t));
    s = acsi_servo_init(sp);
    for k = 1:numel(t)
        y(k) = s.omega_meas;
        s = acsi_servo_step(s, u(k), sp);
    end
    G_virtual(i) = lockin(t, y, w0, amplitude, t(end) - periods * 2 * pi / w0);
end

db = @(G) 20 * log10(abs(G));
deg = @(G) angle(G) * 180 / pi;
result = table(hw.omega_rad_s, hw.magnitude_db, db(G_lin), db(G_virtual), hw.phase_deg, deg(G_lin), deg(G_virtual), ...
               'VariableNames', {'omega', 'hw_dB', 'pt1_dB', 'virtual_dB', 'hw_deg', 'pt1_deg', 'virtual_deg'});
disp(result);
fprintf('mean |magnitude error|: linear PT1 %.2f dB, calibrated virtual servo %.2f dB\n', ...
        mean(abs(db(G_lin) - hw.magnitude_db)), mean(abs(db(G_virtual) - hw.magnitude_db)));

%% Bode plot
wg = logspace(-1, 2, 400)';
Gg = sp.K ./ (1 + 1j * wg * sp.T);
figure('Name', 'Servo frequency validation');
tiledlayout(2, 1, 'TileSpacing', 'compact');
nexttile;
semilogx(wg, db(Gg), 'LineWidth', 1.4);
hold on;
semilogx(hw.omega_rad_s, hw.magnitude_db, 'ko', 'MarkerFaceColor', 'k');
semilogx(hw.omega_rad_s, db(G_virtual), 's', 'MarkerSize', 8, 'LineWidth', 1.4);
grid on;
ylabel('|G| [dB]');
legend('hardware PT1', 'hardware sine tests', 'virtual servo (lock-in)', 'Location', 'southwest');
nexttile;
semilogx(wg, deg(Gg), 'LineWidth', 1.4);
hold on;
semilogx(hw.omega_rad_s, hw.phase_deg, 'ko', 'MarkerFaceColor', 'k');
semilogx(hw.omega_rad_s, deg(G_virtual), 's', 'MarkerSize', 8, 'LineWidth', 1.4);
grid on;
xlabel('\omega [rad/s]');
ylabel('phase [deg]');

function G = lockin(t, y, w0, amplitude, t_start)
    % Correlate the output with sin and cos over the largest whole number of periods after t_start.
    keep = t >= t_start;
    t = t(keep);
    y = y(keep);
    period = 2 * pi / w0;
    n_periods = floor((t(end) - t(1)) / period + 1e-9);
    keep = t <= t(1) + n_periods * period + 1e-12;
    t = t(keep);
    y = y(keep);
    window = t(end) - t(1);
    G = 2 / (amplitude * window) * (trapz(t, y .* sin(w0 * t)) + 1j * trapz(t, y .* cos(w0 * t)));
end
