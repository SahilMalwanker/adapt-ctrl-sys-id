%% Recursive least squares for the servo PT1
% Same experiment as servo_identification, processed sample by sample. The
% estimate converges to the batch least-squares solution for lambda = 1;
% a forgetting factor lambda < 1 keeps the estimator alert to plant changes
% at the price of noisier estimates (memory ~ 1 / (1 - lambda) samples).
% Python counterpart: adapt_ctrl_sys_id.identification.RecursiveLeastSquares

if exist('acsi_params', 'file') ~= 2
    error('Run setup_adapt_ctrl_sys_id from the matlab folder first.');
end
p = acsi_params();
sp = p.servo;

t = (0:sp.ts:15)';
u = acsi_servo_excitation(t);
w = zeros(size(t));
s = acsi_servo_init(sp);
for k = 1:numel(t)
    w(k) = s.omega_meas;
    s = acsi_servo_step(s, u(k), sp);
end

lambdas = [1, 0.995];
K_hat = nan(numel(t), numel(lambdas));
T_hat = nan(numel(t), numel(lambdas));
for j = 1:numel(lambdas)
    lambda = lambdas(j);
    theta = [0; 0];                   % [a1; b1] of w(k) = -a1 w(k-1) + b1 u(k-1)
    P = 1e4 * eye(2);
    for k = 2:numel(t)
        phi = [-w(k - 1); u(k - 1)];
        gain = P * phi / (lambda + phi' * P * phi);
        theta = theta + gain * (w(k) - phi' * theta);
        P = (P - gain * (phi' * P)) / lambda;
        a1 = theta(1);
        if a1 > -1 && a1 < 0          % only a pole in (0, 1) maps to a PT1
            K_hat(k, j) = theta(2) / (1 + a1);
            T_hat(k, j) = -sp.ts / log(-a1);
        end
    end
    fprintf('lambda = %.3f: final K = %.4f, T = %.4f s\n', lambda, K_hat(end, j), T_hat(end, j));
end

figure('Name', 'Servo RLS');
tiledlayout(2, 1, 'TileSpacing', 'compact');
nexttile;
plot(t, K_hat, 'LineWidth', 1.3);
yline(sp.K, 'k--');
ylim([0, 40]);
grid on;
ylabel('K estimate [rad/s/V]');
legend(compose('\\lambda = %.3f', lambdas), 'Location', 'southeast');
nexttile;
plot(t, T_hat, 'LineWidth', 1.3);
yline(sp.T, 'k--');
ylim([0, 0.8]);
grid on;
xlabel('t [s]');
ylabel('T estimate [s]');
