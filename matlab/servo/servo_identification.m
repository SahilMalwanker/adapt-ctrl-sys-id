%% PT1 identification of the servo speed
% Excite the calibrated virtual servo with the standard voltage profile and
% estimate G(s) = K / (T s + 1) twice:
%   * equation error (ARX least squares) - fast, but biased by sensor noise
%   * output error (simulation error)    - what the model is used for
% then compare both with the estimate obtained on the physical servo.
% Python counterpart: adapt_ctrl_sys_id.pipeline.servo_study

if exist('acsi_params', 'file') ~= 2
    error('Run setup_adapt_ctrl_sys_id from the matlab folder first.');
end
p = acsi_params();
sp = p.servo;

%% Experiment on the virtual servo (2 ms loop, encoder + filtered speed)
t = (0:sp.ts:15)';
u = acsi_servo_excitation(t);
w = zeros(size(t));
s = acsi_servo_init(sp);
for k = 1:numel(t)
    w(k) = s.omega_meas;            % sample k is taken before u(k) acts
    s = acsi_servo_step(s, u(k), sp);
end

%% Equation error: w(k) = -a1 w(k-1) + b1 u(k-1)
ab = [-w(1:end - 1), u(1:end - 1)] \ w(2:end);
a1 = ab(1);
b1 = ab(2);
T_arx = -sp.ts / log(-a1);
K_arx = b1 / (1 + a1);

%% Output error: minimise ||w - G u||
pt1 = @(K, T) filter([0, K * (1 - exp(-sp.ts / T))], [1, -exp(-sp.ts / T)], u);
cost = @(q) sum((pt1(q(1), exp(q(2))) - w).^2);
q = fminsearch(cost, [K_arx, log(T_arx)], optimset('TolX', 1e-8, 'TolFun', 1e-8));
K_oe = q(1);
T_oe = exp(q(2));

report = @(label, K, T) fprintf('%-22s K = %.4f, T = %.4f s, fit %.2f %%\n', label, K, T, ...
                                acsi_fit_percent(w, pt1(K, T)));
report('ARX (equation error):', K_arx, T_arx);
report('output error:', K_oe, T_oe);
report('hardware estimate:', sp.K, sp.T);
fprintf('corner frequency of the hardware model 1/T = %.4f rad/s\n', 1 / sp.T);

% zero-order-hold model for the 2 ms real-time loop (what c2d(tf(K, [T 1]), ts, 'zoh') returns)
pole = exp(-sp.ts / T_oe);
fprintf('discrete model: w(k) = %.6f w(k-1) + %.6f u(k-1)\n', pole, K_oe * (1 - pole));

%% Plot
figure('Name', 'Servo identification');
tiledlayout(2, 1, 'TileSpacing', 'compact');
nexttile;
plot(t, w, 'k', t, pt1(K_oe, T_oe), '-', t, pt1(sp.K, sp.T), '--', 'LineWidth', 1.2);
grid on;
ylabel('\omega [rad/s]');
legend('virtual servo', 'output-error PT1', 'hardware PT1', 'Location', 'northeast');
title('Speed response to the excitation profile');
nexttile;
stairs(t, u, 'LineWidth', 1.2);
grid on;
xlabel('t [s]');
ylabel('u [V]');
