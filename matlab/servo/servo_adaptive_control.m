%% Adaptive cascade control: the disk becomes 1.85x heavier at t = 4.9 s
% Fixed gains (designed for the nominal disk) against an indirect adaptive
% cascade. Every 10 samples an RLS estimator updates a model of the encoder
% increments d(m) = theta(m) - theta(m-1); for a zero-order-held
% K / (s (T s + 1)) they obey exactly
%
%     d(m) = a d(m-1) + b1 u(m-1) + b2 u(m-2),   a = exp(-h/T),  b1 + b2 = K h (1 - a)
%
% Using increments instead of the filtered speed keeps the encoder
% quantisation noise from biasing the estimate. The estimate (K, T) is
% projected into plausible bounds and pushed through the same actuator-aware
% design rule as servo_cascade_design. Python counterpart:
% adapt_ctrl_sys_id.servo.AdaptiveCascadeController

if exist('acsi_params', 'file') ~= 2
    error('Run setup_adapt_ctrl_sys_id from the matlab folder first.');
end
p = acsi_params();
sp = p.servo;
ratio = 3 * sqrt(0.7 - sp.T);     % heavier disk: T2 / T1 = 1.851
t_switch = 4.9;
r0 = pi / 2;
t = (0:sp.ts:12)';
ref = r0 * (mod(t, 2) < 1);       % square wave 0 <-> pi/2, period 2 s

nominal = design(sp.K, sp.T, sp.u_max, r0);
names = {'fixed', 'adaptive'};
logs = struct();
for j = 1:2
    ad = adaptive_init(sp.K, sp.T, sp.ts, sp.u_max, r0);
    s = acsi_servo_init(sp);
    n = numel(t);
    theta = zeros(n, 1);
    u = zeros(n, 1);
    T_hat = zeros(n, 1);
    for k = 1:n
        if t(k) >= t_switch
            s.inertia_scale = ratio;
        end
        if j == 1
            u(k) = saturate(nominal.kd * (nominal.kp * (ref(k) - s.theta_meas) - s.omega_meas), sp.u_max);
            T_hat(k) = sp.T;
        else
            [u(k), ad] = adaptive_step(ad, ref(k), s.theta_meas, s.omega_meas);
            T_hat(k) = ad.T;
        end
        theta(k) = s.theta_meas;
        s = acsi_servo_step(s, u(k), sp);
    end
    logs.(names{j}) = struct('theta', theta, 'u', u, 'T_hat', T_hat);
    shoots = zeros(1, 3);
    edges = [6, 8, 10];
    for e = 1:3
        window = t >= edges(e) & t < edges(e) + 0.95;
        shoots(e) = 100 * max(0, max(theta(window)) - r0) / r0;
    end
    after = t >= t_switch;
    fprintf('%-8s overshoot after the change %.1f %%, IAE %.3f rad s\n', names{j}, mean(shoots), ...
            sum(abs(ref(after) - theta(after))) * sp.ts);
end

figure('Name', 'Adaptive servo');
tiledlayout(3, 1, 'TileSpacing', 'compact');
nexttile;
plot(t, ref, 'k--', t, logs.fixed.theta, t, logs.adaptive.theta, 'LineWidth', 1.3);
xline(t_switch, ':', 'heavier disk');
grid on;
ylabel('\theta [rad]');
legend('reference', 'fixed gains', 'adaptive', 'Location', 'southeast');
nexttile;
plot(t, logs.fixed.u, t, logs.adaptive.u, 'LineWidth', 1.1);
grid on;
ylabel('u [V]');
nexttile;
plot(t, sp.tau * (1 + (ratio - 1) * (t >= t_switch)), 'k--', t, logs.adaptive.T_hat, 'LineWidth', 1.4);
grid on;
xlabel('t [s]');
ylabel('T estimate [s]');
legend('virtual servo time constant', 'RLS estimate', 'Location', 'southeast');

%% Local functions
function g = design(K, T, u_max, r0)
    s_vel = 0.95 * -sqrt(2 * K * u_max / (r0 * T));
    [g.kd, g.kp] = acsi_cascade_gains(K, T, s_vel);
end

function u = saturate(u, u_max)
    u = min(u_max, max(-u_max, u));
end

function ad = adaptive_init(K, T, ts, u_max, r0)
    ad.decimation = 10;
    ad.h = ts * ad.decimation;
    ad.lambda = 0.98;
    ad.max_trace = 1;
    ad.u_max = u_max;
    ad.r0 = r0;
    a = exp(-ad.h / T);
    lag = T * (1 - a);
    ad.est = [a; K * (ad.h - lag) / ad.h; K * (lag - a * ad.h) / ad.h];   % [a; b1/h; b2/h]
    ad.P = 1e-2 * eye(3);
    ad.K = K;
    ad.T = T;
    g = design(K, T, u_max, r0);
    ad.kd = g.kd;
    ad.kp = g.kp;
    ad.k = 0;
    ad.u_sum = 0;
    ad.theta_prev = NaN;
    ad.rate_prev = NaN;
    ad.u_prev = NaN;
end

function [u, ad] = adaptive_step(ad, ref, theta, omega)
    if mod(ad.k, ad.decimation) == 0
        if ~isnan(ad.theta_prev)
            rate = (theta - ad.theta_prev) / ad.h;
            u_mean = ad.u_sum / ad.decimation;
            if ~isnan(ad.rate_prev)
                phi = [ad.rate_prev; u_mean; ad.u_prev];
                Pphi = ad.P * phi;
                gain = Pphi / (ad.lambda + phi' * Pphi);
                ad.est = ad.est + gain * (rate - phi' * ad.est);
                ad.P = (ad.P - gain * Pphi') / ad.lambda;
                ad.P = (ad.P + ad.P') / 2;
                if trace(ad.P) > ad.max_trace
                    ad.P = ad.P * ad.max_trace / trace(ad.P);
                end
                a = ad.est(1);
                c = ad.est(2) + ad.est(3);
                if a > 0 && a < 1 && c > 0      % otherwise keep the last valid model
                    ad.K = min(max(c / (1 - a), 5), 80);
                    ad.T = min(max(-ad.h / log(a), 0.05), 3);
                    g = design(ad.K, ad.T, ad.u_max, ad.r0);
                    ad.kd = g.kd;
                    ad.kp = g.kp;
                end
            end
            ad.rate_prev = rate;
            ad.u_prev = u_mean;
        end
        ad.theta_prev = theta;
        ad.u_sum = 0;
    end
    u = saturate(ad.kd * (ad.kp * (ref - theta) - omega), ad.u_max);
    ad.u_sum = ad.u_sum + u;
    ad.k = ad.k + 1;
end
