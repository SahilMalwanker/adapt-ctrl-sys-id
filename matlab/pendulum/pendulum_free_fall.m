%% Free fall of the propeller pendulum: inertia, damping and air drag
% The pendulum is held at about 91 deg and released with the rotor off.
%   1. geometry:     l(x) and J(x) from masses and lengths
%   2. energy bound: J <= 2 m g l (1 - cos phi0) / phid_max^2
%   3. envelope:     exponential decay of the swing peaks -> viscous d
%   4. fit:          J, d, quadratic drag c and the release instant by output error
% The geometry-only model drifts out of phase within a few swings; the fitted
% model follows the whole 60 s record. Python counterpart: adapt_ctrl_sys_id.pendulum.fit_free_fall

if exist('acsi_params', 'file') ~= 2
    error('Run setup_adapt_ctrl_sys_id from the matlab folder first.');
end
p = acsi_params();
g = p.geometry;
lab = acsi_pendulum_coeffs(p.lab, g);
fprintf('geometry: l(x) = %.4f m, J(x) = %.5f kg m^2, a0 = m g l = %.4f N m\n', lab.l, lab.J, lab.a0);

%% Record (potentiometer angle; the IMU axis points the other way -> sign flip)
d = acsi_read('pendulum_free_fall');
[t, phi_deg, imu_deg_s] = acsi_resample(d.t_s, 0.01, d.phi_deg, -d.imu_rate_deg_s);
phi = deg2rad(phi_deg);
phid = deg2rad(imu_deg_s);

phi0 = median(phi(1:max(10, floor(numel(phi) / 200))));
t_rel0 = t(find(abs(phi - phi0) > deg2rad(1), 1));
after = t > t_rel0;
J_bound = 2 * lab.a0 * (1 - cos(phi0)) / max(abs(phid(after)))^2;
fprintf('release at %.2f s from %.2f deg; energy bound J <= %.5f kg m^2\n', t_rel0, rad2deg(phi0), J_bound);

%% Envelope of the positive swing peaks
[tp, ap] = swing_peaks(t(after), phi(after));
coef = polyfit(tp, log(ap), 1);
sigma = -coef(1);
fprintf('envelope decay %.4f 1/s -> d = 2 J sigma = %.5f N m s (geometry J)\n', sigma, 2 * lab.J * sigma);

%% Output-error fit of J, d, c and the release instant
% z holds scaled parameters [J, d, t_release, c] ./ scale so that fminsearch
% sees numbers of similar size; d and c are kept non-negative.
scale = [0.01, 0.001, 0.1, 1e-4];
model = @(z) free_fall(t, phi0, z(3) * scale(3), lab.a0, z(1) * scale(1), max(z(2), 0) * scale(2), ...
                       max(z(4), 0) * scale(4));
z0 = [lab.J, 0.004, t_rel0 - 0.05, 5e-4] ./ scale;
opts = optimset('MaxFunEvals', 3000, 'MaxIter', 3000, 'TolX', 1e-6, 'TolFun', 1e-10);
z = fminsearch(@(z) sum((model(z) - phi).^2), z0, opts);
J_fit = z(1) * scale(1);
d_fit = max(z(2), 0) * scale(2);
t_rel = z(3) * scale(3);
c_fit = max(z(4), 0) * scale(4);
phi_cal = model(z);
phi_geo = free_fall(t, phi0, t_rel0, lab.a0, lab.J, p.lab.viscous, 0);
fprintf('fit: J = %.5f kg m^2, d = %.5f N m s, c = %.2g N m s^2\n', J_fit, d_fit, c_fit);
fprintf('(Python reference: J = 0.05372, d = 0.00211, c = 5.1e-4)\n');
fprintf('NRMSE fit: geometry model %.1f %%, calibrated model %.1f %%\n', ...
        acsi_fit_percent(phi, phi_geo), acsi_fit_percent(phi, phi_cal));

%% IMU cross-check: integrated rate vs potentiometer
pot_rate = gradient(movmean(phi_deg, 7), t);
lin = polyfit(pot_rate, imu_deg_s, 1);
drift = polyfit(t, cumtrapz(t, imu_deg_s) + phi_deg(1) - phi_deg, 1);
fprintf('IMU vs potentiometer: scale %.3f, bias %+.2f deg/s, integrated drift %+.2f deg/s\n', lin(1), lin(2), drift(1));

%% Plot
figure('Name', 'Free fall');
tiledlayout(2, 1, 'TileSpacing', 'compact');
nexttile;
plot(t, phi_deg, 'k', t, rad2deg(phi_geo), '--', t, rad2deg(phi_cal), 'LineWidth', 1.1);
grid on;
ylabel('\phi [deg]');
legend('measured', 'geometry model', 'calibrated model', 'Location', 'northeast');
title('Full record');
nexttile;
win = t > 38 & t < 46;
plot(t(win), phi_deg(win), 'k', t(win), rad2deg(phi_geo(win)), '--', t(win), rad2deg(phi_cal(win)), 'LineWidth', 1.5);
grid on;
xlabel('t [s]');
ylabel('\phi [deg]');
title('38-46 s: the geometry model is out of phase');

%% Local functions
function phi_sim = free_fall(t, phi0, t_rel, a0, J, d, c)
    % Unpowered pendulum held at phi0 until t_rel:  J phi'' = -a0 sin(phi) - d phi' - c phi' |phi'|
    phi_sim = phi0 * ones(size(t));
    after = t > t_rel;
    if any(after)
        rhs = @(~, x) [x(2); (-a0 * sin(x(1)) - d * x(2) - c * x(2) * abs(x(2))) / J];
        [~, x] = ode45(rhs, [t_rel; t(after)], [phi0; 0], odeset('RelTol', 1e-8, 'AbsTol', 1e-10));
        phi_sim(after) = x(2:end, 1);
    end
end

function [tp, ap] = swing_peaks(t, phi)
    % Positive local maxima above 3 deg, at least 0.5 s apart.
    is_peak = [false; phi(2:end - 1) > phi(1:end - 2) & phi(2:end - 1) >= phi(3:end); false];
    is_peak = is_peak & phi > deg2rad(3);
    idx = find(is_peak);
    keep = [true; diff(t(idx)) > 0.5];
    tp = t(idx(keep));
    ap = phi(idx(keep));
end
