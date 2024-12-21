%% Static equilibria: thrust map and centre of gravity
% Constant rotor commands hold the pendulum at an equilibrium where the rotor
% torque balances gravity:  F(u) l_r = m g l(x) sin(phi).
%   * F(u) from the measured angles -> thrust gain k (line through the origin,
%     ignoring the start-up dead band below about 6 %)
%   * the same command at two slider positions x, x' gives the centre of gravity
%         l(x) = (ms/m) (x - x') / (1 - sin(phi)/sin(phi'))
% Python counterpart: adapt_ctrl_sys_id.pendulum.thrust_from_angle / com_from_two_positions

if exist('acsi_params', 'file') ~= 2
    error('Run setup_adapt_ctrl_sys_id from the matlab folder first.');
end
p = acsi_params();
g = p.geometry;
st = acsi_read('pendulum_static_equilibrium');
up = st.x_m == g.x_up;
down = st.x_m == g.x_down;

%% Thrust from the equilibrium angle
first_moment = st.x_m * g.ms + g.lb / 2 * g.mb + g.lr * g.mr;
F = sind(st.phi_deg) * g.g .* first_moment / g.lr;
active = st.u_pct > 5;
k = (st.u_pct(active)' * F(active)) / (st.u_pct(active)' * st.u_pct(active));
fprintf('thrust gain k = %.5f N/%%  ->  b0 = k l_r = %.5f N m/%%\n', k, k * g.lr);

%% Centre of gravity from two slider positions with the same command (30 %)
phi_up = deg2rad(st.phi_deg(up & st.u_pct == 30));
phi_down = deg2rad(st.phi_deg(down & st.u_pct == 30));
com = @(a, b) g.ms / g.m * (g.x_up - g.x_down) / (1 - sin(a) / sin(b));
l_exp = com(phi_up, phi_down);
one = deg2rad(1);
combos = [-1, -1; -1, 1; 1, -1; 1, 1] * one;
band = arrayfun(@(i) com(phi_up + combos(i, 1), phi_down + combos(i, 2)), 1:4);
l_geo = (g.x_up * g.ms + g.lb / 2 * g.mb + g.lr * g.mr) / g.m;
fprintf('centre of gravity: geometry %.4f m, experiment %.4f m (%.4f .. %.4f m for +/-1 deg reading error)\n', ...
        l_geo, l_exp, min(band), max(band));

%% Plot
uu = linspace(0, 70, 200);
figure('Name', 'Static equilibria');
tiledlayout(1, 2, 'TileSpacing', 'compact');
nexttile;
plot(st.u_pct(up), st.phi_deg(up), 'ko', st.u_pct(down), st.phi_deg(down), 's', 'MarkerSize', 7, 'LineWidth', 1.4);
grid on;
xlabel('u [%]');
ylabel('\phi [deg]');
legend('slider up', 'slider down', 'Location', 'northwest');
title('Equilibrium angle');
nexttile;
patch([0, 6, 6, 0], [0, 0, 1.1, 1.1] * max(F), [1, 0.88, 0.88], 'EdgeColor', 'none', 'HandleVisibility', 'off');
hold on;
plot(st.u_pct, F, 'ko', 'MarkerFaceColor', 'k');
plot(uu, k * uu, 'LineWidth', 1.6);
text(0.5, max(F), 'dead band', 'Color', [0.7, 0.1, 0.1]);
grid on;
xlabel('u [%]');
ylabel('F_r [N]');
legend('from equilibria', sprintf('k = %.4f N/%%', k), 'Location', 'northwest');
title('Thrust is linear in the command');
