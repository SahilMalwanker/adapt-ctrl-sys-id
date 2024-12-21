%% Closed-loop PI-P cascade against the pendulum twin
% Start the twin in a terminal first:   adapt-ctrl-sys-id twin serve
% The controller is designed once before the loop (tuning inside the loop
% costs milliseconds per sample and breaks the 10 ms timing), runs the
% incremental PI-P difference equations with gravity feedforward and
% anti-windup, and limits the rotor command to 0..100 %.
% Python counterpart: adapt-ctrl-sys-id twin run

if exist('acsi_params', 'file') ~= 2
    error('Run setup_adapt_ctrl_sys_id from the matlab folder first.');
end
p = acsi_params();
dz = p.design;
c = acsi_pendulum_coeffs(p.calibrated, p.geometry);
lin = acsi_pendulum_linearize(p.calibrated, p.geometry, deg2rad(dz.phi_op_deg));
lin.b = lin.b * 180 / pi;
d = acsi_place_pid(lin, dz.omega, dz.zeta, dz.alpha, dz.ts);
t_end = 120;

client = tcpclient(p.twin.host, p.twin.port, 'Timeout', 5);
configureTerminator(client, 'LF');
writeline(client, 'START');
rec = nan(round(t_end / dz.ts) + 1000, 5);   % t, ref, phi, phid, u
n = 0;
t0 = NaN;
u_pi = 0;
e_prev = 0;
while true
    line = readline(client);
    if isempty(line) || strlength(line) == 0
        break
    end
    v = sscanf(strrep(char(line), ',', '.'), '<PP><t>%f </t><PA>%f </PA><PAD>%f </PAD>');
    if numel(v) < 3
        continue
    end
    if isnan(t0)
        t0 = v(1);
    end
    t = v(1) - t0;
    if t >= t_end
        break
    end
    phi = v(2);
    phid = v(3);
    ref = acsi_reference(t);

    % outer PI loop -> rate set-point, inner P loop on the measured rate
    e = ref - phi;
    u_pi = u_pi + d.q0_pi * e + d.q1_pi * e_prev;
    u_ff = max(0, c.a0 * sind(ref) / c.b0);
    u = d.q0_p * (u_pi - phid) + u_ff;
    u_sat = min(100, max(0, u));
    if u_sat ~= u
        u_pi = (u_sat - u_ff) / d.q0_p + phid;   % anti-windup by conditioning
    end
    e_prev = e;

    writeline(client, sprintf('%f;%f', u_sat, 0));
    n = n + 1;
    rec(n, :) = [t, ref, phi, phid, u_sat];
end
writeline(client, 'EXIT');
clear client;
rec = rec(1:n, :);

window = rec(:, 1) >= 30 & rec(:, 1) <= 100;
fprintf('%d samples: J_e = %.3f deg^2, J_u = %.1f %%^2 (30-100 s)\n', n, ...
        mean((rec(window, 2) - rec(window, 3)).^2), mean(rec(window, 5).^2));

figure('Name', 'Twin PI-P');
tiledlayout(2, 1, 'TileSpacing', 'compact');
nexttile;
plot(rec(:, 1), rec(:, 2), 'k--', rec(:, 1), rec(:, 3), 'LineWidth', 1.3);
grid on;
ylabel('\phi [deg]');
legend('reference', 'twin', 'Location', 'northwest');
nexttile;
plot(rec(:, 1), rec(:, 5), 'LineWidth', 1.1);
grid on;
xlabel('t [s]');
ylabel('u [%]');
