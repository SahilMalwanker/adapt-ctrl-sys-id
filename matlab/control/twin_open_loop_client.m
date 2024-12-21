%% Open-loop session with the pendulum twin + real-time timing analysis
% Start the twin in a terminal first:   adapt-ctrl-sys-id twin serve
% Protocol (one line each, LF-terminated):
%   client -> twin   START
%   twin -> client   <PP><t>12.3400 </t><PA>40.1000 </PA><PAD>-0.5000 </PAD></PP>   (s, deg, deg/s)
%   client -> twin   u;deltax              rotor command [%]; slider offset [mm]
%   client -> twin   EXIT
% The command staircase raises the rotor command and moves the slider down
% in steps; afterwards the sampling intervals show how regular a desktop
% "real-time" loop really is.

if exist('acsi_params', 'file') ~= 2
    error('Run setup_adapt_ctrl_sys_id from the matlab folder first.');
end
p = acsi_params();
t_end = 120;
ts = 0.01;

client = tcpclient(p.twin.host, p.twin.port, 'Timeout', 5);
configureTerminator(client, 'LF');
writeline(client, 'START');
rec = nan(round(t_end / ts) + 1000, 5);      % t, phi, phid, u, deltax
n = 0;
t0 = NaN;
while true
    line = readline(client);
    if isempty(line) || strlength(line) == 0
        break                                 % timeout or twin closed the connection
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
    [u, deltax] = staircase(t);
    writeline(client, sprintf('%f;%f', u, deltax));
    n = n + 1;
    rec(n, :) = [t, v(2), v(3), u, deltax];
end
writeline(client, 'EXIT');
clear client;
rec = rec(1:n, :);

%% Sampling intervals
dt = diff(rec(:, 1));
late = abs(dt - ts) > 0.5e-3;
sorted = sort(dt);
p99 = sorted(ceil(0.99 * numel(sorted)));
fprintf('%d samples, mean interval %.2f ms, %.1f %% off by more than 0.5 ms, p99 %.1f ms, max %.1f ms\n', ...
        n, 1000 * mean(dt), 100 * mean(late), 1000 * p99, 1000 * max(dt));

figure('Name', 'Twin open loop');
tiledlayout(3, 1, 'TileSpacing', 'compact');
nexttile;
plot(rec(:, 1), rec(:, 2), 'LineWidth', 1.3);
grid on;
ylabel('\phi [deg]');
nexttile;
yyaxis left;
stairs(rec(:, 1), rec(:, 4), 'LineWidth', 1.3);
ylabel('u [%]');
yyaxis right;
stairs(rec(:, 1), rec(:, 5), 'LineWidth', 1.3);
ylabel('\Deltax [mm]');
grid on;
nexttile;
stem(rec(2:end, 1), 1000 * (dt - ts), 'Marker', 'none');
grid on;
xlabel('t [s]');
ylabel('interval - 10 ms [ms]');

%% Local functions
function [u, deltax] = staircase(t)
    levels = [0, 5, 10, 20, 30, 40, 50, 0];
    slider = [0, 50, 100, 150, 200, 250, 250, 0];
    edges = [0, 10, 20, 40, 60, 80, 100];
    i = 1 + sum(t >= edges);
    u = levels(i);
    deltax = slider(i);
end
