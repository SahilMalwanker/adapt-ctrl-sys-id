%% Counting chair rotations with a smartphone gyroscope
% Offline: the recorded yaw angle in data/gyro_chair_spin.csv.
% Live:    set use_phone = true, start the phyphox experiment
%          "Gyroscope (rotation rate)" on a phone in the same network, enable
%          its remote access and enter the address phyphox shows.
% A direction reversal is only counted after 0.2 rad of travel (hysteresis),
% so sensor noise around a standstill does not create phantom reversals.
% Python counterpart: adapt_ctrl_sys_id.gyro.count_turns

if exist('acsi_params', 'file') ~= 2
    error('Run setup_adapt_ctrl_sys_id from the matlab folder first.');
end
use_phone = false;
phone = 'http://<phone-ip>:8080/';   % address shown by phyphox under "Allow remote access"
duration = 20;                       % [s] live recording time

if use_phone
    opts = weboptions('ContentType', 'json', 'Timeout', 10);
    webread([phone 'control?cmd=clear'], opts);
    webread([phone 'control?cmd=start'], opts);
    pause(duration);
    webread([phone 'control?cmd=stop'], opts);
    data = webread([phone 'get?gyrZ=full&gyr_time=full'], opts);
    t = data.buffer.gyr_time.buffer(:);
    rate = data.buffer.gyrZ.buffer(:);
    ok = ~isnan(t) & ~isnan(rate);
    t = t(ok);
    yaw = cumtrapz(t, rate(ok));     % rad
else
    rec = acsi_read('gyro_chair_spin');
    t = rec.t_s;
    yaw = rec.yaw_rad;
end

%% Turn statistics
steps = diff(yaw);
positive = sum(steps(steps > 0)) / (2 * pi);
negative = -sum(steps(steps < 0)) / (2 * pi);
reversals = count_reversals(yaw, 0.2);
fprintf('net %+.3f turns, travelled %.3f turns (+%.3f / -%.3f), %d reversal(s)\n', ...
        (yaw(end) - yaw(1)) / (2 * pi), positive + negative, positive, negative, reversals);

figure('Name', 'Gyro turn counter');
plot(t, yaw / (2 * pi), 'LineWidth', 1.8);
yline(0, ':');
grid on;
xlabel('t [s]');
ylabel('turns');
title('Integrated yaw rate');

%% Local functions
function n = count_reversals(angle, hysteresis)
    n = 0;
    direction = 0;
    start = angle(1);
    extreme = start;
    for k = 2:numel(angle)
        value = angle(k);
        if direction == 0
            if abs(value - start) > hysteresis
                direction = sign(value - start);
                extreme = value;
            end
        elseif direction * (value - extreme) > 0
            extreme = value;                       % still moving the same way
        elseif direction * (extreme - value) > hysteresis
            direction = -direction;                % turned around by more than the hysteresis
            extreme = value;
            n = n + 1;
        end
    end
end
