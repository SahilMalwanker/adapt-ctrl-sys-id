function u = acsi_servo_excitation(t)
%ACSI_SERVO_EXCITATION Piecewise-constant identification voltage [V] (data/servo_excitation_profile.csv).
%   Mixes small and large steps of both signs so that the experiment exposes
%   the speed-dependent gain of the servo. The signal is 0 before the first start.

profile = acsi_read('servo_excitation_profile');
starts = profile.t_start_s;
levels = profile.u_V;
u = zeros(size(t));
for k = 1:numel(starts)
    u(t >= starts(k)) = levels(k);
end
end
