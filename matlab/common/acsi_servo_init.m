function s = acsi_servo_init(p, inertia_scale)
%ACSI_SERVO_INIT State of the calibrated virtual servo (at rest, angle 0).
%   p = ACSI_PARAMS().servo; inertia_scale multiplies the disk inertia (default 1).

if nargin < 2
    inertia_scale = 1;
end
s.theta = 0;           % true angle [rad]
s.omega = 0;           % true speed [rad/s]
s.theta_meas = 0;      % encoder angle [rad]
s.omega_meas = 0;      % filtered finite-difference speed [rad/s]
s.quantum = 2 * pi / p.counts;
s.inertia_scale = inertia_scale;
end
