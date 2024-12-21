function [phi, phid, thrust] = acsi_pendulum_simulate(model, geometry, t, u, phi0, substeps)
%ACSI_PENDULUM_SIMULATE Nonlinear pendulum on the grid t with u(k) held on [t(k), t(k+1)).
%   [phi, phid, thrust] = ACSI_PENDULUM_SIMULATE(model, geometry, t, u, phi0, substeps)
%   Angles in rad, rotor command in %. The rotor starts in steady state for u(1)
%   and the pendulum at rest at phi0 (default 0).

if nargin < 5
    phi0 = 0;
end
if nargin < 6
    substeps = 1;
end
t = t(:);
if isscalar(u)
    u = u * ones(size(t));
end
u = u(:);
c = acsi_pendulum_coeffs(model, geometry);
n = numel(t);
phi = zeros(n, 1);
phid = zeros(n, 1);
thrust = zeros(n, 1);
start = min(max(u(1), 0), model.u_max);
if start < model.deadband
    start = 0;
end
x = [phi0; 0; start];
for k = 1:n
    phi(k) = x(1);
    phid(k) = x(2);
    thrust(k) = x(3);
    if k < n
        h = (t(k + 1) - t(k)) / substeps;
        for s = 1:substeps
            x = acsi_pendulum_step(x, u(k), h, model, c);
        end
    end
end
end
