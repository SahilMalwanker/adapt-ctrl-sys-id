function x = acsi_pendulum_step(x, u, h, model, c)
%ACSI_PENDULUM_STEP One RK4 step of the pendulum state x = [phi; phid; w].
%   u      rotor command [%], held over the step
%   h      step length [s]
%   model  ACSI_PARAMS().lab or ACSI_PARAMS().calibrated
%   c      ACSI_PENDULUM_COEFFS(model, geometry)
%
%   J phi'' = b0 w - a0 sin(phi) - (d + d_u w) phi' - c_drag phi' |phi'|
%   tau_r w' = u_eff - w      (u_eff: command limited to 0..u_max, 0 inside the dead band)

target = min(max(u, 0), model.u_max);
if target < model.deadband
    target = 0;
end
if model.rotor_tau <= 0
    x(3) = target;   % no rotor lag: the thrust state follows the command instantly
end
k1 = rhs(x, target, model, c);
k2 = rhs(x + 0.5 * h * k1, target, model, c);
k3 = rhs(x + 0.5 * h * k2, target, model, c);
k4 = rhs(x + h * k3, target, model, c);
x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
end

function dx = rhs(x, target, model, c)
phi = x(1);
phid = x(2);
w = x(3);
damping = (model.viscous + model.thrust_damping * w) * phid + model.drag * phid * abs(phid);
phidd = (c.b0 * w - c.a0 * sin(phi) - damping) / c.J;
if model.rotor_tau > 0
    wd = (target - w) / model.rotor_tau;
else
    wd = 0;
end
dx = [phid; phidd; wd];
end
