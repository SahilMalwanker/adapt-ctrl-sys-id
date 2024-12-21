function c = acsi_pendulum_coeffs(model, geometry)
%ACSI_PENDULUM_COEFFS Coefficients of  a2*phi'' + a1*phi' + a0*sin(phi) = b0*u.
%   model    one of the structs returned by ACSI_PARAMS (lab or calibrated)
%   geometry ACSI_PARAMS().geometry
%
%   l(x) = (x ms + lb/2 mb + lr mr) / m           centre of gravity
%   J(x) = mb lb^2/3 + mr lr^2 + ms x^2 (+ offset)  inertia about the pivot

g = geometry;
first_moment = model.x * g.ms + g.lb / 2 * g.mb + g.lr * g.mr;   % m * l(x)
c.l = first_moment / g.m;
c.J = g.mb * g.lb^2 / 3 + g.mr * g.lr^2 + g.ms * model.x^2 + model.inertia_offset;
c.a2 = c.J;
c.a1 = model.viscous;
c.a0 = g.g * first_moment;
c.b0 = model.thrust_gain * g.lr;
end
