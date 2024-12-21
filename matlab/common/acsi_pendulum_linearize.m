function lin = acsi_pendulum_linearize(model, geometry, phi_op)
%ACSI_PENDULUM_LINEARIZE Small-signal model  a2 dphi'' + a1 dphi' + a0 dphi = b du  around phi_op [rad].
%   The operating input u_op = a0 sin(phi_op) / b0 holds the pendulum at phi_op;
%   the thrust-dependent damping adds d_u * u_op to a1.

c = acsi_pendulum_coeffs(model, geometry);
lin.phi_op = phi_op;
lin.u_op = c.a0 * sin(phi_op) / c.b0;
lin.a2 = c.J;
lin.a1 = model.viscous + model.thrust_damping * lin.u_op;
lin.a0 = c.a0 * cos(phi_op);
lin.b = c.b0;
lin.wn = sqrt(lin.a0 / lin.a2);
lin.zeta = lin.a1 / (2 * sqrt(lin.a0 * lin.a2));
end
