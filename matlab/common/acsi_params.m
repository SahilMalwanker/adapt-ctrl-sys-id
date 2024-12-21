function p = acsi_params()
%ACSI_PARAMS Shared model parameters (identical to the Python package defaults).
%   p.servo      DC servo with inertia disk
%   p.geometry   propeller pendulum masses and lengths
%   p.lab        geometry-only pendulum model (round numbers, hand-tuned damping)
%   p.calibrated pendulum model calibrated on the hardware records
%   p.design     PI-P design point
%   p.twin       TCP address of the virtual twin

% ---------------------------------------------------------------- DC servo
p.servo.K = 25.172;          % PT1 gain identified on the hardware [rad/s/V]
p.servo.T = 0.31919;         % PT1 time constant identified on the hardware [s]
p.servo.K0 = 28.63;          % static map: small-signal gain at low speed [rad/s/V]
p.servo.beta = 3.25e-5;      % static map: quadratic loss [V s^2/rad^2]
p.servo.tau = 0.353;         % calibrated mechanical time constant of the virtual servo [s]
p.servo.u_max = 10;          % amplifier limit [V]
p.servo.counts = 2048;       % encoder counts per revolution
p.servo.vel_filter_tc = 0.004; % speed-estimation low-pass [s]
p.servo.ts = 0.002;          % real-time sample time [s]

% ---------------------------------------------------------------- pendulum geometry
g.mb = 0.300;                % beam mass [kg]
g.mr = 0.087;                % rotor mass [kg]
g.ms = 0.430;                % slider mass [kg]
g.lb = 0.473;                % beam length [m]
g.lr = 0.500;                % rotor arm [m]
g.x_up = 0.09;               % slider position, up [m]
g.x_down = 0.34;             % slider position, down [m]
g.g = 9.81;
g.m = g.mb + g.mr + g.ms;
p.geometry = g;

% ---------------------------------------------------------------- pendulum models
base = struct('x', g.x_up, 'viscous', 0.0045, 'drag', 0, 'thrust_damping', 0, ...
              'thrust_gain', 0.05, 'inertia_offset', 0, 'rotor_tau', 0, 'deadband', 0, 'u_max', 100);
p.lab = base;

cal = base;
cal.viscous = 0.00211;        % viscous friction [N m s]
cal.drag = 5.1e-4;            % quadratic air drag [N m s^2]
cal.thrust_damping = 7.6e-4;  % propeller-inflow damping per % command [N m s/%]
cal.thrust_gain = 0.0526;     % rotor force per % command [N/%]
cal.inertia_offset = 0.00611; % inertia not captured by the geometry [kg m^2]
cal.rotor_tau = 0.044;        % rotor speed lag [s]
cal.deadband = 6.0;           % rotor start-up dead band [%]
p.calibrated = cal;

% ---------------------------------------------------------------- control and twin
p.design = struct('phi_op_deg', 40, 'omega', 8, 'zeta', 0.9, 'alpha', 1, 'ts', 0.01);
p.twin = struct('host', 'localhost', 'port', 10815);
end
