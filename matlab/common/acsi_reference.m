function r = acsi_reference(t)
%ACSI_REFERENCE Pendulum reference profile in degrees.
%   0 deg for 5 s, a 2 deg/s ramp to 40 deg, then plateaus at 40, 30, 50 and
%   40 deg (steps at 40, 65 and 90 s). Works on scalars and arrays.

r = 40 * ones(size(t));
r(t < 90) = 50;
r(t < 65) = 30;
r(t < 40) = 40;
ramp = t < 25;
r(ramp) = 2 * (t(ramp) - 5);
r(t < 5) = 0;
end
