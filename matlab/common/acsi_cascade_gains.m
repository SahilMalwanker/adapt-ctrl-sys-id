function [kd, kp] = acsi_cascade_gains(K, T, s_vel)
%ACSI_CASCADE_GAINS Gains of the P-velocity / P-position cascade for the plant K / (s (T s + 1)).
%   The inner loop Kd moves the velocity pole to s_vel; the outer loop Kp gives
%   the position loop a damping ratio of 1/sqrt(2) with poles s_vel/2 * (1 +/- j):
%
%       Kd = (-s_vel T - 1) / K,      Kp = (1 + Kd K)^2 / (2 T Kd K)

if s_vel >= -1 / T
    error('acsi_cascade_gains:pole', 'velocity pole must be faster than the open-loop pole -1/T');
end
kd = (-s_vel * T - 1) / K;
kp = (1 + kd * K)^2 / (2 * T * kd * K);
end
