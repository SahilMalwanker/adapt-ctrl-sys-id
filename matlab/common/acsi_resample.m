function [tu, varargout] = acsi_resample(t, dt, varargin)
%ACSI_RESAMPLE Linear interpolation of logged signals onto a uniform time grid.
%   [tu, y1u, y2u, ...] = ACSI_RESAMPLE(t, dt, y1, y2, ...) removes repeated
%   time stamps (the logger occasionally writes two samples per tick) and
%   interpolates every signal onto tu = t(1):dt:t(end).

[t, keep] = unique(t(:));
tu = (t(1):dt:t(end)).';
varargout = cell(1, numel(varargin));
for k = 1:numel(varargin)
    y = varargin{k}(:);
    varargout{k} = interp1(t, y(keep), tu, 'linear');
end
end
