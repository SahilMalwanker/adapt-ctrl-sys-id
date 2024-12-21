function fit = acsi_fit_percent(y, yhat)
%ACSI_FIT_PERCENT Normalised RMSE fit in percent: 100 * (1 - ||y - yhat|| / ||y - mean(y)||).
%   Same definition as the fit reported by compare / goodnessOfFit('NRMSE');
%   100 % is a perfect match, 0 % is no better than the mean, negative is worse.

y = y(:);
yhat = yhat(:);
fit = 100 * (1 - norm(y - yhat) / norm(y - mean(y)));
end
