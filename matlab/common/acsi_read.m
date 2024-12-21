function T = acsi_read(name)
%ACSI_READ Read data/<name>.csv into a table; column names follow the CSV header.
%   Time series with irregular time stamps are returned as recorded; use
%   ACSI_RESAMPLE to put them on a uniform grid.

file = fullfile(acsi_data_dir(), [name '.csv']);
if ~isfile(file)
    error('acsi_read:missing', 'No data record named "%s" in %s', name, acsi_data_dir());
end
T = readtable(file, 'VariableNamingRule', 'preserve');
end
