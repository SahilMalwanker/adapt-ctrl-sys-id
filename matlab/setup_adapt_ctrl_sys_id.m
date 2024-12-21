function setup_adapt_ctrl_sys_id()
%SETUP_ADAPT_CTRL_SYS_ID Put the adapt-ctrl-sys-id MATLAB tools on the path.
%   cd into the matlab folder and call setup_adapt_ctrl_sys_id once per session.
%   All scripts read the CSV records from the repository's data folder.

here = fileparts(mfilename('fullpath'));
folders = {'common', 'servo', 'pendulum', 'control', 'simulink'};
addpath(here);
for k = 1:numel(folders)
    addpath(fullfile(here, folders{k}));
end
fprintf('adapt-ctrl-sys-id MATLAB tools on the path, data folder: %s\n', acsi_data_dir());
end
