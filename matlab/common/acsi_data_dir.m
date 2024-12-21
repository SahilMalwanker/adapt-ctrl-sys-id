function folder = acsi_data_dir()
%ACSI_DATA_DIR Absolute path of the repository's data folder.

repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
folder = fullfile(repo, 'data');
end
