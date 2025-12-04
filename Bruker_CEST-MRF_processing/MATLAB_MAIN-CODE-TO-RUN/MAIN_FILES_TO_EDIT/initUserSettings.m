% initUserSettings: Sets fields of the struct settings to include
% information pertaining to the user's computer as well as processing
% preferences
%
%   INPUTS:     None
%
%   OUTPUTS:    
%       configs     -   Struct containing subfields describing user
%                       configuration settings: paths to load/save 
%                       folders, scripts, etc.
%       prefs       -   Struct containing user specific processing options
%
function [configs,prefs]=initUserSettings()
disp(['Loading user configuration settings and processing preferences from '...
    'file initUserSettings.m...'])
%% USER CONFIGURATION SETTINGS
% Base function directory (i.e. the one containing MAIN_FILES_TO_EDIT/, 
% MAIN_FUNCTIONS_TO_RUN/, saved_data_ROIs/, and subfunctions_otherFiles/)
baseFcnDir='/Users/dk384/Documents/Laboratory/MGH_Farrar-Rosen/MATLAB/Image_processing/Bruker_CEST-MRF_processing';

% Directories for loading/saving images and ROIs
configs.load_dir='/Users/dk384/Documents/Laboratory/MGH_Farrar-Rosen/Data';
configs.save_dir=fullfile(baseFcnDir,'saved_data_ROIs'); %shouldn't need to change

% Directory for calling dcm2niix (for T1 + T2 maps):
configs.ext_dir=fullfile(baseFcnDir,'subfunctions_otherFiles','external'); 
    %shouldn't need to change

% Info for calling Python scripts to do dictionary matching
configs.py_dir='/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main';
    %path to directory containing Python code for dictionary matching, etc.
configs.py_file='MRFmatch_B-SL_dk.py'; %name of file to run
configs.conda_env='cbdmrf'; %name of conda environment to activate
    %(NOTE: currently I need to activate this along with conda to get the
    %Python script to work!
configs.py_env='cbdmrfpy'; %name of python environment to activate

configs.bashfn='.bashrc'; %system file in home directory containing conda alias
    %(e.g. .bashrc, .zshrc)

% Name of file to search for and load for MRF, which contains parameters
% maps outputted from Python scripts (make sure it ends with .mat)
configs.MRFfn='quant_maps.mat';
% configs.MRFfn='quant_maps_sim[y,-x,-y].mat';
% configs.MRFfn='quant_maps_T1T2Fixed.mat';
% configs.MRFfn='quant_maps_logKb_T1T2Fixed.mat';

% Cluster configuration
configs.use_cluster = true;  % SET TO true WHEN YOU WANT TO USE CLUSTER
configs.cluster_user = 'cd1052';
configs.cluster_host = 'mlsc.nmr.mgh.harvard.edu';
configs.cluster_dir = '/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main';
configs.cluster_job_script = 'srun.bash';
configs.cluster_large_storage_dir='/autofs/vast/farrar/users/cd1052';
    % Large storage directory (1TB) on cluster for saving dict.mat and quant_maps.mat

% For MATCH_MRF_MULTI: Substrings pertaining to schedule filename that are
% NOT MRF datasets (used to eliminate non-MRF fp(SL)_EPI datasets from
% directory list) - note that these are NOT case-sensitive!
configs.notMRFstr={'quesp','wassr','R1rho'};


%% USER PREFERENCES
% MRF processing parameters
prefs.nPools=2; %number of pools, including water (must be >= 2)

% QUESP processing parameters
prefs.QUESPfcn='Inverse'; %type of QUESP fitting to use (options: 'Regular', 
    %'Inverse (default)', 'OmegaPlot') 
prefs.RsqThreshold=0.95; %value of R^2 to use for thresholding QUESP voxelwise 
% fitting maps 

% WASSR/z-spectral imaging processing parameters
prefs.SNRthresh=3.5; %minimum voxel SNR to keep for processing
end