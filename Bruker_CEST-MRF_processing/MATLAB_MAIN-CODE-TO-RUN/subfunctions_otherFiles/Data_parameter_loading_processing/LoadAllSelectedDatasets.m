% LoadAllSelectedDatasets: Takes in structures specifying the selected scan
% directories to load and what types of datasets they pertain to, along
% with other processing parameters, and returns all processed images (or 
% dummy images, if not selected by user) by calling the respective 
% loading/processing functions:
%   MRFLoad()
%   T1T2Load()
%   QUESP_load_proc()
%   WASSR_load_proc()
%
%   INPUTS:
%       specifiedflg    -   Struct containing logical indicators whether
%                           each type of dataset type has a specified scan
%                           number to process
%       scan_dirs       -   Struct containing scan directories
%                           corresponding with each type of dataset
%       i_flds          -   Struct containing cell arrays of names of the 
%                           images pertaining to how struct 'img' is  
%                           organized, further organized by plotting groups
%       cfg             -   Struct containing subfields describing user
%                           configuration settings: paths to load/save 
%                           folders, scripts, etc.
%       parprefs        -   Struct containing user specific processing 
%                           options
%       PV360flg        -   Logical indicating whether the selected study is
%                           identified as being obtained with ParaVision 360
%                           (true) or an older ParaVision version (false)
%       
%   OUTPUTS:
%       img             -   Struct containing images obtained by processing
%                           all specified datasets
%

function img=LoadAllSelectedDatasets(specifiedflg,scan_dirs,i_flds,cfg,...
    parprefs,PV360flg)

%% LOAD INTO MATLAB: MRF 
if specifiedflg.MRF
    % Set up config struct to include location of MRF dataset
    cfg.loadMRF=fullfile(scan_dirs.base_dir,scan_dirs.MRF,'pdata','1');
%     MRFdirs.pydir=py_dir;
%     MRFdirs.pyfile=py_file;
%     MRFdirs.bash=bashfn;
%     MRFdirs.condaenv=conda_env;
%     MRFdirs.pyenv=py_env;
    disp(['MRF data: loading from ' cfg.MRFfn ' if found...'])
    [img.MRF,info.MRF]=MRF_load_proc(cfg,i_flds.MRF,parprefs,PV360flg);
    img.MRF.size=size(img.MRF.dp);
    disp('MRF data loading and processing complete!')
end


%% LOAD INTO MATLAB: T1 + T2
if specifiedflg.T1map
    disp('T1 map: loading fitted T1 map from scanner-generated DICOMs...')
    img.other.t1wIR=T1T2Load(fullfile(scan_dirs.base_dir,scan_dirs.T1map,...
        'pdata','2','dicom'),cfg.ext_dir);
    img.other.size=size(img.other.t1wIR);
    disp('T1 map loading complete!')
end

if specifiedflg.T2map
    disp('T2 map: loading fitted T1 map from scanner-generated DICOMs...')
    try
        img.other.t2wMSME=T1T2Load(fullfile(scan_dirs.base_dir,scan_dirs.T2map,...
            'pdata','2','dicom'),cfg.ext_dir);
    catch
        img.other.t2wMSME=T1T2Load(fullfile(scan_dirs.base_dir,scan_dirs.T2map,...
            'pdata','3','dicom'),cfg.ext_dir);
    end
    img.other.size=size(img.other.t2wMSME);
    disp('T2 map loading complete!')
end


%% LOAD INTO MATLAB: QUESP
if specifiedflg.QUESP
    disp('QUESP data: loading...')
    [img.other.fsQUESP,img.other.kswQUESP,Rsq,info.QUESP]=QUESP_load_proc(...
        fullfile(scan_dirs.base_dir,scan_dirs.QUESP,'pdata','1'),...
        img.other.t1wIR,parprefs,PV360flg);
    img.other.RsqMask=(Rsq>=parprefs.RsqThreshold);
    img.other.size=size(img.other.fsQUESP);
    disp('QUESP data loading and processing complete!')
end


%% LOAD INTO MATLAB: WASSR
if specifiedflg.WASSR
    disp('WASSR data: loading...')
    [img.other.B0WASSR_Hz,~,~,info.WASSR]=WASSR_load_proc(...
        fullfile(scan_dirs.base_dir,scan_dirs.WASSR,'pdata','1'),parprefs,...
        PV360flg);
    img.other.size=size(img.other.B0WASSR_Hz);
    img.zSpec.size=size(img.other.B0WASSR_Hz); %just in case no z-spec data 
        %are loaded, to prevent an error! This is b/c WASSR shows up in the
        %"zspec" plotting group as well as "other"
    img.zSpec.B0WASSRppm=img.other.B0WASSR_Hz./info.WASSR.omega_0; %copy over to
        % zSpec group, but in ppm, not Hz!
    disp('WASSR data loading and processing complete!')
end


%% LOAD INTO MATLAB: Z-SPEC IMAGING
if specifiedflg.zSpec
    disp('Z-spectroscopic imaging data: loading...')
    try
        B0map=img.zSpec.B0WASSRppm;
    catch
        B0map=[];
    end
    [img.zSpec.img,img.zSpec.M0img,img.zSpec.fitImg,img.zSpec.peakFits,...
        info.zSpec]=zSpec_load_proc(...
        fullfile(scan_dirs.base_dir,scan_dirs.zSpec,'pdata','1'),B0map,...
        parprefs,PV360flg);
    img.zSpec.size=size(img.zSpec.M0img);
    img.zSpec.ppm=info.zSpec.w_offsetPPM;
    % Fill in dummy images of zeros for pools that weren't fit
    for ii=1:numel(i_flds.poolnames)
        if ~isfield(img.zSpec.fitImg,i_flds.poolnames{ii})
            img.zSpec.fitImg.(i_flds.poolnames{ii})=zeros(img.zSpec.size);
        end
    end   
    disp('Z-spectroscopic imaging data loading and processing complete!')

    %% REX MAP: (1/Z - 1) * R1 * cos^2(theta)
    % Rex isolates exchange-dependent relaxation. Requires both z-spec + T1 map.
    % cos^2(theta) = DeltaOmega^2 / (omega1^2 + DeltaOmega^2)
    %   omega1     [rad/s] = 2*pi * 42.577 * satpwr_uT
    %   DeltaOmega [rad/s] = 2*pi * omega0_MHz * ppm
    if specifiedflg.T1map
        disp('Z-spectroscopic imaging data: computing Rex = (1/Z-1)*R1*cos^2(theta)...')
        omega1_rads   = 2*pi * 42.577 * info.zSpec.satpwr_uT;         % scalar [rad/s]
        deltaOmega    = 2*pi * info.zSpec.omega_0 .* img.zSpec.ppm;   % [1 x nppm] [rad/s]
        cos2theta     = deltaOmega.^2 ./ (omega1_rads^2 + deltaOmega.^2); % [1 x nppm]
        cos2theta(~isfinite(cos2theta)) = 0;    % zero at ppm=0

        R1map = 1 ./ double(img.other.t1wIR);  % [nx x ny] [s^-1]
        R1map(~isfinite(R1map)) = 0;            % mask voxels where T1=0

        nppm      = length(img.zSpec.ppm);
        R1map3D   = repmat(R1map,    [1, 1, nppm]);
        cos2theta3D = repmat(reshape(cos2theta, 1, 1, nppm), ...
                             [size(img.zSpec.img,1), size(img.zSpec.img,2), 1]);

        RexImg = (1 ./ double(img.zSpec.img) - 1) .* R1map3D .* cos2theta3D;
        RexImg(~isfinite(RexImg)) = 0;
        img.zSpec.RexImg = RexImg;
        disp('Rex map computation complete!')

        %% REX SPECTRUM FITTING (same pools as Z-spectrum, no 1-Z inversion)
        disp('Z-spectroscopic imaging data: fitting Rex spectra voxelwise...')
        zppars_rex.pools    = {'water','NOE','MT','amide'};
        zppars_rex.peaktype = 'Pseudo-Voigt';
        zppars_rex.water1st = false;

        % Derive SNR mask from processed zImg (masked voxels are all-zero)
        Thmask_rex      = any(double(img.zSpec.img) ~= 0, 3);
        ThmaskIdxVec_rex = find(reshape(Thmask_rex, [], 1));

        % Reshape and select unmasked voxels
        RexSelVox = reshape(RexImg, prod(size(RexImg,[1,2])), []);
        RexSelVox = RexSelVox(ThmaskIdxVec_rex, :);

        % Fit Rex spectra directly (invertflg=false: no 1-Z inversion)
        [RexFittedAmpls, RexFittedPeaksIndiv, RexFittedPeaksAll] = ...
            fitAllZspec(img.zSpec.ppm, RexSelVox, zppars_rex, false);

        % Fill in fitted amplitude maps and peak-curve maps
        nPools_rex = numel(zppars_rex.pools);
        for ii = 1:nPools_rex
            pool = zppars_rex.pools{ii};
            img.zSpec.RexFitImg.(pool) = zeros(img.zSpec.size);
            img.zSpec.RexFitImg.(pool)(ThmaskIdxVec_rex) = RexFittedAmpls(ii,:);

            RexPeakVec = zeros(prod(size(img.zSpec.img,[1,2])), nppm);
            for jj = 1:numel(ThmaskIdxVec_rex)
                RexPeakVec(ThmaskIdxVec_rex(jj),:) = RexFittedPeaksIndiv(ii,jj,:);
            end
            img.zSpec.RexPeakFits.(pool) = reshape(RexPeakVec, size(RexImg));
        end
        RexAllVec = zeros(prod(size(img.zSpec.img,[1,2])), nppm);
        for ii = 1:numel(ThmaskIdxVec_rex)
            RexAllVec(ThmaskIdxVec_rex(ii),:) = RexFittedPeaksAll(ii,:);
        end
        img.zSpec.RexPeakFits.all = reshape(RexAllVec, size(RexImg));
        disp('Rex spectrum fitting complete!')
    end
end


%% DUMMY IMAGES FOR NONSELECTED DATASET TYPES
% Generate dummy images to fill in any unspecified dataset images
grps=fieldnames(img);

if ~prod(cell2mat(struct2cell(specifiedflg))) 
    disp('Filling in non-specified image datasets with dummy images...')

    % Generate dummy images of all zeros, using the default size determined
    % for each group
    for ii=1:numel(grps)
        dummysize=img.(grps{ii}).size;
%         if ~isfield(img,grps{ii})
%             img.(grps{ii})=struct;
%         end
        for jj=1:numel(i_flds.(grps{ii}))
            if ~isfield(img.(grps{ii}),i_flds.(grps{ii}){jj})
                if strcmp(i_flds.(grps{ii}){jj},'fitImg') %make dummy image for each fitable pool
                    for kk=1:numel(i_flds.poolnames)
                        img.zSpec.fitImg.(i_flds.poolnames{kk})=zeros(dummysize);
                    end
                elseif ~strcmp(i_flds.(grps{ii}){jj},'avgZspec') % avoid img.zSpec.avgZspec 
                    % -- it will be initialized with the first ROI drawn!
                    img.(grps{ii}).(i_flds.(grps{ii}){jj})=zeros(dummysize);
                end               
            end
        end
    end
end

% Also generate dummy images for error maps
if sum(strcmp(grps,'MRF'))>0
    img.ErrorMaps.size=img.MRF.size;
elseif sum(strcmp(grps,'other'))>0
    img.ErrorMaps.size=img.other.size;
else
    img.ErrorMaps.size=img.(grps{1}).size;
end
dummysize=img.ErrorMaps.size;
for ii=1:numel(i_flds.ErrorMaps)
    img.ErrorMaps.(i_flds.ErrorMaps{ii})=zeros(dummysize);
end
end
