% Diagnostic script to check peakFits structure
% Run this first to understand what data is available

% Load your saved data
load('/home/user/BrukerCESTmethod_processingPipeline/Bruker_CEST-MRF_processing/MATLAB_MAIN-CODE-TO-RUN/saved_data_ROIs/img_roi_data.mat');

disp('=== Checking img.zSpec.peakFits structure ===');
if isfield(img.zSpec, 'peakFits')
    disp('peakFits exists!');
    disp('Fields in peakFits:');
    disp(fieldnames(img.zSpec.peakFits));

    % Check the size/structure
    disp(' ');
    disp('Structure details:');
    disp(img.zSpec.peakFits);
else
    disp('peakFits does not exist in img.zSpec');
end

disp(' ');
disp('=== Checking img.zSpec.fitImg structure ===');
if isfield(img.zSpec, 'fitImg')
    disp('fitImg exists!');
    if isstruct(img.zSpec.fitImg)
        disp('Fields in fitImg:');
        disp(fieldnames(img.zSpec.fitImg));
    else
        disp(['fitImg size: ' num2str(size(img.zSpec.fitImg))]);
    end
else
    disp('fitImg does not exist in img.zSpec');
end

disp(' ');
disp('=== Checking img.zSpec.MTRimg structure ===');
if isfield(img.zSpec, 'MTRimg')
    disp(['MTRimg size: ' num2str(size(img.zSpec.MTRimg))]);
else
    disp('MTRimg does not exist in img.zSpec');
end

disp(' ');
disp('=== Available fields in img.zSpec ===');
disp(fieldnames(img.zSpec));
