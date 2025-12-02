%% Clinical 3T DICOM Adapter for Bruker CEST-MRF Pipeline
% This function adapts 3T clinical scanner DICOM data to work with the
% Bruker CEST-MRF processing pipeline's dictionary simulation and matching
%
% Purpose: Bridge between your clinical DICOM data and the repository's
%          powerful Bloch-McConnell simulation capabilities
%
% INPUTS:
%   dicom_dir       - Path to folder containing CEST DICOM files
%   wassr_offsets   - WASSR frequency offsets in Hz (1×N array)
%   cest_offsets    - CEST frequency offsets in Hz (1×M array)
%   S0_index        - Index of S0 reference image (default: 2)
%   output_dir      - Directory to save converted data (default: pwd)
%
% OUTPUTS:
%   saved_file      - Path to generated 'acquired_data.mat' file
%                     compatible with Python MRF matching
%
% USAGE EXAMPLE:
%   wassr_Hz = [240, 192, 144, 96, 48, 0, -48, -96, -144, -192, -240];
%   cest_Hz = [896, 864, ..., -896];  % Your 49 offsets
%   saved_file = Clinical_3T_DICOM_Adapter('/path/to/dicoms', wassr_Hz, cest_Hz);
%
% Then run Python matching:
%   MRFmatch_B-SL_dk.py (called from MATLAB via system command)
%
% Author: Integration adapter for BrukerCESTmethod_processingPipeline
% Date: 2025-12-02

function saved_file = Clinical_3T_DICOM_Adapter(dicom_dir, wassr_offsets, cest_offsets, S0_index, output_dir)

if nargin < 4
    S0_index = 2;
end
if nargin < 5
    output_dir = pwd;
end

fprintf('========================================\n');
fprintf('Clinical 3T DICOM Adapter\n');
fprintf('========================================\n');

%% Step 1: Load DICOM Data
fprintf('Loading DICOM files from: %s\n', dicom_dir);
dicomFiles = dir(fullfile(dicom_dir, '*.dcm'));
numFiles = length(dicomFiles);
fprintf('Found %d DICOM files\n', numFiles);

if numFiles == 0
    error('No DICOM files found in directory: %s', dicom_dir);
end

% Read dimensions from first file
firstFile = fullfile(dicom_dir, dicomFiles(1).name);
temp = dicomread(firstFile);
[xDim, yDim] = size(temp);

% Load all DICOM images
dicomVolume = zeros(xDim, yDim, numFiles);
fprintf('Loading %d DICOM files...\n', numFiles);
for i = 1:numFiles
    filePath = fullfile(dicom_dir, dicomFiles(i).name);
    dicomVolume(:,:,i) = double(dicomread(filePath));
end

%% Step 2: Extract Scanner Parameters
fprintf('Extracting scanner parameters from DICOM header...\n');
info = dicominfo(firstFile);

% Field strength
if isfield(info, 'MagneticFieldStrength')
    B0_field = info.MagneticFieldStrength;
else
    warning('Magnetic field strength not found in DICOM. Assuming 3T.');
    B0_field = 3.0;
end
fprintf('  B0 field strength: %.1f Tesla\n', B0_field);

% Larmor frequency
f0_MHz = 42.577 * B0_field;  % MHz
fprintf('  Larmor frequency: %.3f MHz\n', f0_MHz);

% TR and TE
if isfield(info, 'RepetitionTime')
    TR = info.RepetitionTime;  % ms
else
    warning('TR not found in DICOM. Using default: 5000 ms');
    TR = 5000;
end

if isfield(info, 'EchoTime')
    TE = info.EchoTime;  % ms
else
    warning('TE not found in DICOM. Using default: 30 ms');
    TE = 30;
end
fprintf('  TR: %.1f ms, TE: %.1f ms\n', TR, TE);

%% Step 3: Extract WASSR and CEST volumes
fprintf('Extracting WASSR and CEST data...\n');

% Calculate indices
num_wassr = length(wassr_offsets);
num_cest = length(cest_offsets);

wassr_indices = S0_index + 1 : S0_index + num_wassr;
cest_indices = wassr_indices(end) + 1 : wassr_indices(end) + num_cest;

if cest_indices(end) > numFiles
    error('Not enough DICOM files! Expected %d, found %d', cest_indices(end), numFiles);
end

S0_image = dicomVolume(:,:,S0_index);
wasserVolume = dicomVolume(:,:,wassr_indices);
cestVolume = dicomVolume(:,:,cest_indices);

% Convert Hz to ppm
wassr_offsets_ppm = wassr_offsets / f0_MHz;
cest_offsets_ppm = cest_offsets / f0_MHz;

fprintf('  WASSR images: %d (offsets: %.1f to %.1f ppm)\n', ...
    num_wassr, min(wassr_offsets_ppm), max(wassr_offsets_ppm));
fprintf('  CEST images: %d (offsets: %.1f to %.1f ppm)\n', ...
    num_cest, min(cest_offsets_ppm), max(cest_offsets_ppm));

%% Step 4: Calculate B0 Map from WASSR
fprintf('Calculating B0 map from WASSR data...\n');
B0_map_ppm = zeros(xDim, yDim);

for i = 1:xDim
    for j = 1:yDim
        wassr_spectrum = squeeze(wasserVolume(i,j,:));
        [~, min_idx] = min(wassr_spectrum);
        B0_map_ppm(i,j) = wassr_offsets_ppm(min_idx);
    end
end

% Smooth and clip B0 map
B0_map_ppm = imgaussfilt(B0_map_ppm, 2);
B0_map_ppm(B0_map_ppm < -2) = -2;
B0_map_ppm(B0_map_ppm > 2) = 2;

fprintf('  B0 map: mean = %.3f ppm, std = %.3f ppm\n', ...
    mean(B0_map_ppm(:)), std(B0_map_ppm(:)));

%% Step 5: Apply B0 Correction to CEST Data
fprintf('Applying B0 correction to CEST data...\n');

% Detect phantom outline for masking
Sm = imgaussfilt(S0_image, 4);
Smn = mat2gray(Sm);
bw_phantom = imbinarize(Smn);
CCp = bwconncomp(bw_phantom);
numPix = cellfun(@numel, CCp.PixelIdxList);
[~, idxMax] = max(numPix);
phantom_outline = false(size(bw_phantom));
phantom_outline(CCp.PixelIdxList{idxMax}) = true;
phantom_outline = imfill(phantom_outline, "holes");

cestVolume_corrected = zeros(size(cestVolume));

for i = 1:xDim
    for j = 1:yDim
        if phantom_outline(i,j)
            B0_offset = B0_map_ppm(i,j);
            shifted_ppm = cest_offsets_ppm - B0_offset;
            original_spectrum = squeeze(cestVolume(i,j,:));
            corrected_spectrum = interp1(shifted_ppm, original_spectrum, ...
                cest_offsets_ppm, 'linear', 'extrap');
            cestVolume_corrected(i,j,:) = corrected_spectrum;
        else
            cestVolume_corrected(i,j,:) = cestVolume(i,j,:);
        end
    end
end

fprintf('  B0 correction applied to %d voxels\n', sum(phantom_outline(:)));

%% Step 6: Prepare Data for Python MRF Matching
fprintf('Preparing data structure for Python MRF matching...\n');

% Image dimensions
image_dims = [xDim, yDim, num_cest];

% MRF acquisition parameters (adapt to your sequence)
% NOTE: Adjust these based on your actual CEST-MRF schedule!
M0_norm = 0;  % 0 = S0 normalization, 1 = max signal normalization
num_meas = num_cest;  % Number of CEST measurements

% Frequency offsets for simulation (in ppm, sorted from negative to positive)
[offsets_sorted, sort_idx] = sort(cest_offsets_ppm);
offsets_Hz_sorted = cest_offsets(sort_idx);

% Reorder CEST volume to match sorted offsets
cestVolume_sorted = cestVolume_corrected(:,:,sort_idx);

% Normalize by S0
M0_map = S0_image;
M0_map(M0_map == 0) = 1;  % Avoid division by zero
normalized_images = zeros(size(cestVolume_sorted));
for i = 1:num_cest
    normalized_images(:,:,i) = cestVolume_sorted(:,:,i) ./ M0_map;
end

%% Step 7: Create Dictionary Parameters Structure
fprintf('Creating dictionary parameters structure...\n');

% Call DictConfigParams to get default dictionary settings
% You can modify these in DictConfigParams.m for your specific phantom
seq_info.B0 = B0_field;
prefs.nPools = 2;  % Start with 2-pool model (water + CEST)

dictpars = DictConfigParams(seq_info, prefs);

fprintf('  Dictionary configuration:\n');
fprintf('    Water T1 range: %.2f - %.2f s\n', min(dictpars.water_t1), max(dictpars.water_t1));
fprintf('    Water T2 range: %.2f - %.2f s\n', min(dictpars.water_t2), max(dictpars.water_t2));
fprintf('    CEST Kex range: %d - %d Hz\n', min(dictpars.cest_amine_k), max(dictpars.cest_amine_k));
fprintf('    CEST conc range: %.1f - %.1f mM\n', ...
    min(dictpars.cest_amine_sol_conc), max(dictpars.cest_amine_sol_conc));

%% Step 8: Create Acquisition Schedule Structure
fprintf('Creating acquisition schedule structure...\n');

% B1 saturation power (adjust for your sequence!)
% Extract from DICOM or set manually
B1_power_uT = 2.0;  % Default value, ADJUST THIS!
sat_duration_s = 2.0;  % Saturation pulse duration in seconds, ADJUST THIS!

fprintf('  ⚠ WARNING: Using default saturation parameters!\n');
fprintf('    B1 power: %.1f µT\n', B1_power_uT);
fprintf('    Saturation duration: %.1f s\n', sat_duration_s);
fprintf('    YOU SHOULD UPDATE THESE VALUES!\n');

% Create schedule structure
acq_schedule.num_meas = num_meas;
acq_schedule.offsets_ppm = offsets_sorted;
acq_schedule.offsets_Hz = offsets_Hz_sorted;
acq_schedule.B1_power_uT = ones(num_meas, 1) * B1_power_uT;
acq_schedule.sat_duration_s = ones(num_meas, 1) * sat_duration_s;
acq_schedule.TR_s = TR / 1000;  % Convert ms to s
acq_schedule.TE_s = TE / 1000;  % Convert ms to s

%% Step 9: Save to acquired_data.mat Format
fprintf('Saving to acquired_data.mat format...\n');

% Create output directory if it doesn't exist
output_subdir = fullfile(output_dir, 'INPUT_FILES');
if ~exist(output_subdir, 'dir')
    mkdir(output_subdir);
end

saved_file = fullfile(output_subdir, 'acquired_data.mat');

% Save in the format expected by Python code
save(saved_file, ...
    'normalized_images', ...  % [xDim × yDim × num_meas] normalized CEST data
    'M0_map', ...             % [xDim × yDim] S0 reference image
    'B0_map_ppm', ...         % [xDim × yDim] B0 shift map
    'dictpars', ...           % Dictionary parameters structure
    'acq_schedule', ...       % Acquisition schedule
    'image_dims', ...         % Image dimensions
    'M0_norm', ...            % Normalization flag
    'phantom_outline', ...    % Mask
    '-v7.3');

fprintf('✓ Saved: %s\n', saved_file);
fprintf('  File size: %.1f MB\n', dir(saved_file).bytes / 1e6);

%% Step 10: Generate Visualization
fprintf('Generating preview images...\n');

fig = figure('Position', [100, 100, 1400, 600]);

subplot(2,3,1);
imagesc(S0_image);
axis image off;
colormap(gca, gray);
colorbar;
title('S0 Reference Image');

subplot(2,3,2);
imagesc(B0_map_ppm);
axis image off;
colormap(gca, jet);
colorbar;
title('B0 Map (ppm)');
caxis([-0.5 0.5]);

subplot(2,3,3);
imagesc(phantom_outline);
axis image off;
colormap(gca, gray);
title('Phantom Mask');

subplot(2,3,4);
imagesc(normalized_images(:,:,round(num_cest/2)));
axis image off;
colormap(gca, gray);
colorbar;
title(sprintf('Sample CEST Image (%.1f ppm)', offsets_sorted(round(num_cest/2))));

subplot(2,3,5);
% Plot central voxel Z-spectrum
center_x = round(xDim/2);
center_y = round(yDim/2);
zspec = squeeze(normalized_images(center_x, center_y, :));
plot(offsets_sorted, zspec, 'o-', 'LineWidth', 2);
xlabel('Offset (ppm)');
ylabel('Normalized Signal');
title('Central Voxel Z-spectrum');
grid on;
xlim([min(offsets_sorted) max(offsets_sorted)]);

subplot(2,3,6);
histogram(B0_map_ppm(phantom_outline), 50);
xlabel('B0 Shift (ppm)');
ylabel('Voxel Count');
title('B0 Distribution');
grid on;

sgtitle('Clinical 3T DICOM Adapter - Data Preview', 'FontSize', 14, 'FontWeight', 'bold');

preview_file = fullfile(output_dir, 'DICOM_adapter_preview.png');
saveas(fig, preview_file);
fprintf('✓ Saved preview: %s\n', preview_file);

%% Summary
fprintf('\n========================================\n');
fprintf('DICOM Adapter Complete!\n');
fprintf('========================================\n');
fprintf('Next steps:\n');
fprintf('  1. Review the preview image: DICOM_adapter_preview.png\n');
fprintf('  2. Verify acquisition parameters (B1, saturation duration)\n');
fprintf('  3. Adjust dictionary parameters in DictConfigParams.m if needed\n');
fprintf('  4. Run Python MRF matching:\n');
fprintf('     >> MRFmatch(dirstruct, prefs, false)\n');
fprintf('     (from MATLAB after setting up Python environment)\n');
fprintf('\n');
fprintf('Output file: %s\n', saved_file);
fprintf('========================================\n');

end
