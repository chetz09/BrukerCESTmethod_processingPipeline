%% COMPLETE 3T CEST Phantom Analysis with Advanced Z-Spectrum Fitting
% This script performs complete analysis of 24-tube phantom data acquired on
% a 3T clinical scanner, including WASSR B0 correction and advanced multi-peak
% Lorentzian fitting using functions from the Bruker CEST-MRF repository.
%
% REQUIREMENTS:
% 1. Copy these files from the repository to your working directory:
%    - zspecMultiPeakFit.m
%    - zspecSetLPeakBounds.m
%    - Custom_3T_Phantom_PeakBounds.m (provided)
%
% Phantom Layout (24 tubes):
% Tubes 1-3:   Iopamidol 20mM pH 6.2, 6.8, 7.4
% Tubes 4-6:   Iopamidol 50mM pH 6.2, 6.8, 7.4
% Tubes 7-9:   Creatine 20mM pH 6.2, 6.8, 7.4
% Tubes 10-12: Creatine 50mM pH 6.2, 6.8, 7.4
% Tubes 13-15: Taurine 20mM pH 6.2, 6.8, 7.4
% Tubes 16-18: Taurine 50mM pH 6.2, 6.8, 7.4
% Tubes 19-21: PLL 0.1% pH 6.2, 6.8, 7.4
% Tubes 22-24: PBS (blank) pH 6.2, 6.8, 7.4
%
% Author: Integration of Bruker repository functions with 3T clinical data
% Date: 2025-12-02

clearvars; clc; close all;
tic;

Starting_Directory = pwd;

%% Add paths for fitting functions
fprintf('========================================\n');
fprintf('SETUP: Adding fitting function paths\n');
fprintf('========================================\n');

% Check if fitting functions are available
if ~exist('zspecMultiPeakFit', 'file')
    fprintf('⚠ WARNING: zspecMultiPeakFit.m not found!\n');
    fprintf('Please copy the following files to this directory:\n');
    fprintf('  - zspecMultiPeakFit.m\n');
    fprintf('  - zspecSetLPeakBounds.m\n');
    fprintf('  - Custom_3T_Phantom_PeakBounds.m\n\n');
    fprintf('From: Bruker_CEST-MRF_processing/MATLAB_MAIN-CODE-TO-RUN/\n');
    fprintf('      subfunctions_otherFiles/Data_parameter_loading_processing/\n');
    fprintf('      zspec_fitting_subfunctions/\n\n');
    use_advanced_fitting = false;
else
    fprintf('✓ Z-spectrum fitting functions found!\n\n');
    use_advanced_fitting = true;
end

%% Configuration
tube_labels = {
    'Iopamidol 20mM pH6.2', 'Iopamidol 20mM pH6.8', 'Iopamidol 20mM pH7.4', ...
    'Iopamidol 50mM pH6.2', 'Iopamidol 50mM pH6.8', 'Iopamidol 50mM pH7.4', ...
    'Creatine 20mM pH6.2', 'Creatine 20mM pH6.8', 'Creatine 20mM pH7.4', ...
    'Creatine 50mM pH6.2', 'Creatine 50mM pH6.8', 'Creatine 50mM pH7.4', ...
    'Taurine 20mM pH6.2', 'Taurine 20mM pH6.8', 'Taurine 20mM pH7.4', ...
    'Taurine 50mM pH6.2', 'Taurine 50mM pH6.8', 'Taurine 50mM pH7.4', ...
    'PLL 0.1% pH6.2', 'PLL 0.1% pH6.8', 'PLL 0.1% pH7.4', ...
    'PBS pH6.2', 'PBS pH6.8', 'PBS pH7.4'
};

%% Step 1: Load DICOM Data
disp('========================================');
disp('STEP 1: Load CEST Z-spectrum Data');
disp('========================================');
disp('Select CEST scan directory containing 63 DICOM files');
DIR_CEST = uigetdir(pwd, 'Select CEST DICOM folder');
if DIR_CEST == 0
    error('No folder selected. Exiting.');
end

dicomFiles = dir(fullfile(DIR_CEST, '*.dcm'));
numFiles = length(dicomFiles);
fprintf('Found %d DICOM files\n', numFiles);

if numFiles ~= 63
    warning('Expected 63 DICOM files but found %d. Proceeding with caution.', numFiles);
end

firstFile = fullfile(DIR_CEST, dicomFiles(1).name);
temp = dicomread(firstFile);
[xDim, yDim] = size(temp);

dicomVolume = zeros(xDim, yDim, numFiles);
fprintf('Loading DICOM files...\n');
for i = 1:numFiles
    filePath = fullfile(DIR_CEST, dicomFiles(i).name);
    dicomVolume(:,:,i) = double(dicomread(filePath));
end

%% Step 2: Define Frequency Offsets
disp('========================================');
disp('STEP 2: Defining Frequency Offsets');
disp('========================================');

B0_field = 3.0;  % Tesla
f0_MHz = 42.577 * B0_field;  % Larmor frequency

% WASSR images: indices 4-14 (11 images)
wassr_offsets_Hz = [240, 192, 144, 96, 48, 0, -48, -96, -144, -192, -240];
wassr_indices = 4:14;

% CEST images: indices 15-63 (49 images)
cest_offsets_Hz = [896, 864, 832, 800, 768, 736, 704, 672, 640, 608, 576, 544, ...
                   512, 480, 448, 416, 384, 352, 320, 288, 256, 192, 128, 64, 0, ...
                   -64, -128, -192, -256, -288, -320, -352, -384, -416, -448, ...
                   -480, -512, -544, -576, -608, -640, -672, -704, -736, -768, ...
                   -800, -832, -864, -896];
cest_indices = 15:63;

% S0 reference image (index 2)
S0_index = 2;
S0_image = dicomVolume(:,:,S0_index);

% Convert Hz to ppm
wassr_offsets_ppm = wassr_offsets_Hz / f0_MHz;
cest_offsets_ppm = cest_offsets_Hz / f0_MHz;

% Extract volumes
wasserVolume = dicomVolume(:,:,wassr_indices);
zspecVolume = dicomVolume(:,:,cest_indices);
ppmOffsets = cest_offsets_ppm';

fprintf('✓ Loaded complete dataset:\n');
fprintf('  - CEST PPM range: %.2f to %.2f ppm\n', min(ppmOffsets), max(ppmOffsets));

%% Step 3: Calculate B0 Map from WASSR
disp('========================================');
disp('STEP 3: Calculate B0 Map from WASSR');
disp('========================================');

B0_map_ppm = zeros(xDim, yDim);
for i = 1:xDim
    for j = 1:yDim
        wassr_spectrum = squeeze(wasserVolume(i,j,:));
        [~, min_idx] = min(wassr_spectrum);
        B0_map_ppm(i,j) = wassr_offsets_ppm(min_idx);
    end
end

B0_map_ppm = imgaussfilt(B0_map_ppm, 2);
B0_map_ppm(B0_map_ppm < -2) = -2;
B0_map_ppm(B0_map_ppm > 2) = 2;

fprintf('✓ B0 map calculated: mean = %.3f ppm\n', mean(B0_map_ppm(:)));

%% Step 4: Detect Phantom Outline
disp('========================================');
disp('STEP 4: Detect Phantom Outline');
disp('========================================');

Sm = imgaussfilt(S0_image, 4);
Smn = mat2gray(Sm);
bw_phantom = imbinarize(Smn);

CCp = bwconncomp(bw_phantom);
numPix = cellfun(@numel, CCp.PixelIdxList);
[~, idxMax] = max(numPix);
phantom_outline = false(size(bw_phantom));
phantom_outline(CCp.PixelIdxList{idxMax}) = true;
phantom_outline = imfill(phantom_outline, "holes");
phantom_outline = imopen(phantom_outline, strel('disk', 5));

fprintf('✓ Phantom outline detected\n');

%% Step 5: Load or Draw Tube Masks
disp('========================================');
disp('STEP 5: Load or Draw Tube Masks');
disp('========================================');

if exist('BMC_tubeMasks.mat', 'file')
    fprintf('Found existing BMC_tubeMasks.mat file.\n');
    choice = questdlg('Load existing tube masks or redraw?', ...
        'Load Masks', 'Load existing', 'Redraw all', 'Load existing');

    if strcmp(choice, 'Load existing')
        load('BMC_tubeMasks.mat', 'tubeMasks', 'tube_info');
        numTubes = size(tubeMasks, 3);
        fprintf('✓ Loaded %d tube masks\n', numTubes);
        use_loaded_masks = true;
    else
        use_loaded_masks = false;
    end
else
    fprintf('No existing masks found.\n');
    use_loaded_masks = false;
end

if ~use_loaded_masks
    error('Please run your tube drawing code first to create BMC_tubeMasks.mat');
end

%% Step 6: Apply B0 Correction to Z-spectra
disp('========================================');
disp('STEP 6: Apply B0 Correction');
disp('========================================');

zspecVolume_corrected = zeros(size(zspecVolume));
for i = 1:xDim
    for j = 1:yDim
        if phantom_outline(i,j)
            B0_offset = B0_map_ppm(i,j);
            shifted_ppm = ppmOffsets - B0_offset;
            original_spectrum = squeeze(zspecVolume(i,j,:));
            corrected_spectrum = interp1(shifted_ppm, original_spectrum, ...
                ppmOffsets, 'linear', 'extrap');
            zspecVolume_corrected(i,j,:) = corrected_spectrum;
        else
            zspecVolume_corrected(i,j,:) = zspecVolume(i,j,:);
        end
    end
end

fprintf('✓ B0 correction applied\n');

%% Step 7: Extract Z-spectra and Calculate MTR Asymmetry
disp('========================================');
disp('STEP 7: Extract Z-spectra & MTRasym');
disp('========================================');

results = struct();
results.tube_number = (1:numTubes)';
results.tube_label = tube_labels';
results.num_voxels = zeros(numTubes, 1);
results.CEST_at_3_5ppm = zeros(numTubes, 1);
results.CEST_at_4_3ppm = zeros(numTubes, 1);
results.CEST_at_5_5ppm = zeros(numTubes, 1);
results.CEST_at_1_9ppm = zeros(numTubes, 1);
results.CEST_at_2_7ppm = zeros(numTubes, 1);
results.CEST_at_2_8ppm = zeros(numTubes, 1);
results.B0_mean_ppm = zeros(numTubes, 1);

all_zspectra = zeros(numTubes, length(ppmOffsets));

for t = 1:numTubes
    currentMask = tubeMasks(:,:,t);
    results.num_voxels(t) = sum(currentMask(:));
    results.B0_mean_ppm(t) = mean(B0_map_ppm(currentMask), 'omitnan');

    % Extract mean Z-spectrum
    tube_zspec = zeros(length(ppmOffsets), 1);
    for offset_idx = 1:length(ppmOffsets)
        img_slice = zspecVolume_corrected(:,:,offset_idx);
        tube_zspec(offset_idx) = mean(img_slice(currentMask), 'omitnan');
    end

    S0_tube = mean(S0_image(currentMask), 'omitnan');
    tube_zspec_norm = tube_zspec / S0_tube;
    all_zspectra(t, :) = tube_zspec_norm;

    % Calculate MTR asymmetry at key offsets
    idx_pos_3_5 = find(abs(ppmOffsets - 3.5) == min(abs(ppmOffsets - 3.5)), 1);
    idx_neg_3_5 = find(abs(ppmOffsets + 3.5) == min(abs(ppmOffsets + 3.5)), 1);
    results.CEST_at_3_5ppm(t) = 100 * (tube_zspec_norm(idx_neg_3_5) - tube_zspec_norm(idx_pos_3_5));

    idx_pos_4_3 = find(abs(ppmOffsets - 4.3) == min(abs(ppmOffsets - 4.3)), 1);
    idx_neg_4_3 = find(abs(ppmOffsets + 4.3) == min(abs(ppmOffsets + 4.3)), 1);
    results.CEST_at_4_3ppm(t) = 100 * (tube_zspec_norm(idx_neg_4_3) - tube_zspec_norm(idx_pos_4_3));

    idx_pos_1_9 = find(abs(ppmOffsets - 1.9) == min(abs(ppmOffsets - 1.9)), 1);
    idx_neg_1_9 = find(abs(ppmOffsets + 1.9) == min(abs(ppmOffsets + 1.9)), 1);
    results.CEST_at_1_9ppm(t) = 100 * (tube_zspec_norm(idx_neg_1_9) - tube_zspec_norm(idx_pos_1_9));

    idx_pos_5_5 = find(abs(ppmOffsets - 5.5) == min(abs(ppmOffsets - 5.5)), 1);
    idx_neg_5_5 = find(abs(ppmOffsets + 5.5) == min(abs(ppmOffsets + 5.5)), 1);
    results.CEST_at_5_5ppm(t) = 100 * (tube_zspec_norm(idx_neg_5_5) - tube_zspec_norm(idx_pos_5_5));

    idx_pos_2_7 = find(abs(ppmOffsets - 2.7) == min(abs(ppmOffsets - 2.7)), 1);
    idx_neg_2_7 = find(abs(ppmOffsets + 2.7) == min(abs(ppmOffsets + 2.7)), 1);
    results.CEST_at_2_7ppm(t) = 100 * (tube_zspec_norm(idx_neg_2_7) - tube_zspec_norm(idx_pos_2_7));

    idx_pos_2_8 = find(abs(ppmOffsets - 2.8) == min(abs(ppmOffsets - 2.8)), 1);
    idx_neg_2_8 = find(abs(ppmOffsets + 2.8) == min(abs(ppmOffsets + 2.8)), 1);
    results.CEST_at_2_8ppm(t) = 100 * (tube_zspec_norm(idx_neg_2_8) - tube_zspec_norm(idx_pos_2_8));
end

fprintf('✓ MTR asymmetry calculated for all tubes\n');

%% Step 8: Advanced Multi-Peak Lorentzian Fitting
disp('========================================');
disp('STEP 8: Advanced Multi-Peak Fitting');
disp('========================================');

if use_advanced_fitting
    fprintf('Performing advanced multi-peak Lorentzian fitting...\n');

    % Initialize storage
    results.fitted_amplitude = zeros(numTubes, 1);
    results.fitted_FWHM = zeros(numTubes, 1);
    results.fitted_offset_ppm = zeros(numTubes, 1);
    results.fitted_amplitude_CI_lower = zeros(numTubes, 1);
    results.fitted_amplitude_CI_upper = zeros(numTubes, 1);
    results.fitted_offset_CI_lower = zeros(numTubes, 1);
    results.fitted_offset_CI_upper = zeros(numTubes, 1);
    results.fit_quality_resnorm = zeros(numTubes, 1);
    results.fitted_amplitude_peak2 = zeros(numTubes, 1);
    results.fitted_offset_ppm_peak2 = zeros(numTubes, 1);

    for t = 1:numTubes
        fprintf('  Tube %d/%d: %s\n', t, numTubes, tube_labels{t});

        tube_zspec_norm = all_zspectra(t, :)';
        OneMinZ = 1 - tube_zspec_norm;

        % Determine tube type
        if t <= 6  % Iopamidol
            pNames = {'water', 'amine', 'amide'};
            pPars = Custom_3T_Phantom_PeakBounds('iopamidol');
            fixedVals.water = [NaN, NaN, 0, NaN];
            fixedVals.amine = [NaN, NaN, NaN, NaN];
            fixedVals.amide = [NaN, NaN, NaN, NaN];
            tube_type = 'iopamidol';

        elseif t <= 12  % Creatine
            pNames = {'water', 'amine'};
            pPars = Custom_3T_Phantom_PeakBounds('creatine');
            fixedVals.water = [NaN, NaN, 0, NaN];
            fixedVals.amine = [NaN, NaN, NaN, NaN];
            tube_type = 'creatine';

        elseif t <= 18  % Taurine
            pNames = {'water', 'amine'};
            pPars = Custom_3T_Phantom_PeakBounds('taurine');
            fixedVals.water = [NaN, NaN, 0, NaN];
            fixedVals.amine = [NaN, NaN, NaN, NaN];
            tube_type = 'taurine';

        elseif t <= 21  % PLL
            pNames = {'water', 'amide', 'amine'};
            pPars = Custom_3T_Phantom_PeakBounds('PLL');
            fixedVals.water = [NaN, NaN, 0, NaN];
            fixedVals.amide = [NaN, NaN, NaN, NaN];
            fixedVals.amine = [NaN, NaN, NaN, NaN];
            tube_type = 'PLL';

        else  % PBS
            pNames = {'water', 'amine'};
            pPars = Custom_3T_Phantom_PeakBounds('PBS');
            fixedVals.water = [NaN, NaN, 0, NaN];
            fixedVals.amine = [NaN, NaN, NaN, NaN];
            tube_type = 'PBS';
        end

        % Perform fitting
        try
            [EP, CI, Residual, ~, ~] = zspecMultiPeakFit(ppmOffsets, OneMinZ, ...
                pNames, pPars, fixedVals, NaN, false);

            % Store results based on tube type
            if strcmp(tube_type, 'PLL')
                results.fitted_amplitude(t) = EP.amide(1);
                results.fitted_FWHM(t) = EP.amide(2);
                results.fitted_offset_ppm(t) = EP.amide(3);
                results.fitted_amplitude_CI_lower(t) = CI.amide(1,1);
                results.fitted_amplitude_CI_upper(t) = CI.amide(1,2);
                results.fitted_offset_CI_lower(t) = CI.amide(3,1);
                results.fitted_offset_CI_upper(t) = CI.amide(3,2);
                results.fitted_amplitude_peak2(t) = EP.amine(1);
                results.fitted_offset_ppm_peak2(t) = EP.amine(3);

            elseif strcmp(tube_type, 'iopamidol')
                results.fitted_amplitude(t) = EP.amine(1);
                results.fitted_FWHM(t) = EP.amine(2);
                results.fitted_offset_ppm(t) = EP.amine(3);
                results.fitted_amplitude_CI_lower(t) = CI.amine(1,1);
                results.fitted_amplitude_CI_upper(t) = CI.amine(1,2);
                results.fitted_offset_CI_lower(t) = CI.amine(3,1);
                results.fitted_offset_CI_upper(t) = CI.amine(3,2);
                results.fitted_amplitude_peak2(t) = EP.amide(1);
                results.fitted_offset_ppm_peak2(t) = EP.amide(3);

            else
                results.fitted_amplitude(t) = EP.amine(1);
                results.fitted_FWHM(t) = EP.amine(2);
                results.fitted_offset_ppm(t) = EP.amine(3);
                results.fitted_amplitude_CI_lower(t) = CI.amine(1,1);
                results.fitted_amplitude_CI_upper(t) = CI.amine(1,2);
                results.fitted_offset_CI_lower(t) = CI.amine(3,1);
                results.fitted_offset_CI_upper(t) = CI.amine(3,2);
            end

            results.fit_quality_resnorm(t) = sum(Residual.^2);

            fprintf('    ✓ Fitted: Amp=%.4f, Offset=%.3f ppm, FWHM=%.3f ppm\n', ...
                results.fitted_amplitude(t), results.fitted_offset_ppm(t), ...
                results.fitted_FWHM(t));

        catch ME
            warning('    ⚠ Fitting failed: %s', ME.message);
            results.fitted_amplitude(t) = NaN;
            results.fitted_FWHM(t) = NaN;
            results.fitted_offset_ppm(t) = NaN;
        end
    end

    fprintf('✓ Advanced fitting complete!\n');
else
    fprintf('⚠ Skipping advanced fitting (functions not found)\n');
end

%% Step 9: Generate Plots
disp('========================================');
disp('STEP 9: Generate Plots');
disp('========================================');

% Plot 1: Z-spectra
fig1 = figure('Position', [50, 50, 2000, 1200]);
for t = 1:numTubes
    subplot(4, 6, t);
    plot(ppmOffsets, all_zspectra(t,:), 'bo-', 'LineWidth', 1.5, ...
        'MarkerSize', 4, 'MarkerFaceColor', 'b');
    xlabel('Offset (ppm)', 'FontSize', 8);
    ylabel('Z', 'FontSize', 8);
    title(sprintf('T%d: %s', t, strrep(tube_labels{t}, '_', ' ')), ...
        'FontSize', 8, 'Interpreter', 'none');
    grid on;
    xlim([min(ppmOffsets) max(ppmOffsets)]);
    ylim([0 1.2]);
    set(gca, 'FontSize', 7);
end
sgtitle('Z-spectra for All 24 Tubes', 'FontSize', 14, 'FontWeight', 'bold');
saveas(fig1, 'Zspectra_all_tubes.png');

% Plot 2: Fitting Results
if use_advanced_fitting
    fig2 = figure('Position', [100, 100, 1400, 900]);

    subplot(2,3,1);
    bar(1:numTubes, results.fitted_amplitude);
    hold on;
    errorbar(1:numTubes, results.fitted_amplitude, ...
        results.fitted_amplitude - results.fitted_amplitude_CI_lower, ...
        results.fitted_amplitude_CI_upper - results.fitted_amplitude, ...
        'k.', 'LineWidth', 1.5);
    xlabel('Tube Number');
    ylabel('Fitted Amplitude');
    title('Fitted CEST Amplitude (95% CI)');
    grid on;

    subplot(2,3,2);
    bar(1:numTubes, results.fitted_offset_ppm);
    hold on;
    errorbar(1:numTubes, results.fitted_offset_ppm, ...
        results.fitted_offset_ppm - results.fitted_offset_CI_lower, ...
        results.fitted_offset_CI_upper - results.fitted_offset_ppm, ...
        'k.', 'LineWidth', 1.5);
    xlabel('Tube Number');
    ylabel('Chemical Shift (ppm)');
    title('Fitted Peak Offset (95% CI)');
    grid on;
    ylim([0 6]);

    subplot(2,3,3);
    bar(1:numTubes, results.fitted_FWHM);
    xlabel('Tube Number');
    ylabel('FWHM (ppm)');
    title('Fitted Peak Width');
    grid on;

    subplot(2,3,4);
    bar(1:numTubes, results.fit_quality_resnorm);
    xlabel('Tube Number');
    ylabel('Residual Norm');
    title('Fit Quality');
    grid on;
    set(gca, 'YScale', 'log');

    subplot(2,3,5);
    plot(1:numTubes, results.fitted_amplitude * 100, 'bo-', 'LineWidth', 2, ...
        'MarkerFaceColor', 'b', 'DisplayName', 'Fitted Amp × 100');
    hold on;
    plot(1:6, results.CEST_at_4_3ppm(1:6), 'ro-', 'LineWidth', 2, ...
        'MarkerFaceColor', 'r', 'DisplayName', 'MTRasym @ 4.3ppm');
    xlabel('Tube Number');
    ylabel('CEST Effect (%)');
    title('Fitted vs MTRasym');
    legend;
    grid on;

    subplot(2,3,6);
    iopamidol_idx = 1:6;
    pll_idx = 19:21;
    bar(iopamidol_idx, results.fitted_amplitude_peak2(iopamidol_idx) * 100, ...
        'FaceColor', [0.8 0.4 0.4]);
    hold on;
    bar(pll_idx, results.fitted_amplitude_peak2(pll_idx) * 100, ...
        'FaceColor', [0.4 0.8 0.4]);
    xlabel('Tube Number');
    ylabel('Secondary Peak Amp × 100');
    title('Secondary Peaks');
    grid on;

    sgtitle('Advanced Multi-Peak Fitting Results', 'FontSize', 16, 'FontWeight', 'bold');
    saveas(fig2, 'Advanced_Fitting_Results.png');
end

%% Step 10: Export Results
disp('========================================');
disp('STEP 10: Export Results');
disp('========================================');

T = struct2table(results);
writetable(T, 'CEST_Phantom_Results.csv');
fprintf('✓ Saved: CEST_Phantom_Results.csv\n');

save('CEST_Phantom_Workspace.mat', 'results', 'tubeMasks', 'B0_map_ppm', ...
    'zspecVolume_corrected', 'ppmOffsets', 'all_zspectra');
fprintf('✓ Saved: CEST_Phantom_Workspace.mat\n');

%% Summary
disp('========================================');
disp('ANALYSIS COMPLETE!');
disp('========================================');
fprintf('Total time: %.1f seconds\n', toc);
fprintf('\nGenerated Files:\n');
fprintf('  1. CEST_Phantom_Results.csv - All results\n');
fprintf('  2. Zspectra_all_tubes.png - Z-spectra plots\n');
if use_advanced_fitting
    fprintf('  3. Advanced_Fitting_Results.png - Fitting analysis\n');
end
fprintf('  4. CEST_Phantom_Workspace.mat - Full workspace\n');
fprintf('\nKey Results:\n');
fprintf('  - Tubes analyzed: %d\n', numTubes);
fprintf('  - B0 correction: Applied (WASSR-based)\n');
if use_advanced_fitting
    fprintf('  - Multi-peak fitting: COMPLETED\n');
    fprintf('  - Mean fit quality: %.2e\n', mean(results.fit_quality_resnorm, 'omitnan'));
else
    fprintf('  - Multi-peak fitting: SKIPPED (copy fitting functions)\n');
end
disp('========================================');
