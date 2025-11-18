% Enhanced z-spectra plotting with peak fits and MTR
% This script plots measured z-spectra, fitted peaks, and MTR for all ROIs

% Load your saved data
load('/home/user/BrukerCESTmethod_processingPipeline/Bruker_CEST-MRF_processing/MATLAB_MAIN-CODE-TO-RUN/saved_data_ROIs/img_roi_data.mat');

% Setup figure
figure('Name', 'Z-Spectra with Peak Fits and MTR', 'Position', [100 100 1400 900]);
colors = lines(5);

% Number of ROIs
num_rois = length(roi);

%% Plot 1: Z-Spectra with fitted peaks
subplot(2,2,[1 2]);
hold on;

for i = 1:num_rois
    % Calculate measured z-spectrum for this ROI (your working code)
    zspec_roi = squeeze(mean(mean(img.zSpec.img .* repmat(roi(i).mask, [1 1 size(img.zSpec.img,3)]), 1), 2));

    % Plot measured data
    plot(img.zSpec.ppm, zspec_roi, 'o-', 'LineWidth', 2, 'Color', colors(i,:), ...
         'DisplayName', sprintf('ROI %d - Measured', i));

    % Plot fitted data from peakFits (which contains 64x64x49 z-spectra)
    if isfield(img.zSpec, 'peakFits') && isstruct(img.zSpec.peakFits)
        % Plot total fit if 'all' field exists
        if isfield(img.zSpec.peakFits, 'all')
            fit_all_roi = squeeze(mean(mean(img.zSpec.peakFits.all .* repmat(roi(i).mask, [1 1 size(img.zSpec.peakFits.all,3)]), 1), 2));
            plot(img.zSpec.ppm, fit_all_roi, '--', 'LineWidth', 1.5, 'Color', colors(i,:), ...
                 'DisplayName', sprintf('ROI %d - Total Fit', i));
        end

        % Plot individual peak contributions (amide, MT, NOE)
        if isfield(img.zSpec.peakFits, 'amide')
            amide_roi = squeeze(mean(mean(img.zSpec.peakFits.amide .* repmat(roi(i).mask, [1 1 size(img.zSpec.peakFits.amide,3)]), 1), 2));
            plot(img.zSpec.ppm, amide_roi, ':', 'LineWidth', 1.5, 'Color', colors(i,:), ...
                 'DisplayName', sprintf('ROI %d - Amide', i));
        end

        if isfield(img.zSpec.peakFits, 'MT')
            mt_roi = squeeze(mean(mean(img.zSpec.peakFits.MT .* repmat(roi(i).mask, [1 1 size(img.zSpec.peakFits.MT,3)]), 1), 2));
            plot(img.zSpec.ppm, mt_roi, ':', 'LineWidth', 1.5, 'Color', colors(i,:)*0.8, ...
                 'DisplayName', sprintf('ROI %d - MT', i));
        end

        if isfield(img.zSpec.peakFits, 'NOE')
            noe_roi = squeeze(mean(mean(img.zSpec.peakFits.NOE .* repmat(roi(i).mask, [1 1 size(img.zSpec.peakFits.NOE,3)]), 1), 2));
            plot(img.zSpec.ppm, noe_roi, ':', 'LineWidth', 1.5, 'Color', colors(i,:)*0.6, ...
                 'DisplayName', sprintf('ROI %d - NOE', i));
        end
    end
end

xlabel('Offset (ppm)', 'FontSize', 12);
ylabel('Normalized Signal', 'FontSize', 12);
title('Z-Spectra: Measured Data and Peak Fits', 'FontSize', 14);
legend('Location', 'eastoutside', 'FontSize', 9);
grid on;
xlim([min(img.zSpec.ppm) max(img.zSpec.ppm)]);
hold off;

%% Plot 2: MTR Asymmetry (2D image, single value per voxel)
subplot(2,2,3);

if isfield(img.zSpec, 'MTRimg') && ~isempty(img.zSpec.MTRimg)
    % MTRimg is 64x64 (2D), so extract mean value per ROI
    mtr_values = zeros(1, num_rois);
    mtr_std = zeros(1, num_rois);

    for i = 1:num_rois
        mtr_in_roi = img.zSpec.MTRimg(roi(i).mask);
        mtr_values(i) = mean(mtr_in_roi, 'omitnan');
        mtr_std(i) = std(mtr_in_roi, 'omitnan');
    end

    bar(1:num_rois, mtr_values, 'FaceColor', [0.3 0.6 0.8]);
    hold on;
    errorbar(1:num_rois, mtr_values, mtr_std, 'k.', 'LineWidth', 1.5);
    hold off;

    xlabel('ROI', 'FontSize', 12);
    ylabel('MTR Asymmetry', 'FontSize', 12);
    title('Mean MTR Asymmetry per ROI', 'FontSize', 14);
    set(gca, 'XTick', 1:num_rois);
    grid on;

    % Add text labels with values
    for i = 1:num_rois
        text(i, mtr_values(i)+mtr_std(i), sprintf('%.3f', mtr_values(i)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 9);
    end
else
    text(0.5, 0.5, 'MTR data not available', 'HorizontalAlignment', 'center', ...
         'FontSize', 14, 'Units', 'normalized');
    axis off;
end

%% Plot 3: CEST Peak Heights (calculated from fitted z-spectra)
subplot(2,2,4);

if isfield(img.zSpec, 'peakFits') && isstruct(img.zSpec.peakFits)
    % Calculate peak heights from the fitted z-spectra (max saturation = min value)
    peak_names_list = {'amide', 'MT', 'NOE'};
    peak_data = [];
    peak_labels = {};

    for pn = 1:length(peak_names_list)
        if isfield(img.zSpec.peakFits, peak_names_list{pn})
            % Extract peak heights for each ROI
            roi_heights = zeros(1, num_rois);

            for i = 1:num_rois
                % Get fitted z-spectrum for this pool and ROI
                peak_zspec_roi = squeeze(mean(mean(img.zSpec.peakFits.(peak_names_list{pn}) .* repmat(roi(i).mask, [1 1 size(img.zSpec.peakFits.(peak_names_list{pn}),3)]), 1), 2));

                % Peak height is the maximum saturation effect (minimum value from baseline of 1)
                % CEST effect = 1 - min(signal), so higher effect = deeper saturation
                roi_heights(i) = 1 - min(peak_zspec_roi);
            end

            peak_data = [peak_data; roi_heights];
            peak_labels{end+1} = peak_names_list{pn};
        end
    end

    if ~isempty(peak_data)
        bar(peak_data', 'grouped');
        xlabel('ROI', 'FontSize', 12);
        ylabel('CEST Effect (1 - S/S0)', 'FontSize', 12);
        title('CEST Peak Heights by ROI', 'FontSize', 14);
        legend(peak_labels, 'Location', 'best');
        grid on;
        set(gca, 'XTick', 1:num_rois);
    else
        text(0.5, 0.5, 'Peak data not available', 'HorizontalAlignment', 'center', ...
             'FontSize', 14, 'Units', 'normalized');
        axis off;
    end
else
    text(0.5, 0.5, 'Peak fit data not available', 'HorizontalAlignment', 'center', ...
         'FontSize', 14, 'Units', 'normalized');
    axis off;
end

%% Print summary statistics
fprintf('\n=== Z-Spectra Summary Statistics ===\n');
for i = 1:num_rois
    fprintf('\nROI %d:\n', i);

    % Mean z-spectrum signal
    zspec_roi = squeeze(mean(mean(img.zSpec.img .* repmat(roi(i).mask, [1 1 size(img.zSpec.img,3)]), 1), 2));
    fprintf('  Mean z-spectrum signal: %.4f ± %.4f\n', mean(zspec_roi), std(zspec_roi));

    % MTR asymmetry (2D image)
    if isfield(img.zSpec, 'MTRimg') && ~isempty(img.zSpec.MTRimg)
        mtr_in_roi = img.zSpec.MTRimg(roi(i).mask);
        fprintf('  MTR Asymmetry: %.4f ± %.4f\n', mean(mtr_in_roi, 'omitnan'), std(mtr_in_roi, 'omitnan'));
    end

    % CEST peak heights from fitted spectra
    if isfield(img.zSpec, 'peakFits')
        if isfield(img.zSpec.peakFits, 'amide')
            amide_zspec = squeeze(mean(mean(img.zSpec.peakFits.amide .* repmat(roi(i).mask, [1 1 size(img.zSpec.peakFits.amide,3)]), 1), 2));
            fprintf('  Amide CEST effect: %.4f\n', 1 - min(amide_zspec));
        end
        if isfield(img.zSpec.peakFits, 'MT')
            mt_zspec = squeeze(mean(mean(img.zSpec.peakFits.MT .* repmat(roi(i).mask, [1 1 size(img.zSpec.peakFits.MT,3)]), 1), 2));
            fprintf('  MT CEST effect: %.4f\n', 1 - min(mt_zspec));
        end
        if isfield(img.zSpec.peakFits, 'NOE')
            noe_zspec = squeeze(mean(mean(img.zSpec.peakFits.NOE .* repmat(roi(i).mask, [1 1 size(img.zSpec.peakFits.NOE,3)]), 1), 2));
            fprintf('  NOE CEST effect: %.4f\n', 1 - min(noe_zspec));
        end
    end
end

fprintf('\nPlotting complete!\n');
