% Z-spectra plotting with fitted peaks (Amide, MT, NOE) and MTR asymmetry
% This script plots measured z-spectra with individual peak fits for all ROIs

% Load your saved data
load('/home/user/BrukerCESTmethod_processingPipeline/Bruker_CEST-MRF_processing/MATLAB_MAIN-CODE-TO-RUN/saved_data_ROIs/img_roi_data.mat');

% Setup colors
colors = lines(5);

% Number of ROIs
num_rois = length(roi);

% ROI names
roi_names = {'GA\_BSA\_5mM', 'GA\_BSA\_10mM', 'GA\_BSA\_20mM', 'GA\_BSA\_30mM', 'GA\_30mM'};

%% Figure 1: Z-Spectra with fitted peaks
figure('Name', 'Z-Spectra with Peak Fits');
hold on;

for i = 1:num_rois
    % Calculate measured z-spectrum for this ROI
    zspec_roi = squeeze(mean(mean(img.zSpec.img .* repmat(roi(i).mask, [1 1 size(img.zSpec.img,3)]), 1), 2));

    % Plot measured data
    plot(img.zSpec.ppm, zspec_roi, 'o-', 'LineWidth', 2, 'Color', colors(i,:), ...
         'DisplayName', sprintf('%s - Measured', roi_names{i}));

    % Plot individual peak fits (amide, MT, NOE) - all with solid lines
    if isfield(img.zSpec, 'peakFits') && isstruct(img.zSpec.peakFits)

        if isfield(img.zSpec.peakFits, 'amide')
            amide_roi = squeeze(mean(mean(img.zSpec.peakFits.amide .* repmat(roi(i).mask, [1 1 size(img.zSpec.peakFits.amide,3)]), 1), 2));
            plot(img.zSpec.ppm, amide_roi, '-', 'LineWidth', 1.5, 'Color', colors(i,:), ...
                 'DisplayName', sprintf('%s - Amide', roi_names{i}));
        end

        if isfield(img.zSpec.peakFits, 'MT')
            mt_roi = squeeze(mean(mean(img.zSpec.peakFits.MT .* repmat(roi(i).mask, [1 1 size(img.zSpec.peakFits.MT,3)]), 1), 2));
            plot(img.zSpec.ppm, mt_roi, '-', 'LineWidth', 1.5, 'Color', colors(i,:), ...
                 'DisplayName', sprintf('%s - MT', roi_names{i}));
        end

        if isfield(img.zSpec.peakFits, 'NOE')
            noe_roi = squeeze(mean(mean(img.zSpec.peakFits.NOE .* repmat(roi(i).mask, [1 1 size(img.zSpec.peakFits.NOE,3)]), 1), 2));
            plot(img.zSpec.ppm, noe_roi, '-', 'LineWidth', 1.5, 'Color', colors(i,:), ...
                 'DisplayName', sprintf('%s - NOE', roi_names{i}));
        end
    end
end

xlabel('Offset (ppm)', 'FontSize', 12);
ylabel('Normalized Signal', 'FontSize', 12);
title('Z-Spectra with Peak Fits (Amide, MT, NOE)', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 8);
grid on;
xlim([min(img.zSpec.ppm) max(img.zSpec.ppm)]);
hold off;

%% Figure 2: MTR asymmetry separate plot
figure('Name', 'MTR Asymmetry');

if isfield(img.zSpec, 'MTRimg') && ~isempty(img.zSpec.MTRimg)
    mtr_values = zeros(1, num_rois);
    mtr_std = zeros(1, num_rois);

    for i = 1:num_rois
        mtr_in_roi = img.zSpec.MTRimg(roi(i).mask);
        mtr_values(i) = mean(mtr_in_roi, 'omitnan');
        mtr_std(i) = std(mtr_in_roi, 'omitnan');
    end

    % Plot as line graph with error bars
    hold on;
    for i = 1:num_rois
        errorbar(i, mtr_values(i), mtr_std(i), 'o', 'MarkerSize', 8, ...
                 'LineWidth', 2, 'Color', colors(i,:), 'MarkerFaceColor', colors(i,:), ...
                 'DisplayName', roi_names{i});
    end
    plot(1:num_rois, mtr_values, '-', 'LineWidth', 1.5, 'Color', [0.5 0.5 0.5], ...
         'HandleVisibility', 'off');
    hold off;

    xlabel('Sample', 'FontSize', 12);
    ylabel('MTR Asymmetry', 'FontSize', 12);
    title('MTR Asymmetry per Sample', 'FontSize', 14);
    set(gca, 'XTick', 1:num_rois, 'XTickLabel', roi_names);
    xtickangle(45);
    legend('Location', 'best');
    grid on;
else
    text(0.5, 0.5, 'MTR data not available', 'HorizontalAlignment', 'center', ...
         'FontSize', 14, 'Units', 'normalized');
    axis off;
end

fprintf('\nPlotting complete!\n');
