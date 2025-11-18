% Z-spectra plotting with fitted peaks (Amide, MT, NOE)
% This script plots measured z-spectra with individual peak fits for all ROIs

% Load your saved data
load('/home/user/BrukerCESTmethod_processingPipeline/Bruker_CEST-MRF_processing/MATLAB_MAIN-CODE-TO-RUN/saved_data_ROIs/img_roi_data.mat');

% Setup figure
figure('Name', 'Z-Spectra with Peak Fits');
colors = lines(5);

% Number of ROIs
num_rois = length(roi);

%% Plot Z-Spectra with fitted peaks
hold on;

for i = 1:num_rois
    % Calculate measured z-spectrum for this ROI
    zspec_roi = squeeze(mean(mean(img.zSpec.img .* repmat(roi(i).mask, [1 1 size(img.zSpec.img,3)]), 1), 2));

    % Plot measured data
    plot(img.zSpec.ppm, zspec_roi, 'o-', 'LineWidth', 2, 'Color', colors(i,:), ...
         'DisplayName', sprintf('ROI %d - Measured', i));

    % Plot individual peak fits (amide, MT, NOE) - all with solid lines
    if isfield(img.zSpec, 'peakFits') && isstruct(img.zSpec.peakFits)

        if isfield(img.zSpec.peakFits, 'amide')
            amide_roi = squeeze(mean(mean(img.zSpec.peakFits.amide .* repmat(roi(i).mask, [1 1 size(img.zSpec.peakFits.amide,3)]), 1), 2));
            plot(img.zSpec.ppm, amide_roi, '-', 'LineWidth', 1.5, 'Color', colors(i,:), ...
                 'DisplayName', sprintf('ROI %d - Amide', i));
        end

        if isfield(img.zSpec.peakFits, 'MT')
            mt_roi = squeeze(mean(mean(img.zSpec.peakFits.MT .* repmat(roi(i).mask, [1 1 size(img.zSpec.peakFits.MT,3)]), 1), 2));
            plot(img.zSpec.ppm, mt_roi, '-', 'LineWidth', 1.5, 'Color', colors(i,:), ...
                 'DisplayName', sprintf('ROI %d - MT', i));
        end

        if isfield(img.zSpec.peakFits, 'NOE')
            noe_roi = squeeze(mean(mean(img.zSpec.peakFits.NOE .* repmat(roi(i).mask, [1 1 size(img.zSpec.peakFits.NOE,3)]), 1), 2));
            plot(img.zSpec.ppm, noe_roi, '-', 'LineWidth', 1.5, 'Color', colors(i,:), ...
                 'DisplayName', sprintf('ROI %d - NOE', i));
        end
    end
end

xlabel('Offset (ppm)', 'FontSize', 12);
ylabel('Normalized Signal', 'FontSize', 12);
title('Z-Spectra with Peak Fits (Amide, MT, NOE)', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([min(img.zSpec.ppm) max(img.zSpec.ppm)]);
hold off;

fprintf('\nPlotting complete!\n');
