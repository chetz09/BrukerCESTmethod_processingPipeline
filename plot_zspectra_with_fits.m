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

    % Try to plot fitted data if fitImg exists
    if isfield(img.zSpec, 'fitImg')
        if isstruct(img.zSpec.fitImg)
            % If fitImg is a struct with individual pool fits
            fields = fieldnames(img.zSpec.fitImg);

            % Plot total fit
            if ismember('total', fields) || ismember('Total', fields)
                field_name = 'total';
                if ~ismember('total', fields)
                    field_name = 'Total';
                end
                fit_roi = squeeze(mean(mean(img.zSpec.fitImg.(field_name) .* repmat(roi(i).mask, [1 1 size(img.zSpec.fitImg.(field_name),3)]), 1), 2));
                plot(img.zSpec.ppm, fit_roi, '--', 'LineWidth', 1.5, 'Color', colors(i,:), ...
                     'DisplayName', sprintf('ROI %d - Total Fit', i));
            end

            % Plot individual peaks (Amide, MT, NOE) with lighter shades
            peak_names = {'Amide', 'MT', 'NOE', 'amide', 'mt', 'noe'};
            for pn = 1:length(peak_names)
                if ismember(peak_names{pn}, fields)
                    peak_roi = squeeze(mean(mean(img.zSpec.fitImg.(peak_names{pn}) .* repmat(roi(i).mask, [1 1 size(img.zSpec.fitImg.(peak_names{pn}),3)]), 1), 2));
                    plot(img.zSpec.ppm, peak_roi, ':', 'LineWidth', 1, 'Color', colors(i,:)*0.7, ...
                         'DisplayName', sprintf('ROI %d - %s', i, peak_names{pn}));
                end
            end
        else
            % If fitImg is just an array (total fit)
            fit_roi = squeeze(mean(mean(img.zSpec.fitImg .* repmat(roi(i).mask, [1 1 size(img.zSpec.fitImg,3)]), 1), 2));
            plot(img.zSpec.ppm, fit_roi, '--', 'LineWidth', 1.5, 'Color', colors(i,:), ...
                 'DisplayName', sprintf('ROI %d - Fit', i));
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

%% Plot 2: MTR Asymmetry
subplot(2,2,3);
hold on;

if isfield(img.zSpec, 'MTRimg') && ~isempty(img.zSpec.MTRimg)
    for i = 1:num_rois
        % Calculate MTR for this ROI
        mtr_roi = squeeze(mean(mean(img.zSpec.MTRimg .* repmat(roi(i).mask, [1 1 size(img.zSpec.MTRimg,3)]), 1), 2));

        % Find ppm values for MTR (should be half the size of full spectrum)
        if length(mtr_roi) < length(img.zSpec.ppm)
            % MTR is typically calculated for positive ppm only
            ppm_positive = img.zSpec.ppm(img.zSpec.ppm >= 0);
            if length(mtr_roi) == length(ppm_positive)
                plot(ppm_positive, mtr_roi, 'o-', 'LineWidth', 2, 'Color', colors(i,:), ...
                     'DisplayName', sprintf('ROI %d', i));
            else
                plot(mtr_roi, 'o-', 'LineWidth', 2, 'Color', colors(i,:), ...
                     'DisplayName', sprintf('ROI %d', i));
            end
        else
            plot(img.zSpec.ppm, mtr_roi, 'o-', 'LineWidth', 2, 'Color', colors(i,:), ...
                 'DisplayName', sprintf('ROI %d', i));
        end
    end
    xlabel('Offset (ppm)', 'FontSize', 12);
    ylabel('MTR Asymmetry', 'FontSize', 12);
    title('Magnetization Transfer Ratio (MTR) Asymmetry', 'FontSize', 14);
    legend('Location', 'best');
    grid on;
else
    text(0.5, 0.5, 'MTR data not available', 'HorizontalAlignment', 'center', ...
         'FontSize', 14, 'Units', 'normalized');
    axis off;
end
hold off;

%% Plot 3: Peak amplitudes comparison
subplot(2,2,4);

if isfield(img.zSpec, 'peakFits') && isstruct(img.zSpec.peakFits)
    % Try to extract peak parameters if available
    peak_names = {'Amide', 'MT', 'NOE', 'amide', 'mt', 'noe'};
    peak_data = [];
    peak_labels = {};

    for pn = 1:length(peak_names)
        if isfield(img.zSpec.peakFits, peak_names{pn})
            if isstruct(img.zSpec.peakFits.(peak_names{pn}))
                % Check for amplitude or height field
                if isfield(img.zSpec.peakFits.(peak_names{pn}), 'amp')
                    peak_img = img.zSpec.peakFits.(peak_names{pn}).amp;
                elseif isfield(img.zSpec.peakFits.(peak_names{pn}), 'amplitude')
                    peak_img = img.zSpec.peakFits.(peak_names{pn}).amplitude;
                elseif isfield(img.zSpec.peakFits.(peak_names{pn}), 'height')
                    peak_img = img.zSpec.peakFits.(peak_names{pn}).height;
                else
                    continue;
                end

                % Extract ROI values
                roi_values = zeros(1, num_rois);
                for i = 1:num_rois
                    roi_values(i) = mean(peak_img(roi(i).mask), 'all', 'omitnan');
                end
                peak_data = [peak_data; roi_values];
                peak_labels{end+1} = peak_names{pn};
            end
        end
    end

    if ~isempty(peak_data)
        bar(peak_data');
        xlabel('ROI', 'FontSize', 12);
        ylabel('Peak Amplitude', 'FontSize', 12);
        title('CEST Peak Amplitudes by ROI', 'FontSize', 14);
        legend(peak_labels, 'Location', 'best');
        grid on;
        set(gca, 'XTick', 1:num_rois);
    else
        text(0.5, 0.5, 'Peak amplitude data not available', 'HorizontalAlignment', 'center', ...
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

    % MTR at Amide peak (3.5 ppm)
    if isfield(img.zSpec, 'MTRimg') && ~isempty(img.zSpec.MTRimg)
        [~, amide_idx] = min(abs(img.zSpec.ppm - 3.5));
        if amide_idx <= size(img.zSpec.MTRimg, 3)
            mtr_amide = img.zSpec.MTRimg(:,:,amide_idx) .* roi(i).mask;
            fprintf('  MTR at Amide (3.5 ppm): %.4f ± %.4f\n', mean(mtr_amide(roi(i).mask)), std(mtr_amide(roi(i).mask)));
        end
    end
end

fprintf('\nPlotting complete!\n');
