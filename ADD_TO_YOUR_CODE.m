%% =======================================================================
%% PLUG-AND-PLAY: Add this code block after Step 7 in your existing script
%% =======================================================================
%
% This code performs advanced multi-peak Lorentzian fitting using the
% functions from the Bruker CEST-MRF repository.
%
% REQUIREMENTS:
% 1. Copy these files to your working directory:
%    - zspecMultiPeakFit.m
%    - zspecSetLPeakBounds.m
%    - Custom_3T_Phantom_PeakBounds.m (created for your phantom)
%
% 2. Make sure they're on your MATLAB path:
%    >> addpath(pwd);  % if files are in current directory
%
% INSERT THIS CODE AFTER "Step 7: Bloch-McConnell Fitting for Each Tube"
%% =======================================================================

%% Step 7b: Advanced Multi-Peak Lorentzian Fitting
disp('========================================');
disp('STEP 7b: Advanced Multi-Peak Fitting');
disp('========================================');

% Check if fitting functions are available
if ~exist('zspecMultiPeakFit', 'file')
    warning('zspecMultiPeakFit.m not found! Skipping advanced fitting.');
    warning('Copy fitting functions from BrukerCESTmethod_processingPipeline repository.');
else
    fprintf('Performing advanced multi-peak Lorentzian fitting for all tubes...\n');

    % Initialize storage for fitted parameters
    results.fitted_amplitude = zeros(numTubes, 1);
    results.fitted_FWHM = zeros(numTubes, 1);
    results.fitted_offset_ppm = zeros(numTubes, 1);
    results.fitted_amplitude_CI_lower = zeros(numTubes, 1);
    results.fitted_amplitude_CI_upper = zeros(numTubes, 1);
    results.fitted_offset_CI_lower = zeros(numTubes, 1);
    results.fitted_offset_CI_upper = zeros(numTubes, 1);
    results.fit_quality_resnorm = zeros(numTubes, 1);

    % For Iopamidol: also store second peak parameters
    results.fitted_amplitude_peak2 = zeros(numTubes, 1);
    results.fitted_offset_ppm_peak2 = zeros(numTubes, 1);

    for t = 1:numTubes
        fprintf('  Tube %d/%d: %s\n', t, numTubes, tube_labels{t});

        % Get normalized Z-spectrum for this tube
        tube_zspec_norm = all_zspectra(t, :)';  % Make sure it's a column vector
        OneMinZ = 1 - tube_zspec_norm;

        % Determine tube type and get appropriate peak bounds
        if t <= 6  % Iopamidol
            tube_type = 'iopamidol';
            pNames = {'water', 'amine', 'amide'};  % Two CEST peaks
            pPars = Custom_3T_Phantom_PeakBounds('iopamidol');
            fixedVals.water = [NaN, NaN, 0, NaN];  % Fix water at 0 ppm
            fixedVals.amine = [NaN, NaN, NaN, NaN];  % Let 4.3 ppm peak vary
            fixedVals.amide = [NaN, NaN, NaN, NaN];  % Let 5.5 ppm peak vary

        elseif t <= 12  % Creatine
            tube_type = 'creatine';
            pNames = {'water', 'amine'};
            pPars = Custom_3T_Phantom_PeakBounds('creatine');
            fixedVals.water = [NaN, NaN, 0, NaN];
            fixedVals.amine = [NaN, NaN, NaN, NaN];

        elseif t <= 18  % Taurine
            tube_type = 'taurine';
            pNames = {'water', 'amine'};
            pPars = Custom_3T_Phantom_PeakBounds('taurine');
            fixedVals.water = [NaN, NaN, 0, NaN];
            fixedVals.amine = [NaN, NaN, NaN, NaN];

        elseif t <= 21  % PLL
            tube_type = 'PLL';
            pNames = {'water', 'amide', 'amine'};  % Two peaks
            pPars = Custom_3T_Phantom_PeakBounds('PLL');
            fixedVals.water = [NaN, NaN, 0, NaN];
            fixedVals.amide = [NaN, NaN, NaN, NaN];  % 3.5 ppm
            fixedVals.amine = [NaN, NaN, NaN, NaN];  % 2.7 ppm

        else  % PBS
            tube_type = 'PBS';
            pNames = {'water', 'amine'};
            pPars = Custom_3T_Phantom_PeakBounds('PBS');
            fixedVals.water = [NaN, NaN, 0, NaN];
            fixedVals.amine = [NaN, NaN, NaN, NaN];
        end

        % Perform the multi-peak fit (PlotDispFlag = false for speed)
        try
            [EstimatedParams, CI, Residual, Sum_All_P, Indiv_P] = ...
                zspecMultiPeakFit(ppmOffsets, OneMinZ, pNames, pPars, fixedVals, NaN, false);

            % Store main CEST peak parameters (amine for most, amide for PLL)
            if strcmp(tube_type, 'PLL')
                % For PLL, use amide peak (3.5 ppm) as primary
                results.fitted_amplitude(t) = EstimatedParams.amide(1);
                results.fitted_FWHM(t) = EstimatedParams.amide(2);
                results.fitted_offset_ppm(t) = EstimatedParams.amide(3);
                results.fitted_amplitude_CI_lower(t) = CI.amide(1,1);
                results.fitted_amplitude_CI_upper(t) = CI.amide(1,2);
                results.fitted_offset_CI_lower(t) = CI.amide(3,1);
                results.fitted_offset_CI_upper(t) = CI.amide(3,2);

                % Store amine peak (2.7 ppm) as secondary
                results.fitted_amplitude_peak2(t) = EstimatedParams.amine(1);
                results.fitted_offset_ppm_peak2(t) = EstimatedParams.amine(3);

            elseif strcmp(tube_type, 'iopamidol')
                % For Iopamidol, amine is 4.3 ppm (primary)
                results.fitted_amplitude(t) = EstimatedParams.amine(1);
                results.fitted_FWHM(t) = EstimatedParams.amine(2);
                results.fitted_offset_ppm(t) = EstimatedParams.amine(3);
                results.fitted_amplitude_CI_lower(t) = CI.amine(1,1);
                results.fitted_amplitude_CI_upper(t) = CI.amine(1,2);
                results.fitted_offset_CI_lower(t) = CI.amine(3,1);
                results.fitted_offset_CI_upper(t) = CI.amine(3,2);

                % Store amide peak (5.5 ppm) as secondary
                results.fitted_amplitude_peak2(t) = EstimatedParams.amide(1);
                results.fitted_offset_ppm_peak2(t) = EstimatedParams.amide(3);

            else
                % For all others (creatine, taurine, PBS)
                results.fitted_amplitude(t) = EstimatedParams.amine(1);
                results.fitted_FWHM(t) = EstimatedParams.amine(2);
                results.fitted_offset_ppm(t) = EstimatedParams.amine(3);
                results.fitted_amplitude_CI_lower(t) = CI.amine(1,1);
                results.fitted_amplitude_CI_upper(t) = CI.amine(1,2);
                results.fitted_offset_CI_lower(t) = CI.amine(3,1);
                results.fitted_offset_CI_upper(t) = CI.amine(3,2);
            end

            % Calculate fit quality (residual norm)
            results.fit_quality_resnorm(t) = sum(Residual.^2);

            fprintf('    ✓ Fitted: Amplitude=%.4f, Offset=%.3f ppm, FWHM=%.3f ppm\n', ...
                results.fitted_amplitude(t), results.fitted_offset_ppm(t), ...
                results.fitted_FWHM(t));

        catch ME
            warning('    ⚠ Fitting failed for Tube %d: %s', t, ME.message);
            results.fitted_amplitude(t) = NaN;
            results.fitted_FWHM(t) = NaN;
            results.fitted_offset_ppm(t) = NaN;
            results.fit_quality_resnorm(t) = NaN;
        end
    end

    fprintf('✓ Advanced fitting complete for all tubes!\n\n');

    %% Plot fitted parameters
    figure('Position', [100, 100, 1400, 900]);

    subplot(2,3,1);
    bar(1:numTubes, results.fitted_amplitude);
    hold on;
    errorbar(1:numTubes, results.fitted_amplitude, ...
        results.fitted_amplitude - results.fitted_amplitude_CI_lower, ...
        results.fitted_amplitude_CI_upper - results.fitted_amplitude, ...
        'k.', 'LineWidth', 1.5);
    xlabel('Tube Number');
    ylabel('Fitted Amplitude');
    title('Fitted CEST Peak Amplitude (with 95% CI)');
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
    title('Fitted Peak Offset (with 95% CI)');
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
    title('Fit Quality (lower is better)');
    grid on;
    set(gca, 'YScale', 'log');

    subplot(2,3,5);
    % Compare fitted amplitude vs simple MTR asymmetry
    plot(1:numTubes, results.fitted_amplitude * 100, 'bo-', 'LineWidth', 2, ...
        'MarkerFaceColor', 'b', 'DisplayName', 'Fitted Amplitude × 100');
    hold on;
    % Plot your original CEST measurements for comparison
    if t <= 6
        plot(1:6, results.CEST_at_4_3ppm(1:6), 'ro-', 'LineWidth', 2, ...
            'MarkerFaceColor', 'r', 'DisplayName', 'MTRasym @ 4.3 ppm');
    end
    xlabel('Tube Number');
    ylabel('CEST Effect (%)');
    title('Fitted vs MTRasym Comparison');
    legend('Location', 'best');
    grid on;

    subplot(2,3,6);
    % Show secondary peaks for Iopamidol and PLL
    iopamidol_indices = 1:6;
    pll_indices = 19:21;
    bar(iopamidol_indices, results.fitted_amplitude_peak2(iopamidol_indices) * 100, ...
        'FaceColor', [0.8 0.4 0.4]);
    hold on;
    bar(pll_indices, results.fitted_amplitude_peak2(pll_indices) * 100, ...
        'FaceColor', [0.4 0.8 0.4]);
    xlabel('Tube Number');
    ylabel('Secondary Peak Amplitude × 100');
    title('Secondary Peaks (Iopamidol 5.5ppm, PLL 2.7ppm)');
    grid on;

    sgtitle('Advanced Multi-Peak Fitting Results', 'FontSize', 16, 'FontWeight', 'bold');
    saveas(gcf, 'Advanced_Fitting_Results.png');
    saveas(gcf, 'Advanced_Fitting_Results.fig');

    fprintf('✓ Saved: Advanced_Fitting_Results.png/fig\n');
end

%% Continue with your existing Step 8, 9, etc...
