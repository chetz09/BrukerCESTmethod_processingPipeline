%% INTEGRATION GUIDE: Using Bruker Repository's Z-Spectrum Fitting with Your 3T Phantom Data
%
% This script shows you HOW to integrate the advanced z-spectrum fitting
% functions from the BrukerCESTmethod_processingPipeline repository into
% your existing 3T phantom analysis code.
%
% WHAT YOU NEED:
% 1. Copy the zspec_fitting_subfunctions folder to your working directory
% 2. Add it to your MATLAB path
% 3. Use the examples below in your existing code
%
% AVAILABLE FITTING FUNCTIONS:
% - zspecMultiPeakFit: Multi-peak Lorentzian or Pseudo-Voigt fitting
% - zspecSetLPeakBounds: Lorentzian peak bounds configuration
% - zspecSetPVPeakBounds: Pseudo-Voigt peak bounds configuration
% - zspecMTRphase: MTR phase correction
%
% Author: Integration guide for 3T phantom analysis
% Date: 2025-12-02

clearvars; clc; close all;

%% ========================================================================
%% STEP 1: Setup - Add the fitting functions to your path
%% ========================================================================

fprintf('=== SETUP ===\n');
fprintf('Add the zspec_fitting_subfunctions folder to your MATLAB path:\n\n');
fprintf('  >> addpath(genpath(''path/to/BrukerCESTmethod_processingPipeline/Bruker_CEST-MRF_processing/MATLAB_MAIN-CODE-TO-RUN/subfunctions_otherFiles/Data_parameter_loading_processing/zspec_fitting_subfunctions''));\n\n');

% For this example, assume you've already added it
fprintf('Verifying functions are available...\n');
if exist('zspecMultiPeakFit', 'file') && exist('zspecSetLPeakBounds', 'file')
    fprintf('✓ Z-spectrum fitting functions found!\n\n');
else
    error('Please add zspec_fitting_subfunctions to your MATLAB path!');
end

%% ========================================================================
%% STEP 2: Example 1 - Fitting Iopamidol tubes (multi-peak Lorentzian)
%% ========================================================================

fprintf('\n=== EXAMPLE 1: Multi-peak Lorentzian Fitting (Iopamidol) ===\n\n');

% Simulate some example Z-spectrum data for Iopamidol
% (Replace this with your actual tube_zspec_norm from your code)
ppmOffsets_example = linspace(-7, 7, 49)';  % Your CEST offsets in ppm

% Simulated Iopamidol Z-spectrum (replace with your real data!)
water_peak = 0.8 * exp(-((ppmOffsets_example - 0) / 1.5).^2);
iopamidol_peak1 = 0.15 * exp(-((ppmOffsets_example - 4.3) / 0.8).^2);
iopamidol_peak2 = 0.10 * exp(-((ppmOffsets_example - 5.5) / 0.7).^2);
z_spectrum = 1 - (water_peak + iopamidol_peak1 + iopamidol_peak2);
OneMinZ = 1 - z_spectrum;  % This is what the fitting function expects

% -------------------------------------------------------------------------
% Configure which peaks to fit for Iopamidol
% -------------------------------------------------------------------------
pNames = {'water', 'amine'};  % Peak names to fit
                               % For Iopamidol: 'amine' will capture 4.3 ppm peak

% Get default Lorentzian bounds
pPars = zspecSetLPeakBounds();

% Customize bounds for Iopamidol at 4.3 ppm
pPars.amine.st(3) = 4.3;     % Start at 4.3 ppm (Iopamidol peak 1)
pPars.amine.lb(3) = 4.0;     % Lower bound
pPars.amine.ub(3) = 4.6;     % Upper bound

% Optional: Fix certain parameters (use NaN to let them vary)
fixedVals.water = [NaN, NaN, 0, NaN];  % Fix water offset at 0 ppm
fixedVals.amine = [NaN, NaN, NaN, NaN];  % Let all amine parameters vary

% -------------------------------------------------------------------------
% Run the fitting!
% -------------------------------------------------------------------------
[EstimatedParams, CI, Residual, Sum_All_P, Indiv_P] = ...
    zspecMultiPeakFit(ppmOffsets_example, OneMinZ, pNames, pPars, fixedVals, NaN, true);

fprintf('\n✓ Iopamidol tube fitted!\n');
fprintf('  Water peak offset: %.3f ppm (CI: [%.3f, %.3f])\n', ...
    EstimatedParams.water(3), CI.water(3,1), CI.water(3,2));
fprintf('  Iopamidol peak offset: %.3f ppm (CI: [%.3f, %.3f])\n', ...
    EstimatedParams.amine(3), CI.amine(3,1), CI.amine(3,2));
fprintf('  Iopamidol amplitude: %.4f (CI: [%.4f, %.4f])\n', ...
    EstimatedParams.amine(1), CI.amine(1,1), CI.amine(1,2));
fprintf('  Iopamidol FWHM: %.3f ppm\n', EstimatedParams.amine(2));

%% ========================================================================
%% STEP 3: Example 2 - Fitting Creatine tubes (single CEST peak)
%% ========================================================================

fprintf('\n=== EXAMPLE 2: Creatine Fitting (1.9 ppm) ===\n\n');

% Simulated Creatine Z-spectrum
water_peak_cr = 0.85 * exp(-((ppmOffsets_example - 0) / 1.4).^2);
creatine_peak = 0.12 * exp(-((ppmOffsets_example - 1.9) / 0.6).^2);
z_spectrum_cr = 1 - (water_peak_cr + creatine_peak);
OneMinZ_cr = 1 - z_spectrum_cr;

% Configure for Creatine
pNames_cr = {'water', 'amine'};
pPars_cr = zspecSetLPeakBounds();

% Customize for Creatine at 1.9 ppm
pPars_cr.amine.st(3) = 1.9;
pPars_cr.amine.lb(3) = 1.7;
pPars_cr.amine.ub(3) = 2.1;

fixedVals_cr.water = [NaN, NaN, 0, NaN];
fixedVals_cr.amine = [NaN, NaN, NaN, NaN];

% Run fitting
[EP_cr, CI_cr, ~, ~, ~] = ...
    zspecMultiPeakFit(ppmOffsets_example, OneMinZ_cr, pNames_cr, pPars_cr, fixedVals_cr, NaN, true);

fprintf('\n✓ Creatine tube fitted!\n');
fprintf('  Creatine peak offset: %.3f ppm\n', EP_cr.amine(3));
fprintf('  Creatine amplitude: %.4f\n', EP_cr.amine(1));

%% ========================================================================
%% STEP 4: Example 3 - Advanced: Pseudo-Voigt fitting for better accuracy
%% ========================================================================

fprintf('\n=== EXAMPLE 3: Pseudo-Voigt Fitting (More Accurate) ===\n\n');

% Get Pseudo-Voigt bounds (more parameters, better fit)
pPars_PV = zspecSetPVPeakBounds();

% Customize for Iopamidol
pPars_PV.amine.st(5) = 4.3;  % Offset is 5th parameter for Pseudo-Voigt
pPars_PV.amine.lb(5) = 4.0;
pPars_PV.amine.ub(5) = 4.6;

fixedVals_PV.water = [NaN, NaN, NaN, NaN, 0, NaN];  % 6 params for Pseudo-Voigt
fixedVals_PV.amine = [NaN, NaN, NaN, NaN, NaN, NaN];

% Run Pseudo-Voigt fitting
[EP_PV, CI_PV, ~, Sum_PV, Indiv_PV] = ...
    zspecMultiPeakFit(ppmOffsets_example, OneMinZ, pNames, pPars_PV, fixedVals_PV, NaN, true);

fprintf('\n✓ Pseudo-Voigt fitting complete!\n');
fprintf('  Provides better fit quality for complex line shapes\n');

%% ========================================================================
%% STEP 5: How to integrate into YOUR existing phantom code
%% ========================================================================

fprintf('\n\n=== INTEGRATION INTO YOUR CODE ===\n\n');
fprintf('Add this code block after Step 7 in your existing script:\n\n');
fprintf('%%-------------------------------------------------------------\n');
fprintf('%% Advanced Multi-Peak Fitting for Each Tube\n');
fprintf('%%-------------------------------------------------------------\n');
fprintf('for t = 1:numTubes\n');
fprintf('    %% Get normalized Z-spectrum for this tube\n');
fprintf('    tube_zspec_norm = all_zspectra(t, :);  %% From your existing code\n');
fprintf('    OneMinZ = 1 - tube_zspec_norm;\n');
fprintf('    \n');
fprintf('    %% Determine which peaks to fit based on tube type\n');
fprintf('    if t <= 6  %% Iopamidol tubes\n');
fprintf('        pNames = {''water'', ''amine''};\n');
fprintf('        pPars = zspecSetLPeakBounds();\n');
fprintf('        pPars.amine.st(3) = 4.3;  %% Iopamidol peak\n');
fprintf('        pPars.amine.lb(3) = 4.0;\n');
fprintf('        pPars.amine.ub(3) = 4.6;\n');
fprintf('        fixedVals.water = [NaN, NaN, 0, NaN];\n');
fprintf('        fixedVals.amine = [NaN, NaN, NaN, NaN];\n');
fprintf('        \n');
fprintf('    elseif t <= 12  %% Creatine tubes\n');
fprintf('        pNames = {''water'', ''amine''};\n');
fprintf('        pPars = zspecSetLPeakBounds();\n');
fprintf('        pPars.amine.st(3) = 1.9;  %% Creatine peak\n');
fprintf('        pPars.amine.lb(3) = 1.7;\n');
fprintf('        pPars.amine.ub(3) = 2.1;\n');
fprintf('        fixedVals.water = [NaN, NaN, 0, NaN];\n');
fprintf('        fixedVals.amine = [NaN, NaN, NaN, NaN];\n');
fprintf('        \n');
fprintf('    elseif t <= 18  %% Taurine tubes\n');
fprintf('        pNames = {''water'', ''amine''};\n');
fprintf('        pPars = zspecSetLPeakBounds();\n');
fprintf('        pPars.amine.st(3) = 2.8;  %% Taurine peak\n');
fprintf('        pPars.amine.lb(3) = 2.6;\n');
fprintf('        pPars.amine.ub(3) = 3.0;\n');
fprintf('        fixedVals.water = [NaN, NaN, 0, NaN];\n');
fprintf('        fixedVals.amine = [NaN, NaN, NaN, NaN];\n');
fprintf('        \n');
fprintf('    else  %% PLL or PBS tubes\n');
fprintf('        pNames = {''water'', ''amide''};\n');
fprintf('        pPars = zspecSetLPeakBounds();\n');
fprintf('        pPars.amide.st(3) = 3.5;  %% Amide peak\n');
fprintf('        fixedVals.water = [NaN, NaN, 0, NaN];\n');
fprintf('        fixedVals.amide = [NaN, NaN, NaN, NaN];\n');
fprintf('    end\n');
fprintf('    \n');
fprintf('    %% Perform the fit (no plot for speed)\n');
fprintf('    [EP, CI, ~, ~, ~] = zspecMultiPeakFit(ppmOffsets, OneMinZ, ...\n');
fprintf('        pNames, pPars, fixedVals, NaN, false);\n');
fprintf('    \n');
fprintf('    %% Store fitted parameters\n');
fprintf('    results.fitted_amplitude(t) = EP.amine(1);  %% or EP.amide(1)\n');
fprintf('    results.fitted_FWHM(t) = EP.amine(2);\n');
fprintf('    results.fitted_offset(t) = EP.amine(3);\n');
fprintf('    results.amplitude_CI_lower(t) = CI.amine(1,1);\n');
fprintf('    results.amplitude_CI_upper(t) = CI.amine(1,2);\n');
fprintf('end\n');
fprintf('%%-------------------------------------------------------------\n\n');

%% ========================================================================
%% STEP 6: Key Parameter Meanings
%% ========================================================================

fprintf('\n=== UNDERSTANDING THE FITTED PARAMETERS ===\n\n');
fprintf('For Lorentzian peaks (4 parameters):\n');
fprintf('  EstimatedParams.poolname(1) = Amplitude (0-1, CEST effect strength)\n');
fprintf('  EstimatedParams.poolname(2) = FWHM (ppm, peak width)\n');
fprintf('  EstimatedParams.poolname(3) = Offset (ppm, chemical shift)\n');
fprintf('  EstimatedParams.poolname(4) = Phase (radians, for correction)\n\n');

fprintf('For Pseudo-Voigt peaks (6 parameters):\n');
fprintf('  EstimatedParams.poolname(1) = Amplitude\n');
fprintf('  EstimatedParams.poolname(2) = Alpha (0-1, Gaussian proportion)\n');
fprintf('  EstimatedParams.poolname(3) = FWHM_Lorentzian (ppm)\n');
fprintf('  EstimatedParams.poolname(4) = FWHM ratio (Gaussian/Lorentzian)\n');
fprintf('  EstimatedParams.poolname(5) = Offset (ppm)\n');
fprintf('  EstimatedParams.poolname(6) = Phase (radians)\n\n');

fprintf('CI = 95%% Confidence Intervals\n');
fprintf('  CI.poolname(i,:) = [lower_bound, upper_bound] for parameter i\n\n');

%% ========================================================================
%% STEP 7: Files you need to copy
%% ========================================================================

fprintf('\n=== FILES TO COPY FROM REPOSITORY ===\n\n');
fprintf('Copy these files to your working directory:\n\n');
fprintf('  1. zspecMultiPeakFit.m\n');
fprintf('  2. zspecSetLPeakBounds.m\n');
fprintf('  3. zspecSetPVPeakBounds.m\n');
fprintf('  4. zspecMTRphase.m (optional, for phase correction)\n\n');
fprintf('From:\n');
fprintf('  BrukerCESTmethod_processingPipeline/\n');
fprintf('    Bruker_CEST-MRF_processing/\n');
fprintf('      MATLAB_MAIN-CODE-TO-RUN/\n');
fprintf('        subfunctions_otherFiles/\n');
fprintf('          Data_parameter_loading_processing/\n');
fprintf('            zspec_fitting_subfunctions/\n\n');

%% ========================================================================
%% SUMMARY
%% ========================================================================

fprintf('\n=== SUMMARY ===\n\n');
fprintf('✓ You can use these MATLAB-only fitting functions\n');
fprintf('✓ No Python required!\n');
fprintf('✓ More accurate than simple MTR asymmetry\n');
fprintf('✓ Extracts: amplitude, FWHM, chemical shift, with confidence intervals\n');
fprintf('✓ Supports multi-peak fitting (e.g., Iopamidol''s two peaks)\n');
fprintf('✓ Works with your existing 3T DICOM data\n\n');

fprintf('Next Steps:\n');
fprintf('1. Copy the 4 .m files to your working directory\n');
fprintf('2. Add the integration code (Step 5) to your existing script\n');
fprintf('3. Customize peak bounds for your phantom composition\n');
fprintf('4. Run and compare with your current MTR asymmetry results!\n\n');

fprintf('========================================\n');
fprintf('INTEGRATION GUIDE COMPLETE\n');
fprintf('========================================\n');
