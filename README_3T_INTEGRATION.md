# Using Bruker Repository Z-Spectrum Fitting Functions with Your 3T Phantom Data

## Quick Summary

You can use the advanced **MATLAB-only** z-spectrum fitting functions from this repository with your 3T clinical scanner data - **NO PYTHON REQUIRED!**

## What You Get

✅ **Advanced multi-peak Lorentzian or Pseudo-Voigt fitting**
✅ **Accurate parameter extraction**: amplitude, FWHM, chemical shift
✅ **95% confidence intervals** for all fitted parameters
✅ **Better than simple MTR asymmetry** for quantification
✅ **Works directly with your 3T DICOM data**

## Files You Need (Copy These)

From this repository, copy these 4 files to your working directory:

```
BrukerCESTmethod_processingPipeline/
  Bruker_CEST-MRF_processing/
    MATLAB_MAIN-CODE-TO-RUN/
      subfunctions_otherFiles/
        Data_parameter_loading_processing/
          zspec_fitting_subfunctions/
            ├── zspecMultiPeakFit.m          ⭐ Main fitting function
            ├── zspecSetLPeakBounds.m         ⭐ Lorentzian peak bounds
            ├── zspecSetPVPeakBounds.m        ⭐ Pseudo-Voigt peak bounds
            └── zspecMTRphase.m               ⭐ Phase correction (optional)
```

**Plus** these custom files created for your 24-tube phantom:
- `Custom_3T_Phantom_PeakBounds.m` - Pre-configured for your tubes
- `ADD_TO_YOUR_CODE.m` - Ready-to-use code snippet

## Quick Start (3 Steps)

### Step 1: Copy Files

```bash
cd /your/working/directory
cp /path/to/BrukerCESTmethod_processingPipeline/Bruker_CEST-MRF_processing/MATLAB_MAIN-CODE-TO-RUN/subfunctions_otherFiles/Data_parameter_loading_processing/zspec_fitting_subfunctions/*.m .
cp /path/to/BrukerCESTmethod_processingPipeline/Custom_3T_Phantom_PeakBounds.m .
```

### Step 2: Add to MATLAB Path

```matlab
addpath(pwd);  % or addpath('/your/working/directory')
```

### Step 3: Add Fitting Code to Your Script

Open `ADD_TO_YOUR_CODE.m` and copy the entire code block into your existing script **after Step 7**.

That's it! Run your script and you'll get advanced multi-peak fitting.

## What Each Function Does

### `zspecMultiPeakFit.m`

**Main fitting function** that performs multi-peak Lorentzian or Pseudo-Voigt fitting.

**Usage:**
```matlab
[EstimatedParams, CI, Residual, Sum_All_P, Indiv_P] = ...
    zspecMultiPeakFit(ppmOffsets, OneMinZ, pNames, pPars, fixedVals, NaN, PlotFlag);
```

**Inputs:**
- `ppmOffsets` - Your frequency offsets in ppm (column vector)
- `OneMinZ` - 1 minus normalized Z-spectrum (column vector)
- `pNames` - Cell array of peak names, e.g., `{'water', 'amine'}`
- `pPars` - Structure with peak fitting parameters (from `zspecSetLPeakBounds()`)
- `fixedVals` - Structure specifying which parameters to fix (use `NaN` to let vary)
- `ppm_wt` - Weight specific ppm values (use `NaN` for no weighting)
- `PlotFlag` - true/false to show fit plot

**Outputs:**
- `EstimatedParams` - Fitted parameter values for each peak
- `CI` - 95% confidence intervals for each parameter
- `Residual` - Fitting residuals
- `Sum_All_P` - Sum of all fitted peaks
- `Indiv_P` - Individual fitted peaks

### `Custom_3T_Phantom_PeakBounds.m`

**Pre-configured peak bounds** for your 24-tube phantom.

**Usage:**
```matlab
pPars = Custom_3T_Phantom_PeakBounds('iopamidol');  % For Iopamidol tubes
pPars = Custom_3T_Phantom_PeakBounds('creatine');   % For Creatine tubes
pPars = Custom_3T_Phantom_PeakBounds('taurine');    % For Taurine tubes
pPars = Custom_3T_Phantom_PeakBounds('PLL');        % For PLL tubes
pPars = Custom_3T_Phantom_PeakBounds('PBS');        % For PBS tubes
```

Returns optimized fitting parameters for each tube type.

## Example: Fitting a Single Tube

```matlab
% Your data (from existing code)
tube_zspec_norm = all_zspectra(1, :)';  % Tube 1 (Iopamidol 20mM pH 6.2)
OneMinZ = 1 - tube_zspec_norm;

% Get pre-configured bounds for Iopamidol
pNames = {'water', 'amine', 'amide'};  % Iopamidol has 2 CEST peaks
pPars = Custom_3T_Phantom_PeakBounds('iopamidol');

% Fix water offset at 0 ppm, let CEST peaks vary
fixedVals.water = [NaN, NaN, 0, NaN];
fixedVals.amine = [NaN, NaN, NaN, NaN];  % 4.3 ppm peak
fixedVals.amide = [NaN, NaN, NaN, NaN];  % 5.5 ppm peak

% Fit!
[EP, CI, ~, Sum, Indiv] = zspecMultiPeakFit(ppmOffsets, OneMinZ, ...
    pNames, pPars, fixedVals, NaN, true);  % true = show plot

% Extract results
fprintf('Iopamidol 4.3 ppm peak:\n');
fprintf('  Amplitude: %.4f (95%% CI: [%.4f, %.4f])\n', ...
    EP.amine(1), CI.amine(1,1), CI.amine(1,2));
fprintf('  Offset: %.3f ppm (95%% CI: [%.3f, %.3f])\n', ...
    EP.amine(3), CI.amine(3,1), CI.amine(3,2));
fprintf('  FWHM: %.3f ppm\n', EP.amine(2));
```

## Parameter Meanings

For **Lorentzian peaks** (4 parameters):
```
EstimatedParams.poolname(1) = Amplitude (0-1, CEST effect strength)
EstimatedParams.poolname(2) = FWHM (ppm, peak width)
EstimatedParams.poolname(3) = Offset (ppm, chemical shift)
EstimatedParams.poolname(4) = Phase (radians, for correction)
```

For **Pseudo-Voigt peaks** (6 parameters):
```
EstimatedParams.poolname(1) = Amplitude
EstimatedParams.poolname(2) = Alpha (0-1, Gaussian proportion)
EstimatedParams.poolname(3) = FWHM_Lorentzian (ppm)
EstimatedParams.poolname(4) = FWHM ratio (Gaussian/Lorentzian)
EstimatedParams.poolname(5) = Offset (ppm)
EstimatedParams.poolname(6) = Phase (radians)
```

## Your 24-Tube Phantom Configuration

| Tubes | Chemical | Concentration | Chemical Shift | Peak Name Used |
|-------|----------|---------------|----------------|----------------|
| 1-3   | Iopamidol | 20mM | 4.3 ppm, 5.5 ppm | amine, amide |
| 4-6   | Iopamidol | 50mM | 4.3 ppm, 5.5 ppm | amine, amide |
| 7-9   | Creatine | 20mM | 1.9 ppm | amine |
| 10-12 | Creatine | 50mM | 1.9 ppm | amine |
| 13-15 | Taurine | 20mM | 2.8 ppm | amine |
| 16-18 | Taurine | 50mM | 2.8 ppm | amine |
| 19-21 | PLL | 0.1% | 3.5 ppm, 2.7 ppm | amide, amine |
| 22-24 | PBS | - | - | - |

## Advantages Over Simple MTR Asymmetry

| Feature | MTRasym | Multi-Peak Fitting |
|---------|---------|-------------------|
| **Accuracy** | Approximate | High precision |
| **Overlapping peaks** | Cannot separate | Separates peaks |
| **Statistical confidence** | None | 95% CI provided |
| **Line shape** | Assumes symmetry | Fits actual shape |
| **B0 sensitivity** | High | Corrects for shifts |
| **Concentration quantification** | Linear assumption | Non-linear model |

## Troubleshooting

### "zspecMultiPeakFit not found"
- Make sure you copied all 4 `.m` files to your working directory
- Run `addpath(pwd)` or add path to the folder containing the files

### Fitting fails for some tubes
- Check that `OneMinZ` is a column vector
- Verify ppm offsets match your data
- Try adjusting bounds in `Custom_3T_Phantom_PeakBounds.m`

### Confidence intervals are very wide
- Indicates poor fit quality
- May need more data points around the peak
- Try constraining FWHM or other parameters

### Fits look wrong
- Set `PlotFlag = true` to visualize the fit
- Check that peak names match your configuration
- Verify `fixedVals` structure is correct

## Advanced Customization

To modify fitting bounds for a specific tube type, edit `Custom_3T_Phantom_PeakBounds.m`:

```matlab
case 'creatine'
    % Adjust these values for your specific phantom
    pPars.amine.st = [0.12, 0.6, 1.9, 0];  % [Amp, FWHM, Offset, Phase]
    pPars.amine.lb = [0.01, 0.2, 1.7, 0];  % Lower bounds
    pPars.amine.ub = [0.5, 1.5, 2.1, 0];   % Upper bounds
```

## Additional Resources

- **Full integration guide**: `INTEGRATION_GUIDE_3T_Phantom.m` - Run this for detailed examples
- **Ready-to-use code**: `ADD_TO_YOUR_CODE.m` - Copy directly into your script
- **Original functions**: `Bruker_CEST-MRF_processing/MATLAB_MAIN-CODE-TO-RUN/subfunctions_otherFiles/`

## Summary

✅ Copy 4 files from repository
✅ Add `Custom_3T_Phantom_PeakBounds.m`
✅ Insert code from `ADD_TO_YOUR_CODE.m` after Step 7
✅ Run your script with advanced fitting!

**No Python. No complex setup. Pure MATLAB.**

## Questions?

Check the integration guide:
```matlab
run('INTEGRATION_GUIDE_3T_Phantom.m')
```

This will run examples and show you how everything works!

---

Happy fitting! 🎯
