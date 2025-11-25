# CRITICAL FIXES NEEDED FOR ACCURATE MRF RESULTS

## Problem Summary

Your current dictionary configuration has **THREE critical errors** causing both **inaccurate results** and **440M dictionary size**:

### Error 1: Single MT k_sw value (cannot adapt)
```matlab
❌ dictparams.mt_k = 49.25;  % Single value - cannot adapt to BSA exchange
✓  dictparams.mt_k = 30:5:70;  % Range allows matching to find correct value
```
**Impact**: All voxels forced to use exactly 49.25 Hz, preventing accurate BSA modeling

### Error 2: Wrong MT offset for SuperLorentzian
```matlab
❌ dictparams.mt_dw = -27.38/(42.57*9.4);  % = -0.068 ppm (WRONG!)
✓  dictparams.mt_dw = 0;  % Must be 0 ppm for SuperLorentzian
```
**Impact**: Physically incorrect MT modeling (SuperLorentzian centered at water)

### Error 3: Variable MT fraction (causes 440M explosion)
```matlab
❌ dictparams.mt_f = mt_sol_conc * mt_protons / mt_water_conc;  % 40 values!
✓  dictparams.mt_f = 0.05;  % Fixed at 5% BSA
```
**Impact**:
- Creates 40 different mt_f values (11M × 40 = 440M entries!)
- BSA concentration is KNOWN (5%), should not be variable
- Dictionary too large to process

---

## Corrected Parameters (25.4M entries)

```matlab
% ========================================================================
% WATER POOL (reduce resolution to lower size)
% ========================================================================
dictparams.water_t1 = 1.5:0.1:4;              % 26 values (was 0.05 step)
dictparams.water_t2 = 0.05:0.05:1.5;          % 30 values

% ========================================================================
% CEST POOL (Glutamate - amine protons)
% ========================================================================
dictparams.cest_amine_k = 50:100:9000;        % 91 values (was 50 step)
dictparams.cest_amine_sol_conc = 1:1:40;      % 40 values
dictparams.cest_amine_dw = 3.5;               % ppm

% ========================================================================
% MT POOL (5% BSA) - THREE CRITICAL FIXES
% ========================================================================
dictparams.mt_k = 30:5:70;                    % 9 values ✓ FIX 1
dictparams.mt_dw = 0;                         % 0 ppm ✓ FIX 2
dictparams.mt_f = 0.05;                       % Fixed ✓ FIX 3

dictparams.mt_t1 = 1.0;                       % s
dictparams.mt_t2 = 7.338e-06;                 % 7.338 μs (semi-solid)
dictparams.mt_lineshape = 'SuperLorentzian';  % Required for BSA
```

**Dictionary size**: 26 × 30 × 91 × 40 × 9 = **24,429,600 entries (24.4M)**
- File size: ~5.5 GB (vs 95 GB with variable mt_f!)
- Generation time: ~45 minutes
- Matching time: ~2-3 hours (chunked algorithm auto-enables)

---

## Why These Fixes Matter

### Fix 1: MT k_sw range (not single value)
- 5% BSA has exchange rate typically 30-70 Hz
- Different voxels may have slightly different BSA properties
- Range allows matching to find best value per voxel
- Single value forces all voxels to use 49.25 Hz → systematically wrong

### Fix 2: MT dw = 0 (not -0.068)
- SuperLorentzian lineshape models semi-solid MT pool
- Must be centered at water frequency (0 ppm)
- Non-zero offset physically incorrect for this lineshape
- -0.068 ppm offset causes incorrect MT physics

### Fix 3: MT f = 0.05 fixed (not variable)
- Your phantom has 5% BSA - this is KNOWN and FIXED
- Should not vary with glutamate concentration
- Variable mt_f created 40 values → 40× size explosion
- 11M × 40 = 440M entries (what you saw!)

---

## Impact on Results

**Before fixes** (current):
- All matches constrained to mt_k = 49.25 Hz (cannot adapt)
- Wrong MT physics (mt_dw = -0.068)
- Compensation: matching finds "best" water/CEST parameters to compensate
- Result: **All parameters systematically wrong**

**After fixes**:
- MT k_sw can adapt (30-70 Hz range)
- Correct MT physics (mt_dw = 0)
- Fixed BSA fraction (mt_f = 0.05)
- Result: **Accurate quantification of all parameters**

---

## Next Steps

1. **Update dictionary generation script** with corrected parameters above
2. **Regenerate dictionary** (~45 min on cluster)
3. **Run matching** (chunked algorithm will auto-enable for 24M entries)
4. **Compare results** - should see much better agreement with expected values

---

## File Locations

- Corrected parameters: `dict_params_corrected_25M.m`
- Size calculator: `calculate_dict_sizes_simple.py`
- This summary: `CRITICAL_FIXES_NEEDED.md`

---

## Questions?

The key insight: Your mt_f was calculated from concentrations, creating 40 values:
```matlab
mt_f = mt_sol_conc × mt_protons / mt_water_conc  % Creates 40 values!
```

This added an extra dimension: **11M × 40 = 440M** (what you reported!)

For 5% BSA, mt_f should be **fixed at 0.05**, not calculated from glutamate concentrations.
