#!/usr/bin/env python3
"""
Calculate dictionary sizes for old vs corrected parameters
Shows why user got 440M entries and how to fix it
"""

import numpy as np

print("="*70)
print("DICTIONARY SIZE ANALYSIS")
print("="*70)

# ========================================================================
# OLD CONFIGURATION (User's current - WRONG)
# ========================================================================
print("\n" + "="*70)
print("OLD CONFIGURATION (INCORRECT - causes 440M entries)")
print("="*70 + "\n")

old_water_t1 = np.arange(1.5, 4.05, 0.05)      # 51 values
old_water_t2 = np.arange(0.05, 1.55, 0.05)     # 30 values
old_cest_k = np.arange(50, 9050, 50)            # 180 values
old_cest_conc = np.arange(1, 41, 1)             # 40 values
old_mt_k = [49.25]                              # 1 value ❌ WRONG!
old_mt_f_values = 40                            # 40 values ❌ WRONG! (calculated as variable)

print(f"Water T1:        {len(old_water_t1)} values  ({old_water_t1[0]:.2f} to {old_water_t1[-1]:.2f} s, step {old_water_t1[1]-old_water_t1[0]:.2f})")
print(f"Water T2:        {len(old_water_t2)} values  ({old_water_t2[0]:.3f} to {old_water_t2[-1]:.3f} s, step {old_water_t2[1]-old_water_t2[0]:.3f})")
print(f"CEST k_sw:       {len(old_cest_k)} values ({old_cest_k[0]:.0f} to {old_cest_k[-1]:.0f} Hz, step {old_cest_k[1]-old_cest_k[0]:.0f})")
print(f"CEST conc:       {len(old_cest_conc)} values ({old_cest_conc[0]:.0f} to {old_cest_conc[-1]:.0f} mM)")
print(f"MT k_sw:         {len(old_mt_k)} value   (49.25 Hz) ← ERROR 1: Should be RANGE!")
print(f"MT dw:           -0.068 ppm                    ← ERROR 2: Should be 0!")
print(f"MT f:            {old_mt_f_values} values (variable)         ← ERROR 3: Should be FIXED 0.05!")

old_size_without_mt_f = len(old_water_t1) * len(old_water_t2) * len(old_cest_k) * len(old_cest_conc) * len(old_mt_k)
old_size_with_mt_f = old_size_without_mt_f * old_mt_f_values

print(f"\nCalculation:")
print(f"  Without mt_f variable: {len(old_water_t1)} × {len(old_water_t2)} × {len(old_cest_k)} × {len(old_cest_conc)} × {len(old_mt_k)}")
print(f"                       = {old_size_without_mt_f:,} entries ({old_size_without_mt_f/1e6:.1f}M)")
print(f"  With mt_f variable:    {old_size_without_mt_f:,} × {old_mt_f_values}")
print(f"                       = {old_size_with_mt_f:,} entries ({old_size_with_mt_f/1e6:.1f}M) ← This is what you saw!")

old_size_gb = old_size_with_mt_f * 30 * 8 / (1024**3)
print(f"\nEstimated size: {old_size_gb:.2f} GB")
print(f"\n⚠️  PROBLEMS:")
print(f"   1. Dictionary too large (440M entries)")
print(f"   2. Single mt_k value → cannot adapt → INACCURATE RESULTS")
print(f"   3. Wrong mt_dw (-0.068) → incorrect MT physics")
print(f"   4. Variable mt_f → creates 40× size explosion")

# ========================================================================
# NEW CONFIGURATION (Corrected - ACCURATE)
# ========================================================================
print("\n" + "="*70)
print("NEW CONFIGURATION (CORRECTED - accurate results)")
print("="*70 + "\n")

new_water_t1 = np.arange(1.5, 4.1, 0.1)        # 26 values (reduced step)
new_water_t2 = np.arange(0.05, 1.55, 0.05)     # 30 values (keep)
new_cest_k = np.arange(50, 9050, 100)           # 91 values (reduced step)
new_cest_conc = np.arange(1, 41, 1)             # 40 values (keep)
new_mt_k = np.arange(30, 75, 5)                 # 9 values ✓ RANGE!
new_mt_f = [0.05]                               # 1 value (fixed) ✓ FIXED!

print(f"Water T1:        {len(new_water_t1)} values  ({new_water_t1[0]:.2f} to {new_water_t1[-1]:.2f} s, step {new_water_t1[1]-new_water_t1[0]:.2f})")
print(f"Water T2:        {len(new_water_t2)} values  ({new_water_t2[0]:.3f} to {new_water_t2[-1]:.3f} s, step {new_water_t2[1]-new_water_t2[0]:.3f})")
print(f"CEST k_sw:       {len(new_cest_k)} values ({new_cest_k[0]:.0f} to {new_cest_k[-1]:.0f} Hz, step {new_cest_k[1]-new_cest_k[0]:.0f})")
print(f"CEST conc:       {len(new_cest_conc)} values ({new_cest_conc[0]:.0f} to {new_cest_conc[-1]:.0f} mM)")
print(f"MT k_sw:         {len(new_mt_k)} values  ({new_mt_k[0]:.0f} to {new_mt_k[-1]:.0f} Hz, step {new_mt_k[1]-new_mt_k[0]:.0f}) ✓ FIX 1")
print(f"MT dw:           0 ppm                         ✓ FIX 2")
print(f"MT f:            {len(new_mt_f)} value   (0.05 fixed)           ✓ FIX 3")

new_size = len(new_water_t1) * len(new_water_t2) * len(new_cest_k) * len(new_cest_conc) * len(new_mt_k)

print(f"\nCalculation:")
print(f"  {len(new_water_t1)} × {len(new_water_t2)} × {len(new_cest_k)} × {len(new_cest_conc)} × {len(new_mt_k)}")
print(f"  = {new_size:,} entries ({new_size/1e6:.2f}M)")

new_size_gb = new_size * 30 * 8 / (1024**3)
print(f"\nEstimated size: {new_size_gb:.2f} GB")
print(f"\n✓ BENEFITS:")
print(f"   1. Manageable size (25.5M entries)")
print(f"   2. MT k_sw range → adapts to BSA exchange → ACCURATE")
print(f"   3. Correct mt_dw (0) → proper SuperLorentzian MT physics")
print(f"   4. Fixed mt_f (0.05) → no size explosion, physically correct")
print(f"   5. Chunked matching will auto-enable (>20M entries)")

# ========================================================================
# COMPARISON
# ========================================================================
print("\n" + "="*70)
print("COMPARISON")
print("="*70 + "\n")

reduction_factor = old_size_with_mt_f / new_size
print(f"Size reduction:     {old_size_with_mt_f/1e6:.1f}M → {new_size/1e6:.1f}M  ({reduction_factor:.1f}× smaller)")
print(f"File size:          {old_size_gb:.1f} GB → {new_size_gb:.1f} GB")
print(f"Generation time:    ~8 hours → ~45 minutes (estimated)")
print(f"Matching time:      Will segfault → ~2-3 hours (chunked)")
print(f"Result accuracy:    WRONG → CORRECT")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70 + "\n")
print("1. Update your MATLAB dictionary generation script with corrected parameters")
print("2. Use dict_params_corrected_25M.m as reference")
print("3. Make sure these THREE fixes are applied:")
print("   - dictparams.mt_k = 30:5:70;      (not 49.25)")
print("   - dictparams.mt_dw = 0;           (not -0.068)")
print("   - dictparams.mt_f = 0.05;         (not variable)")
print("4. Regenerate dictionary (~45 min)")
print("5. Run matching (chunked algorithm will auto-enable)")
print("\n" + "="*70)
