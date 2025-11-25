#!/usr/bin/env python3
"""
Calculate dictionary sizes for old vs corrected parameters
Shows why user got 440M entries and how to fix it
"""

print("="*70)
print("DICTIONARY SIZE ANALYSIS")
print("="*70)

# ========================================================================
# OLD CONFIGURATION (User's current - WRONG)
# ========================================================================
print("\n" + "="*70)
print("OLD CONFIGURATION (INCORRECT - causes 440M entries)")
print("="*70 + "\n")

# Calculate array lengths
old_n_t1w = int((4.0 - 1.5) / 0.05) + 1           # 51 values
old_n_t2w = int((1.5 - 0.05) / 0.05) + 1          # 30 values
old_n_cest_k = int((9000 - 50) / 50) + 1          # 180 values
old_n_cest_conc = int((40 - 1) / 1) + 1           # 40 values
old_n_mt_k = 1                                     # 1 value ❌ WRONG!
old_n_mt_f = 40                                    # 40 values ❌ WRONG! (calculated as variable)

print(f"Water T1:        {old_n_t1w} values  (1.50 to 4.00 s, step 0.05)")
print(f"Water T2:        {old_n_t2w} values  (0.05 to 1.50 s, step 0.05)")
print(f"CEST k_sw:       {old_n_cest_k} values (50 to 9000 Hz, step 50)")
print(f"CEST conc:       {old_n_cest_conc} values (1 to 40 mM)")
print(f"MT k_sw:         {old_n_mt_k} value   (49.25 Hz) ← ERROR 1: Should be RANGE!")
print(f"MT dw:           -0.068 ppm          ← ERROR 2: Should be 0!")
print(f"MT f:            {old_n_mt_f} values (variable) ← ERROR 3: Should be FIXED 0.05!")

old_size_without_mt_f = old_n_t1w * old_n_t2w * old_n_cest_k * old_n_cest_conc * old_n_mt_k
old_size_with_mt_f = old_size_without_mt_f * old_n_mt_f

print(f"\nCalculation:")
print(f"  Without mt_f variable: {old_n_t1w} × {old_n_t2w} × {old_n_cest_k} × {old_n_cest_conc} × {old_n_mt_k}")
print(f"                       = {old_size_without_mt_f:,} entries ({old_size_without_mt_f/1e6:.1f}M)")
print(f"  With mt_f variable:    {old_size_without_mt_f:,} × {old_n_mt_f}")
print(f"                       = {old_size_with_mt_f:,} entries ({old_size_with_mt_f/1e6:.1f}M)")
print(f"\n  ↑ THIS IS THE 440M YOU SAW: 0/{old_size_with_mt_f:,}")

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

new_n_t1w = int((4.0 - 1.5) / 0.1) + 1            # 26 values (reduced step)
new_n_t2w = int((1.5 - 0.05) / 0.05) + 1          # 30 values (keep)
new_n_cest_k = int((9000 - 50) / 100) + 1         # 91 values (reduced step)
new_n_cest_conc = int((40 - 1) / 1) + 1           # 40 values (keep)
new_n_mt_k = int((70 - 30) / 5) + 1               # 9 values ✓ RANGE!
new_n_mt_f = 1                                     # 1 value (fixed) ✓ FIXED!

print(f"Water T1:        {new_n_t1w} values  (1.5 to 4.0 s, step 0.1)")
print(f"Water T2:        {new_n_t2w} values  (0.05 to 1.5 s, step 0.05)")
print(f"CEST k_sw:       {new_n_cest_k} values (50 to 9000 Hz, step 100)")
print(f"CEST conc:       {new_n_cest_conc} values (1 to 40 mM)")
print(f"MT k_sw:         {new_n_mt_k} values  (30 to 70 Hz, step 5) ✓ FIX 1")
print(f"MT dw:           0 ppm                       ✓ FIX 2")
print(f"MT f:            {new_n_mt_f} value   (0.05 fixed)         ✓ FIX 3")

new_size = new_n_t1w * new_n_t2w * new_n_cest_k * new_n_cest_conc * new_n_mt_k

print(f"\nCalculation:")
print(f"  {new_n_t1w} × {new_n_t2w} × {new_n_cest_k} × {new_n_cest_conc} × {new_n_mt_k}")
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
print("KEY INSIGHT: Why 440M instead of 11M?")
print("="*70)
print(f"\nYour mt_f was being calculated as VARIABLE from concentrations:")
print(f"  mt_f = mt_sol_conc × mt_protons / mt_water_conc")
print(f"\nThis created {old_n_mt_f} different mt_f values (one per concentration)")
print(f"Adding an extra dimension: 11M × {old_n_mt_f} = 440M!")
print(f"\nFor 5% BSA, mt_f should be FIXED at 0.05, not variable.")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70 + "\n")
print("1. Update your MATLAB dictionary generation script:")
print("   ")
print("   dictparams.water_t1 = 1.5:0.1:4;              % 26 values")
print("   dictparams.water_t2 = 0.05:0.05:1.5;          % 30 values")
print("   dictparams.cest_amine_k = 50:100:9000;        % 91 values")
print("   dictparams.cest_amine_sol_conc = 1:1:40;      % 40 values")
print("   ")
print("   dictparams.mt_k = 30:5:70;                    % 9 values ✓")
print("   dictparams.mt_dw = 0;                         % 0 ppm ✓")
print("   dictparams.mt_f = 0.05;                       % Fixed ✓")
print("   dictparams.mt_t2 = 7.338e-06;                 % 7.338 μs")
print("   dictparams.mt_lineshape = 'SuperLorentzian';")
print("")
print("2. Regenerate dictionary (~45 min)")
print("3. Run matching (chunked algorithm will auto-enable)")
print("4. Get accurate results!")
print("\n" + "="*70)
