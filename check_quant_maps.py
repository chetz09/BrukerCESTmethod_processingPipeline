#!/usr/bin/env python3
"""
Check quantitative maps to diagnose empty visualization plots
"""
import scipy.io as sio
import numpy as np
import sys
import os

# Find quant_maps.mat
possible_paths = [
    'OUTPUT_FILES/quant_maps.mat',
    '../OUTPUT_FILES/quant_maps.mat',
    'quant_maps.mat',
]

if len(sys.argv) > 1:
    possible_paths.insert(0, sys.argv[1])

print("Searching for quant_maps.mat...")
quant_file = None
for path in possible_paths:
    if os.path.exists(path):
        quant_file = path
        break

if quant_file is None:
    print("\nERROR: Could not find quant_maps.mat")
    print("Please provide path as argument: python check_quant_maps.py /path/to/quant_maps.mat")
    sys.exit(1)

print(f"\nFound quant maps at: {quant_file}\n")

# Load data
quant_maps = sio.loadmat(quant_file)

print("=" * 60)
print("QUANTITATIVE MAPS SUMMARY")
print("=" * 60)

# Show all keys
keys = [k for k in quant_maps.keys() if not k.startswith('__')]
print(f"\nAvailable maps: {keys}\n")

# Check each map
for key in keys:
    data = quant_maps[key]
    print(f"\n{key.upper()}:")
    print(f"  Shape: {data.shape}")
    print(f"  Data type: {data.dtype}")
    print(f"  Min value: {np.min(data):.6f}")
    print(f"  Max value: {np.max(data):.6f}")
    print(f"  Mean value: {np.mean(data):.6f}")
    print(f"  Median value: {np.median(data):.6f}")
    print(f"  Non-zero voxels: {np.count_nonzero(data)} / {data.size} ({100*np.count_nonzero(data)/data.size:.1f}%)")

    # Check for NaN or Inf
    if np.any(np.isnan(data)):
        print(f"  ⚠️ WARNING: Contains {np.sum(np.isnan(data))} NaN values!")
    if np.any(np.isinf(data)):
        print(f"  ⚠️ WARNING: Contains {np.sum(np.isinf(data))} Inf values!")

# Check dot product specifically
if 'dp' in quant_maps:
    print("\n" + "=" * 60)
    print("DOT PRODUCT ANALYSIS (Matching Quality)")
    print("=" * 60)
    dp = quant_maps['dp']

    print(f"\nDot product statistics:")
    print(f"  Min: {np.min(dp):.6f}")
    print(f"  Max: {np.max(dp):.6f}")
    print(f"  Mean: {np.mean(dp):.6f}")
    print(f"  Median: {np.median(dp):.6f}")
    print(f"  Std: {np.std(dp):.6f}")

    # Check various threshold levels
    print(f"\nVoxels passing different thresholds:")
    thresholds = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9997, 0.99974]
    for thresh in thresholds:
        n_pass = np.sum(dp > thresh)
        pct = 100 * n_pass / dp.size
        marker = " ← CURRENT THRESHOLD" if abs(thresh - 0.99974) < 1e-6 else ""
        print(f"  > {thresh:.5f}: {n_pass:6d} voxels ({pct:5.1f}%){marker}")

    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)

    current_thresh = 0.99974
    n_current = np.sum(dp > current_thresh)

    if n_current == 0:
        print(f"\n⚠️ PROBLEM FOUND: No voxels pass the current threshold ({current_thresh:.5f})")
        print(f"   This is why your plots are empty!")
        print(f"\n💡 SOLUTION:")
        print(f"   Lower the mask threshold in MRFmatch_B-SL_dk.py line 196")

        # Suggest appropriate threshold
        if np.max(dp) < 0.95:
            suggested = 0.90
        elif np.max(dp) < 0.98:
            suggested = 0.95
        else:
            suggested = 0.98

        n_suggested = np.sum(dp > suggested)
        pct_suggested = 100 * n_suggested / dp.size
        print(f"\n   Suggested threshold: {suggested:.2f}")
        print(f"   This would show {n_suggested} voxels ({pct_suggested:.1f}%)")
        print(f"\n   Change line 196 from:")
        print(f"       mask = quant_maps['dp'] > 0.99974")
        print(f"   to:")
        print(f"       mask = quant_maps['dp'] > {suggested:.2f}")

    elif n_current < dp.size * 0.05:  # Less than 5% of voxels
        print(f"\n⚠️ WARNING: Only {n_current} voxels ({100*n_current/dp.size:.1f}%) pass threshold")
        print(f"   Your plots may be very sparse")
        print(f"\n💡 SUGGESTION: Consider lowering threshold to 0.95 or 0.98")
    else:
        print(f"\n✓ Threshold looks reasonable: {n_current} voxels pass ({100*n_current/dp.size:.1f}%)")

# Check if all maps are zeros
print("\n" + "=" * 60)
print("DATA VALIDITY CHECK")
print("=" * 60)

all_zero_maps = []
for key in ['fs', 'ksw', 't1w', 't2w']:
    if key in quant_maps:
        if np.all(quant_maps[key] == 0):
            all_zero_maps.append(key)

if all_zero_maps:
    print(f"\n⚠️ ERROR: These maps are all zeros: {all_zero_maps}")
    print(f"   This suggests matching failed or didn't run")
    print(f"\n💡 Check:")
    print(f"   1. Did matching complete successfully?")
    print(f"   2. Check for errors in the Python output")
    print(f"   3. Verify dictionary has 'sig' key (use check_dict_structure.py)")
else:
    print(f"\n✓ All maps contain non-zero values")

print("\n" + "=" * 60)
