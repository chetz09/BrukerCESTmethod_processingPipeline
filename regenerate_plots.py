#!/usr/bin/env python3
"""
Regenerate visualization plots from existing quant_maps.mat
Use this to quickly fix empty plots without rerunning matching (which takes hours!)

Usage:
    python regenerate_plots.py [threshold]

Examples:
    python regenerate_plots.py           # Uses 0.95 threshold
    python regenerate_plots.py 0.98      # Uses 0.98 threshold
    python regenerate_plots.py 0.90      # Uses 0.90 threshold
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Blue-viridis colormap (from original code)
from matplotlib.colors import ListedColormap
cdict_bviridis = {'red': ((0.0, 0.28, 0.28),
                           (1.0, 0.28, 0.28)),
                  'green': ((0.0, 0.0, 0.0),
                            (1.0, 0.95, 0.95)),
                  'blue': ((0.0, 0.3, 0.3),
                           (1.0, 0.99, 0.99))}
b_viridis = ListedColormap(plt.cm.viridis(np.linspace(0, 1, 256)) * np.array([0.28, 0.95, 0.99, 1]))

# Parse threshold from command line
if len(sys.argv) > 1:
    mask_threshold = float(sys.argv[1])
    print(f"Using threshold from command line: {mask_threshold}")
else:
    mask_threshold = 0.95  # Default
    print(f"Using default threshold: {mask_threshold}")

# Load quantitative maps
quant_maps_file = 'OUTPUT_FILES/quant_maps.mat'
if not os.path.exists(quant_maps_file):
    print(f"\nERROR: {quant_maps_file} not found!")
    print("Make sure you're in the correct directory and matching has completed.")
    sys.exit(1)

print(f"Loading {quant_maps_file}...")
quant_maps = sio.loadmat(quant_maps_file)

# Check available keys
available_keys = [k for k in quant_maps.keys() if not k.startswith('__')]
print(f"Available maps: {available_keys}")

# Create mask
if 'dp' not in quant_maps:
    print("\nERROR: 'dp' (dot product) not found in quant_maps.mat!")
    sys.exit(1)

mask = quant_maps['dp'] > mask_threshold

# Save mask
mask_fn = 'mask.npy'
np.save(mask_fn, mask)
print(f"Mask saved to {mask_fn}")

# Report statistics
n_voxels_shown = np.sum(mask)
total_voxels = mask.size
pct_shown = 100 * n_voxels_shown / total_voxels

print(f"\nVisualization settings:")
print(f"  Mask threshold: {mask_threshold}")
print(f"  Voxels shown: {n_voxels_shown} / {total_voxels} ({pct_shown:.1f}%)")
print(f"  Dot product range: [{np.min(quant_maps['dp']):.3f}, {np.max(quant_maps['dp']):.3f}]")

if n_voxels_shown == 0:
    print("\n⚠️ WARNING: No voxels pass this threshold!")
    print("   Try a lower threshold (e.g., 0.90, 0.85)")
    print(f"\n   Run: python regenerate_plots.py 0.90")
    sys.exit(1)
elif n_voxels_shown < total_voxels * 0.05:
    print(f"\n⚠️ WARNING: Only {pct_shown:.1f}% of voxels shown - plots may be sparse")
    print(f"   Consider lowering threshold")

# Create visualization
print("\nGenerating plots...")

fig_fn = 'OUTPUT_FILES/dot_product_results.eps'
fig, axes = plt.subplots(1, 3, figsize=(30, 25))

color_maps = [b_viridis, 'magma', 'magma']
data_keys = ['fs', 'ksw', 'dp']
titles = ['[L-arg] (mM)', 'k$_{sw}$ (s$^{-1}$)', 'Dot product']
clim_list = [(0, 120), (0, 500), (0.90, 1)]
tick_list = [np.arange(0, 140, 20), np.arange(0, 600, 100), np.arange(0.90, 1.01, 0.02)]

for ax, color_map, key, title, clim, ticks in zip(axes.flat, color_maps, data_keys, titles, clim_list, tick_list):
    if key not in quant_maps:
        print(f"⚠️ WARNING: '{key}' not found in quant_maps, skipping...")
        continue

    # Scale factor for concentration (fs)
    scale_factor = 110e3 / 3 if key == 'fs' else 1.0

    # Convert to proper numpy arrays
    data_array = np.array(quant_maps[key], dtype=np.float64, copy=True)
    mask_array = np.array(mask, dtype=np.float64, copy=True)
    vals = np.ascontiguousarray(data_array * float(scale_factor) * mask_array)

    plot = ax.imshow(vals, cmap=color_map)
    plot.set_clim(*clim)
    ax.set_title(title, fontsize=25)

    # Colorbar
    tick_array = np.ascontiguousarray(ticks, dtype=np.float64)
    cb = plt.colorbar(plot, ax=ax, ticks=tick_array.tolist(), orientation='vertical', fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=25)
    ax.set_axis_off()

plt.tight_layout()

# Save as EPS
print(f"Saving to {fig_fn}...")
plt.savefig(fig_fn, format="eps", dpi=300)
plt.close()

print(f"\n✓ SUCCESS: Plots saved to {fig_fn}")
print(f"\nTo try a different threshold, run:")
print(f"  python regenerate_plots.py 0.98   # More strict (fewer voxels)")
print(f"  python regenerate_plots.py 0.90   # More permissive (more voxels)")
