"""
Extract dp-based mask from quant_maps.mat and save as mask.npy
"""
import numpy as np
import scipy.io as sio
import os

# Load quant_maps.mat
# Check if using large storage
large_storage = os.environ.get('LARGE_STORAGE_DIR', None)

if large_storage and os.path.exists(large_storage):
    quant_maps_fn = os.path.join(large_storage, 'MRF_OUTPUT', 'quant_maps.mat')
else:
    quant_maps_fn = 'OUTPUT_FILES/quant_maps.mat'

print(f'Loading quant_maps from: {quant_maps_fn}')
quant_maps = sio.loadmat(quant_maps_fn)

# Extract dp and create mask
dp = quant_maps['dp']
threshold = 0.95  # Adjust if needed

print(f'dp shape: {dp.shape}')
print(f'dp range: [{dp.min():.4f}, {dp.max():.4f}]')
print(f'Creating mask with threshold: {threshold}')

mask = dp > threshold

# Count masked pixels
n_masked = np.sum(mask)
n_total = mask.size
percent_masked = 100 * n_masked / n_total

print(f'Masked pixels: {n_masked}/{n_total} ({percent_masked:.2f}%)')

# Save mask
mask_fn = 'mask.npy'
np.save(mask_fn, mask)
print(f'Mask saved to: {mask_fn}')
