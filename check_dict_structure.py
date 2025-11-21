#!/usr/bin/env python3
"""
Quick diagnostic to check dictionary structure
"""
import scipy.io as sio
import sys
import os

# Try to find dict.mat in common locations
possible_paths = [
    'OUTPUT_FILES/dict.mat',
    '../OUTPUT_FILES/dict.mat',
    '../../OUTPUT_FILES/dict.mat',
]

# Check if path provided as argument
if len(sys.argv) > 1:
    dict_path = sys.argv[1]
    possible_paths.insert(0, dict_path)

print("Searching for dict.mat...")
dict_file = None
for path in possible_paths:
    if os.path.exists(path):
        dict_file = path
        break

if dict_file is None:
    print("\nERROR: Could not find dict.mat")
    print("Please provide path as argument: python check_dict_structure.py /path/to/dict.mat")
    sys.exit(1)

print(f"\nFound dictionary at: {dict_file}")
print(f"File size: {os.path.getsize(dict_file) / (1024**3):.2f} GB")

# Load and check structure
print("\nLoading dictionary...")
try:
    dict_data = sio.loadmat(dict_file)

    print("\n=== Dictionary Keys ===")
    keys = [k for k in dict_data.keys() if not k.startswith('__')]
    for key in sorted(keys):
        shape = dict_data[key].shape if hasattr(dict_data[key], 'shape') else 'scalar'
        dtype = dict_data[key].dtype if hasattr(dict_data[key], 'dtype') else type(dict_data[key])
        print(f"  {key:20s}: shape={shape}, dtype={dtype}")

    # Check for critical 'sig' key
    if 'sig' in dict_data:
        print(f"\n✓ 'sig' key EXISTS - shape: {dict_data['sig'].shape}")
        print("Dictionary should work for matching!")
    else:
        print("\n✗ 'sig' key MISSING - this is the problem!")
        print("Dictionary is incomplete. You need to regenerate it.")

    # Count entries
    if 'sig' in dict_data:
        n_entries = dict_data['sig'].shape[1] if len(dict_data['sig'].shape) > 1 else dict_data['sig'].shape[0]
        print(f"\nDictionary entries: {n_entries:,}")

except Exception as e:
    print(f"\nERROR loading dictionary: {e}")
    sys.exit(1)
