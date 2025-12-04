#!/bin/bash

echo "=========================================="
echo "MRF Dictionary Simulation + Matching"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "=========================================="

# Check if LARGE_STORAGE_DIR is set
if [ -n "$LARGE_STORAGE_DIR" ]; then
    echo "✓ Large storage enabled: $LARGE_STORAGE_DIR"
    export LARGE_STORAGE_DIR  # Make sure it's exported for Python
else
    echo "ℹ Using default OUTPUT_FILES directory"
fi
echo "=========================================="

# Show GPU info
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
fi

# Activate conda + venv
source ~/.bashrc
conda activate cbdmrf
source ~/cbdmrfpy/bin/activate

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Numpy: $(python -c 'import numpy; print(numpy.__version__)')"
echo "=========================================="

# Navigate to code directory
cd ~/molecular-mrf-main/molecular-mrf-main

# Show where files will be saved
if [ -n "$LARGE_STORAGE_DIR" ]; then
    echo "Output files will be saved to:"
    echo "  - dict.mat: $LARGE_STORAGE_DIR/MRF_OUTPUT/"
    echo "  - quant_maps.mat: $LARGE_STORAGE_DIR/MRF_OUTPUT/"
else
    echo "Output files will be saved to:"
    echo "  - dict.mat: OUTPUT_FILES/"
    echo "  - quant_maps.mat: OUTPUT_FILES/"
fi
echo "=========================================="

# Run the Python script directly
echo "Starting MRF processing..."
python MRFmatch_B-SL_dk.py

# Check if processing completed successfully
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "✓ MRF processing completed successfully"

    # Show output file sizes
    if [ -n "$LARGE_STORAGE_DIR" ]; then
        echo "Output files created:"
        ls -lh $LARGE_STORAGE_DIR/MRF_OUTPUT/ 2>/dev/null || echo "  No files in large storage"
    else
        echo "Output files created:"
        ls -lh OUTPUT_FILES/*.mat 2>/dev/null || echo "  No .mat files in OUTPUT_FILES"
    fi
else
    echo "=========================================="
    echo "✗ MRF processing failed with exit code $?"
fi

echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="
