#!/bin/bash

echo "=========================================="
echo "MRF Processing with CNN/Dictionary Option"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "=========================================="

# Check processing mode
if [ -n "$USE_CNN" ] && [ "$USE_CNN" = "1" ]; then
    echo "✓ MODE: CNN Inference"
    echo "  - No dictionary generation needed"
    echo "  - Using pre-trained neural network"
    echo "  - Expected time: ~3 minutes"
    echo "  - Expected RAM: ~8 GB"
    export USE_CNN=1
else
    echo "ℹ MODE: Traditional Dictionary Matching"
    echo "  - Will generate full dictionary"
    echo "  - Expected time: 60+ hours"
    echo "  - Expected RAM: 200+ GB"
    export USE_CNN=0
fi

# Check if LARGE_STORAGE_DIR is set
if [ -n "$LARGE_STORAGE_DIR" ]; then
    echo "✓ Large storage enabled: $LARGE_STORAGE_DIR"
    export LARGE_STORAGE_DIR
else
    echo "ℹ Using default OUTPUT_FILES directory"
fi
echo "=========================================="

# Show GPU info
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
    echo "=========================================="
fi

# Activate conda + venv
source ~/.bashrc
conda activate cbdmrf
source ~/cbdmrfpy/bin/activate

# Verify environment
echo ""
echo "Environment Check:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"

# Check for PyTorch (needed for CNN)
if [ "$USE_CNN" = "1" ]; then
    echo "  PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
    echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'UNKNOWN')"
    if python -c 'import torch; torch.cuda.is_available()' 2>/dev/null | grep -q "True"; then
        echo "  ✓ GPU acceleration available"
    else
        echo "  ⚠ WARNING: GPU not available, will use CPU (slower)"
    fi
fi

echo "=========================================="

# Navigate to code directory
cd ~/molecular-mrf-main/molecular-mrf-main

# Show where files will be saved
echo ""
echo "Output Configuration:"
if [ -n "$LARGE_STORAGE_DIR" ]; then
    echo "  Output directory: $LARGE_STORAGE_DIR/MRF_OUTPUT/"
    echo "  - quant_maps.mat"
    if [ "$USE_CNN" != "1" ]; then
        echo "  - dict.mat"
    fi
else
    echo "  Output directory: OUTPUT_FILES/"
    echo "  - quant_maps.mat"
    if [ "$USE_CNN" != "1" ]; then
        echo "  - dict.mat"
    fi
fi
echo "=========================================="

# Run the Python script
echo ""
echo "Starting MRF processing..."
echo ""

if [ "$USE_CNN" = "1" ]; then
    # Use CNN version
    if [ -f "MRFmatch_B-SL_dk_CNN.py" ]; then
        python MRFmatch_B-SL_dk_CNN.py
        EXIT_CODE=$?
    else
        echo "✗ ERROR: MRFmatch_B-SL_dk_CNN.py not found!"
        echo "  Please ensure the CNN script is uploaded to the cluster."
        exit 1
    fi
else
    # Use original dictionary matching version
    python MRFmatch_B-SL_dk.py
    EXIT_CODE=$?
fi

# Check if processing completed successfully
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "✓ MRF PROCESSING COMPLETED SUCCESSFULLY"
    echo "=========================================="

    # Show output file sizes
    echo ""
    echo "Output Files Created:"
    if [ -n "$LARGE_STORAGE_DIR" ]; then
        ls -lh $LARGE_STORAGE_DIR/MRF_OUTPUT/ 2>/dev/null || echo "  (No files in large storage)"
    else
        ls -lh OUTPUT_FILES/*.mat 2>/dev/null || echo "  (No .mat files in OUTPUT_FILES)"
    fi

    # Show processing summary
    echo ""
    echo "Processing Summary:"
    if [ "$USE_CNN" = "1" ]; then
        echo "  ✓ Used CNN inference (fast mode)"
        echo "  ✓ No dictionary file generated (saved ~580 GB)"
    else
        echo "  ✓ Used dictionary matching (traditional mode)"
        if [ -n "$LARGE_STORAGE_DIR" ]; then
            DICT_SIZE=$(ls -lh $LARGE_STORAGE_DIR/MRF_OUTPUT/dict.mat 2>/dev/null | awk '{print $5}')
            if [ -n "$DICT_SIZE" ]; then
                echo "  ✓ Dictionary size: $DICT_SIZE"
            fi
        fi
    fi
else
    echo "=========================================="
    echo "✗ MRF PROCESSING FAILED"
    echo "  Exit code: $EXIT_CODE"
    echo "=========================================="
    echo ""
    echo "Common issues:"
    if [ "$USE_CNN" = "1" ]; then
        echo "  - CNN model file not found (check unsupervised_example/model/NN_model_UL.pth)"
        echo "  - PyTorch not installed (conda install pytorch)"
        echo "  - Data format mismatch (check acquired_data.mat shape)"
    else
        echo "  - Out of memory (increase --mem in SLURM script)"
        echo "  - Dictionary too large (reduce parameter ranges)"
        echo "  - Python dependencies missing"
    fi
    echo ""
    echo "Check the error messages above for details."
fi

echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="

exit $EXIT_CODE
