#!/bin/bash
#SBATCH --job-name=MRF_CNN
#SBATCH --output=slurm_mrf_cnn_%j.out
#SBATCH --error=slurm_mrf_cnn_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# SLURM Job Script for MRF CNN Processing
# This script runs the integrated MRF processing with CNN inference on the cluster
#
# Environment variables that can be passed via sbatch --export:
#   USE_CNN: Set to 1 for CNN mode, 0 for dictionary matching (default: 0)
#   NUM_GPUS: Number of GPUs to use (default: 1)
#   MODEL_PATH: Path to pre-trained model (default: OUTPUT_FILES/trained_model.pt)
#   LARGE_STORAGE_DIR: Optional path to large storage directory for output

echo "================================"
echo "MRF CNN PROCESSING ON CLUSTER"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "================================"

# Print environment variables
echo ""
echo "Configuration:"
echo "  USE_CNN: ${USE_CNN:-0}"
echo "  NUM_GPUS: ${NUM_GPUS:-1}"
echo "  MODEL_PATH: ${MODEL_PATH:-OUTPUT_FILES/trained_model.pt}"
echo "  NUM_WORKERS: ${NUM_WORKERS:-18}"
if [ -n "$LARGE_STORAGE_DIR" ]; then
    echo "  LARGE_STORAGE_DIR: $LARGE_STORAGE_DIR"
fi
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
    echo ""
fi

# Load required modules (adjust based on your cluster setup)
# Uncomment and modify as needed:
# module load cuda/11.8
# module load cudnn/8.6
# module load python/3.9

# Activate conda environment (adjust path as needed)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

# Or activate virtualenv (adjust path as needed)
# source /path/to/venv/bin/activate

# Set default values if not provided
export USE_CNN=${USE_CNN:-0}
export NUM_GPUS=${NUM_GPUS:-1}
export NUM_WORKERS=${NUM_WORKERS:-18}
export MODEL_PATH=${MODEL_PATH:-OUTPUT_FILES/trained_model.pt}

# Change to script directory
cd "$(dirname "$0")" || exit 1

# Print Python environment info
echo "Python Environment:"
echo "  Python: $(which python)"
echo "  Version: $(python --version)"
echo ""

# Check PyTorch installation and CUDA availability
python << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print("")
EOF

# Set up large storage directory if specified
if [ -n "$LARGE_STORAGE_DIR" ]; then
    echo "Setting up large storage directory..."
    mkdir -p "$LARGE_STORAGE_DIR/MRF_OUTPUT"

    # Create symbolic link to large storage for output
    if [ -d "OUTPUT_FILES" ]; then
        mv OUTPUT_FILES OUTPUT_FILES.backup_$(date +%s)
    fi
    ln -sf "$LARGE_STORAGE_DIR/MRF_OUTPUT" OUTPUT_FILES
    echo "  Output will be written to: $LARGE_STORAGE_DIR/MRF_OUTPUT"
    echo ""
fi

# Run the integrated MRF processing script
echo "================================"
echo "Starting MRF Processing..."
echo "================================"
echo ""

START_TIME=$(date +%s)

# Choose script based on mode
if [ "$USE_CNN" -eq 1 ]; then
    echo "Running in CNN INFERENCE mode"
    python MRFmatch_integrated.py
else
    echo "Running in DICTIONARY MATCHING mode"
    python MRFmatch_B-SL_dk.py
fi

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "MRF Processing COMPLETED"
    echo "Exit code: $EXIT_CODE"
    echo "Elapsed time: $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds"

    # List output files
    echo ""
    echo "Output files:"
    if [ -d "OUTPUT_FILES" ]; then
        ls -lh OUTPUT_FILES/quant_maps.mat 2>/dev/null || echo "  Warning: quant_maps.mat not found"
        ls -lh OUTPUT_FILES/*.eps 2>/dev/null || echo "  No EPS files found"
        ls -lh OUTPUT_FILES/*.pt 2>/dev/null || echo "  No model files found"
    else
        echo "  Warning: OUTPUT_FILES directory not found"
    fi
else
    echo "MRF Processing FAILED"
    echo "Exit code: $EXIT_CODE"
    echo "Check error log for details: slurm_mrf_cnn_${SLURM_JOB_ID}.err"
fi
echo "================================"
echo "Finished: $(date)"
echo "================================"

exit $EXIT_CODE
