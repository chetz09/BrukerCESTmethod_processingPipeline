#!/bin/bash
#SBATCH --job-name=MRF_CNN
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================
# MRF PROCESSING MODE SELECTION
# ============================================================
# Set USE_CNN=1 for CNN mode (RECOMMENDED - fast, low memory)
# Set USE_CNN=0 for dictionary mode (slow, high memory)
# ============================================================

# Default to CNN mode (change to 0 for dictionary mode)
export USE_CNN=1

# ============================================================
# RESOURCE ALLOCATION (automatically adjusted based on mode)
# ============================================================

if [ "$USE_CNN" = "1" ]; then
    # CNN MODE - Minimal resources needed
    #SBATCH --cpus-per-task=4
    #SBATCH --gres=gpu:1
    #SBATCH --mem=16G
    #SBATCH --time=0-01:00:00
    #SBATCH --partition=gpu

    echo "Configured for CNN mode:"
    echo "  - CPUs: 4"
    echo "  - GPUs: 1"
    echo "  - Memory: 16 GB"
    echo "  - Time limit: 1 hour"
else
    # DICTIONARY MODE - High resources needed
    #SBATCH --cpus-per-task=32
    #SBATCH --mem=1000G
    #SBATCH --time=0-80:00:00
    #SBATCH --partition=compute

    echo "Configured for Dictionary mode:"
    echo "  - CPUs: 32"
    echo "  - Memory: 1000 GB"
    echo "  - Time limit: 80 hours"
fi

# Set large storage directory
export LARGE_STORAGE_DIR=/autofs/vast/farrar/users/cd1052

# Run the processing script
bash run_cnn.bash
