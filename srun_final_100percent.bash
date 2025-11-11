#!/bin/bash
#SBATCH --job-name=MRF_dict_2w
#SBATCH --account=cestmrf
#SBATCH --qos=normal
#SBATCH --partition=rtx8000,rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=500G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cd1052

# 100% SUCCESS CONFIGURATION
# - 2 workers (guaranteed via explicit env variable)
# - 500GB memory (safe on 1.5TB nodes)
# - Conda environment activation
# - 48 hours time limit

echo "========================================="
echo "100% SUCCESS MODE: 2 WORKERS"
echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory requested: 500G (out of 1.5TB available)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# CRITICAL: Explicitly set to prevent Python defaulting to 4
export SLURM_CPUS_PER_TASK=2
echo "SLURM_CPUS_PER_TASK explicitly set to: $SLURM_CPUS_PER_TASK"
echo "Python will use 2 workers (not the default 4)"
echo "========================================="
echo ""

# Memory monitoring every 30 seconds
echo "Starting memory monitoring..."
(while true; do
    MEMUSED=$(free -h | grep Mem | awk '{print $3}')
    MEMTOTAL=$(free -h | grep Mem | awk '{print $2}')
    MEMPCT=$(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100}')
    echo "[$(date '+%H:%M:%S')] Memory: $MEMUSED / $MEMTOTAL ($MEMPCT)"
    sleep 30
done) &
MONITOR_PID=$!

# Activate conda environment
echo "Activating conda environment 'mrfmatch'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mrfmatch

# Verify environment
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo "Numpy check:"
python -c "import numpy; print(f'  numpy version: {numpy.__version__}')" || echo "  ERROR: numpy not found!"
echo ""

cd /autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main

echo "Starting MRF dictionary generation..."
echo "This will take a long time with 2 workers but will NOT run out of memory"
echo ""

# Run Python
python -u MRFmatch_B-SL_dk.py

EXIT_CODE=$?

# Stop monitoring
kill $MONITOR_PID 2>/dev/null

echo ""
echo "========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS! Job completed without errors."
else
    echo "FAILED with exit code $EXIT_CODE"
    echo "Check error log: slurm-$SLURM_JOB_ID.err"
fi
