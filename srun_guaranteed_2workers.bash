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

# GUARANTEED 2 WORKERS MODE
# Explicitly sets environment variable to prevent defaulting to 4 workers
# Uses 500GB to be absolutely safe (1.5TB available on these nodes)

echo "========================================="
echo "GUARANTEED 2 WORKERS MODE"
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
echo "Python MUST use 2 workers (not the default 4)"
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

cd /autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main

echo ""
echo "Starting Python script with 2 workers..."
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
    echo "SUCCESS!"
else
    echo "FAILED with exit code $EXIT_CODE"
fi
