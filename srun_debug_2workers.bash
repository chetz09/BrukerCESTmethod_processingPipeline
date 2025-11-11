#!/bin/bash
#SBATCH --job-name=MRF_dict_DEBUG
#SBATCH --account=cestmrf
#SBATCH --qos=normal
#SBATCH --partition=rtx8000,rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cd1052

# DEBUG MODE - Verify settings and monitor memory aggressively
echo "========================================="
echo "DEBUG MODE: 2 WORKERS"
echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory requested: 400G"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""
echo "Environment check:"
echo "SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK"
echo "Python should use $SLURM_CPUS_PER_TASK workers"
echo "========================================="
echo ""

# Aggressive memory monitoring every 10 seconds
echo "Starting aggressive memory monitoring (every 10 seconds)..."
(while true; do
    MEMUSED=$(free -h | grep Mem | awk '{print $3}')
    MEMTOTAL=$(free -h | grep Mem | awk '{print $2}')
    MEMPCT=$(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100}')
    echo "[$(date '+%H:%M:%S')] Memory: $MEMUSED / $MEMTOTAL ($MEMPCT)"
    sleep 10
done) &
MONITOR_PID=$!

cd /autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main

# Add Python debugging
echo ""
echo "Starting Python script..."
echo "If num_workers != 2, there's a problem with SLURM_CPUS_PER_TASK"
echo ""

# Run with explicit environment variable
export SLURM_CPUS_PER_TASK=2
python -u MRFmatch_B-SL_dk.py

EXIT_CODE=$?

# Stop monitoring
kill $MONITOR_PID 2>/dev/null

echo ""
echo "========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================="
