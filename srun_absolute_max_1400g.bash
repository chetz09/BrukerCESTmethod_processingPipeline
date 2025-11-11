#!/bin/bash
#SBATCH --job-name=MRF_dict_50M
#SBATCH --account=cestmrf
#SBATCH --qos=normal
#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=1400G
#SBATCH --time=12:00:00
#SBATCH --output=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.out
#SBATCH --error=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cd1052

# ABSOLUTE MAXIMUM MEMORY - 1.4TB
# rtx8000 nodes have 1.5TB available
# Previous job OOM killed at 900GB
# This uses ~93% of available node memory
# LAST ATTEMPT with 16 workers

echo "========================================="
echo "MAXIMUM MEMORY MODE - 1.4TB"
echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: rtx8000 (1.5TB nodes)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 1400G (93% of node capacity)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="
echo ""
echo "WARNING: This is maximum possible memory"
echo "If this OOMs, must reduce workers or dictionary size"
echo ""

# Aggressive memory monitoring
echo "Monitoring memory every 30 seconds..."
(while true; do
    MEMUSED=$(free -h | grep Mem | awk '{print $3}')
    MEMTOTAL=$(free -h | grep Mem | awk '{print $2}')
    MEMPCT=$(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100}')
    echo "[$(date)] Memory: $MEMUSED / $MEMTOTAL ($MEMPCT)"
    sleep 30
done) &
MONITOR_PID=$!

# Load environment
# module load anaconda3
# conda activate your_env_name

cd /autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main

# Run Python - 16 workers
python MRFmatch_B-SL_dk.py

# Stop monitoring
kill $MONITOR_PID 2>/dev/null

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,ReqMem,Elapsed,State
