#!/bin/bash
#SBATCH --job-name=MRF_dict_50M_4w
#SBATCH --account=cestmrf
#SBATCH --qos=normal
#SBATCH --partition=dgx-a100,rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=350G
#SBATCH --time=18:00:00
#SBATCH --output=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.out
#SBATCH --error=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cd1052

# CONSERVATIVE 4 WORKERS
# Previous run: 1400GB with 18 workers = OOM killed
# Strategy: Reduce workers to 4 = 4.5x reduction
# Memory estimate: 1400GB / 4.5 ≈ 311GB, requesting 350GB for safety
# Balance between speed and memory usage

echo "========================================="
echo "CONSERVATIVE MODE: 4 WORKERS"
echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 350G"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="
echo ""
echo "Using 4 workers for balance of speed and memory"
echo "Python will read SLURM_CPUS_PER_TASK=4 automatically"
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

# Run Python - will automatically use 4 workers from SLURM_CPUS_PER_TASK
python MRFmatch_B-SL_dk.py

# Stop monitoring
kill $MONITOR_PID 2>/dev/null

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
echo "Memory stats:"
sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,ReqMem,Elapsed,State
