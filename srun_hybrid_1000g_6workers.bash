#!/bin/bash
#SBATCH --job-name=MRF_dict_50M
#SBATCH --account=cestmrf
#SBATCH --qos=normal
#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --mem=1000G
#SBATCH --time=16:00:00
#SBATCH --output=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.out
#SBATCH --error=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cd1052

# HIGH MEMORY + LOW WORKERS HYBRID
# Strategy: 1TB memory with only 6 workers
# Reduces memory multiplication from parallelism
# Should definitely complete

echo "========================================="
echo "HYBRID APPROACH - 1TB / 6 WORKERS"
echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: rtx8000"
echo "Node: $(hostname)"
echo "CPUs: 6 (reduced for memory efficiency)"
echo "Memory: 1000G"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="
echo ""
echo "Balancing memory and workers for guaranteed completion"
echo "Expected runtime: 8-12 hours"
echo ""

# Memory monitoring
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

# Run Python - 6 workers
python MRFmatch_B-SL_dk.py

# Stop monitoring
kill $MONITOR_PID 2>/dev/null

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,ReqMem,Elapsed,State
