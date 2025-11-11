#!/bin/bash
#SBATCH --job-name=MRF_dict_50M
#SBATCH --account=cestmrf
#SBATCH --qos=normal
#SBATCH --partition=dgx-a100,rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.out
#SBATCH --error=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cd1052

# EMERGENCY: ONLY 2 WORKERS
# Your job used 500GB with 16 workers and still OOM killed!
# Reducing to 2 workers should bring memory down to ~60-100GB
# Will be MUCH slower but should complete

echo "========================================="
echo "EMERGENCY MODE: 2 WORKERS ONLY"
echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK (MINIMAL - to prevent OOM)"
echo "Memory: 200G"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="
echo ""
echo "WARNING: This will be VERY slow with only 2 workers"
echo "But it should complete without OOM killing"
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

# Run Python - will use 2 workers automatically
python MRFmatch_B-SL_dk.py

# Stop monitoring
kill $MONITOR_PID 2>/dev/null

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
echo "Memory stats:"
sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,ReqMem,Elapsed,State
