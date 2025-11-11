#!/bin/bash
#SBATCH --job-name=MRF_dict_50M
#SBATCH --account=cestmrf
#SBATCH --qos=normal
#SBATCH --partition=dgx-a100,rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=700G
#SBATCH --time=08:00:00
#SBATCH --output=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.out
#SBATCH --error=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cd1052

# BALANCED HIGH MEMORY - 700GB with 12 workers
# Conservative option: fewer workers than 16 but more memory
# 700GB should be plenty, and 12 workers is a good balance
# Should complete faster than emergency scripts

echo "========================================="
echo "BALANCED HIGH MEMORY - 700GB / 12 WORKERS"
echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 700G"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="
echo ""

# Monitor memory usage
echo "Monitoring memory every 60 seconds..."
(while true; do
    MEMUSED=$(free -h | grep Mem | awk '{print $3}')
    MEMTOTAL=$(free -h | grep Mem | awk '{print $2}')
    MEMPCT=$(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100}')
    echo "[$(date)] Memory: $MEMUSED / $MEMTOTAL ($MEMPCT)"
    sleep 60
done) &
MONITOR_PID=$!

# Load environment
# module load anaconda3
# conda activate your_env_name

cd /autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main

# Run Python - will use 12 workers automatically
python MRFmatch_B-SL_dk.py

# Stop monitoring
kill $MONITOR_PID 2>/dev/null

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
echo "Memory stats:"
sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,ReqMem,Elapsed,State
