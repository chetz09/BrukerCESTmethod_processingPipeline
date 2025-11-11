#!/bin/bash
#SBATCH --job-name=MRF_dict_50M
#SBATCH --account=cestmrf
#SBATCH --qos=normal
#SBATCH --partition=pubgpu-req,pubgpu,rtx6000,rtx8000,dgx-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=250G
#SBATCH --time=06:00:00
#SBATCH --output=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.out
#SBATCH --error=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cd1052

# HIGH MEMORY VERSION - 250GB
# Previous job OOM killed with 100GB
# Increased to 250GB for 50M dictionary generation

echo "========================================="
echo "Job started at: $(date)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 250G"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="
echo ""

# Monitor memory usage
echo "Monitoring memory every 60 seconds..."
(while true; do
    echo "[$(date)] Memory usage: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
    sleep 60
done) &
MONITOR_PID=$!

# Load environment
# module load anaconda3
# conda activate your_env_name

cd /autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main

# Run Python
python MRFmatch_B-SL_dk.py

# Stop monitoring
kill $MONITOR_PID 2>/dev/null

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
echo "Check actual memory used:"
echo "sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,ReqMem,Elapsed,State"
