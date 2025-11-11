#!/bin/bash
#SBATCH --job-name=MRF_dict_50M
#SBATCH --account=cestmrf
#SBATCH --qos=normal
#SBATCH --partition=pubgpu-req,pubgpu,rtx6000,rtx8000,dgx-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.out
#SBATCH --error=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cd1052

# OPTIMAL 16 CPU VERSION
# Maximum CPUs that work with pubgpu (idle GPU partition)
# Your Python code will automatically use all 16 workers via SLURM_CPUS_PER_TASK
#
# Partitions prioritized for idle nodes:
# 1. pubgpu-req (idle, unlimited CPUs)
# 2. pubgpu (idle, max 16 CPUs) ← This sets our limit
# 3. rtx6000, rtx8000, dgx-a100 (fallback, unlimited)

echo "========================================="
echo "Job started at: $(date)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "QOS: $SLURM_JOB_QOS"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 100G"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="
echo ""

# Load environment
# module load anaconda3
# conda activate your_env_name

cd /autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main

# Python will automatically use 16 workers from SLURM_CPUS_PER_TASK
python MRFmatch_B-SL_dk.py

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
echo "Check resource usage:"
echo "sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,Elapsed,State,Partition"
