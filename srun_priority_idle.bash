#!/bin/bash
#SBATCH --job-name=MRF_dict_50M
#SBATCH --partition=pubgpu-req,pubgpu,west,rtx6000,rtx8000,dgx-a100
#SBATCH --account=cestmrf
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.out
#SBATCH --error=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cd1052

# PRIORITY IDLE NODES - GPU version
# Partitions listed in priority order (idle first):
# 1. pubgpu-req (2 idle GPU nodes)
# 2. pubgpu (1 idle GPU node)
# 3. west (1 idle node)
# 4. rtx6000, rtx8000, dgx-a100 (fallback)
#
# Optimized resources: 18 CPUs, 100GB RAM, 4h
# Should schedule within MINUTES to HOURS!

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 100G"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Load environment
# module load anaconda3
# conda activate your_env_name

cd /autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main

python MRFmatch_B-SL_dk.py

echo ""
echo "Job completed at: $(date)"
echo "Check usage: sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,Elapsed,State"
