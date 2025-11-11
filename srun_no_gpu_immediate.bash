#!/bin/bash
#SBATCH --job-name=MRF_dict_50M
#SBATCH --partition=basic,west,pubcpu,lcncpu
#SBATCH --account=cestmrf
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.out
#SBATCH --error=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cd1052

# NO GPU - IMMEDIATE SCHEDULING
# Targets CPU-only partitions with IDLE nodes:
# - basic: 7 idle nodes
# - west: 1 idle node
# - lcncpu, pubcpu: additional options
#
# Should schedule IMMEDIATELY or within MINUTES!
# Only use if your Python code doesn't need GPU

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 100G"
echo ""

# Load environment
# module load anaconda3
# conda activate your_env_name

cd /autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main

python MRFmatch_B-SL_dk.py

echo ""
echo "Job completed at: $(date)"
echo "Check usage: sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,Elapsed,State"
