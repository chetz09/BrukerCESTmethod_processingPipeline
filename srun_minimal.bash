#!/bin/bash
#SBATCH --job-name=MRF_dict_50M_minimal
#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.out
#SBATCH --error=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cd1052

# MINIMAL RESOURCE VERSION - Fastest queue time
# Changes from original:
# - Reduced CPUs from 32 to 18 (exactly matches num_workers in Python)
# - Reduced memory from 512G to 64G
# - Reduced time from 6h to 4h
# - Test this first to see if it completes successfully

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: 64G"
echo ""

# Load your modules/conda environment here
# Example:
# module load anaconda3
# conda activate your_env_name

# Change to working directory
cd /autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main

# Run your Python script
# Replace with your actual command
python MRFmatch_B-SL_dk.py

# Print completion info
echo ""
echo "Job completed at: $(date)"
