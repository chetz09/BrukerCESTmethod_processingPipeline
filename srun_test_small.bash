#!/bin/bash
#SBATCH --job-name=MRF_test_small
#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.out
#SBATCH --error=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cd1052

# TEST VERSION - Very small resources for quick testing
# Use this to:
# 1. Test if your script runs correctly
# 2. Monitor actual resource usage
# 3. Get through queue quickly
#
# After running, check memory usage with:
# sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,Elapsed

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: 32G"
echo ""
echo "NOTE: This is a test run with reduced workers"
echo "Modify num_workers in your Python config for this test"
echo ""

# Load your modules/conda environment here
# Example:
# module load anaconda3
# conda activate your_env_name

# Change to working directory
cd /autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main

# Run your Python script with modified config
# You'll need to modify the Python code to use fewer workers for testing
# Or pass num_workers as an argument if your script supports it
python MRFmatch_B-SL_dk.py

# Print completion info
echo ""
echo "Job completed at: $(date)"
echo "Check resource usage with: sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,Elapsed"
