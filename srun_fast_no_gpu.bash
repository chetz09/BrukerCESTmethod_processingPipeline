#!/bin/bash
#SBATCH --job-name=MRF_dict_50M
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=125G
#SBATCH --time=06:00:00
#SBATCH --output=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.out
#SBATCH --error=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cd1052

# FASTEST VERSION - No GPU, targets IDLE nodes in basic partition
# Should schedule IMMEDIATELY or within minutes!
# Use this if your job doesn't actually require GPU

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: 125G"
echo ""

# Load your modules/conda environment here
# module load anaconda3
# conda activate your_env_name

# Change to working directory
cd /autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main

# Run your Python script
python MRFmatch_B-SL_dk.py

# Print completion info
echo ""
echo "Job completed at: $(date)"
echo "Check resource usage with: sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,Elapsed,State"
