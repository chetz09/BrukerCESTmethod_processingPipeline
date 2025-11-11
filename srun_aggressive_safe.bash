#!/bin/bash
#SBATCH --job-name=MRF_dict_50M
#SBATCH --partition=rtx8000,dgx-a100,rtx6000
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

# AGGRESSIVE OPTIMIZATION - Minimal resources for fastest queue
# Only uses GPU partitions likely accessible to cestmrf account
# Reduced: CPUs (18), Memory (100G), Time (4h)

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "QOS: $SLURM_JOB_QOS"
echo "Running on node: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: 100G"
echo "GPU: $CUDA_VISIBLE_DEVICES"
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
