#!/bin/bash
#SBATCH --job-name=MRF_dict_50M
#SBATCH --partition=pubgpu,pubgpu-req,rtx6000,rtx8000,dgx-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus=1
#SBATCH --mem=125G
#SBATCH --time=06:00:00
#SBATCH --output=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.out
#SBATCH --error=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cd1052

# FAST VERSION with GPU - Targets partitions with idle/available GPU nodes
# Includes pubgpu and pubgpu-req which have idle nodes (hours not days!)
# Reduced time to 6 hours for faster scheduling

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: 125G"
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
