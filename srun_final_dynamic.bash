#!/bin/bash
#SBATCH --job-name=MRF_dict_50M
#SBATCH --account=cestmrf
#SBATCH --qos=normal
#SBATCH --partition=pubgpu-req,pubgpu,rtx6000,rtx8000,dgx-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.out
#SBATCH --error=/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cd1052

# DYNAMIC VERSION - Automatically uses available CPUs
# Modifies Python config on-the-fly to match allocated CPUs
# No need to manually edit Python code!

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

# Temporarily modify Python code to use available CPUs
echo "Dynamically setting num_workers to $SLURM_CPUS_PER_TASK"
sed -i.bak "s/config\['num_workers'\] = [0-9]\+/config['num_workers'] = $SLURM_CPUS_PER_TASK/" MRFmatch_B-SL_dk.py

# Run Python
python MRFmatch_B-SL_dk.py

# Restore original file
mv MRFmatch_B-SL_dk.py.bak MRFmatch_B-SL_dk.py

echo ""
echo "Job completed at: $(date)"
echo "Check usage: sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,Elapsed,State"
