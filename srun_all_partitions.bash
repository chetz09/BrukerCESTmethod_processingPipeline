#!/bin/bash
#SBATCH --job-name=MRF_dict_50M
#SBATCH --partition=pubgpu-req,pubgpu,basic,west,rtx6000,rtx8000,dgx-a100,lcnrtx,lcnv100,lcna40,lcna100
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

# ALL PARTITIONS - MAXIMUM FLEXIBILITY
# Searches ALL accessible partitions, prioritizing idle ones first
# Idle partitions listed first for fastest scheduling
#
# May or may not get GPU depending on availability
# Should schedule within MINUTES to HOURS!

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 100G"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "GPU: $CUDA_VISIBLE_DEVICES"
else
    echo "GPU: None (running on CPU-only node)"
fi
echo ""

# Load environment
# module load anaconda3
# conda activate your_env_name

cd /autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main

python MRFmatch_B-SL_dk.py

echo ""
echo "Job completed at: $(date)"
echo "Check usage: sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,Elapsed,State"
