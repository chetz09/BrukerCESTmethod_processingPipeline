# Cluster Configuration Guide

This document explains the updated configuration for running MRF processing on the cluster with large storage support.

## Overview

The system now supports:
1. **Cluster processing** with SLURM job submission
2. **Large storage directory** (1TB) for saving dict.mat and quant_maps.mat files
3. **Automatic file transfer** between local machine, cluster, and large storage

## Configuration Changes

### 1. initUserSettings.m

Updated paths and added cluster configuration:

```matlab
% Python directory on cluster
configs.py_dir='/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main';

% Large storage directory (1TB) for dict.mat and quant_maps.mat
configs.large_storage_dir='/autofs/vast/farrar/users/cd1052';

% Cluster configuration
configs.use_cluster = true;  % Set to false for local processing
configs.cluster_user = 'cd1052';
configs.cluster_host = 'mlsc.nmr.mgh.harvard.edu';
configs.cluster_dir = '/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main';
configs.cluster_job_script = 'srun.bash';

% Bash configuration file
configs.bashfn='.bashrc';
```

### 2. Python Script (MRFmatch_B-SL_dk.py)

Modified to check for `LARGE_STORAGE_DIR` environment variable:

- If set: Saves dict.mat and quant_maps.mat to `$LARGE_STORAGE_DIR/MRF_OUTPUT/`
- If not set: Uses default `OUTPUT_FILES/` directory

### 3. MRFmatch.m

Enhanced with cluster support and large storage integration:

**Cluster Mode:**
1. Transfers acquired_data.mat to cluster
2. Submits SLURM job with `LARGE_STORAGE_DIR` environment variable
3. Monitors job completion
4. Downloads results from large storage or OUTPUT_FILES
5. Moves results to final data directory

**Local Mode:**
- Sets `LARGE_STORAGE_DIR` environment variable before running Python
- Python automatically uses large storage if available

## Directory Structure

### On Cluster

```
/autofs/homes/001/cd1052/molecular-mrf-main/molecular-mrf-main/
├── INPUT_FILES/
│   └── acquired_data.mat           (transferred from local)
├── OUTPUT_FILES/
│   ├── scenario.yaml
│   └── acq_protocol.seq
├── MRFmatch_B-SL_dk.py
└── srun.bash                        (SLURM job script)

/autofs/vast/farrar/users/cd1052/
└── MRF_OUTPUT/
    ├── dict.mat                     (large file, saved here)
    ├── quant_maps.mat               (saved here)
    └── dot_product_results.eps      (saved here)
```

## Workflow

1. **MATLAB generates** acquired_data.mat from Bruker 2dseq file
2. **MATLAB transfers** acquired_data.mat to cluster INPUT_FILES/
3. **MATLAB submits** SLURM job with `LARGE_STORAGE_DIR=/autofs/vast/farrar/users/cd1052`
4. **Python reads** acquired_data.mat from INPUT_FILES/
5. **Python saves**:
   - Small files (scenario.yaml, acq_protocol.seq) → OUTPUT_FILES/
   - Large files (dict.mat, quant_maps.mat) → $LARGE_STORAGE_DIR/MRF_OUTPUT/
6. **MATLAB monitors** job completion
7. **MATLAB downloads** results from large storage back to local OUTPUT_FILES/
8. **MATLAB moves** results to final data directory

## Storage Savings

By saving dict.mat and quant_maps.mat to the 1TB storage location:
- **dict.mat**: Dictionary files can be 100s of MB to several GB
- **quant_maps.mat**: Quantification maps typically 10-100 MB
- **Total savings**: Potentially several GB per dataset

## SLURM Job Script

Your `srun.bash` script on the cluster should activate the conda environment and run Python. Example:

```bash
#!/bin/bash
#SBATCH --job-name=mrf_match
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G

# The LARGE_STORAGE_DIR environment variable is set by MATLAB when submitting the job
# and will be automatically available to Python

source ~/.bashrc
conda activate mrfmatch
source dkpymrf/bin/activate

python MRFmatch_B-SL_dk.py
```

## Testing

To test the configuration:

1. **Local mode** (set `configs.use_cluster = false` in initUserSettings.m):
   - Will use large storage if `configs.large_storage_dir` is set

2. **Cluster mode** (set `configs.use_cluster = true`):
   - Will transfer files to cluster
   - Submit job with environment variable
   - Download from large storage

## Troubleshooting

### Files not found in large storage
- Check that `LARGE_STORAGE_DIR` environment variable is set
- Verify the directory exists and has write permissions: `/autofs/vast/farrar/users/cd1052`

### SLURM job fails
- Check job status: `ssh cd1052@mlsc.nmr.mgh.harvard.edu "squeue -u cd1052"`
- View job output: `ssh cd1052@mlsc.nmr.mgh.harvard.edu "cat slurm-*.out"`

### SCP transfer fails
- Verify SSH key authentication is set up
- Test manual connection: `ssh cd1052@mlsc.nmr.mgh.harvard.edu`

## Notes

- The system automatically creates `MRF_OUTPUT/` directory in large storage if it doesn't exist
- Old result files are cleaned before each run to prevent stale data
- Downloaded files are verified for existence and size
- The system falls back to OUTPUT_FILES if large storage is not configured
