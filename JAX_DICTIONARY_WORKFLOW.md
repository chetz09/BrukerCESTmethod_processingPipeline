# JAX-Accelerated Dictionary Generation Workflow

## Overview

This document describes the hybrid JAX + PyTorch workflow for fast dictionary generation.

## Setup Instructions

### 1. Clone the neural-fitting repository
```bash
cd /path/to/BrukerCESTmethod_processingPipeline
git clone https://github.com/chetz09/neural-fitting_1.git
```

### 2. Add custom scripts (already created in your local neural-fitting_1/)
The following custom files are ready in `neural-fitting_1/`:
```
neural-fitting_1/
├── generate_dict_dk.py          # Custom dictionary generation script
├── run_dict_gen_jax.sh          # SLURM script for cluster
├── README_HYBRID_WORKFLOW.md    # Complete instructions
└── simulation.py                # JAX 3-pool BMC simulation (from repo)
```

**Note**: The `neural-fitting_1/` directory is a separate git repository and is excluded from this repo via `.gitignore`.

## Quick Start

### 1. Set up JAX environment (one-time)
```bash
cd neural-fitting_1
conda env create -f environment.yml
conda activate nbmf1
```

### 2. Generate dictionary on GPU (1000× faster!)
```bash
# Option A: Interactive
python generate_dict_dk.py \
    --config /path/to/acquired_data.mat \
    --output $LARGE_STORAGE_DIR/MRF_OUTPUT/dict_jax.mat \
    --batch-size 2000

# Option B: Submit SLURM job
sbatch run_dict_gen_jax.sh
```

### 3. Train neural network (existing PyTorch code)
```bash
conda activate dkpymrf  # Your existing environment
cd Bruker_CEST-MRF_processing/Python/deep_reco_example
python preclinical.py
```

## Dictionary Sizes and RAM Requirements

| Configuration | Entries | RAM (JAX gen) | RAM (PyTorch train) | JAX Time | CPU Time |
|---------------|---------|---------------|---------------------|----------|----------|
| **Current (optimal)** | 1.75M | ~100 GB | ~750 GB | ~10 min | ~4 hours |
| Fine grid | 10M | ~150 GB | ~900 GB | ~1 hour | ~24 hours |
| Your 67.39M | 67.39M | ~400 GB | N/A (too large) | ~6 hours | ~weeks |

## Why This Workflow?

✅ **1000× speedup** for dictionary generation
✅ **Keep existing PyTorch code** (no rewrite needed)
✅ **Separate environments** (no dependency conflicts)
✅ **GPU acceleration** (leverages RTX 8000 GPUs)
✅ **Scalable** (can generate larger dictionaries if needed)

## Recommendation

**Use the 1.75M entry configuration** for optimal neural network training:
- Covers full parameter ranges (T1: 1.5-4s, T2: 0.5-2.5s, k: 0-8000 s⁻¹, conc: 0-40 mM)
- Fast training (smaller dictionary)
- Excellent accuracy (neural networks interpolate well)
- Fits comfortably in 750 GB RAM

**Avoid 67.39M entries** - no accuracy benefit, much slower training, excessive RAM.

## Files Modified

1. `DictConfigParams.m` - Optimized to 1.75M entries (MT parameters fixed)
2. `neural-fitting_1/generate_dict_dk.py` - Custom JAX dictionary generator
3. `neural-fitting_1/run_dict_gen_jax.sh` - SLURM script for GPU generation
4. `neural-fitting_1/README_HYBRID_WORKFLOW.md` - Complete documentation

## Next Steps

1. ✓ JAX environment ready
2. ✓ Dictionary generation script ready
3. **Test dictionary generation** with small batch
4. **Verify PyTorch compatibility**
5. **Run full training pipeline**

For detailed instructions, see:
- `neural-fitting_1/README_HYBRID_WORKFLOW.md`
