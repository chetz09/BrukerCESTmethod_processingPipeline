# MRF Processing with Neural Network Integration

This document describes the integrated MRF processing pipeline that supports both traditional dictionary matching and neural network-based inference.

## Overview

The integrated system provides two processing modes:

1. **Dictionary Matching Mode** (Traditional): Generates a full dictionary and performs dot product matching
2. **Neural Network Mode** (Fast): Uses a fully connected neural network for direct parameter estimation

The neural network approach is particularly useful when:
- Dictionary size becomes very large (causing memory issues)
- Fast inference is required
- GPU acceleration is available

## Architecture

### Neural Network Architecture

The network is a **3-layer fully connected network (MLP)**, not a CNN:

```
Input: Signal vector (e.g., 30 measurements)
  ↓
Linear Layer 1: sig_n → 300
  ↓
ReLU Activation
  ↓
Linear Layer 2: 300 → 300
  ↓
ReLU Activation
  ↓
Linear Layer 3: 300 → 2 (fs, ksw)
  ↓
Output: Concentration (fs) and Exchange Rate (ksw)
```

### Key Files

| File | Description |
|------|-------------|
| `MRFmatch_integrated.py` | Main integrated processing script (replaces `MRFmatch_B-SL_dk.py`) |
| `train_mrf_network.py` | Standalone script for pre-training the neural network |
| `run_mrf_cnn_cluster.sh` | SLURM job script for cluster processing with GPU |
| `deep_reco_example/model.py` | Neural network architecture definition |
| `deep_reco_example/dataset.py` | PyTorch dataset for MRF dictionary |
| `deep_reco_example/configs.py` | Configuration templates |

## Usage

### Method 1: Local Processing

#### Option A: Dictionary Matching (Traditional)

```bash
cd Python
export USE_CNN=0
python MRFmatch_integrated.py
```

#### Option B: Neural Network Inference

```bash
cd Python

# Train the network (first time only)
python train_mrf_network.py --epochs 100 --batch-size 512

# Run inference with the trained model
export USE_CNN=1
export MODEL_PATH=OUTPUT_FILES/trained_model.pt
python MRFmatch_integrated.py
```

### Method 2: Cluster Processing (via MATLAB)

The MATLAB function `MRFmatch_4.m` now supports CNN mode through the `dirstruct.use_cnn` flag.

#### Setting up in MATLAB:

```matlab
% In your MATLAB script:
dirstruct.use_cluster = true;  % Enable cluster processing
dirstruct.use_cnn = true;      % Enable CNN mode
dirstruct.num_gpus = 1;        % Number of GPUs to use
dirstruct.cluster_job_script_cnn = 'run_mrf_cnn_cluster.sh';

% Call MRF processing
MRFmatch_4(dirstruct, prefs, PV360flg);
```

#### For traditional dictionary matching on cluster:

```matlab
dirstruct.use_cluster = true;
dirstruct.use_cnn = false;     % Use dictionary matching
dirstruct.cluster_job_script = 'run_mrf_cluster.sh';  % Your existing script

MRFmatch_4(dirstruct, prefs, PV360flg);
```

### Method 3: Direct Cluster Submission

```bash
# Submit CNN processing job
sbatch --export=ALL,USE_CNN=1,NUM_GPUS=1 run_mrf_cnn_cluster.sh

# Check job status
squeue -u $USER

# Check output
tail -f slurm_mrf_cnn_*.out
```

## Training the Neural Network

### Quick Start

```bash
python train_mrf_network.py
```

This will:
1. Load configuration from `INPUT_FILES/acquired_data.mat`
2. Generate a training dictionary
3. Train the network (with early stopping)
4. Save the trained model to `OUTPUT_FILES/trained_model.pt`
5. Generate a training loss curve

### Advanced Training Options

```bash
python train_mrf_network.py \
    --batch-size 1024 \
    --epochs 200 \
    --lr 0.0005 \
    --noise-std 0.003 \
    --patience 15 \
    --output custom_model.pt \
    --seed 42
```

#### Training Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch-size` | 512 | Training batch size |
| `--epochs` | 100 | Maximum number of epochs |
| `--lr` | 0.0003 | Learning rate for Adam optimizer |
| `--noise-std` | 0.002 | Noise level added during training |
| `--patience` | 10 | Early stopping patience (epochs) |
| `--output` | `OUTPUT_FILES/trained_model.pt` | Output path for trained model |
| `--seed` | 2024 | Random seed for reproducibility |

### Training Tips

1. **Dictionary Size**: The network trains on a simulated dictionary. Larger dictionaries → better generalization but longer training time.

2. **Noise Level**: Adding noise during training (`--noise-std`) improves robustness to measurement noise.

3. **Early Stopping**: Training stops automatically when loss plateaus, preventing overfitting.

4. **GPU Usage**: Training automatically uses GPU if available. Check with:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_CNN` | 0 | Set to 1 for neural network mode, 0 for dictionary matching |
| `NUM_GPUS` | 1 | Number of GPUs to use for processing |
| `NUM_WORKERS` | 18 | Number of CPU workers for dictionary generation |
| `MODEL_PATH` | `OUTPUT_FILES/trained_model.pt` | Path to trained model file |
| `LARGE_STORAGE_DIR` | - | Optional path to large storage directory |

## Output Files

### Standard Outputs

| File | Description |
|------|-------------|
| `OUTPUT_FILES/quant_maps.mat` | Quantitative parameter maps (fs, ksw, dp*) |
| `OUTPUT_FILES/quant_maps_results.eps` | Visualization of results |
| `OUTPUT_FILES/mask.npy` | Mask for valid pixels |
| `OUTPUT_FILES/scenario.yaml` | Generated scenario configuration |
| `OUTPUT_FILES/acq_protocol.seq` | Acquisition protocol sequence |
| `OUTPUT_FILES/dict.mat` | Generated dictionary (if applicable) |

*Note: `dp` (dot product) is only available in dictionary matching mode.

### Model Files (Neural Network Mode)

| File | Description |
|------|-------------|
| `OUTPUT_FILES/trained_model.pt` | Trained neural network checkpoint |
| `OUTPUT_FILES/trained_model_training_curve.png` | Training loss curve |

## Performance Comparison

### Dictionary Matching Mode
- **Pros**:
  - Accurate (theoretical optimal)
  - Provides dot product quality metric
- **Cons**:
  - Memory intensive (OOM for large dictionaries)
  - Slow for large dictionaries
  - CPU bound

### Neural Network Mode
- **Pros**:
  - Fast inference (~10-100x faster)
  - Fixed memory footprint
  - GPU acceleration
  - No OOM issues
- **Cons**:
  - Requires training step
  - Approximation (not theoretical optimal)
  - No built-in quality metric

### Typical Performance

| Mode | Dictionary Size | Processing Time | GPU Memory |
|------|----------------|-----------------|------------|
| Dict Matching | 100K entries | ~30-60 sec | N/A |
| Dict Matching | 1M entries | ~5-10 min | N/A |
| Dict Matching | 10M entries | OOM / Very slow | N/A |
| Neural Network | Any size | ~1-5 sec | ~2-4 GB |

*Times are approximate and depend on hardware and dataset size.

## Troubleshooting

### Issue: "CUDA out of memory"

**For Dictionary Mode:**
```bash
# Switch to neural network mode
export USE_CNN=1
python MRFmatch_integrated.py
```

**For Neural Network Training:**
```bash
# Reduce batch size
python train_mrf_network.py --batch-size 256
```

### Issue: "Model file not found"

Train the network first:
```bash
python train_mrf_network.py
```

Or specify correct path:
```bash
export MODEL_PATH=/path/to/your/model.pt
python MRFmatch_integrated.py
```

### Issue: "Dictionary generation too slow"

Increase number of workers:
```bash
export NUM_WORKERS=32
python MRFmatch_integrated.py
```

### Issue: "Poor reconstruction quality"

1. **Check training loss**: Should converge to < 0.001
2. **Retrain with more epochs**:
   ```bash
   python train_mrf_network.py --epochs 200 --patience 20
   ```
3. **Increase dictionary coverage**: Modify parameter ranges in `acquired_data.mat`

## Integration with Existing Pipeline

### Changes to MRFmatch_4.m

The MATLAB function now supports CNN mode (lines 103-112):

```matlab
if isfield(dirstruct,'use_cnn') && dirstruct.use_cnn
    fprintf('  MODE: CNN INFERENCE (Fast, No OOM!)\n');
    fprintf('  Using %d GPU(s)\n', dirstruct.num_gpus);
    job_script = dirstruct.cluster_job_script_cnn;
    use_cnn_flag = 1;
else
    fprintf('  MODE: TRADITIONAL DICTIONARY MATCHING\n');
    job_script = dirstruct.cluster_job_script;
    use_cnn_flag = 0;
end
```

### Backward Compatibility

- Existing pipelines using `MRFmatch_B-SL_dk.py` continue to work
- `MRFmatch_integrated.py` can replace `MRFmatch_B-SL_dk.py` with `USE_CNN=0`
- All output formats remain the same

## Example Workflows

### Workflow 1: First-time User (Local)

```bash
# 1. Prepare acquired data (done by MATLAB)
# 2. Generate and train network
cd Python
python train_mrf_network.py

# 3. Run inference
export USE_CNN=1
python MRFmatch_integrated.py

# 4. Results are in OUTPUT_FILES/
ls -lh OUTPUT_FILES/quant_maps.mat
```

### Workflow 2: Regular Processing (Cluster)

```matlab
% In MATLAB:
dirstruct.use_cluster = true;
dirstruct.use_cnn = true;
dirstruct.num_gpus = 1;
dirstruct.cluster_job_script_cnn = 'run_mrf_cnn_cluster.sh';

% Process entire study
PROC_MRF_STUDY(dirstruct, prefs, PV360flg);
```

### Workflow 3: Compare Methods

```bash
# Run both methods and compare
export USE_CNN=0
python MRFmatch_integrated.py
mv OUTPUT_FILES/quant_maps.mat OUTPUT_FILES/quant_maps_dict.mat

export USE_CNN=1
python MRFmatch_integrated.py
mv OUTPUT_FILES/quant_maps.mat OUTPUT_FILES/quant_maps_cnn.mat

# Compare in MATLAB or Python
```

## Requirements

### Python Packages

```
torch >= 1.9.0
numpy >= 1.19.0
scipy >= 1.5.0
matplotlib >= 3.3.0
```

### Hardware Requirements

**Minimum:**
- CPU: 8+ cores
- RAM: 32 GB
- GPU: Optional (enables neural network mode)

**Recommended for Neural Network:**
- GPU: NVIDIA GPU with 8+ GB VRAM
- CUDA: 11.0 or higher
- cuDNN: 8.0 or higher

## References

- Original MRF-CEST method: [relevant paper]
- Neural network approach based on: Cohen et al., "MR fingerprinting Deep RecOnstruction NEtwork (DRONE)", Magn Reson Med, 2018

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review SLURM output logs: `slurm_mrf_cnn_*.out` and `slurm_mrf_cnn_*.err`
3. Contact: [your contact info]

## License

[Your license information]
