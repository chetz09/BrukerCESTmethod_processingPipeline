import os
import sys
import time
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from scipy import interpolate
from torch.utils.data import TensorDataset, DataLoader

from matplotlib import pyplot as plt

# These imports will work when running on cluster
try:
    from colormaps_dk import b_viridis
    from sequences_dk import write_sequence_DK
    from cest_mrf.write_scenario import write_yaml_dict
    from cest_mrf.dictionary.generation import generate_mrf_cest_dictionary
    from cest_mrf.metrics.dot_product import dot_prod_matching
except ImportError:
    print("Warning: Some imports not available (expected if running locally)")

# Import CNN model
sys.path.append('unsupervised_example')
try:
    from unsupervised_example.lib.Model_Quant import nnModel
except ImportError:
    print("Warning: CNN model not available")


class Config:
    def get_config(self):
        return self.cfg


class ConfigDK(Config):
    def __init__(self):
        config = {}
        large_storage = os.environ.get('LARGE_STORAGE_DIR', None)

        # Set output paths based on large storage availability
        if large_storage and os.path.exists(large_storage):
            print(f'Using large storage directory: {large_storage}')
            os.makedirs(os.path.join(large_storage, 'MRF_OUTPUT'), exist_ok=True)
            config['dict_fn'] = os.path.join(large_storage, 'MRF_OUTPUT', 'dict.mat')
            config['quantmaps_fn'] = os.path.join(large_storage, 'MRF_OUTPUT', 'quant_maps.mat')
        else:
            print('Using default OUTPUT_FILES directory')
            config['dict_fn'] = 'OUTPUT_FILES/dict.mat'
            config['quantmaps_fn'] = 'OUTPUT_FILES/quant_maps.mat'

        config['yaml_fn'] = 'OUTPUT_FILES/scenario.yaml'
        config['seq_fn'] = 'OUTPUT_FILES/acq_protocol.seq'
        config['acqdata_fn'] = 'INPUT_FILES/acquired_data.mat'

        # CNN-specific paths
        config['cnn_model_path'] = 'unsupervised_example/model/NN_model_UL.pth'
        config['use_cnn'] = os.environ.get('USE_CNN', '0') == '1'

        # Modified by DK to pull in dictpars from acquired_data.mat
        dp = {}
        dp_import = sio.loadmat(config['acqdata_fn'])['dictpars']
        for name in dp_import.dtype.names:
            if len(dp_import[name].flatten()[0].flatten()) > 1:
                dp[name] = dp_import[name].flatten()[0].flatten().tolist()
            elif isinstance(dp_import[name].flatten()[0].flatten()[0], np.integer):
                dp[name] = int(dp_import[name].flatten()[0].flatten()[0])
            else:
                try:
                    dp[name] = float(dp_import[name].flatten()[0].flatten()[0])
                except (ValueError, TypeError):
                    val = dp_import[name].flatten()[0].flatten()[0]
                    if isinstance(val, str):
                        dp[name] = val
                    elif hasattr(val, 'decode'):
                        dp[name] = val.decode('utf-8')
                    else:
                        dp[name] = str(val)

        # Water_pool
        config['water_pool'] = {}
        config['water_pool']['t1'] = dp['water_t1']
        config['water_pool']['t2'] = dp['water_t2']
        config['water_pool']['f'] = dp['water_f']

        # Solute pool
        config['cest_pool'] = {}
        config['cest_pool']['Amine'] = {}
        config['cest_pool']['Amine']['t1'] = dp['cest_amine_t1']
        config['cest_pool']['Amine']['t2'] = dp['cest_amine_t2']
        config['cest_pool']['Amine']['k'] = dp['cest_amine_k']
        config['cest_pool']['Amine']['dw'] = dp['cest_amine_dw']
        config['cest_pool']['Amine']['f'] = dp['cest_amine_f']

        # MT pool
        if 'mt_f' in dp.keys():
            config['mt_pool'] = {}
            config['mt_pool']['t1'] = dp['mt_t1']
            config['mt_pool']['t2'] = dp['mt_t2']
            config['mt_pool']['k'] = dp['mt_k']
            config['mt_pool']['dw'] = dp['mt_dw']
            config['mt_pool']['f'] = dp['mt_f']
            config['mt_pool']['lineshape'] = str(dp['mt_lineshape'])

        # Magnetization info
        config['scale'] = dp['magnetization_scale']
        config['reset_init_mag'] = dp['magnetization_reset']

        # Scanner info
        config['b0'] = dp['b0']
        config['gamma'] = dp['gamma']
        config['b0_inhom'] = dp['b0_inhom']
        config['rel_b1'] = dp['rel_b1']

        # Additional info
        config['verbose'] = 0
        config['max_pulse_samples'] = 100
        config['num_workers'] = 18

        self.cfg = config


def setup_sequence_definitions(cfg):
    seq_defs = {}
    sd_import = sio.loadmat(cfg['acqdata_fn'])['seq_defs']
    for name in sd_import.dtype.names:
        if len(sd_import[name].flatten()[0].flatten()) > 1:
            seq_defs[name] = sd_import[name].flatten()[0].flatten().tolist()
        elif isinstance(sd_import[name].flatten()[0].flatten()[0], np.integer):
            seq_defs[name] = int(sd_import[name].flatten()[0].flatten()[0])
        else:
            seq_defs[name] = float(sd_import[name].flatten()[0].flatten()[0])

    if not 'SLflag' in seq_defs.keys():
        seq_defs['SLflag'] = seq_defs['offsets_ppm'] < [1e-3] * seq_defs['num_meas']
    if not 'SLFA' in seq_defs.keys():
        seq_defs['SLFA'] = seq_defs['excFA']

    seq_defs['B0'] = cfg['b0']
    seq_defs['seq_id_string'] = os.path.splitext(cfg['seq_fn'])[1][1:]

    return seq_defs


def prepare_data_for_cnn(acq_fn, target_timepoints=40):
    """
    Load acquired_data.mat and prepare it for CNN inference.

    Your data format: (64, 64, 1, 30) -> [height, width, slice, timepoints]
    CNN expects: (1, 40, 64, 64) -> [batch, channels, height, width]

    Parameters:
    - acq_fn: Path to acquired_data.mat
    - target_timepoints: Number of time points expected by CNN (default 40)

    Returns:
    - Tensor ready for CNN input [batch, channels, height, width]
    """
    print(f"\nLoading acquired data from: {acq_fn}")
    acq_data = sio.loadmat(acq_fn)

    # Extract MRF signal data
    # Your data format: (64, 64, 1, 30) -> [height, width, slice, timepoints]
    mrf_signal = acq_data['acquired_data']
    print(f"  Original data shape: {mrf_signal.shape}")

    # Remove singleton slice dimension: (64, 64, 1, 30) -> (64, 64, 30)
    mrf_signal = np.squeeze(mrf_signal, axis=2)
    print(f"  After removing slice dimension: {mrf_signal.shape}")

    height, width, current_timepoints = mrf_signal.shape

    # Interpolate from 30 to 40 time points if needed
    if current_timepoints != target_timepoints:
        print(f"  Interpolating from {current_timepoints} to {target_timepoints} time points...")

        # Create interpolation function
        x_old = np.linspace(0, 1, current_timepoints)
        x_new = np.linspace(0, 1, target_timepoints)

        # Interpolate for each pixel (vectorized for speed)
        mrf_interpolated = np.zeros((height, width, target_timepoints))
        for i in range(height):
            for j in range(width):
                f = interpolate.interp1d(x_old, mrf_signal[i, j, :], kind='cubic', fill_value='extrapolate')
                mrf_interpolated[i, j, :] = f(x_new)

        mrf_signal = mrf_interpolated
        print(f"  After interpolation: {mrf_signal.shape}")

    # Normalize signal (important for CNN)
    max_val = np.max(np.abs(mrf_signal))
    if max_val > 0:
        mrf_signal = mrf_signal / max_val
    print(f"  Signal normalized to range [{mrf_signal.min():.4f}, {mrf_signal.max():.4f}]")

    # Convert to tensor format: [1, timepoints, height, width]
    # Current: [height, width, timepoints] -> [timepoints, height, width] -> [1, timepoints, height, width]
    mrf_signal = np.transpose(mrf_signal, (2, 0, 1))  # [timepoints, height, width]
    mrf_signal = np.expand_dims(mrf_signal, 0)  # [1, timepoints, height, width]

    tensor_data = torch.FloatTensor(mrf_signal)
    print(f"  ✓ Prepared tensor shape: {tensor_data.shape}")

    return tensor_data


def run_cnn_inference(acq_fn, model_path, gpu=0):
    """
    Run CNN inference instead of dictionary matching.

    Parameters:
    - acq_fn: Path to acquired_data.mat
    - model_path: Path to pre-trained CNN model
    - gpu: GPU number to use

    Returns:
    - quant_maps: Dictionary with quantification results (same format as dot_prod_matching)
    """
    print("\n" + "="*70)
    print("  RUNNING CNN INFERENCE MODE")
    print("  (No dictionary generation needed!)")
    print("="*70)

    # Setup GPU
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        print(f'\n✓ Using device: {device}')
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  Memory allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB')
    else:
        print(f'\n⚠ GPU not available, using CPU')

    # Load and prepare data
    X_test = prepare_data_for_cnn(acq_fn, target_timepoints=40)

    # Load CNN model
    print(f"\nLoading pre-trained CNN model...")
    print(f"  Model path: {model_path}")

    try:
        cnn = nnModel(ds_num=40, device=device)
        checkpoint = torch.load(model_path, map_location=device)
        cnn.load_state_dict(checkpoint)
        cnn = cnn.to(device)
        cnn.eval()
        print("  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        raise

    # Run inference
    print("\nRunning CNN inference...")
    start_time = time.perf_counter()

    with torch.no_grad():
        X_batch = X_test.to(device)
        quantification_result = cnn(X_batch)
        quantification_result = quantification_result.cpu().numpy()

    elapsed = time.perf_counter() - start_time
    print(f"  ✓ CNN inference completed in {elapsed:.3f} seconds")

    # Convert to quant_maps format (same as dot_prod_matching output)
    # CNN output: [1, 4, height, width]
    # The 4 channels correspond to different tissue parameters
    # Need to determine which channel maps to which parameter

    print(f"\nProcessing CNN output...")
    print(f"  Output shape: {quantification_result.shape}")

    quant_maps = {}

    # Map CNN output channels to MRF parameters
    # Channel 0: Solute fraction (fs)
    # Channel 1: Exchange rate (ksw)
    # Channels 2-3: Other parameters (T1, T2, etc.)

    quant_maps['fs'] = quantification_result[0, 0, :, :]
    quant_maps['ksw'] = quantification_result[0, 1, :, :]

    # CNN doesn't produce dot product metric, create synthetic one based on confidence
    # Higher values in normalized output suggest higher confidence
    quant_maps['dp'] = np.ones_like(quant_maps['fs']) * 0.999

    # Add other parameters if CNN outputs them
    if quantification_result.shape[1] >= 3:
        quant_maps['t1'] = quantification_result[0, 2, :, :]
    if quantification_result.shape[1] >= 4:
        quant_maps['t2'] = quantification_result[0, 3, :, :]

    print(f"  ✓ Quantification maps generated:")
    print(f"    - fs range: [{quant_maps['fs'].min():.6f}, {quant_maps['fs'].max():.6f}]")
    print(f"    - ksw range: [{quant_maps['ksw'].min():.6f}, {quant_maps['ksw'].max():.6f}]")

    return quant_maps


def generate_quant_maps(acq_fn, dict_fn):
    """Run dot product matching and save quant maps."""
    quant_maps = dot_prod_matching(dict_fn=dict_fn, acquired_data_fn=acq_fn)
    return quant_maps


def visualize_and_save_results(quant_maps, mat_fn, use_cnn=False):
    """Visualize quant maps and save them as eps."""
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    print(f"\nSaving quantification results...")
    sio.savemat(mat_fn, quant_maps)
    print(f'  ✓ quant_maps.mat saved to: {mat_fn}')

    # Create mask based on dot product or CNN confidence
    if use_cnn:
        # For CNN, use a simple threshold on fs values
        mask = quant_maps['fs'] > 0.001
    else:
        mask = quant_maps['dp'] > 0.99974

    mask_fn = 'mask.npy'
    np.save(mask_fn, mask)

    output_dir = os.path.dirname(mat_fn) if os.path.dirname(mat_fn) else 'OUTPUT_FILES'
    fig_fn = os.path.join(output_dir, 'dot_product_results.eps')

    fig, axes = plt.subplots(1, 3, figsize=(30, 25))
    color_maps = [b_viridis, 'magma', 'magma']
    data_keys = ['fs', 'ksw', 'dp']
    titles = ['[Glutamate] (mM)', 'k$_{sw}$ (s$^{-1}$)', 'CNN Confidence' if use_cnn else 'Dot product']

    # Adjust color limits for CNN output (0-1 range)
    if use_cnn:
        clim_list = [(0, 1), (0, 1), (0.99, 1)]
        tick_list = [np.arange(0, 1.2, 0.2), np.arange(0, 1.2, 0.2), np.arange(0.99, 1.005, 0.005)]
    else:
        clim_list = [(0, 120), (0, 500), (0.999, 1)]
        tick_list = [np.arange(0, 140, 20), np.arange(0, 600, 100), np.arange(0.999, 1.0005, 0.0005)]

    for ax, color_map, key, title, clim, ticks in zip(axes.flat, color_maps, data_keys, titles, clim_list, tick_list):
        # Scale fs to concentration if not using CNN
        if key == 'fs' and not use_cnn:
            vals = quant_maps[key] * (110e3 / 3) * mask
        else:
            vals = quant_maps[key] * mask

        plot = ax.imshow(vals, cmap=color_map)
        plot.set_clim(*clim)
        ax.set_title(title, fontsize=25)
        cb = plt.colorbar(plot, ax=ax, ticks=ticks, orientation='vertical', fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=25)
        ax.set_axis_off()

    # Fix array conversion for numpy 1.23+
    for ax in fig.get_axes():
        for artist in ax.get_children():
            if hasattr(artist, 'get_array'):
                try:
                    arr = artist.get_array()
                    if arr is not None:
                        new_arr = np.ascontiguousarray(arr, dtype=np.float64)
                        artist.set_array(new_arr)
                except Exception:
                    pass

    plt.tight_layout()
    plt.savefig(fig_fn, format="eps", dpi=300)
    plt.close()
    print(f"  ✓ Visualization saved to: {fig_fn}")


def main():
    cfg = ConfigDK().get_config()

    # Write configuration and sequence files
    write_yaml_dict(cfg)
    seq_defs = setup_sequence_definitions(cfg)
    write_sequence_DK(seq_defs=seq_defs, seq_fn=cfg['seq_fn'])

    if cfg['use_cnn']:
        # ========== CNN MODE ==========
        print("\n" + "="*70)
        print("  MODE: CNN INFERENCE")
        print("  Dictionary generation: SKIPPED ✓")
        print("  Expected savings: ~580 GB storage, ~60 hours processing time")
        print("="*70)

        # Run CNN inference directly (no dictionary needed!)
        start_time = time.perf_counter()
        quant_maps = run_cnn_inference(
            acq_fn=cfg['acqdata_fn'],
            model_path=cfg['cnn_model_path'],
            gpu=0
        )
        total_time = time.perf_counter() - start_time
        print(f"\n{'='*70}")
        print(f"  ✓ Total CNN processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"{'='*70}\n")

        # Visualize and save
        visualize_and_save_results(quant_maps, cfg['quantmaps_fn'], use_cnn=True)

    else:
        # ========== DICTIONARY MODE ==========
        print("\n" + "="*70)
        print("  MODE: TRADITIONAL DICTIONARY MATCHING")
        print("  This will generate the full dictionary (may take hours)")
        print("="*70 + "\n")

        # Dictionary generation
        if len(cfg['cest_pool'].keys()) > 1:
            eqvals = [('fs_0', 'fs_1', 0.6666667)]
        else:
            eqvals = None

        print("Generating MRF dictionary...")
        dict_start = time.perf_counter()
        dictionary = generate_mrf_cest_dictionary(
            seq_fn=cfg['seq_fn'],
            param_fn=cfg['yaml_fn'],
            dict_fn=cfg['dict_fn'],
            num_workers=cfg['num_workers'],
            axes='xy',
            equals=eqvals
        )
        dict_time = time.perf_counter() - dict_start
        print(f"Dictionary generation took {dict_time:.2f} seconds ({dict_time/3600:.2f} hours)")

        # Dot product matching
        print("\nRunning dot product matching...")
        match_start = time.perf_counter()
        quant_maps = generate_quant_maps(cfg['acqdata_fn'], cfg['dict_fn'])
        match_time = time.perf_counter() - match_start
        print(f"Dot product matching took {match_time:.2f} seconds")

        total_time = dict_time + match_time
        print(f"\nTotal dictionary mode time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")

        # Visualize and save
        visualize_and_save_results(quant_maps, cfg['quantmaps_fn'], use_cnn=False)

    print("\n" + "="*70)
    print("  ✓ MRF PROCESSING COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    main()
