#!/usr/bin/env python3
"""
Integrated MRF Processing Script
Supports both traditional dictionary matching and CNN-based inference.

This script can be run in two modes:
1. Dictionary Matching Mode (default): Generate dictionary and perform dot product matching
2. CNN Inference Mode: Use a pre-trained neural network for fast parameter estimation

The mode is controlled by the USE_CNN environment variable.
"""

import os
import sys
import time
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

# Import required modules
from colormaps_dk import b_viridis
from sequences_dk import write_sequence_DK
from cest_mrf.write_scenario import write_yaml_dict
from cest_mrf.dictionary.generation import generate_mrf_cest_dictionary
from cest_mrf.metrics.dot_product import dot_prod_matching

# Import deep learning components
from deep_reco_example.model import Network
from deep_reco_example.dataset import DatasetMRF
from utils.normalization import normalize_range, un_normalize_range
from utils.seed import set_seed


class Config:
    def get_config(self):
        return self.cfg


class ConfigDK(Config):
    """Configuration class that reads parameters from acquired_data.mat"""
    def __init__(self):
        config = {}
        config['yaml_fn'] = 'OUTPUT_FILES/scenario.yaml'
        config['seq_fn'] = 'OUTPUT_FILES/acq_protocol.seq'
        config['dict_fn'] = 'OUTPUT_FILES/dict.mat'
        config['acqdata_fn'] = 'INPUT_FILES/acquired_data.mat'
        config['quantmaps_fn'] = 'OUTPUT_FILES/quant_maps.mat'

        # Load dictionary parameters from acquired_data.mat
        dp = {}
        dp_import = sio.loadmat(config['acqdata_fn'])['dictpars']
        for name in dp_import.dtype.names:
            if len(dp_import[name].flatten()[0].flatten()) > 1:
                dp[name] = dp_import[name].flatten()[0].flatten().tolist()
            elif isinstance(dp_import[name].flatten()[0].flatten()[0], np.integer):
                dp[name] = int(dp_import[name].flatten()[0].flatten()[0])
            else:
                dp[name] = float(dp_import[name].flatten()[0].flatten()[0])

        # Water pool
        config['water_pool'] = {}
        config['water_pool']['t1'] = dp['water_t1']
        config['water_pool']['t2'] = dp['water_t2']
        config['water_pool']['f'] = dp['water_f']

        # Solute pool (Amine)
        config['cest_pool'] = {}
        config['cest_pool']['Amine'] = {}
        config['cest_pool']['Amine']['t1'] = dp['cest_amine_t1']
        config['cest_pool']['Amine']['t2'] = dp['cest_amine_t2']
        config['cest_pool']['Amine']['k'] = dp['cest_amine_k']
        config['cest_pool']['Amine']['dw'] = dp['cest_amine_dw']
        config['cest_pool']['Amine']['f'] = dp['cest_amine_f']

        # MT pool (if present)
        if 'mt_f' in dp.keys():
            config['mt_pool'] = {}
            config['mt_pool']['t1'] = dp['mt_t1']
            config['mt_pool']['t2'] = dp['mt_t2']
            config['mt_pool']['k'] = dp['mt_k']
            config['mt_pool']['dw'] = dp['mt_dw']
            config['mt_pool']['f'] = dp['mt_f']
            config['mt_pool']['lineshape'] = dp['mt_lineshape']

        # Initial magnetization
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
        config['num_workers'] = int(os.environ.get('NUM_WORKERS', 18))

        self.cfg = config


def setup_sequence_definitions(cfg):
    """Read sequence definitions from acquired_data.mat"""
    seq_defs = {}
    sd_import = sio.loadmat(cfg['acqdata_fn'])['seq_defs']
    for name in sd_import.dtype.names:
        if len(sd_import[name].flatten()[0].flatten()) > 1:
            seq_defs[name] = sd_import[name].flatten()[0].flatten().tolist()
        elif isinstance(sd_import[name].flatten()[0].flatten()[0], np.integer):
            seq_defs[name] = int(sd_import[name].flatten()[0].flatten()[0])
        else:
            seq_defs[name] = float(sd_import[name].flatten()[0].flatten()[0])

    # Add SLflag if not present
    if 'SLflag' not in seq_defs.keys():
        seq_defs['SLflag'] = seq_defs['offsets_ppm'] < [1e-3] * seq_defs['num_meas']

    # Add SLFA if not present
    if 'SLFA' not in seq_defs.keys():
        seq_defs['SLFA'] = seq_defs['excFA']

    seq_defs['B0'] = cfg['b0']
    seq_defs['seq_id_string'] = os.path.splitext(cfg['seq_fn'])[1][1:]

    return seq_defs


def preprocess_dict(dictionary):
    """Preprocess the dictionary for use with neural network"""
    dictionary['sig'] = np.array(dictionary['sig'])
    for key in dictionary.keys():
        if key != 'sig':
            dictionary[key] = np.expand_dims(np.squeeze(np.array(dictionary[key])), 0)
    print(f"Dictionary shape: {dictionary['sig'].shape}")
    return dictionary


def define_min_max(dictionary):
    """Define min/max parameters for normalization"""
    min_fs = np.min(dictionary['fs_0'])
    min_ksw = np.min(dictionary['ksw_0'].transpose().astype(np.float32))
    max_fs = np.max(dictionary['fs_0'])
    max_ksw = np.max(dictionary['ksw_0'].transpose().astype(np.float32))

    min_param_tensor = torch.tensor(np.hstack((min_fs, min_ksw)), requires_grad=False)
    max_param_tensor = torch.tensor(np.hstack((max_fs, max_ksw)), requires_grad=False)

    return min_param_tensor, max_param_tensor


def load_and_preprocess_acquired_data(data_fn, sig_n):
    """Load and preprocess acquired MRF data"""
    acquired_data = sio.loadmat(data_fn)['acquired_data'].astype(np.float32)
    _, c_acq_data, w_acq_data = np.shape(acquired_data)

    # Reshape to (sig_n x pixels)
    acquired_data = np.reshape(acquired_data, (sig_n, c_acq_data * w_acq_data), order='F')

    # L2 normalization
    acquired_data = acquired_data / np.sqrt(np.sum(acquired_data ** 2, axis=0))

    # Transpose for NN compatibility (each row is a trajectory)
    acquired_data = acquired_data.T

    acquired_data = torch.from_numpy(acquired_data).float()
    acquired_data.requires_grad = False

    return acquired_data, c_acq_data, w_acq_data


def train_network_on_dictionary(dictionary, sig_n, device, learning_rate=0.0003, batch_size=512,
                                 num_epochs=100, noise_std=0.002, patience=10, min_delta=0.01):
    """Train neural network on the dictionary"""
    print("\n" + "="*60)
    print("TRAINING NEURAL NETWORK ON DICTIONARY")
    print("="*60)

    # Prepare data
    dataset = DatasetMRF(dictionary)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # Get normalization parameters
    min_param_tensor, max_param_tensor = define_min_max(dictionary)

    # Initialize network
    reco_net = Network(sig_n).to(device)
    optimizer = torch.optim.Adam(reco_net.parameters(), lr=learning_rate)

    # Training loop
    t0 = time.time()
    loss_per_epoch = []
    patience_counter = 0
    min_loss = 100.0

    print(f"Training for up to {num_epochs} epochs (early stopping enabled)")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}, Noise std: {noise_std}")
    print("-"*60)

    for epoch in range(num_epochs):
        reco_net.train()
        cum_loss = 0

        for counter, dict_params in enumerate(train_loader, 0):
            cur_fs, cur_ksw, cur_norm_sig = dict_params

            target = torch.stack((cur_fs, cur_ksw), dim=1)
            target = normalize_range(original_array=target, original_min=min_param_tensor,
                                     original_max=max_param_tensor, new_min=-1, new_max=1).to(device).float()

            # Add noise to input signals
            noised_sig = cur_norm_sig + torch.randn(cur_norm_sig.size()) * noise_std
            noised_sig = noised_sig.to(device).float()

            # Forward pass
            prediction = reco_net(noised_sig)

            # Compute loss (MSE)
            loss = torch.mean((prediction - target) ** 2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cum_loss += loss.item()

        # Average loss for this epoch
        avg_loss = cum_loss / (counter + 1)
        loss_per_epoch.append(avg_loss)

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            elapsed_min = (time.time() - t0) / 60
            print(f"Epoch {epoch+1:3d}/{num_epochs}: Loss = {avg_loss:.6f} "
                  f"(Elapsed: {elapsed_min:.1f} min)")

        # Early stopping check
        if (min_loss - avg_loss) / min_loss > min_delta:
            min_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    training_time = time.time() - t0
    print("-"*60)
    print(f"Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")

    # Save checkpoint
    checkpoint_fn = 'OUTPUT_FILES/trained_model.pt'
    torch.save({
        'model_state_dict': reco_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_per_epoch': loss_per_epoch,
        'min_param_tensor': min_param_tensor,
        'max_param_tensor': max_param_tensor,
        'sig_n': sig_n,
    }, checkpoint_fn)
    print(f"Model checkpoint saved to: {checkpoint_fn}")
    print("="*60 + "\n")

    return reco_net, min_param_tensor, max_param_tensor


def cnn_inference(acquired_data, c_acq_data, w_acq_data, sig_n, device,
                  model_path=None, dictionary=None):
    """Perform CNN-based inference on acquired data"""
    print("\n" + "="*60)
    print("CNN INFERENCE MODE")
    print("="*60)

    # Check if pre-trained model exists
    if model_path and os.path.exists(model_path):
        print(f"Loading pre-trained model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        reco_net = Network(sig_n).to(device)
        reco_net.load_state_dict(checkpoint['model_state_dict'])
        min_param_tensor = checkpoint['min_param_tensor']
        max_param_tensor = checkpoint['max_param_tensor']
        print("Pre-trained model loaded successfully")
    elif dictionary is not None:
        print("No pre-trained model found. Training new model on dictionary...")
        reco_net, min_param_tensor, max_param_tensor = train_network_on_dictionary(
            dictionary, sig_n, device)
    else:
        raise ValueError("Either model_path or dictionary must be provided for CNN inference")

    # Perform inference
    print("\nPerforming inference on acquired data...")
    reco_net.eval()
    start_time = time.perf_counter()

    with torch.no_grad():
        inputs = acquired_data.to(device).float()
        outputs = reco_net(inputs)

    # Un-normalize outputs
    outputs = un_normalize_range(outputs, original_min=min_param_tensor.to(device),
                                 original_max=max_param_tensor.to(device),
                                 new_min=-1, new_max=1)

    # Reshape to image format
    quant_map_fs = outputs.cpu().detach().numpy()[:, 0]
    quant_map_fs = quant_map_fs.T
    quant_map_fs = np.reshape(quant_map_fs, (c_acq_data, w_acq_data), order='F')

    quant_map_ksw = outputs.cpu().detach().numpy()[:, 1]
    quant_map_ksw = quant_map_ksw.T
    quant_map_ksw = np.reshape(quant_map_ksw, (c_acq_data, w_acq_data), order='F')

    inference_time = time.perf_counter() - start_time
    print(f"Inference completed in {inference_time:.3f} seconds")
    print("="*60 + "\n")

    quant_maps = {'fs': quant_map_fs, 'ksw': quant_map_ksw}

    return quant_maps


def dictionary_matching_mode(cfg, dict_fn, acq_fn):
    """Traditional dictionary simulation and dot product matching"""
    print("\n" + "="*60)
    print("DICTIONARY MATCHING MODE")
    print("="*60)

    # Dot product matching
    start_time = time.perf_counter()
    quant_maps = dot_prod_matching(dict_fn=dict_fn, acquired_data_fn=acq_fn)
    matching_time = time.perf_counter() - start_time

    print(f"Dot product matching completed in {matching_time:.3f} seconds")
    print("="*60 + "\n")

    return quant_maps


def visualize_and_save_results(quant_maps, mat_fn, create_mask=True):
    """Visualize quantitative maps and save them"""
    # Save .mat file
    sio.savemat(mat_fn, quant_maps)
    print(f"Quantitative maps saved to: {mat_fn}")

    # Create mask based on dot product (if available) or use default mask
    if 'dp' in quant_maps and create_mask:
        mask = quant_maps['dp'] > 0.99974
        mask_fn = 'OUTPUT_FILES/mask.npy'
        np.save(mask_fn, mask)
        print(f"Mask saved to: {mask_fn}")
    else:
        # For CNN mode without dot product, create a simple mask based on non-zero values
        mask = (quant_maps['fs'] > 0) & (quant_maps['ksw'] > 0)
        mask_fn = 'OUTPUT_FILES/mask.npy'
        np.save(mask_fn, mask)
        print(f"Basic mask saved to: {mask_fn}")

    # Create visualization
    fig_fn = 'OUTPUT_FILES/quant_maps_results.eps'

    if 'dp' in quant_maps:
        # Full visualization with dot product
        fig, axes = plt.subplots(1, 3, figsize=(30, 25))
        color_maps = [b_viridis, 'magma', 'magma']
        data_keys = ['fs', 'ksw', 'dp']
        titles = ['[L-arg] (mM)', 'k$_{sw}$ (s$^{-1}$)', 'Dot product']
        clim_list = [(0, 120), (0, 500), (0.999, 1)]
        tick_list = [np.arange(0, 140, 20), np.arange(0, 600, 100),
                     np.arange(0.999, 1.0005, 0.0005)]

        for ax, color_map, key, title, clim, ticks in zip(axes.flat, color_maps, data_keys,
                                                           titles, clim_list, tick_list):
            vals = quant_maps[key] * (key == 'fs' and 110e3 / 3 or 1) * mask
            plot = ax.imshow(vals, cmap=color_map)
            plot.set_clim(*clim)
            ax.set_title(title, fontsize=25)
            cb = plt.colorbar(plot, ax=ax, ticks=ticks, orientation='vertical',
                              fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=25)
            ax.set_axis_off()
    else:
        # CNN mode - only fs and ksw
        fig, axes = plt.subplots(1, 2, figsize=(20, 25))
        color_maps = [b_viridis, 'magma']
        data_keys = ['fs', 'ksw']
        titles = ['[L-arg] (mM)', 'k$_{sw}$ (s$^{-1}$)']
        clim_list = [(0, 120), (0, 500)]
        tick_list = [np.arange(0, 140, 20), np.arange(0, 600, 100)]

        for ax, color_map, key, title, clim, ticks in zip(axes.flat, color_maps, data_keys,
                                                           titles, clim_list, tick_list):
            vals = quant_maps[key] * (key == 'fs' and 110e3 / 3 or 1) * mask
            plot = ax.imshow(vals, cmap=color_map)
            plot.set_clim(*clim)
            ax.set_title(title, fontsize=25)
            cb = plt.colorbar(plot, ax=ax, ticks=ticks, orientation='vertical',
                              fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=25)
            ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(fig_fn, format="eps")
    plt.close()
    print(f"Results visualization saved to: {fig_fn}")


def main():
    """Main function to run integrated MRF processing"""
    print("\n" + "="*70)
    print("INTEGRATED MRF PROCESSING")
    print("="*70)

    # Parse environment variables
    use_cnn = int(os.environ.get('USE_CNN', 0))
    num_gpus = int(os.environ.get('NUM_GPUS', 1))
    model_path = os.environ.get('MODEL_PATH', 'OUTPUT_FILES/trained_model.pt')

    print(f"Mode: {'CNN INFERENCE' if use_cnn else 'DICTIONARY MATCHING'}")
    if use_cnn:
        print(f"Number of GPUs: {num_gpus}")
        print(f"Model path: {model_path}")

    # Set random seed for reproducibility
    set_seed(2024)

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPUs: {torch.cuda.device_count()}")

    # Load configuration
    cfg = ConfigDK().get_config()

    # Write configuration and sequence files
    print("\nWriting configuration files...")
    write_yaml_dict(cfg)
    seq_defs = setup_sequence_definitions(cfg)
    write_sequence_DK(seq_defs=seq_defs, seq_fn=cfg['seq_fn'])
    print("Configuration files written")

    # Dictionary generation
    print("\nGenerating dictionary...")
    start_dict_gen = time.perf_counter()

    if len(cfg['cest_pool'].keys()) > 1:
        eqvals = [('fs_0', 'fs_1', 0.6666667)]
    else:
        eqvals = None

    dictionary = generate_mrf_cest_dictionary(
        seq_fn=cfg['seq_fn'],
        param_fn=cfg['yaml_fn'],
        dict_fn=cfg['dict_fn'],
        num_workers=cfg['num_workers'],
        axes='xy',
        equals=eqvals
    )

    dict_gen_time = time.perf_counter() - start_dict_gen
    print(f"Dictionary generation completed in {dict_gen_time:.1f} seconds")

    # Preprocess dictionary
    dictionary = preprocess_dict(dictionary)

    # Get signal dimension from dictionary
    sig_n = dictionary['sig'].shape[0]
    print(f"Signal dimension (number of measurements): {sig_n}")

    # Choose processing mode
    if use_cnn:
        # CNN INFERENCE MODE
        acquired_data, c_acq_data, w_acq_data = load_and_preprocess_acquired_data(
            cfg['acqdata_fn'], sig_n)

        quant_maps = cnn_inference(
            acquired_data, c_acq_data, w_acq_data, sig_n, device,
            model_path=model_path if os.path.exists(model_path) else None,
            dictionary=dictionary
        )

        # Visualize and save results (without dot product for CNN mode)
        visualize_and_save_results(quant_maps, cfg['quantmaps_fn'], create_mask=True)

    else:
        # DICTIONARY MATCHING MODE
        quant_maps = dictionary_matching_mode(cfg, cfg['dict_fn'], cfg['acqdata_fn'])

        # Visualize and save results (with dot product)
        visualize_and_save_results(quant_maps, cfg['quantmaps_fn'], create_mask=True)

    print("\n" + "="*70)
    print("PROCESSING COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    main()
