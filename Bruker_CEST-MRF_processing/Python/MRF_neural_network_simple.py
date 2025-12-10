#!/usr/bin/env python3
"""
Simple MRF Neural Network Training and Inference
Based on deep_reco_example/preclinical.py structure

This script:
1. Loads configuration from MATLAB-generated acquired_data.mat
2. Generates a dictionary (training data)
3. Trains a neural network on the dictionary
4. Performs inference on acquired data
5. Saves quantitative maps
"""

import os
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
import tqdm

from sequences_dk import write_sequence_DK
from cest_mrf.write_scenario import write_yaml_dict
from cest_mrf.dictionary.generation import generate_mrf_cest_dictionary

from utils.normalization import normalize_range, un_normalize_range
from utils.colormaps import b_viridis
from utils.seed import set_seed

from deep_reco_example.dataset import DatasetMRF
from deep_reco_example.model import Network


class Config:
    def get_config(self):
        return self.cfg


class ConfigDK(Config):
    """Load configuration from MATLAB-generated acquired_data.mat"""
    def __init__(self):
        config = {}

        # Check for large storage directory
        large_storage = os.environ.get('LARGE_STORAGE_DIR', None)

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
        config['num_gpus'] = int(os.environ.get('NUM_GPUS', '3'))  # Support multiple GPUs

        # Load dictionary parameters from acquired_data.mat
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

        # Water pool
        config['water_pool'] = {}
        config['water_pool']['t1'] = dp['water_t1']
        config['water_pool']['t2'] = dp['water_t2']
        config['water_pool']['f'] = dp['water_f']

        # CEST pool(s)
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

        # Magnetization
        config['scale'] = dp['magnetization_scale']
        config['reset_init_mag'] = dp['magnetization_reset']

        # Scanner
        config['b0'] = dp['b0']
        config['gamma'] = dp['gamma']
        config['b0_inhom'] = dp['b0_inhom']
        config['rel_b1'] = dp['rel_b1']

        # Processing
        config['verbose'] = 0
        config['max_pulse_samples'] = 100
        config['num_workers'] = 18

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

    if 'SLflag' not in seq_defs.keys():
        seq_defs['SLflag'] = seq_defs['offsets_ppm'] < [1e-3] * seq_defs['num_meas']
    if 'SLFA' not in seq_defs.keys():
        seq_defs['SLFA'] = seq_defs['excFA']

    seq_defs['B0'] = cfg['b0']
    seq_defs['seq_id_string'] = os.path.splitext(cfg['seq_fn'])[1][1:]

    return seq_defs


def main():
    print("\n" + "="*60)
    print("MRF NEURAL NETWORK - SIMPLE VERSION")
    print("="*60 + "\n")

    # ===== PARAMETERS =====
    sig_n = 30  # Signal dimension (number of measurements)

    # Training hyperparameters
    learning_rate = 0.0003
    batch_size = 512
    num_epochs = 100
    noise_std = 0.002

    # Early stopping
    patience = 10
    min_delta = 0.01

    # Device
    device = initialize_device()
    print(f"Using device: {device}\n")

    # Paths
    data_folder = 'INPUT_FILES'
    output_folder = 'OUTPUT_FILES'
    os.makedirs(output_folder, exist_ok=True)

    # ===== STEP 1: Load Config from MATLAB =====
    print("Step 1: Loading configuration from acquired_data.mat...")
    cfg = ConfigDK().get_config()
    print(f"  B0: {cfg['b0']} T")
    print(f"  Num workers: {cfg['num_workers']}")
    print(f"  Num GPUs: {cfg['num_gpus']}")
    print(f"  CEST pools: {len(cfg['cest_pool'])}")
    print(f"  Dictionary output: {cfg['dict_fn']}")
    print(f"  Quant maps output: {cfg['quantmaps_fn']}")

    # Write YAML and sequence files
    write_yaml_dict(cfg)
    seq_defs = setup_sequence_definitions(cfg)
    write_sequence_DK(seq_defs=seq_defs, seq_fn=cfg['seq_fn'])
    print("  Config files written\n")

    # ===== STEP 2: Generate Dictionary (Training Data) =====
    print("Step 2: Generating dictionary (this creates the training data)...")
    dictionary = generate_dict(cfg)
    sig_n = dictionary['sig'].shape[0]  # Update sig_n from actual data
    print(f"  Signal dimension: {sig_n}")
    print(f"  Dictionary size: {dictionary['sig'].shape[1]} entries\n")

    # ===== STEP 3: Get Normalization Parameters =====
    print("Step 3: Computing normalization parameters...")
    min_param_tensor, max_param_tensor = define_min_max(dictionary)
    print(f"  fs range: [{min_param_tensor[0]:.6f}, {max_param_tensor[0]:.6f}]")
    print(f"  ksw range: [{min_param_tensor[1]:.1f}, {max_param_tensor[1]:.1f}]\n")

    # ===== STEP 4: Train Network =====
    print("Step 4: Training neural network...")
    train_loader = prepare_dataloader(dictionary, batch_size=batch_size)
    reco_net = Network(sig_n).to(device)
    optimizer = torch.optim.Adam(reco_net.parameters(), lr=learning_rate)

    reco_net = train_network(
        train_loader, reco_net, optimizer, device, learning_rate, num_epochs,
        noise_std, min_param_tensor, max_param_tensor, patience, min_delta
    )
    print("")

    # ===== STEP 5: Load and Preprocess Acquired Data =====
    print("Step 5: Loading acquired data...")
    data_fn = os.path.join(data_folder, 'acquired_data.mat')
    eval_data, c_acq_data, w_acq_data = load_and_preprocess_data(data_fn, sig_n)
    print(f"  Image size: {c_acq_data} x {w_acq_data}\n")

    # ===== STEP 6: Inference =====
    print("Step 6: Running inference...")
    quant_maps = evaluate_network(
        reco_net, eval_data, device, min_param_tensor, max_param_tensor,
        c_acq_data=c_acq_data, w_acq_data=w_acq_data
    )
    print("  Inference complete\n")

    # ===== STEP 7: Save Results =====
    print("Step 7: Saving results...")

    # Try to load mask from dot product (if available), otherwise create simple mask
    # Note: Neural network doesn't produce dot product metric (only available in dictionary matching)
    # If you previously ran dictionary matching, it will use that mask (threshold typically 0.95)
    try:
        mask = np.load('OUTPUT_FILES/mask.npy')
        print("  Using existing mask from OUTPUT_FILES/mask.npy")
    except:
        print("  Creating simple mask (non-zero values)")
        # Simple mask: include pixels with valid parameter estimates
        mask = (quant_maps['fs'] > 0) & (quant_maps['ksw'] > 0)
        np.save('OUTPUT_FILES/mask.npy', mask)
        print("  Tip: For better masking, first run dictionary matching to generate mask with dot product > 0.95")

    save_and_plot_results(quant_maps, cfg['quantmaps_fn'], output_folder, mask)

    print("\n" + "="*60)
    print("COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nResults saved:")
    print(f"  - {cfg['quantmaps_fn']}")
    print(f"  - {output_folder}/deep_reco_results.eps")
    print(f"  - {output_folder}/checkpoint.pt (trained model)")
    print(f"  - {output_folder}/mask.npy")
    print("="*60 + "\n")


def load_and_preprocess_data(data_fn, sig_n):
    """Load acquired data from MATLAB .mat file"""
    acquired_data = sio.loadmat(data_fn)['acquired_data'].astype(np.float32)
    _, c_acq_data, w_acq_data = np.shape(acquired_data)

    # Reshape to (sig_n x pixels)
    acquired_data = np.reshape(acquired_data, (sig_n, c_acq_data * w_acq_data), order='F')

    # L2 normalization
    acquired_data = acquired_data / np.sqrt(np.sum(acquired_data ** 2, axis=0))

    # Transpose for NN (each row = one pixel trajectory)
    acquired_data = acquired_data.T

    acquired_data = torch.from_numpy(acquired_data).float()
    acquired_data.requires_grad = False

    return acquired_data, c_acq_data, w_acq_data


def initialize_device():
    """Initialize device (GPU/CPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    return device


def prepare_dataloader(data, batch_size):
    """Prepare DataLoader for training"""
    dataset = DatasetMRF(data)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)


def train_network(train_loader, reco_net, optimizer, device, learning_rate, num_epochs,
                  noise_std, min_param_tensor, max_param_tensor, patience, min_delta):
    """Train the network"""
    t0 = time.time()
    loss_per_epoch = []
    patience_counter = 0
    min_loss = 100

    pbar = tqdm.tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        cum_loss = 0

        for counter, dict_params in enumerate(train_loader, 0):
            cur_fs, cur_ksw, cur_norm_sig = dict_params

            # Stack targets
            target = torch.stack((cur_fs, cur_ksw), dim=1)
            target = normalize_range(
                original_array=target,
                original_min=min_param_tensor,
                original_max=max_param_tensor,
                new_min=-1, new_max=1
            ).to(device).float()

            # Add noise to input signals
            noised_sig = cur_norm_sig + torch.randn(cur_norm_sig.size()) * noise_std
            noised_sig = noised_sig.to(device).float()

            # Forward pass
            prediction = reco_net(noised_sig)

            # Loss (MSE)
            loss = torch.mean((prediction - target) ** 2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cum_loss += loss.item()

        # Average loss for this epoch
        loss_per_epoch.append(cum_loss / (counter + 1))

        pbar.set_description(f'Epoch: {epoch + 1}/{num_epochs}, Loss = {loss_per_epoch[-1]:.6f}')
        pbar.update(1)

        # Early stopping check
        if (min_loss - loss_per_epoch[-1]) / min_loss > min_delta:
            min_loss = loss_per_epoch[-1]
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > patience:
            print('\n  Early stopping triggered!')
            break

    pbar.close()
    print(f"  Training time: {time.time() - t0:.1f} seconds")

    # Save checkpoint
    torch.save({
        'model_state_dict': reco_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_per_epoch': loss_per_epoch,
        'min_param_tensor': min_param_tensor,
        'max_param_tensor': max_param_tensor,
        'sig_n': reco_net.l1.in_features,
    }, 'OUTPUT_FILES/checkpoint.pt')

    return reco_net


def evaluate_network(reco_net, data, device, min_param_tensor, max_param_tensor,
                     c_acq_data=30, w_acq_data=126):
    """Evaluate the network on acquired data"""
    reco_net.eval()

    with torch.no_grad():
        inputs = data.to(device).float()
        outputs = reco_net(inputs)

    # Un-normalize outputs
    outputs = un_normalize_range(
        outputs,
        original_min=min_param_tensor.to(device),
        original_max=max_param_tensor.to(device),
        new_min=-1, new_max=1
    )

    # Reshape to image format
    quant_map_fs = outputs.cpu().detach().numpy()[:, 0]
    quant_map_fs = quant_map_fs.T
    quant_map_fs = np.reshape(quant_map_fs, (c_acq_data, w_acq_data), order='F')

    quant_map_ksw = outputs.cpu().detach().numpy()[:, 1]
    quant_map_ksw = quant_map_ksw.T
    quant_map_ksw = np.reshape(quant_map_ksw, (c_acq_data, w_acq_data), order='F')

    quant_maps = {'fs': quant_map_fs, 'ksw': quant_map_ksw}

    return quant_maps


def save_and_plot_results(quant_maps, quantmaps_fn, output_folder, mask):
    """Save quantitative maps and generate plots"""
    os.makedirs(output_folder, exist_ok=True)

    # Save .mat file (use provided path or default)
    sio.savemat(quantmaps_fn, quant_maps)
    print(f"  Saved: {quantmaps_fn}")

    # Generate plot
    fig_fn = os.path.join(output_folder, 'deep_reco_results.eps')
    plt.figure(figsize=(10, 5))

    # Concentration map
    plt.subplot(121)
    plt.imshow(quant_maps['fs'] * 110e3/3 * mask, cmap=b_viridis, clim=(0, 120))
    plt.colorbar(ticks=np.arange(0, 121, 20), fraction=0.046, pad=0.04)
    plt.title('[L-arg] (mM)')
    plt.axis("off")

    # Exchange rate map
    plt.subplot(122)
    plt.imshow(quant_maps['ksw'] * mask, cmap='magma', clim=(0, 500))
    plt.colorbar(ticks=np.arange(0, 501, 100), fraction=0.046, pad=0.04)
    plt.title('k$_{sw}$ (s$^{-1}$)')
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(fig_fn, format="eps")
    plt.close()
    print(f"  Saved: {fig_fn}")


def generate_dict(cfg):
    """
    Generate MRF dictionary (training data for neural network)

    This is ESSENTIAL because:
    - The neural network needs training examples
    - Dictionary = simulated signals for known parameter combinations
    - Without this, the network has nothing to learn from

    For 3-pool systems (e.g., glutamate + BSA):
    - Use 'equals' parameter to constrain pool relationships
    - Network still outputs 2 values (fs, ksw) representing the total
    """
    yaml_fn = cfg['yaml_fn']
    seq_fn = cfg['seq_fn']
    dict_fn = cfg['dict_fn']

    # Check if we have multiple CEST pools
    if len(cfg['cest_pool'].keys()) > 1:
        # For glutamate + BSA or other 2-pool systems
        # This constrains fs_1 = 0.6667 * fs_0 (or your desired ratio)
        eqvals = [('fs_0', 'fs_1', 0.6666667)]
        print("  Multi-pool system detected: using constraint fs_0:fs_1 = 3:2")
    else:
        eqvals = None

    dictionary = generate_mrf_cest_dictionary(
        seq_fn=seq_fn,
        param_fn=yaml_fn,
        dict_fn=dict_fn,
        num_workers=cfg['num_workers'],
        axes='xy',  # 'xy' if readout is simulated, 'z' otherwise
        equals=eqvals
    )

    return preprocess_dict(dictionary)


def preprocess_dict(dictionary):
    """Preprocess dictionary for neural network training"""
    dictionary['sig'] = np.array(dictionary['sig'])
    for key in dictionary.keys():
        if key != 'sig':
            dictionary[key] = np.expand_dims(np.squeeze(np.array(dictionary[key])), 0)

    print(f"  Dictionary signal shape: {dictionary['sig'].shape}")
    return dictionary


def define_min_max(dictionary):
    """Get min/max values for normalization"""
    min_fs = np.min(dictionary['fs_0'])
    min_ksw = np.min(dictionary['ksw_0'].transpose().astype(np.float32))
    max_fs = np.max(dictionary['fs_0'])
    max_ksw = np.max(dictionary['ksw_0'].transpose().astype(np.float32))

    min_param_tensor = torch.tensor(np.hstack((min_fs, min_ksw)), requires_grad=False)
    max_param_tensor = torch.tensor(np.hstack((max_fs, max_ksw)), requires_grad=False)

    return min_param_tensor, max_param_tensor


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    set_seed(2024)
    main()
