#!/usr/bin/env python3
"""
Standalone Training Script for MRF Neural Network

This script trains a neural network on a simulated MRF dictionary.
The trained model can then be used for fast inference on acquired data.

Usage:
    python train_mrf_network.py

Environment Variables:
    NUM_WORKERS: Number of parallel workers for dictionary generation (default: 18)
    BATCH_SIZE: Training batch size (default: 512)
    NUM_EPOCHS: Maximum number of training epochs (default: 100)
    LEARNING_RATE: Learning rate for Adam optimizer (default: 0.0003)
"""

import os
import sys
import time
import argparse
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

# Import required modules
from sequences_dk import write_sequence_DK
from cest_mrf.write_scenario import write_yaml_dict
from cest_mrf.dictionary.generation import generate_mrf_cest_dictionary

# Import deep learning components
from deep_reco_example.model import Network
from deep_reco_example.dataset import DatasetMRF
from utils.normalization import normalize_range
from utils.seed import set_seed


class ConfigDK:
    """Configuration class that reads parameters from acquired_data.mat"""
    def __init__(self):
        config = {}
        config['yaml_fn'] = 'OUTPUT_FILES/scenario.yaml'
        config['seq_fn'] = 'OUTPUT_FILES/acq_protocol.seq'
        config['dict_fn'] = 'OUTPUT_FILES/dict_training.mat'
        config['acqdata_fn'] = 'INPUT_FILES/acquired_data.mat'

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

    def get_config(self):
        return self.cfg


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


def preprocess_dict(dictionary):
    """Preprocess the dictionary"""
    dictionary['sig'] = np.array(dictionary['sig'])
    for key in dictionary.keys():
        if key != 'sig':
            dictionary[key] = np.expand_dims(np.squeeze(np.array(dictionary[key])), 0)
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


def train_network(train_loader, sig_n, device, min_param_tensor, max_param_tensor,
                  learning_rate=0.0003, num_epochs=100, noise_std=0.002,
                  patience=10, min_delta=0.01, save_path='OUTPUT_FILES/trained_model.pt'):
    """Train the neural network"""
    print("\n" + "="*70)
    print("TRAINING NEURAL NETWORK")
    print("="*70)

    # Initialize network
    reco_net = Network(sig_n).to(device)
    optimizer = torch.optim.Adam(reco_net.parameters(), lr=learning_rate)

    print(f"Network architecture:")
    print(f"  Input dimension: {sig_n}")
    print(f"  Hidden layers: 300 -> 300")
    print(f"  Output dimension: 2 (fs, ksw)")
    print(f"\nTraining parameters:")
    print(f"  Max epochs: {num_epochs}")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Noise std: {noise_std}")
    print(f"  Early stopping: patience={patience}, min_delta={min_delta}")
    print(f"  Device: {device}")
    print("-"*70)

    # Training loop
    t0 = time.time()
    loss_per_epoch = []
    patience_counter = 0
    min_loss = float('inf')

    for epoch in range(num_epochs):
        reco_net.train()
        cum_loss = 0
        num_batches = 0

        for dict_params in train_loader:
            cur_fs, cur_ksw, cur_norm_sig = dict_params

            target = torch.stack((cur_fs, cur_ksw), dim=1)
            target = normalize_range(original_array=target, original_min=min_param_tensor,
                                     original_max=max_param_tensor, new_min=-1, new_max=1).to(device).float()

            # Add noise to input signals
            noised_sig = cur_norm_sig + torch.randn(cur_norm_sig.size()) * noise_std
            noised_sig = noised_sig.to(device).float()

            # Forward pass
            prediction = reco_net(noised_sig)

            # Compute loss
            loss = torch.mean((prediction - target) ** 2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cum_loss += loss.item()
            num_batches += 1

        # Average loss for this epoch
        avg_loss = cum_loss / num_batches
        loss_per_epoch.append(avg_loss)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed_min = (time.time() - t0) / 60
            print(f"Epoch {epoch+1:3d}/{num_epochs}: Loss = {avg_loss:.6f}, "
                  f"Time = {elapsed_min:.1f} min")

        # Early stopping check
        if (min_loss - avg_loss) / max(min_loss, 1e-8) > min_delta:
            min_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best loss: {min_loss:.6f}")
            break

    training_time = time.time() - t0
    print("-"*70)
    print(f"Training completed!")
    print(f"  Total time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    print(f"  Final loss: {loss_per_epoch[-1]:.6f}")
    print(f"  Best loss: {min_loss:.6f}")

    # Save checkpoint
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': reco_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_per_epoch': loss_per_epoch,
        'min_param_tensor': min_param_tensor,
        'max_param_tensor': max_param_tensor,
        'sig_n': sig_n,
        'training_time': training_time,
        'final_loss': loss_per_epoch[-1],
    }, save_path)
    print(f"\nModel saved to: {save_path}")
    print("="*70 + "\n")

    # Plot training curve
    plot_training_curve(loss_per_epoch, save_path.replace('.pt', '_training_curve.png'))

    return reco_net


def plot_training_curve(loss_per_epoch, save_path):
    """Plot and save training curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_per_epoch, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curve saved to: {save_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train MRF neural network')
    parser.add_argument('--batch-size', type=int, default=512, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum training epochs')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--noise-std', type=float, default=0.002, help='Training noise std')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--output', type=str, default='OUTPUT_FILES/trained_model.pt',
                        help='Output model path')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("MRF NEURAL NETWORK TRAINING")
    print("="*70)

    # Set random seed
    set_seed(args.seed)

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Load configuration
    print("\nLoading configuration from acquired_data.mat...")
    cfg = ConfigDK().get_config()

    # Write configuration and sequence files
    print("Writing configuration files...")
    write_yaml_dict(cfg)
    seq_defs = setup_sequence_definitions(cfg)
    write_sequence_DK(seq_defs=seq_defs, seq_fn=cfg['seq_fn'])

    # Generate dictionary
    print("\nGenerating training dictionary...")
    print("This may take some time depending on dictionary size...")
    start_time = time.perf_counter()

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

    dict_time = time.perf_counter() - start_time
    print(f"Dictionary generated in {dict_time:.1f} seconds")

    # Preprocess dictionary
    dictionary = preprocess_dict(dictionary)
    sig_n = dictionary['sig'].shape[0]
    dict_size = dictionary['sig'].shape[1]
    print(f"\nDictionary properties:")
    print(f"  Signal dimension: {sig_n}")
    print(f"  Dictionary size: {dict_size} entries")

    # Get normalization parameters
    min_param_tensor, max_param_tensor = define_min_max(dictionary)
    print(f"  fs range: [{min_param_tensor[0]:.6f}, {max_param_tensor[0]:.6f}]")
    print(f"  ksw range: [{min_param_tensor[1]:.1f}, {max_param_tensor[1]:.1f}]")

    # Create data loader
    dataset = DatasetMRF(dictionary)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=8)

    # Train network
    reco_net = train_network(
        train_loader, sig_n, device, min_param_tensor, max_param_tensor,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        noise_std=args.noise_std,
        patience=args.patience,
        save_path=args.output
    )

    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nTrained model can be used for inference with:")
    print(f"  USE_CNN=1 MODEL_PATH={args.output} python MRFmatch_integrated.py")
    print("="*70 + "\n")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    main()
