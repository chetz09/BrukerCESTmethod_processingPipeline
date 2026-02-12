import os
import sys
import time
import numpy as np
import scipy.io as sio

from matplotlib import pyplot as plt

from colormaps_dk import b_viridis

# from ..dot_prod_example.configs import ConfigPreclinical
from sequences_dk import write_sequence_DK

from cest_mrf.write_scenario import write_yaml_dict
from cest_mrf.dictionary.generation import generate_mrf_cest_dictionary
from cest_mrf.metrics.dot_product import dot_prod_matching

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

        # Modified by DK to pull in dictpars from acquired_data.mat
        dp = {}
        dp_import = sio.loadmat(config['acqdata_fn'])['dictpars']
        for name in dp_import.dtype.names:
            if len(dp_import[name].flatten()[0].flatten()) > 1: #store as list
                dp[name]=dp_import[name].flatten()[0].flatten().tolist()
            elif isinstance(dp_import[name].flatten()[0].flatten()[0],np.integer): #store as single integer value
                dp[name]=int(dp_import[name].flatten()[0].flatten()[0])
            else:
                # Try to convert to float, but if it's a string, keep as string
                try:
                    dp[name]=float(dp_import[name].flatten()[0].flatten()[0])
                except (ValueError, TypeError):
                    # It's a string (like 'SuperLorentzian') - keep as string
                    val = dp_import[name].flatten()[0].flatten()[0]
                    if isinstance(val, str):
                        dp[name] = str(val)
                    elif hasattr(val, 'decode'):  # Handle byte strings
                        dp[name] = val.decode('utf-8')
                    else:
                        dp[name] = str(val)

        # Water_pool
        config['water_pool'] = {}
        config['water_pool']['t1'] = dp['water_t1']
        config['water_pool']['t2'] = dp['water_t2']
        config['water_pool']['f'] = dp['water_f']

        # Solute pool (optional - only if CEST parameters present in data)
        if 'cest_amine_f' in dp:
            config['cest_pool'] = {}
            config['cest_pool']['Amine'] = {}
            config['cest_pool']['Amine']['t1'] = dp['cest_amine_t1']
            config['cest_pool']['Amine']['t2'] = dp['cest_amine_t2']
            config['cest_pool']['Amine']['k'] = dp['cest_amine_k']
            config['cest_pool']['Amine']['dw'] = dp['cest_amine_dw']
            config['cest_pool']['Amine']['f'] = dp['cest_amine_f']

        # MT pool (optional)
        if 'mt_f' in dp:
            config['mt_pool'] = {}
            config['mt_pool']['t1'] = dp['mt_t1']
            config['mt_pool']['t2'] = dp['mt_t2']
            config['mt_pool']['k'] = dp['mt_k']
            config['mt_pool']['dw'] = dp['mt_dw']
            config['mt_pool']['f'] = dp['mt_f']
            config['mt_pool']['lineshape'] = dp['mt_lineshape']

        # Fill initial magnetization info
        # this is important now for the mrf simulation! For the regular pulseq-cest
        # simulation, we usually assume that the magnetization reached a steady
        # state after the readout, which means we can set the magnetization vector
        # to a specific scale, e.g. 0.5. This is because we do not simulate the
        # readout there. For mrf we include the readout in the simulation, which
        # means we need to carry the same magnetization vector through the entire
        # sequence. To avoid that the magnetization vector gets set to the initial
        # value after each readout, we need to set reset_init_mag to false
        config['scale'] = dp['magnetization_scale']
        config['reset_init_mag'] = dp['magnetization_reset']

        # Fill scanner info
        config['b0'] = dp['b0']
        config['gamma'] = dp['gamma']
        config['b0_inhom'] = dp['b0_inhom']
        config['rel_b1'] = dp['rel_b1']

        # Fill additional info
        config['verbose'] = 0
        config['max_pulse_samples'] = 100
        config['num_workers'] = 18

        self.cfg = config

def setup_sequence_definitions(cfg):
    # Read in seq_defs from acquired_data.mat
    seq_defs = {}
    sd_import = sio.loadmat(cfg['acqdata_fn'])['seq_defs']
    for name in sd_import.dtype.names:
        if len(sd_import[name].flatten()[0].flatten()) > 1: #store as list
            seq_defs[name]=sd_import[name].flatten()[0].flatten().tolist()
        elif isinstance(sd_import[name].flatten()[0].flatten()[0],np.integer): #store as single integer value
            seq_defs[name]=int(sd_import[name].flatten()[0].flatten()[0])
        else:
            seq_defs[name]=float(sd_import[name].flatten()[0].flatten()[0])

    # DK edit 8/26/24: Add in 'SLflag' if not imported above
    if not 'SLflag' in seq_defs.keys():
        seq_defs['SLflag']=seq_defs['offsets_ppm'] < [1e-3]*seq_defs['num_meas']
    # DK edit 9/4/24: Add in 'SLFA' if not imported above
    if not 'SLFA' in seq_defs.keys():
        seq_defs['SLFA']=seq_defs['excFA']    #use excitation tip angles, since that's what it was for a while unfortunately....

    seq_defs['B0'] = cfg['b0']  # B0 [T]
    seq_defs['seq_id_string'] = os.path.splitext(cfg['seq_fn'])[1][1:]  # unique seq id

    return seq_defs


def generate_quant_maps(acq_fn, dict_fn):
    """Run dot product matching and save quant maps."""
    # acq_fn = os.path.join(data_f, 'acquired_data.mat')
    quant_maps = dot_prod_matching(dict_fn=dict_fn, acquired_data_fn=acq_fn)
    return quant_maps


def visualize_and_save_results(quant_maps, mat_fn):
    """Visualize quant maps and save them as eps."""
    sio.savemat(mat_fn, quant_maps)
    print('quant_maps.mat saved')

    output_dir = os.path.dirname(mat_fn) if os.path.dirname(mat_fn) else 'OUTPUT_FILES'

    mask = quant_maps['dp'] > 0.99974
    mask_fn = os.path.join(output_dir, 'mask.npy')
    np.save(mask_fn, mask)

    fig_fn = os.path.join(output_dir, 'dot_product_results.eps')

    # Build visualization panels based on available maps
    has_cest = 'fs' in quant_maps
    has_mt = 'fm' in quant_maps

    panels = []
    if has_cest:
        panels.append({'key': 'fs', 'title': 'Glutamate (mM)', 'cmap': b_viridis,
                        'clim': (0, 30), 'ticks': np.arange(0, 35, 5),
                        'scale': 110e3 / 3})
        panels.append({'key': 'ksw', 'title': 'k$_{sw}$ (s$^{-1}$)', 'cmap': 'magma',
                        'clim': (0, 500), 'ticks': np.arange(0, 600, 100),
                        'scale': 1})
    if has_mt:
        panels.append({'key': 't1m', 'title': 'T$_{1,MT}$ (s)', 'cmap': 'inferno',
                        'clim': (0, 1.0), 'ticks': np.arange(0, 1.2, 0.2),
                        'scale': 1})
        panels.append({'key': 't2m', 'title': 'T$_{2,MT}$ ($\mu$s)', 'cmap': 'inferno',
                        'clim': (0, 30), 'ticks': np.arange(0, 35, 5),
                        'scale': 1e6})
        panels.append({'key': 'fm', 'title': 'MT fraction', 'cmap': 'viridis',
                        'clim': (0, 0.05), 'ticks': np.arange(0, 0.06, 0.01),
                        'scale': 1})
        panels.append({'key': 'kmw', 'title': 'k$_{mw}$ (s$^{-1}$)', 'cmap': 'magma',
                        'clim': (0, 100), 'ticks': np.arange(0, 120, 20),
                        'scale': 1})
    panels.append({'key': 'dp', 'title': 'Dot product', 'cmap': 'magma',
                    'clim': (0.999, 1), 'ticks': np.arange(0.999, 1.0005, 0.0005),
                    'scale': 1})

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(10 * n_panels, 25))
    if n_panels == 1:
        axes = [axes]

    for ax, panel in zip(axes.flat, panels):
        vals = quant_maps[panel['key']] * panel['scale'] * mask
        plot = ax.imshow(vals, cmap=panel['cmap'])
        plot.set_clim(*panel['clim'])
        ax.set_title(panel['title'], fontsize=25)
        cb = plt.colorbar(plot, ax=ax, ticks=panel['ticks'], orientation='vertical', fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=25)
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(fig_fn, format="eps")
    plt.close()
    print("Resulting plots saved as EPS")


def main():
    cfg = ConfigDK().get_config()

    # Write configuration and sequence files
    write_yaml_dict(cfg)
    seq_defs = setup_sequence_definitions(cfg)
    write_sequence_DK(seq_defs=seq_defs, seq_fn=cfg['seq_fn'])

    # Dictionary generation
    # eqvals constrains two CEST pools to have related fs values;
    # only needed when there are multiple CEST pools
    if 'cest_pool' in cfg and len(cfg['cest_pool'].keys()) > 1:
        eqvals=[('fs_0','fs_1',0.6666667)]
    else:
        eqvals=None
    dictionary = generate_mrf_cest_dictionary(seq_fn=cfg['seq_fn'], param_fn=cfg['yaml_fn'], dict_fn=cfg['dict_fn'],
                                 num_workers=cfg['num_workers'], axes='xy', equals=eqvals)

    # Dot product matching and quant map generation
    start_time = time.perf_counter()
    quant_maps = generate_quant_maps(cfg['acqdata_fn'], cfg['dict_fn'])
    print(f"Dot product matching took {time.perf_counter() - start_time:.03f} s.")

    # Visualization and saving results
    visualize_and_save_results(quant_maps, cfg['quantmaps_fn'])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    main()
