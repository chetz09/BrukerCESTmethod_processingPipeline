import numpy as np
import scipy.io as sio
import os

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
                #dp[name]=float(dp_import[name].flatten()[0].flatten()[0])
                try:
                    dp[name]=float(dp_import[name].flatten()[0].flatten()[0])
                except (ValueError, TypeError):
                   # It's a string (like 'SuperLorentzian') - keep as string
                   val = dp_import[name].flatten()[0].flatten()[0]
                   if hasattr(val, 'decode'):  # Byte strings
                        dp[name] = str(val.decode('utf-8'))
                   else:
                        # Force to native Python string - handles numpy.str_, MATLAB strings, etc.
                        dp[name] = str(val).strip()

        # Water_pool
        config['water_pool'] = {}
        config['water_pool']['t1'] = dp['water_t1']
        # config['water_pool']['t1'] = config['water_pool']['t1'].tolist()  # vary t1
        config['water_pool']['t2'] = dp['water_t2']
        # config['water_pool']['t2'] = config['water_pool']['t2'].tolist()  # vary t2
        config['water_pool']['f'] = dp['water_f']

        # Solute pool
        config['cest_pool'] = {}
        config['cest_pool']['Amine'] = {}
        config['cest_pool']['Amine']['t1'] = dp['cest_amine_t1']
        config['cest_pool']['Amine']['t2'] = dp['cest_amine_t2']
        config['cest_pool']['Amine']['k'] = dp['cest_amine_k']
        config['cest_pool']['Amine']['dw'] = dp['cest_amine_dw']
        config['cest_pool']['Amine']['f'] = dp['cest_amine_f']
        # config['cest_pool']['Amine']['f'] = config['cest_pool']['Amine']['f'].tolist()

        # Additional CEST pool ("MT")

         #This is for treating MT as an additional CEST pool
         #if 'cest_mt_f' in dp.keys():
         #   config['cest_pool']['MT'] = {}
         #   config['cest_pool']['MT']['t1'] = dp['cest_mt_t1']
         #   config['cest_pool']['MT']['t2'] = dp['cest_mt_t2']
         #   config['cest_pool']['MT']['k'] = dp['cest_mt_k']
         #   config['cest_pool']['MT']['dw'] = dp['cest_mt_dw']
         #   config['cest_pool']['MT']['f'] = dp['cest_mt_f']
        # This is for treating MT as an MT pool
        if 'mt_f' in dp.keys():
            config['mt_pool'] = {}
            config['mt_pool']['t1'] = dp['mt_t1']
            config['mt_pool']['t2'] = dp['mt_t2']
            config['mt_pool']['k'] = dp['mt_k']
            config['mt_pool']['dw'] = dp['mt_dw']
            config['mt_pool']['f'] = dp['mt_f']
            config['mt_pool']['lineshape'] = str(dp['mt_lineshape'])

        # Fill initial magnetization info
        # this is important now for the mrf simulation! For the regular pulseq-cest
        # simulation, we usually assume athat the magnetization reached a steady
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
