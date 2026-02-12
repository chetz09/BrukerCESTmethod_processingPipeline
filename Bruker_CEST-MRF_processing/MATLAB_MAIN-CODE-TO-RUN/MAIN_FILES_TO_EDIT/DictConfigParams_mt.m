% DictConfigParams_mt: Water + MT pool only (no CEST pool).
% For BSA-only phantom experiments to verify MT pool is working.
%
%   INPUT:  seq_info    -   Struct containing pulse sequence info
%           prefs       -   Struct containing user processing options
%   OUTPUT: dictparams  -   Struct containing dictionary simulation values
%
function dictparams = DictConfigParams_mt(seq_info, prefs)
disp('Loading MT-ONLY dictionary simulation settings...')
disp('Water pool + MT pool (no CEST pool)')

% =========================================================================
% Water pool
% =========================================================================
dictparams.water_t1 = 2.5;     % water T1 values, in s (vary)
dictparams.water_t2 = 0.175;  % water T2 values, in s (vary)
dictparams.water_f = 1;               % water proton fraction

% =========================================================================
% NO CEST pool
% =========================================================================

% =========================================================================
% MT pool
% =========================================================================
dictparams.mt_t1 = 0.1:.1:1;           % MT T1, in s (fixed, literature value)
dictparams.mt_t2 = 5e-06:1e-06:30e-06;         % MT T2, in s (fixed, very short for macromolecules)
dictparams.mt_k = 40:2:60;            % MT exchange rate, in Hz (vary)
dictparams.mt_dw = -0.06:0.01:0.06;            % MT chemical shift offset, in ppm (fixed)
dictparams.mt_f = 0.005:0.002:0.025;  % MT proton fraction (vary for diff BSA concentrations)
dictparams.mt_lineshape = 'SuperLorentzian'; % REQUIRED string

% =========================================================================
% Fill initial magnetization info
% =========================================================================
dictparams.magnetization_scale = 1;
dictparams.magnetization_reset = 0;

% =========================================================================
% Fill scanner info
% =========================================================================
dictparams.b0 = seq_info.B0;   % [T]
dictparams.gamma = 267.5153;   % [rad / uT]
dictparams.b0_inhom = 0;
dictparams.rel_b1 = 1;

disp('MT-only dictionary parameters loaded successfully.')
end
