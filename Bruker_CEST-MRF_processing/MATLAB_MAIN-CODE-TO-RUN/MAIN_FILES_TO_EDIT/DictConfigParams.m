% DictConfigParams: Outputs a structure variable dictparams containing the  
% values to simulate for the dictionary generation, to be saved in 
% acquired_data.mat  
%
%   INPUT:  seq_info    -   Struct containing pulse sequence info and
%                           parameter values
%           prefs       -   Struct containing user specific processing options
%   OUTPUT: dictparams  -   Struct containing vector arrays pertaining to
%                           the values to simulate during dictionary
%                           generation
%
function dictparams = DictConfigParams(seq_info,prefs)
disp('Loading dictionary simulation settings from file DictConfigParams.m...')

% Water pool - REDUCED GRID FOR MEMORY EFFICIENCY
dictparams.water_t1 = 1.5:0.1:4; %water T1 values, in s (26 values)
dictparams.water_t2 = 0.5:0.1:2.5; %water T2 values, in s (21 values)
dictparams.water_f = 1; %water proton volume fraction(?)

% Solute pool (Amine/Glutamate)
dictparams.cest_amine_t1 = 2.8;  % fixed solute t1, in s
dictparams.cest_amine_t2 = .04;  % fixed solute t2, in s
dictparams.cest_amine_k = 100:100:8000;  % solute exchange rate, in s^-1 (80 values)
dictparams.cest_amine_dw = 3;  % solute chemical shift offset, in ppm

% solute concentration * protons / water concentration
dictparams.cest_amine_sol_conc = 1:1:40;  % solute concentration, in mM (40 values)
dictparams.cest_amine_protons = 3;
dictparams.cest_amine_water_conc = 110000;  %in mM
dictparams.cest_amine_f = dictparams.cest_amine_sol_conc * ...
    dictparams.cest_amine_protons ./ dictparams.cest_amine_water_conc;

if prefs.nPools > 2
    disp('More than 2 pools specified! Adding additional pool...')
    % CRITICAL: MT pool parameters must be FIXED (not varying) to avoid memory explosion
    % With varying MT parameters, dictionary becomes 22.6B entries (17TB RAM)
    % With fixed MT parameters, dictionary is 1.75M entries (~12GB RAM)
    dictparams.cest_mt_t1 = 2.8;      % FIXED - same as amine
    dictparams.cest_mt_t2 = 0.04;     % FIXED - same as amine
    dictparams.cest_mt_k = 500;       % FIXED - typical MT exchange rate (s^-1)
    dictparams.cest_mt_dw = 0.6;      % FIXED - chemical shift offset (ppm)

    % FIXED MT concentration - adjust this value to match your phantom
    % Typical MT pool is much smaller than amine pool
    dictparams.cest_mt_protons = 2;
    dictparams.cest_mt_water_conc = 110000;
    dictparams.cest_mt_sol_conc = 10;  % FIXED - 10 mM (adjust as needed)
    dictparams.cest_mt_f = dictparams.cest_mt_sol_conc * ...
        dictparams.cest_mt_protons / dictparams.cest_mt_water_conc;
end

% Fill initial magnetization info
dictparams.magnetization_scale = 1;
dictparams.magnetization_reset = 0;

% Fill scanner info
dictparams.b0 = seq_info.B0;  % [T]
dictparams.gamma = 267.5153;  % [rad / uT]
dictparams.b0_inhom = 0;
dictparams.rel_b1 = 1;

% Initial magnetization info: this is important now for the mrf simulation! 
% For the regular pulseq-cest simulation, we usually assume that the 
% magnetization reached a steady state after the readout, which means we 
% can set the magnetization vector to a specific scale, e.g. 0.5. This is 
% because we do not simulate the readout there. For mrf we include the 
% readout in the simulation, which means we need to carry the same 
% magnetization vector through the entire sequence. To avoid that the 
% magnetization vector gets set to the initial value after each readout, we 
% need to set reset_init_mag to false

% Check that size of dictionary won't be too large for 64GB RAM system
% Memory requirements: ~7 KB per entry
% 64GB RAM limit: ~2M entries (leaving room for neural network training)
max_size = 2000000;  % 2 million entries max for 64GB RAM

dict_size = size(dictparams.water_t1,2) * size(dictparams.water_t2,2) * ...
    size(dictparams.cest_amine_k,2) * size(dictparams.cest_amine_f,2);

fprintf('Dictionary size calculation:\n');
fprintf('  water_t1: %d values\n', size(dictparams.water_t1,2));
fprintf('  water_t2: %d values\n', size(dictparams.water_t2,2));
fprintf('  cest_amine_k: %d values\n', size(dictparams.cest_amine_k,2));
fprintf('  cest_amine_f: %d values\n', size(dictparams.cest_amine_f,2));
fprintf('  Total entries: %d (%.2f M)\n', dict_size, dict_size/1e6);
fprintf('  Estimated RAM: %.1f GB\n', dict_size * 7e3 / 1e9);

if dict_size > max_size
    error('Dictionary size is too large! %d entries requires ~%.0f GB RAM. Reduce parameter ranges.', ...
        dict_size, dict_size * 7e3 / 1e9);
end
fprintf('Dictionary size OK for 64GB RAM system.\n');
end