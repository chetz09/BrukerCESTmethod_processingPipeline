% Dictionary Parameter Configuration for ~18.8M entries
% Optimized to avoid segmentation fault while maintaining good resolution

% OPTION 1: 18.37M entries (RECOMMENDED - just under 18.8M)
% Good balance across all parameters
dictpars.water_t1 = 0.5:0.1:12;              % 116 values
dictpars.water_t2 = 0.05:0.1:1.5;            % 15 values
dictpars.cest_amine_k = 100:210:9910;        % 48 values
dictpars.cest_amine_sol_conc = 1:2:40;       % 20 values
dictpars.mt_k = 30:3.7:70;                   % 11 values
% Total: 116 × 15 × 48 × 20 × 11 = 18,374,400 entries
% Estimated size: ~3.5-4.0 GB

fprintf('Dictionary Configuration:\n');
fprintf('  T1 water: %d values (%.2f to %.2f s, step %.2f)\n', ...
    length(dictpars.water_t1), min(dictpars.water_t1), max(dictpars.water_t1), dictpars.water_t1(2)-dictpars.water_t1(1));
fprintf('  T2 water: %d values (%.3f to %.3f s, step %.3f)\n', ...
    length(dictpars.water_t2), min(dictpars.water_t2), max(dictpars.water_t2), dictpars.water_t2(2)-dictpars.water_t2(1));
fprintf('  Amine k: %d values (%d to %d Hz, step %d)\n', ...
    length(dictpars.cest_amine_k), min(dictpars.cest_amine_k), max(dictpars.cest_amine_k), dictpars.cest_amine_k(2)-dictpars.cest_amine_k(1));
fprintf('  Amine conc: %d values (%d to %d mM, step %d)\n', ...
    length(dictpars.cest_amine_sol_conc), min(dictpars.cest_amine_sol_conc), max(dictpars.cest_amine_sol_conc), dictpars.cest_amine_sol_conc(2)-dictpars.cest_amine_sol_conc(1));
fprintf('  MT k: %d values (%d to %.1f Hz, step %.1f)\n', ...
    length(dictpars.mt_k), min(dictpars.mt_k), max(dictpars.mt_k), dictpars.mt_k(2)-dictpars.mt_k(1));

total_entries = length(dictpars.water_t1) * length(dictpars.water_t2) * ...
                length(dictpars.cest_amine_k) * length(dictpars.cest_amine_sol_conc) * ...
                length(dictpars.mt_k);
fprintf('\nTotal dictionary entries: %.2f M\n', total_entries/1e6);
fprintf('Estimated file size: %.2f GB\n', total_entries * 30 * 8 / 1024^3);
fprintf('(Assumes 30 Z-spectrum points, 8 bytes per double)\n');

% ========================================================================
% ALTERNATIVE OPTIONS (comment/uncomment as needed):
% ========================================================================

% OPTION 2: 17.61M entries (More conservative, faster generation)
% dictpars.water_t1 = 0.5:0.1:12;              % 116 values
% dictpars.water_t2 = 0.05:0.1:1.5;            % 15 values
% dictpars.cest_amine_k = 100:220:9880;        % 46 values
% dictpars.cest_amine_sol_conc = 1:2:40;       % 20 values
% dictpars.mt_k = 30:3.7:70;                   % 11 values
% % Total: 116 × 15 × 46 × 20 × 11 = 17,606,400 entries

% OPTION 3: 14.21M entries (Safest option, good parameter coverage)
% dictpars.water_t1 = 0.5:0.15:12;             % 77 values
% dictpars.water_t2 = 0.05:0.075:1.5;          % 20 values
% dictpars.cest_amine_k = 100:250:9850;        % 40 values
% dictpars.cest_amine_sol_conc = 1:1.5:40;     % 27 values
% dictpars.mt_k = 30:2.5:70;                   % 17 values
% % Total: 77 × 20 × 40 × 27 × 17 = 14,212,800 entries

% NOTE: Keep these parameters the same across all options:
dictpars.cest_amine_dw = 3.5;               % ppm (amide proton chemical shift)
dictpars.mt_dw = 0;                          % ppm (MT pool)
dictpars.mt_lineshape = 'SuperLorentzian';   % MT lineshape
