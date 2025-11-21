% check_dictionary_size.m
% Script to calculate and display dictionary size before generation
% Helps avoid creating dictionaries that are too large

clear; clc;

fprintf('=================================================\n');
fprintf('DICTIONARY SIZE CALCULATOR\n');
fprintf('=================================================\n\n');

%% Set your dictionary parameters here (copy from DictConfigParams.m)

% Water pool
water_t1 = 0.5:0.1:12;  % water T1 values, in s
water_t2 = 0.05:0.05:1.5;  % water T2 values, in s

% Solute pool (CEST amine)
cest_amine_t1 = 2.7;  % fixed solute t1, in s (single value)
cest_amine_t2 = 0.04;  % fixed solute t2, in s (single value)
cest_amine_k = 100:200:10000;  % solute exchange rate, in s^-1
cest_amine_sol_conc = 1:1:40;  % solute concentration, in mM

% MT pool (set nPools = 2 to skip MT, or nPools = 3 to include)
nPools = 3;  % 2 = water + amine only, 3 = water + amine + MT
mt_k = 30:2:70;  % MT exchange rate, in s^-1

%% Calculate number of entries for each parameter

n_water_t1 = length(water_t1);
n_water_t2 = length(water_t2);
n_cest_k = length(cest_amine_k);
n_cest_conc = length(cest_amine_sol_conc);

fprintf('Water pool parameters:\n');
fprintf('  water_t1: %d values (%.2f to %.2f s, step %.2f)\n', ...
    n_water_t1, min(water_t1), max(water_t1), water_t1(2)-water_t1(1));
fprintf('  water_t2: %d values (%.3f to %.3f s, step %.3f)\n', ...
    n_water_t2, min(water_t2), max(water_t2), water_t2(2)-water_t2(1));

fprintf('\nCEST amine pool parameters:\n');
fprintf('  cest_amine_t1: %d value (fixed at %.2f s)\n', ...
    length(cest_amine_t1), cest_amine_t1);
fprintf('  cest_amine_t2: %d value (fixed at %.3f s)\n', ...
    length(cest_amine_t2), cest_amine_t2);
fprintf('  cest_amine_k: %d values (%.0f to %.0f s^-1, step %.0f)\n', ...
    n_cest_k, min(cest_amine_k), max(cest_amine_k), cest_amine_k(2)-cest_amine_k(1));
fprintf('  cest_amine_sol_conc: %d values (%.0f to %.0f mM, step %.0f)\n', ...
    n_cest_conc, min(cest_amine_sol_conc), max(cest_amine_sol_conc), ...
    cest_amine_sol_conc(2)-cest_amine_sol_conc(1));

%% Calculate total dictionary size

if nPools == 2
    % 2 pools: water + amine only
    total_entries = n_water_t1 * n_water_t2 * n_cest_k * n_cest_conc;

    fprintf('\n=================================================\n');
    fprintf('DICTIONARY SIZE (2 POOLS: water + amine)\n');
    fprintf('=================================================\n');
    fprintf('Calculation: %d × %d × %d × %d\n', ...
        n_water_t1, n_water_t2, n_cest_k, n_cest_conc);
    fprintf('Total entries: %s\n', addCommas(total_entries));

elseif nPools == 3
    % 3 pools: water + amine + MT
    n_mt_k = length(mt_k);

    fprintf('\nMT pool parameters:\n');
    fprintf('  mt_k: %d values (%.0f to %.0f s^-1, step %.0f)\n', ...
        n_mt_k, min(mt_k), max(mt_k), mt_k(2)-mt_k(1));

    total_entries_2pool = n_water_t1 * n_water_t2 * n_cest_k * n_cest_conc;
    total_entries_3pool = total_entries_2pool * n_mt_k;

    fprintf('\n=================================================\n');
    fprintf('DICTIONARY SIZE (2 POOLS: water + amine)\n');
    fprintf('=================================================\n');
    fprintf('Calculation: %d × %d × %d × %d\n', ...
        n_water_t1, n_water_t2, n_cest_k, n_cest_conc);
    fprintf('Total entries: %s\n', addCommas(total_entries_2pool));

    fprintf('\n=================================================\n');
    fprintf('DICTIONARY SIZE (3 POOLS: water + amine + MT)\n');
    fprintf('=================================================\n');
    fprintf('Calculation: %d × %d × %d × %d × %d\n', ...
        n_water_t1, n_water_t2, n_cest_k, n_cest_conc, n_mt_k);
    fprintf('Total entries: %s\n', addCommas(total_entries_3pool));

    total_entries = total_entries_3pool;
else
    error('nPools must be 2 or 3');
end

%% Estimate file size and runtime

fprintf('\n=================================================\n');
fprintf('ESTIMATES\n');
fprintf('=================================================\n');

% Rough file size estimate (each entry ~50-100 KB for z-spectrum)
avg_entry_size_kb = 75;  % KB per entry (approximate)
file_size_gb = (total_entries * avg_entry_size_kb) / (1024^2);

fprintf('Estimated dictionary file size: %.2f GB\n', file_size_gb);

% Runtime estimate (very rough - depends on hardware)
% Assume ~1000-2000 entries/second on modern CPU with multiprocessing
if total_entries < 10000
    time_estimate = 'Less than 1 minute';
elseif total_entries < 100000
    time_estimate = sprintf('%.1f - %.1f minutes', total_entries/2000/60, total_entries/1000/60);
elseif total_entries < 1000000
    time_estimate = sprintf('%.1f - %.1f hours', total_entries/2000/3600, total_entries/1000/3600);
else
    time_estimate = sprintf('%.1f - %.1f hours (VERY LONG!)', total_entries/2000/3600, total_entries/1000/3600);
end

fprintf('Estimated generation time: %s\n', time_estimate);

%% Assessment and recommendations

fprintf('\n=================================================\n');
fprintf('ASSESSMENT\n');
fprintf('=================================================\n');

if total_entries < 10000
    fprintf('✓ EXCELLENT: Very small dictionary, will generate quickly\n');
    fprintf('  Status: RECOMMENDED\n');
elseif total_entries < 100000
    fprintf('✓ GOOD: Reasonable dictionary size\n');
    fprintf('  Status: RECOMMENDED\n');
elseif total_entries < 1000000
    fprintf('⚠ MODERATE: Large dictionary, may take 1-4 hours\n');
    fprintf('  Status: ACCEPTABLE but consider reducing parameters\n');
elseif total_entries < 10000000
    fprintf('⚠ LARGE: Very large dictionary, will take many hours\n');
    fprintf('  Status: RISKY - consider reducing parameters\n');
elseif total_entries < 100000000
    fprintf('✗ VERY LARGE: Extremely large dictionary\n');
    fprintf('  Status: NOT RECOMMENDED - high risk of memory issues\n');
else
    fprintf('✗✗ MASSIVE: Dictionary too large!\n');
    fprintf('  Status: WILL LIKELY FAIL - must reduce parameters\n');
end

%% Suggestions for optimization

if total_entries > 1000000
    fprintf('\n=================================================\n');
    fprintf('OPTIMIZATION SUGGESTIONS\n');
    fprintf('=================================================\n');

    % Check which parameter contributes most to size
    [~, max_idx] = max([n_water_t1, n_water_t2, n_cest_k, n_cest_conc]);
    param_names = {'water_t1', 'water_t2', 'cest_amine_k', 'cest_amine_sol_conc'};
    param_values = [n_water_t1, n_water_t2, n_cest_k, n_cest_conc];

    fprintf('Largest parameter: %s (%d values)\n', param_names{max_idx}, param_values(max_idx));
    fprintf('\nRecommendations to reduce dictionary size:\n');

    if n_water_t1 > 20
        fprintf('  • Reduce water_t1 range or increase step size\n');
        fprintf('    Current: %d values, try: 8-15 values\n', n_water_t1);
        fprintf('    Example: water_t1 = 2.0:0.2:3.5 (8 values for 9.4T)\n');
    end

    if n_water_t2 > 10
        fprintf('  • Reduce water_t2 range or increase step size\n');
        fprintf('    Current: %d values, try: 3-5 values\n', n_water_t2);
        fprintf('    Example: water_t2 = 0.05:0.05:0.15 (3 values for 9.4T)\n');
    end

    if n_cest_k > 30
        fprintf('  • Reduce cest_amine_k range or increase step size\n');
        fprintf('    Current: %d values, try: 15-25 values\n', n_cest_k);
        fprintf('    Example: cest_amine_k = 100:300:7000 (24 values)\n');
    end

    if n_cest_conc > 30
        fprintf('  • Reduce concentration range or increase step size\n');
        fprintf('    Current: %d values, try: 15-20 values\n', n_cest_conc);
        fprintf('    Example: cest_amine_sol_conc = 2:2:40 (20 values)\n');
    end

    if nPools == 3 && n_mt_k > 15
        fprintf('  • Reduce mt_k range or increase step size\n');
        fprintf('    Current: %d values, try: 8-12 values\n', n_mt_k);
        fprintf('    Example: mt_k = 30:5:70 (9 values)\n');
    end
end

fprintf('\n=================================================\n\n');

%% Helper function to format numbers with commas
function str = addCommas(num)
    str = num2str(num);
    if num >= 1000
        % Add commas for thousands
        str_with_commas = '';
        len = length(str);
        for i = 1:len
            str_with_commas = [str(end-i+1) str_with_commas];
            if mod(i, 3) == 0 && i < len
                str_with_commas = [',' str_with_commas];
            end
        end
        str = [str_with_commas ' (' formatSize(num) ')'];
    end
end

function str = formatSize(num)
    if num < 1e3
        str = sprintf('%.0f', num);
    elseif num < 1e6
        str = sprintf('%.1fK', num/1e3);
    elseif num < 1e9
        str = sprintf('%.2fM', num/1e6);
    else
        str = sprintf('%.2fB', num/1e9);
    end
end
