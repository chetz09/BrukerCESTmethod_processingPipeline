% MRFmatch: Generates acquired_data.mat with raw data from 2dseq file, then
% calls Python to perform dictionary simulation and matching (either locally
% or on cluster), and finally places the resulting file in the same data
% directory as the 2dseq file.
%
%   INPUTS:
%       dirstruct   -   Struct containing paths to required directories
%       prefs       -   Struct containing user specific processing options
%       PV360flg    -   Logical; if true, will process according to
%                       ParaVision 360 format (default false)
%
%   OUTPUTS:    None (results are saved in a .mat file)
%
function MRFmatch(dirstruct,prefs,PV360flg)

disp(['MRF data: Generating acquired_data.mat for dictionary '...
     'simulation and matching...'])

% Generate acquired_data.mat from 2dseq file
read2dseq(dirstruct.loadMRF,'dictmatch',prefs,PV360flg);

% Copy acquired_data.mat to Python INPUT_FILES directory
copyfile(fullfile(dirstruct.loadMRF,'acquired_data.mat'),...
    fullfile(dirstruct.py_dir,'INPUT_FILES'));

%% Check if using cluster or local processing
if isfield(dirstruct,'use_cluster') && dirstruct.use_cluster
    %% ===== CLUSTER PROCESSING MODE =====
    disp(' ')
    disp('========================================')
    disp('CLUSTER PROCESSING MODE')
    disp('========================================')

    %% STEP 0: Clean old results to prevent reuse
    disp('Step 0: Cleaning previous results...')

    % Clean OUTPUT_FILES directory
    output_dir = fullfile(dirstruct.py_dir,'OUTPUT_FILES');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    old_files = dir(fullfile(output_dir, 'quant_maps*.mat'));
    for i = 1:length(old_files)
        delete(fullfile(old_files(i).folder, old_files(i).name));
        fprintf('  ✓ Removed: %s\n', old_files(i).name);
    end

    old_files = dir(fullfile(output_dir, 'dot_product_results.*'));
    for i = 1:length(old_files)
        delete(fullfile(old_files(i).folder, old_files(i).name));
        fprintf('  ✓ Removed: %s\n', old_files(i).name);
    end

    % Clean large storage directory if specified
    if isfield(dirstruct,'cluster_large_storage_dir') && ~isempty(dirstruct.cluster_large_storage_dir)
        large_output_dir = fullfile(dirstruct.cluster_large_storage_dir, 'MRF_OUTPUT');
        if exist(large_output_dir, 'dir')
            old_files = dir(fullfile(large_output_dir, 'quant_maps*.mat'));
            for i = 1:length(old_files)
                delete(fullfile(old_files(i).folder, old_files(i).name));
                fprintf('  ✓ Removed: %s\n', old_files(i).name);
            end

            old_files = dir(fullfile(large_output_dir, 'dot_product_results.*'));
            for i = 1:length(old_files)
                delete(fullfile(old_files(i).folder, old_files(i).name));
                fprintf('  ✓ Removed: %s\n', old_files(i).name);
            end
        end
    end

    disp('✓ Old results cleaned - ready for fresh results')
    disp('========================================')

    %% STEP 1: Transfer acquired_data.mat to cluster
    disp('Step 1/4: Transferring data to cluster...')

    local_acq = fullfile(dirstruct.py_dir,'INPUT_FILES','acquired_data.mat');
    remote_input = sprintf('%s@%s:%s/INPUT_FILES/', ...
        dirstruct.cluster_user, ...
        dirstruct.cluster_host, ...
        dirstruct.cluster_dir);

    scp_cmd = sprintf('scp %s %s', local_acq, remote_input);
    fprintf('Transferring acquired_data.mat... ');
    [status, result] = system(scp_cmd);

    if status ~= 0
        fprintf('\n=== ERROR DETAILS ===\n');
        fprintf('Status code: %d\n', status);
        fprintf('Error message:\n%s\n', result);
        fprintf('====================\n');
        error('Failed to transfer acquired_data.mat: %s', result);
    end

    disp('✓ Transfer complete')

    %% STEP 2: Submit SLURM job with large storage environment variable
    disp('Step 2/4: Submitting job to cluster...')

    % Prepare environment variable for large storage if specified
    if isfield(dirstruct,'cluster_large_storage_dir') && ~isempty(dirstruct.cluster_large_storage_dir)
        fprintf('  Setting LARGE_STORAGE_DIR=%s\n', dirstruct.cluster_large_storage_dir);
        ssh_cmd = sprintf('ssh %s@%s "cd %s && sbatch --export=ALL,LARGE_STORAGE_DIR=%s %s"', ...
            dirstruct.cluster_user, ...
            dirstruct.cluster_host, ...
            dirstruct.cluster_dir, ...
            dirstruct.cluster_large_storage_dir, ...
            dirstruct.cluster_job_script);
    else
        ssh_cmd = sprintf('ssh %s@%s "cd %s && sbatch %s"', ...
            dirstruct.cluster_user, ...
            dirstruct.cluster_host, ...
            dirstruct.cluster_dir, ...
            dirstruct.cluster_job_script);
    end

    [status, result] = system(ssh_cmd);
    if status ~= 0
        error('Failed to submit cluster job: %s', result);
    end

    % Extract job ID
    job_id = regexp(result, 'Submitted batch job (\d+)', 'tokens');
    if isempty(job_id)
        error('Could not parse job ID from: %s', result);
    end
    job_id = job_id{1}{1};

    fprintf('✓ Job submitted! Job ID: %s\n', job_id);

    %% STEP 3: Monitor job status with robust error handling
    disp('Step 3/4: Monitoring job status...')

    check_interval = 60;  % Check every 60 seconds
    max_minutes = 240;     % 4 hours max (for large dictionaries)
    max_checks = ceil(max_minutes * 60 / check_interval);

    fprintf('Will monitor for up to %.1f hours (%d checks every %d seconds)\n', ...
        max_minutes/60, max_checks, check_interval);
    disp('Press Ctrl+C to stop monitoring and download results manually')
    disp(' ')

    job_running = true;
    check_count = 0;

    % Determine where to check for output files
    if isfield(dirstruct,'cluster_large_storage_dir') && ~isempty(dirstruct.cluster_large_storage_dir)
        check_path = sprintf('%s/MRF_OUTPUT/quant_maps.mat', dirstruct.cluster_large_storage_dir);
    else
        check_path = sprintf('%s/OUTPUT_FILES/quant_maps.mat', dirstruct.cluster_dir);
    end

    while job_running && check_count < max_checks
        pause(check_interval);
        check_count = check_count + 1;

        % Method 1: Check if output files exist on cluster (most reliable)
        check_output_cmd = sprintf('ssh %s@%s "test -f %s && echo DONE"', ...
            dirstruct.cluster_user, ...
            dirstruct.cluster_host, ...
            check_path);

        [~, file_check] = system(check_output_cmd);

        if contains(file_check, 'DONE')
            disp('✓ Output files detected! Job completed!')
            pause(5);  % Wait 5s for file system sync
            job_running = false;
            break;
        end

        % Method 2: Check squeue (with error handling)
        check_cmd = sprintf('ssh %s@%s "squeue -j %s -h -o ''%%%%T'' 2>&1"', ...
            dirstruct.cluster_user, ...
            dirstruct.cluster_host, ...
            job_id);

        [status, queue_result] = system(check_cmd);
        queue_result = strtrim(queue_result);

        % Handle different squeue responses
        if status ~= 0 || contains(queue_result, 'Invalid job')
            disp('✓ Job completed and removed from queue!')
            pause(5);  % Wait for file sync
            job_running = false;
        elseif isempty(queue_result)
            disp('✓ Job no longer in queue!')
            pause(5);
            job_running = false;
        else
            if mod(check_count, 4) == 0  % Print every 4 minutes
                elapsed_min = check_count * check_interval / 60;
                fprintf('[%.1f min] Job status: %s (check #%d/%d)\n', ...
                    elapsed_min, queue_result, check_count, max_checks);
            end
        end
    end

    if check_count >= max_checks
        warning('Monitoring timeout reached after %.1f hours.', max_minutes/60);
        disp('Job may still be running. Check manually with:');
        fprintf('  ssh %s@%s "squeue -u %s"\n', ...
            dirstruct.cluster_user, dirstruct.cluster_host, dirstruct.cluster_user);

        response = input('Continue waiting? (y/n): ', 's');
        if strcmpi(response, 'y')
            disp('Continuing to monitor...');
            max_checks = max_checks + 60;  % Add another hour
            job_running = true;
        else
            error('Monitoring stopped. Download results manually when job completes.');
        end
    end

    %% STEP 4: Download results with verification
    disp('Step 4/4: Downloading results from cluster...')

    % Determine source and destination for files
    if isfield(dirstruct,'cluster_large_storage_dir') && ~isempty(dirstruct.cluster_large_storage_dir)
        remote_dir = sprintf('%s/MRF_OUTPUT', dirstruct.cluster_large_storage_dir);
        local_output = fullfile(dirstruct.py_dir,'OUTPUT_FILES');
        fprintf('Retrieving files from large storage: %s\n', remote_dir);
    else
        remote_dir = sprintf('%s/OUTPUT_FILES', dirstruct.cluster_dir);
        local_output = fullfile(dirstruct.py_dir,'OUTPUT_FILES');
        fprintf('Retrieving files from OUTPUT_FILES\n');
    end

    % Check what files exist on cluster
    check_cmd = sprintf('ssh %s@%s "ls -lh %s/quant_maps.mat %s/dot_product_results.* 2>&1"', ...
        dirstruct.cluster_user, ...
        dirstruct.cluster_host, ...
        remote_dir, ...
        remote_dir);

    [~, file_list] = system(check_cmd);
    fprintf('Files on cluster:\n%s\n', file_list);

    % Download quant_maps.mat
    remote_quant = sprintf('%s@%s:%s/quant_maps.mat', ...
        dirstruct.cluster_user, ...
        dirstruct.cluster_host, ...
        remote_dir);

    local_quant = fullfile(local_output, 'quant_maps.mat');

    fprintf('Downloading quant_maps.mat... ');
    scp_cmd = sprintf('scp %s %s/', remote_quant, local_output);
    [status, result] = system(scp_cmd);

    if status ~= 0
        error('Failed to download quant_maps.mat: %s', result);
    end

    % Verify file exists
    if ~exist(local_quant, 'file')
        error('Download appeared successful but file not found: %s', local_quant);
    end

    file_info = dir(local_quant);
    fprintf('✓ (%.1f KB, modified: %s)\n', file_info.bytes/1024, ...
        datestr(file_info.datenum, 'HH:MM:SS'));

    % Download dot_product_results
    fprintf('Downloading dot_product_results... ');
    remote_dpr = sprintf('%s@%s:%s/dot_product_results.*', ...
        dirstruct.cluster_user, ...
        dirstruct.cluster_host, ...
        remote_dir);

    scp_cmd = sprintf('scp %s %s/', remote_dpr, local_output);
    [status, result] = system(scp_cmd);

    if status == 0
        disp('✓');
    else
        warning('Failed to download dot_product_results: %s', result);
    end

    disp('========================================')
    disp('✓ CLUSTER PROCESSING COMPLETED')
    disp('========================================')
    disp(' ')

else
    %% ===== LOCAL PROCESSING MODE =====
    disp(['MRF data: Calling Python locally to perform dictionary simulation '...
        'and matching...'])

    home=pwd;
    cd(dirstruct.py_dir)

    % Set environment variable for large storage if specified
    if isfield(dirstruct,'cluster_large_storage_dir') && ~isempty(dirstruct.cluster_large_storage_dir)
        env_cmd = sprintf('export LARGE_STORAGE_DIR=%s;', dirstruct.cluster_large_storage_dir);
        fprintf('Using large storage directory: %s\n', dirstruct.cluster_large_storage_dir);
    else
        env_cmd = '';
    end

    % Run Python with both conda and venv activated
    system(['source ~/' dirstruct.bashfn ';'...
       'conda activate ' dirstruct.conda_env ';'...
       'source ' dirstruct.py_env '/bin/activate;'...
       env_cmd ...
       'python ' dirstruct.py_file ';']);

    cd(home);
end

%% Move results to data directory (same for both local and cluster)
disp('Moving results to data directory...')

% Move quant_maps.mat
source_quant = fullfile(dirstruct.py_dir,'OUTPUT_FILES','quant_maps.mat');
dest_quant = fullfile(dirstruct.loadMRF, dirstruct.MRFfn);

if exist(source_quant, 'file')
    % Ensure destination directory exists
    dest_dir = fileparts(dest_quant);
    if ~exist(dest_dir, 'dir')
        mkdir(dest_dir);
    end

    movefile(source_quant, dest_quant, 'f');
    fprintf('  ✓ Moved quant_maps.mat to: %s\n', dest_quant);
else
    error('quant_maps.mat not found in OUTPUT_FILES!');
end

% Move dot_product_results files
dpr_files = dir(fullfile(dirstruct.py_dir,'OUTPUT_FILES','dot_product_results.*'));

if ~isempty(dpr_files)
    % Handle custom naming if needed
    if ~strcmp(dirstruct.MRFfn,'quant_maps.mat')
        % Extract name modifier
        if contains(dirstruct.MRFfn,'quant_maps')
            namepart=extractBetween(dirstruct.MRFfn,'quant_maps','.mat');
        else
            namepart=extractBefore(dirstruct.MRFfn,'.mat');
        end
        if iscell(namepart)
            namepart=namepart{:};
        end
        if ~isempty(namepart)
            if ~startsWith(namepart,'_')
                namepart=['_' namepart];
            end
        end

        % Rename and move each dot_product_results file
        for i = 1:length(dpr_files)
            [~, ~, ext] = fileparts(dpr_files(i).name);
            new_name = ['dot_product_results' namepart ext];
            movefile(fullfile(dpr_files(i).folder, dpr_files(i).name), ...
                fullfile(dirstruct.loadMRF, new_name), 'f');
            fprintf('  ✓ Moved %s to: %s\n', dpr_files(i).name, ...
                fullfile(dirstruct.loadMRF, new_name));
        end
    else
        % Default naming - move all dot_product_results files
        for i = 1:length(dpr_files)
            movefile(fullfile(dpr_files(i).folder, dpr_files(i).name), ...
                dirstruct.loadMRF, 'f');
            fprintf('  ✓ Moved %s to: %s\n', dpr_files(i).name, dirstruct.loadMRF);
        end
    end
else
    warning('No dot_product_results files found in OUTPUT_FILES');
end

disp('✓ MRF data: Dictionary matching completed and results saved.')

end
