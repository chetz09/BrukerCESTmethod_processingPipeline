function [roi, detectedCenters, detectedRadii] = autoDetectPhantomTubes(img, settings, currentROI, options)
% autoDetectPhantomTubes: Automatically detects circular phantom tubes in images
% and creates ROI structures for each detected tube
%
%   INPUTS:
%       img         - Struct containing images (must have at least one image group)
%       settings    - Struct with plotting settings (contains .plotgrp field)
%       currentROI  - Existing ROI struct (can be empty)
%       options     - Optional struct with detection parameters:
%                     .minRadius - Minimum tube radius in pixels (default: 10)
%                     .maxRadius - Maximum tube radius in pixels (default: 50)
%                     .sensitivity - Detection sensitivity 0-1 (default: 0.9)
%                     .edgeThreshold - Edge detection threshold (default: auto)
%                     .method - 'hough' or 'regionprops' (default: 'hough')
%
%   OUTPUTS:
%       roi             - Updated ROI struct with detected tubes
%       detectedCenters - Nx2 array of [x,y] centers for detected tubes
%       detectedRadii   - Nx1 array of radii for detected tubes
%

%% Parse inputs and set defaults
if nargin < 4 || isempty(options)
    options = struct();
end

% Set default parameters
if ~isfield(options, 'minRadius'), options.minRadius = 10; end
if ~isfield(options, 'maxRadius'), options.maxRadius = 50; end
if ~isfield(options, 'sensitivity'), options.sensitivity = 0.9; end
if ~isfield(options, 'method'), options.method = 'hough'; end

% Initialize output
roi = currentROI;
if isfield(roi, 'name')
    nROI = numel(roi);
else
    nROI = 0;
end

%% Select image to use for detection
% Try to use the most informative image available
if isfield(img, settings.plotgrp)
    % Use the current display group
    imgStruct = img.(settings.plotgrp);

    % Priority order for image selection within group
    if isfield(imgStruct, 'fs')
        detectionImg = imgStruct.fs;  % Concentration map - usually good contrast
    elseif isfield(imgStruct, 't2w')
        detectionImg = imgStruct.t2w;  % T2-weighted
    elseif isfield(imgStruct, 't1w')
        detectionImg = imgStruct.t1w;  % T1-weighted
    elseif isfield(imgStruct, 'dp')
        detectionImg = imgStruct.dp;   % Dot product
    elseif isfield(imgStruct, 'M0img')
        detectionImg = imgStruct.M0img;  % M0 image
    elseif isfield(imgStruct, 'avgZspec')
        detectionImg = mean(imgStruct.avgZspec, 3);  % Average z-spectrum
    else
        % Use first available field
        flds = fieldnames(imgStruct);
        detectionImg = imgStruct.(flds{1});
        if ndims(detectionImg) > 2
            detectionImg = detectionImg(:,:,1);
        end
    end
else
    error('No valid image group found for tube detection');
end

% Ensure 2D image
if ndims(detectionImg) > 2
    detectionImg = detectionImg(:,:,1);
end

%% Image preprocessing
% Normalize to 0-1 range
detectionImg = double(detectionImg);
detectionImg = detectionImg - min(detectionImg(:));
detectionImg = detectionImg / max(detectionImg(:));

% Apply median filter to reduce noise
detectionImg = medfilt2(detectionImg, [3 3]);

%% Detect tubes based on selected method
switch lower(options.method)
    case 'hough'
        % Use Circular Hough Transform for robust circle detection
        [detectedCenters, detectedRadii] = detectTubesHough(detectionImg, options);

    case 'regionprops'
        % Use morphological operations and regionprops
        [detectedCenters, detectedRadii] = detectTubesRegionProps(detectionImg, options);

    otherwise
        error('Invalid detection method. Use ''hough'' or ''regionprops''');
end

%% Create ROI structures for each detected tube
if isempty(detectedCenters)
    warning('No tubes detected. Try adjusting detection parameters.');
    return;
end

fprintf('Detected %d phantom tubes.\n', size(detectedCenters, 1));

% Get image size for mask creation
imgHeight = size(detectionImg, 1);
imgWidth = size(detectionImg, 2);

% Create ROI for each detected tube
for tubeIdx = 1:size(detectedCenters, 1)
    center = detectedCenters(tubeIdx, :);
    radius = detectedRadii(tubeIdx);

    % Create circular polygon coordinates
    theta = linspace(0, 2*pi, 50);
    x = center(1) + radius * cos(theta);
    y = center(2) + radius * sin(theta);
    coords = [x', y'];

    % Create binary mask
    [X, Y] = meshgrid(1:imgWidth, 1:imgHeight);
    mask = ((X - center(1)).^2 + (Y - center(2)).^2) <= radius^2;

    % Add to ROI structure
    roi(nROI + tubeIdx).coords = coords;
    roi(nROI + tubeIdx).mask = mask;
    roi(nROI + tubeIdx).name = sprintf('Tube%d', tubeIdx);
    roi(nROI + tubeIdx).nomConc = NaN;
    roi(nROI + tubeIdx).nomExch = NaN;
    roi(nROI + tubeIdx).autoDetected = true;  % Flag for auto-detection
    roi(nROI + tubeIdx).center = center;
    roi(nROI + tubeIdx).radius = radius;
end

end


%% HELPER FUNCTION: Circular Hough Transform Detection
function [centers, radii] = detectTubesHough(img, options)
    % Edge detection
    if isfield(options, 'edgeThreshold')
        edges = edge(img, 'Canny', options.edgeThreshold);
    else
        edges = edge(img, 'Canny');
    end

    % Circular Hough Transform
    [centers, radii] = imfindcircles(img, [options.minRadius, options.maxRadius], ...
        'ObjectPolarity', 'bright', ...
        'Sensitivity', options.sensitivity, ...
        'Method', 'TwoStage', ...
        'EdgeThreshold', 0.1);

    % If no bright circles found, try dark circles
    if isempty(centers)
        [centers, radii] = imfindcircles(img, [options.minRadius, options.maxRadius], ...
            'ObjectPolarity', 'dark', ...
            'Sensitivity', options.sensitivity, ...
            'Method', 'TwoStage', ...
            'EdgeThreshold', 0.1);
    end

    % Sort by radius (largest first) for consistency
    if ~isempty(centers)
        [radii, sortIdx] = sort(radii, 'descend');
        centers = centers(sortIdx, :);
    end
end


%% HELPER FUNCTION: Region Properties Detection
function [centers, radii] = detectTubesRegionProps(img, options)
    % Threshold image (Otsu's method)
    level = graythresh(img);
    BW = imbinarize(img, level);

    % Try inverted if needed
    if sum(BW(:)) > numel(BW) / 2
        BW = ~BW;
    end

    % Morphological operations to clean up
    BW = imfill(BW, 'holes');
    BW = bwareaopen(BW, round(pi * options.minRadius^2));  % Remove small objects

    % Get region properties
    stats = regionprops(BW, 'Centroid', 'Area', 'Circularity', 'EquivDiameter');

    % Filter by circularity and size
    validIdx = ([stats.Circularity] > 0.7) & ...  % Reasonably circular
               ([stats.EquivDiameter]/2 >= options.minRadius) & ...
               ([stats.EquivDiameter]/2 <= options.maxRadius);

    stats = stats(validIdx);

    if isempty(stats)
        centers = [];
        radii = [];
        return;
    end

    % Extract centers and radii
    centers = reshape([stats.Centroid], 2, [])';  % Nx2 array
    radii = [stats.EquivDiameter]' / 2;  % Nx1 array

    % Sort by area (largest first)
    areas = [stats.Area]';
    [~, sortIdx] = sort(areas, 'descend');
    centers = centers(sortIdx, :);
    radii = radii(sortIdx);
end
