# Automatic Tube Detection for CEST Phantom Analysis

## Overview

The automatic tube detection feature eliminates the need for manual ROI (Region of Interest) drawing by automatically identifying and segmenting phantom tubes in CEST-MRF images.

## Features

- **Automatic phantom outline detection**: Identifies the phantom boundary using Otsu thresholding and morphological operations
- **Background subtraction**: Uses Gaussian smoothing to estimate and remove background signal
- **Connected component analysis**: Identifies individual tubes based on size and shape
- **Circular tube ordering**: Automatically sorts detected tubes by their angular position around the phantom center
- **Adjustable parameters**: Fine-tune detection sensitivity through GUI controls

## Usage

### Basic Usage

1. Load your CEST-MRF study data using `PROC_MRF_STUDY.m`
2. **Select the appropriate display group:**
   - **Recommended: zSpec group with M0 image** (best for tube detection)
   - Alternative: MRF group (uses dot product or T1w/T2w images)
3. In the interactive GUI, click the **"Auto-detect tubes"** button
4. The algorithm will automatically detect all phantom tubes and create ROIs
5. Review the detected ROIs overlaid on your images
6. If needed, adjust detection parameters (see below) and re-run

### Which Image is Used for Detection?

The algorithm automatically selects the best image based on your active display group:

- **zSpec group**: Uses **M0 reference image** (recommended - cleanest signal without saturation)
- **MRF group**: Uses dot product image, or falls back to T1w/T2w
- **other group**: Uses T1w IR, T2w MSME, or B0 maps

The GUI will display "Detecting on: [image type]" to show which image is being used.

### Detection Parameters

The GUI provides two adjustable parameters for fine-tuning detection:

- **Gauss sigma** (default: 4)
  - Controls the amount of Gaussian smoothing applied
  - Higher values = more smoothing (better for noisy images)
  - Lower values = less smoothing (better for sharp tube boundaries)

- **Min tube pixels** (default: 10)
  - Minimum number of pixels required for a region to be considered a tube
  - Increase this to filter out small noise artifacts
  - Decrease this to detect smaller tubes

### Replace vs. Append ROIs

When auto-detection is run and ROIs already exist, you'll be prompted to:
- **Replace**: Delete all existing ROIs and use only the newly detected tubes
- **Append**: Keep existing ROIs and add the newly detected tubes
- **Cancel**: Abort the detection and keep existing ROIs

## Algorithm Details

The automatic tube detection algorithm consists of six main steps:

### Step 1: Phantom Outline Detection
- Apply Gaussian smoothing to reduce noise
- Normalize image intensity to [0,1]
- Apply Otsu's automatic thresholding
- Extract largest connected component (phantom boundary)
- Fill holes and smooth outline with morphological operations

### Step 2: Background Subtraction
- Estimate background by heavy Gaussian filtering
- Subtract background from original image
- Binarize result using Otsu's method
- Mask out regions outside phantom outline

### Step 3: Connected Component Filtering
- Identify all connected components in binary image
- Remove largest component (often phantom interior, not individual tubes)
- Filter out very small components (noise)
- Fill holes within remaining components
- Remove tubes touching image borders

### Step 4: Tube Labeling
- Label each remaining connected component as a tube
- Extract centroid coordinates for each tube

### Step 5: Circular Ordering
- Calculate phantom center from outline
- Compute angular position of each tube relative to center
- Sort tubes counter-clockwise from rightmost position
- Assign sequential tube indices (Tube1, Tube2, ...)

### Step 6: ROI Structure Creation
- Extract boundary coordinates for each tube
- Create binary mask for each ROI
- Store all ROI information compatible with existing pipeline
- Initialize nominal concentration and exchange rate fields to NaN

## Output

Each detected tube generates an ROI structure with the following fields:
- `coords`: Boundary coordinates [x, y] for visualization
- `mask`: Binary mask indicating tube pixels
- `name`: Tube identifier (e.g., "Tube1", "Tube2", ...)
- `nomConc`: Nominal concentration (initialized to NaN)
- `nomExch`: Nominal exchange rate (initialized to NaN)
- `center`: [x, y] coordinates of tube centroid
- `tubeIndex`: Sequential tube number in circular order

## Troubleshooting

### "No suitable image found" error
- **For zSpec group**: Ensure M0 reference image is loaded
  - M0 image is the unsaturated reference scan
  - Check that your data loading includes M0 data
- **For MRF group**: Ensure dot product or T1w/T2w images are available
- Try switching to a different display group (use dropdown at top of GUI)

### No tubes detected
- **Best practice**: Use zSpec group with M0 image for cleanest detection
- Try adjusting the **Gauss sigma** parameter:
  - Increase for noisy images (try 5-8)
  - Decrease for high-contrast images (try 2-3)
- Adjust **Min tube pixels** based on your tube size
- Verify the correct image group is selected (MRF, other, or zSpec)
- Check that tubes are visible in the displayed image

### Too many false detections
- Increase **Min tube pixels** to filter out small artifacts
- Ensure phantom is properly positioned (not touching image borders)

### Tubes numbered incorrectly
- The circular ordering sorts tubes counter-clockwise from the rightmost position
- Tube numbering starts from angular position 0° (right side of phantom)
- If consistent ordering is needed, manually rename ROIs after detection

### Detection takes too long
- Reduce image resolution before detection (if acceptable for your analysis)
- Use lower **Gauss sigma** value for faster processing

## Integration with Existing Workflow

The automatic tube detection is fully integrated with the existing CEST-MRF processing pipeline:

1. Detected ROIs are stored in the same format as manually drawn ROIs
2. All downstream analysis functions work identically
3. ROI statistics are calculated automatically after detection
4. Results can be saved and loaded like manual ROIs
5. Detected ROIs can be manually edited or refined if needed

## Function Reference

### `autoDetectTubes.m`
Core detection algorithm that processes images and returns ROI structures.

**Syntax:**
```matlab
roi = autoDetectTubes(img, imageSize, gaussSigma, minTubePixels, ...
                      roiNamePrefix, sortCircular)
```

**Parameters:**
- `img`: 2D image matrix to analyze
- `imageSize`: [height, width] of the image
- `gaussSigma`: Sigma for Gaussian smoothing (default: 4)
- `minTubePixels`: Minimum pixels per tube (default: 10)
- `roiNamePrefix`: Prefix for ROI names (default: 'Tube')
- `sortCircular`: Sort tubes by circular position (default: true)

### `imageDispGUI.m` - `autoDetectROIs()` callback
GUI callback function that:
- Selects appropriate image for detection
- Calls `autoDetectTubes()` with user parameters
- Handles ROI replacement/appending
- Updates GUI displays
- Shows success/error messages

## Technical Requirements

- MATLAB Image Processing Toolbox
- Functions used: `imgaussfilt`, `imbinarize`, `bwconncomp`, `regionprops`, `bwboundaries`, `imfill`, `imopen`, `imclearborder`

## Version History

- **v1.0** (2025): Initial implementation with background subtraction and circular ordering

## Contact

For questions, issues, or feature requests related to automatic tube detection, please contact the repository maintainer.
