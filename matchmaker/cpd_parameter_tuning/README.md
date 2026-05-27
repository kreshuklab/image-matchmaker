# CPD Hyperparameter Tuning

Optuna-based grid search over CPD (Coherent Point Drift) nonrigid registration parameters,
evaluated via Landmark Registration Error (LRE) on user-provided corresponding landmarks.

## Parameters being tuned

| Parameter | Role |
|-----------|------|
| `beta` | Gaussian kernel width — controls smoothness/locality of the displacement field. Larger = smoother, more global deformation. |
| `w` | Outlier weight — fraction of points treated as noise. Higher = more robust to outliers but less accurate. |
| `lmd` (lambda) | Regularization strength — stiffness of the deformation. Larger = stiffer, less flexible. |
| `maxiter` | Maximum EM iterations. |

Search ranges can be suggested automatically from point-cloud statistics using
`suggest_cpd_ranges.py` (see below).

## Evaluation metric

**Landmark Registration Error (LRE)** — mean Euclidean distance (µm) between corresponding
landmark positions in the fixed image and the CPD-registered moving image. Lower is better.

## Landmark input format

Two CSV files are required, one for the fixed image and one for the moving image:

```
name,x,y,z
landmark_a,52.48,68.48,54.04
landmark_b,62.08,66.88,56.00
...
```

- `name`: unique identifier for the landmark.
- `x`, `y`, `z`: landmark position in physical units (µm).
- Correspondence between fixed and moving landmarks is established by matching `name` values. Only landmarks whose name appears in **both** files are used; landmarks present in only one file are ignored with a warning.

Both files must use the same coordinate space as the segmentation volumes passed to the pipeline.

## Workflow

### 1. Embed landmarks and run alignment

Landmarks are embedded as small spheres (filled, radius=3 voxels) with high label IDs into
the **raw (unaligned) instance segmentations** of both images, before any registration takes
place. The script then computes SVD prealignment and elastix rigid alignment itself, carrying
the landmark labels through both transforms together with the nuclei.

Both images are processed together in a single call since SVD prealignment requires both
volumes simultaneously.

```bash
python matchmaker/cpd_parameter_tuning/add_landmarks.py \
    --fixed_path <fixed.n5> \
    --fixed_key <raw_fixed_key> \
    --fixed_output_key <fixed_aligned_with_lm_key> \
    --fixed_landmarks_csv <fixed_landmarks.csv> \
    --moving_path <moving.n5> \
    --moving_key <raw_moving_key> \
    --moving_output_key <moving_aligned_with_lm_key> \
    --moving_landmarks_csv <moving_landmarks.csv> \
    --axis_orientation <X|Y|Z|IDENTITY|auto> \
    --log_dir <output_dir>
```

### 2. (Optional) Suggest parameter search ranges

Derives a `beta` range from point-cloud spatial extent and prints a YAML block ready to
paste into the optimization config.

```bash
python matchmaker/cpd_parameter_tuning/suggest_cpd_ranges.py \
    --path <fixed.n5> \
    --key <prealigned_key> \
    --x_res <µm> --y_res <µm> --z_res <µm>
```

### 3. Run the grid search

```bash
python matchmaker/cpd_parameter_tuning/cpd_optimization.py \
    --config <cpd_optimization_config.yaml>
```

The grid search runs all parameter combinations defined in the config.
Results are written to the `log_dir` specified in the config.

## Outputs

| File | Description |
|------|-------------|
| `best_cpd_params.yaml` | Best parameters in pipeline-config format, ready to paste into the main pipeline config |
| `study_results.csv` | All trials sorted by mean LRE |
| `trial_NNNN.json` | Per-trial results including per-landmark distances |
| `landmark_label_ids.json` | Mapping of landmark names to label IDs (written by `add_landmarks.py`, required by the grid search) |

## Config file

The optimization is configured via a YAML file. See
`data/brain_matching/cpd_optimization_config.yaml` for a full example. Key sections:

```yaml
fixed_image:
  path: <path to .n5>
  aligned_key: <dataset key with landmarks embedded>
  x_res: <µm>
  y_res: <µm>
  z_res: <µm>

moving_image:
  # same structure as fixed_image

landmarks:
  fixed: <fixed_landmarks.csv>
  moving: <moving_landmarks.csv>

log_dir: <output directory>

optuna:
  study_name: <name>
  search_space:
    w: [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2]
    beta: [50.0, 100.0, 200.0]
    lmd: [0.01, 0.1, 1.0]
    maxiter: [100, 150]
```
