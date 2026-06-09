# CPD Hyperparameter Tuning

Optuna-based grid search over CPD (Coherent Point Drift) nonrigid registration parameters,
evaluated on user-provided corresponding landmarks.

## Parameters

| Parameter | Role |
|-----------|------|
| `beta` | Gaussian kernel width — controls smoothness/locality of the displacement field. Larger = smoother, more global deformation. |
| `w` | Outlier weight — fraction of points treated as noise. Higher = more robust to outliers but less accurate. |
| `lmd` | Regularization strength. Larger = stiffer, penalizes deformation more strongly. |
| `maxiter` | Maximum EM iterations. |

### Default CPD parameters

saved in `default_cpd_params.yaml`

```
beta = 100
w = 1e-5
lmd = 0.1
maxiter = 100
```

### Default CPD grid search ranges

saved in `default_cpd_ranges.yaml`

```
DEFAULT_SEARCH_SPACE = {
    "w":       [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "beta":    [10.0, 50.0, 100.0, 200.0],
    "lmd":     [0.01, 0.1, 1.0, 10.0],
    "maxiter": [100],
}
```

### Dataset specific CPD grid search ranges

run `suggest_cpd_ranges.py`: output `dataset_cpd_ranges.yaml

Suggest dataset-specific beta parameter search ranges for use in cpd_optimization_config.yaml (set `search_space: "dataset-specific"`).

Reads a segmentation and computes basic point-cloud statistics (spatial extent, point density)
to derive a beta search range proportional to the data scale.  Prints a YAML block that can
be pasted into the optuna.search_space section of the optimization config.

Usage:
    python matchmaker/cpd_parameter_tuning/suggest_cpd_ranges.py \
        --path data/brain_matching/igor_fixed_image.n5 \
        --key svd_prealignment \
        --x_res 0.4 --y_res 0.4 --z_res 0.4`


## Snakemake workflow

Snakemake workflow for CPD parameter optimization via Optuna.

Separate from the main registration pipeline. Performs:
  1. Embed anatomical landmarks into the input segmentations as single-voxel labels.
  2. Apply the stored SVD prealignment transform to the fixed (Igor) image with landmarks.
  3. Apply the stored SVD + rigid transforms to the moving (Seymour) image with landmarks.
  4. Optionally compute dataset-specific beta ranges from the aligned fixed segmentation.
  5. Run an Optuna grid search over CPD parameters, evaluating each combination by the
     mean Landmark Registration Error (LRE) between corresponding landmarks after CPD.

Output: best_cpd_params.yaml (drop-in replacement for the coherent_point_drift section
of the main registration config).

Usage:
```
snakemake -s workflows/cpd_optimization.smk \
          --configfile data/brain_matching/cpd_optimization_config.yaml \
          --cores 1
```