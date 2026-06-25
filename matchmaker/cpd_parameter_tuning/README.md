# CPD Hyperparameter Tuning

Optuna-based grid search over CPD (Coherent Point Drift) nonrigid registration parameters,
evaluated on user-provided corresponding landmarks.

## Input landmarks

You must supply your own pairs of corresponding landmarks — one CSV for the fixed image and
one for the moving image — and point to them in the `landmarks` section of
`examples/cpd_optimization_config.yaml`. Each CSV has the columns `name, x, y, z`, where
coordinates are in physical µm and `name` matches between the two files so corresponding
landmarks can be paired:

```
name,x,y,z
landmark_1,12.4,8.1,30.0
landmark_2,40.2,15.7,28.5
```

The repository does not ship example landmark CSVs; the example config paths are placeholders.

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

### Dataset-specific CPD grid search ranges

`suggest_cpd_ranges.py` derives a `beta` search range proportional to the data scale.
It reads a segmentation, computes basic point-cloud statistics (spatial extent, point
density), and prints a YAML block that can be pasted into the `optuna.search_space`
section of the optimization config.

When the config sets `search_space: "dataset-specific"`, the Snakemake workflow runs this
step automatically on the aligned fixed segmentation. You can also run it manually:

```
python matchmaker/cpd_parameter_tuning/suggest_cpd_ranges.py \
    --path <segmentation>.n5 \
    --key svd_prealignment \
    --x_res 0.4 --y_res 0.4 --z_res 0.4
```


## Snakemake workflow

Snakemake workflow for CPD parameter optimization via Optuna.

Separate from the main registration pipeline. Performs:
  1. Embed the corresponding landmarks into both input segmentations as labeled spheres.
  2. Apply SVD prealignment to the fixed image (carrying its landmarks).
  3. Apply SVD + rigid alignment to the moving image (carrying its landmarks).
  4. Optionally compute dataset-specific beta ranges from the aligned fixed segmentation
     (when `search_space: "dataset-specific"`).
  5. Run an Optuna grid search over CPD parameters, evaluating each combination by the
     mean Landmark Registration Error (LRE) between corresponding landmarks after CPD.

Output: `best_cpd_params.yaml` (drop-in replacement for the `coherent_point_drift` section
of the main registration config).

Usage:
```
snakemake -s workflows/cpd_optimization.smk \
          --configfile examples/cpd_optimization_config.yaml \
          --cores 1
```