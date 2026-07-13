# CPD Parameter Tuning

The Coherent Point Drift (CPD) step has four parameters (`w`, `beta`, `lmd`,
`maxiter`) whose best values depend on the data. Matchmaker ships an
[Optuna](https://optuna.org/)-based grid search that tunes them automatically by
minimizing the mean **Landmark Registration Error (LRE)** between user-provided
corresponding landmarks after CPD.

This is a **separate** Snakemake workflow (`workflows/cpd_optimization.smk`), not
part of the main registration pipeline. Its output, `best_cpd_params.yaml`, is a
drop-in replacement for the `coherent_point_drift` section of your registration
config.

The source and defaults live under `image_matchmaker/cpd_parameter_tuning/`.

## Input landmarks

You must provide pairs of corresponding landmarks — one CSV for the fixed image and
one for the moving image. Each CSV has the columns `name, x, y, z`, where
coordinates are in physical µm and `name` matches between the two files so
corresponding landmarks can be paired:

```text
name,x,y,z
landmark_1,12.4,8.1,30.0
landmark_2,40.2,15.7,28.5
```

The repository does not ship example landmark CSVs; the paths in the example config
are placeholders. Point the `landmarks` section of the config at your own files.

## Parameters tuned

| Parameter | Role |
|-----------|------|
| `beta` | Gaussian kernel width — smoothness/locality of the displacement field. Larger = smoother, more global. |
| `w` | Outlier weight — fraction of points treated as noise. Higher = more robust but less accurate. |
| `lmd` | Regularization strength — larger = stiffer, penalizes deformation more. |
| `maxiter` | Maximum EM iterations. |

Defaults (when not tuning) are in `default_cpd_params.yaml`; the default grid-search
ranges are in `default_cpd_ranges.yaml`:

```yaml
# default_cpd_ranges.yaml
w:       [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]
beta:    [10.0, 50.0, 100.0, 200.0]
lmd:     [0.01, 0.1, 1.0, 10.0]
maxiter: [100]
```

## Configuration

The optimization is driven by `examples/cpd_optimization_config.yaml`. Key fields:

```yaml
fixed_image:
  path: "data/.../fixed_image.n5"
  input_key: "input"
  aligned_key: "svd_prealignment_with_lm"   # key the SVD-aligned fixed image is written to
  x_res: 1.0
  y_res: 1.0
  z_res: 1.0

moving_image:
  path: "data/.../moving_image.n5"
  input_key: "input"
  aligned_key: "rigid_alignment_with_lm"     # key the rigid-aligned moving image is written to
  x_res: 1.0
  y_res: 1.0
  z_res: 1.0

landmarks:
  fixed: "examples/data/fixed_landmarks.csv"
  moving: "examples/data/moving_landmarks.csv"

prealignment:
  axis_orientation: "auto"

log_dir: "data/.../cpd_optimization"

optuna:
  study_name: "cpd_optimization"
  search_space: "default"
  n_jobs: 1                                    # trials to evaluate in parallel
```

### `optuna.search_space`

Controls which parameter ranges the grid search uses. Three options:

- `"default"` — use the ranges from `default_cpd_ranges.yaml`.
- `"dataset-specific"` — derive the `beta` range from the spatial extent/density of
  the aligned fixed point cloud (computed automatically during optimization); the
  other parameters come from the defaults.
- an explicit dict — specify the ranges directly, e.g.:

  ```yaml
  search_space:
    w: [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2]
    beta: [50.0, 100.0, 200.0]
    lmd: [0.01, 0.1, 1.0]
    maxiter: [150]
  ```

`suggest_cpd_ranges.py` implements the dataset-specific `beta` estimate and can also
be run standalone to print a YAML block you can paste into `search_space`:

```bash
python image_matchmaker/cpd_parameter_tuning/suggest_cpd_ranges.py \
    --path <segmentation>.n5 \
    --key svd_prealignment \
    --x_res 0.4 --y_res 0.4 --z_res 0.4
```

## Parallelization

### `optuna.n_jobs`

Number of trials Optuna evaluates concurrently (default `1`). The point clouds are
extracted once and reused across all trials, so each worker just runs one CPD fit;
raising `n_jobs` runs several parameter combinations at the same time and is the main
way to speed up the search. A good starting point is the number of physical cores
available.

Note that Snakemake's `--cores` does **not** parallelize the search — only a single
`optimize_cpd` job runs, and it fans out internally according to `n_jobs`.

### `LOCAL_TMPDIR` (environment variable)

With `n_jobs > 1` the parallel workers all write to the Optuna SQLite study
database, and SQLite's file locking is unreliable on shared/network filesystems
(e.g. Lustre), causing `database is locked` errors. To avoid this, the study DB is
kept on **node-local disk** during the run and copied back to
`{log_dir}/04_cpd_optimization/optuna_study.db` when it finishes (even on failure).

By default the local copy goes under the system temp directory. On a cluster, set
`LOCAL_TMPDIR` to a node-local scratch path so the DB does not land on the shared
filesystem:

```bash
export LOCAL_TMPDIR=/scratch/$USER   # node-local disk
snakemake -s workflows/cpd_optimization.smk \
          --configfile examples/cpd_optimization_config.yaml \
          --cores 1
```

If a previous run left an `optuna_study.db`, it is seeded into the local copy first,
so an interrupted search resumes where it left off (`load_if_exists`).

## Running it

With the conda environment activated:

```bash
snakemake -s workflows/cpd_optimization.smk \
          --configfile examples/cpd_optimization_config.yaml \
          --cores 1
```

The workflow:

1. Converts the input segmentations to `.n5`.
2. Embeds the corresponding landmarks into both segmentations as labeled spheres.
3. Applies SVD pre-alignment to the fixed image (carrying its landmarks).
4. Applies SVD + rigid alignment to the moving image (carrying its landmarks).
5. Runs the Optuna grid search over CPD parameters, evaluating each combination by
   the mean LRE between corresponding landmarks after CPD.

## Outputs

Written under `{log_dir}/04_cpd_optimization/`:

- `best_cpd_params.yaml` — the best parameters found. Drop it in place of the
  `coherent_point_drift` section of your main registration config.
- `study_results.csv` — every trial with its parameters and resulting mean LRE
  (sorted best-first).
- `trial_XXXX_{xy,xz,yz}.png` — per-trial displacement-field plots, labeled with the
  trial's parameters and LRE.
- `trial_XXXX.json` — per-trial parameters and metrics.
- `optuna_study.db` — the Optuna study database (inspectable, resumable).
