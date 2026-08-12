# CPD Parameter Tuning

The Coherent Point Drift (CPD) step has four parameters (`w`, `beta`, `lmd`,
`maxiter`) whose best values depend on the data. Image-Matchmaker ships an
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
  name: "fixed_image"                        # optional, names the working .n5 in log_dir
  path: "data/.../fixed_image.n5"
  input_key: "input"
  aligned_key: "svd_prealignment_with_lm"   # key the SVD-aligned fixed image is written to
  x_res: 1.0
  y_res: 1.0
  z_res: 1.0

moving_image:
  name: "moving_image"                       # optional, names the working .n5 in log_dir
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

`suggest_cpd_ranges.py` implements the dataset-specific `beta` estimate. The optimization
calls it in-process, so it is not a separate workflow step, but it can also be run
standalone: it writes the full search space to `<log_dir>/dataset_cpd_ranges.yaml` and
echoes the suggested `beta` values, which you can paste into `search_space`:

```bash
python image_matchmaker/cpd_parameter_tuning/suggest_cpd_ranges.py \
    --path <segmentation>.n5 \
    --key svd_prealignment \
    --log_dir <output_dir> \
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
6. Plots the landmark overlays for every stage of the pipeline.

## Outputs

Like the main pipeline, each step writes to its own ordinal-prefixed subfolder of
`log_dir`, so they list in pipeline order:

```text
{log_dir}/
├── fixed_image.n5                     # one key per stage, see below
├── moving_image.n5
├── input_image_<name>.pdf             # slices of each raw input
├── image_matchmaker.log                     # main Snakemake log
├── raw_to_n5.log
├── 01_prepare_landmarks/
│   ├── landmark_label_ids.json        # {landmark name: label id} used by every later step
│   ├── add_landmarks.log
│   └── plots/                         # {fixed,moving}_landmarks_qc.pdf
├── 02_prealignment_with_lm/
│   ├── prealignment_transform.json
│   ├── prealignment.log
│   └── plots/
├── 03_rigid_alignment_with_lm/
│   ├── TransformParameters.0.txt
│   ├── result.0.{mhd,raw}             # Elastix warped result (intermediate, large)
│   ├── rigid_alignment.log
│   ├── elastix_log_rigid.log
│   └── plots/
├── 04_cpd_optimization/               # the grid search, see below
└── 05_landmark_overlays/              # the QC plots to look at first, see below
```

The `.n5` containers gain one key per stage: `input` (raw), `input_with_lm` (landmarks
embedded as labeled spheres), `svd_prealignment_with_lm`, and — moving image only —
`rigid_alignment_with_lm`. Note that pre-alignment writes **both** images under the key
named by `fixed_image.aligned_key`; only rigid alignment introduces a separate key for the
moving image, which is why `moving_image.aligned_key` names the rigid-alignment output.

The landmark QC plots in `01_prepare_landmarks/plots/` are worth a glance before anything
else: they show each segmentation with its embedded landmark spheres, so a landmark that
was mis-specified or fell outside the volume shows up immediately.

Written under `{log_dir}/04_cpd_optimization/`:

- `best_cpd_params.yaml` — the best parameters found. Drop it in place of the
  `coherent_point_drift` section of your main registration config.
- `study_results.csv` — every trial with its parameters and resulting mean LRE
  (sorted best-first).
- `trial_XXXX.pdf` — per-trial displacement-field plot, the xy, xz and yz projections as
  three panels in one figure, labeled with the trial's parameters and LRE.
- `trial_XXXX.json` — per-trial parameters and metrics.
- `registered_pcd.pcd` — the registered point cloud of the best trial, kept so the
  overlay below can be drawn without re-running the winning CPD fit.
- `optuna_study.db` — the Optuna study database (inspectable, resumable).

Written under `{log_dir}/05_landmark_overlays/`:

- `landmark_overlay_{input,prealignment,rigid_alignment,best_cpd}.pdf` — the
  corresponding landmarks fixed-vs-moving, one plot per pipeline stage, each with the xy,
  xz and yz projections as three panels. Every landmark appears twice — red at its fixed
  position, blue at its position at that stage — joined by a line, over a faint cloud of
  all instance centroids for context. The title carries that stage's mean LRE.
- `plot_overlays.log` — the per-stage mean LRE, also logged as a single summary line.

### Reading the overlays

The four plots are the quickest way to see whether the pipeline is behaving, because they
put the same measurement — mean LRE — on every stage:

- `input` is the unregistered baseline. The two clouds sit apart and the connecting lines
  are long; that is expected.
- `prealignment` and `rigid_alignment` should each pull the pairs closer. A stage that
  *increases* the LRE is a real signal, not noise: it means that step is not helping on
  this data, and its parameters (`prealignment.axis_orientation` in particular) are worth
  revisiting.
- `best_cpd` should be the tightest. If it is barely better than `rigid_alignment`, the
  search space is probably in the wrong region — see the `optuna.search_space` section
  above.

Long lines that all point the same way indicate a residual global offset, which pre-alignment
or rigid alignment should have removed. Long lines pointing in scattered directions indicate
either genuinely poor CPD parameters or mis-specified landmark correspondences.

### Redrawing the plots

Only the `best_cpd` overlay depends on the search, and it is drawn from the saved
`registered_pcd.pcd` rather than by re-fitting. So all four can be regenerated cheaply —
delete them and re-run snakemake, which reruns just the `plot_overlays` step:

```bash
rm {log_dir}/05_landmark_overlays/*.pdf
snakemake -s workflows/cpd_optimization.smk --configfile <your config> --cores 4
```

The same script can be called directly, which is useful when tweaking a plot:

```bash
python image_matchmaker/cpd_parameter_tuning/plot_overlays.py \
    --config <your config> \
    --fixed_path {log_dir}/<fixed_name>.n5 \
    --moving_path {log_dir}/<moving_name>.n5
```
