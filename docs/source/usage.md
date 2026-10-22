# Usage

Image-Matchmaker can be used in three ways:

1. **The CLI workflow** — run the whole pipeline from a config file (recommended).
2. **Individual scripts** — run a single pipeline stage from the command line.
3. **Python API** — import and call functions directly.

## 1. CLI workflow (recommended)

This is the recommended way to run image-matchmaker and is covered in the
{doc}`Quick Start <quickstart>`. The CLI uses Snakemake internally to chain the
pipeline steps together.

If you installed Image-Matchmaker from source in a Conda environment, activate that environment before running the commands. For a pip installation, use the Python environment in which the package is installed.

The available workflows are:

```bash
# full registration
image-matchmaker register --config <config.yaml> --cores 8

# apply the resulting transforms to other images
image-matchmaker apply --config <config.yaml> --cores 8
```

The `--config` option is required; no default configuration is provided.

### Command-line options

The same options are available for all three workflows.

| Option | Default | Description |
|---|---|---|
| `--config` | Required | Path to an existing YAML configuration file. |
| `--cores` | `8` | Maximum number of CPU cores Snakemake can use for the workflow. |
| `--work-dir` | `.` | Working directory used by Snakemake. Relative paths in the configuration are resolved against this directory, not necessarily against the directory from which you launch the command. |
| `--unlock` | Disabled | Unlock the Snakemake working directory after a previous run has left it locked. Use this to unlock the directory rather than to run the workflow. |
| `--rerun-incomplete` | Disabled | Rerun jobs whose output files are marked as incomplete. |

For example, to run the registration workflow with a specific working directory:

```bash
image-matchmaker register \
    --config /path/to/config.yaml \
    --work-dir /path/to/workdir \
    --cores 8
```

In this example, relative input and output paths in the configuration are interpreted relative to `/path/to/workdir`. Choose a working directory that makes those paths resolve correctly. If you use relative paths for the configuration file itself, consider providing an absolute path to avoid ambiguity when changing the working directory.

See the {doc}`Configuration Reference <config_ref>` for the config fields and
{doc}`Understanding the Outputs <outputs>` for details about the results produced by each run.

## 2. Running individual scripts

Each pipeline stage is a standalone command-line script under `image_matchmaker/`. The
Snakemake workflow simply chains them together, but you can also run a single stage
yourself. Every script accepts `--help` for the full list of options.

The stages, in pipeline order:

### `data_processing.raw_to_n5` — convert inputs to `.n5`

```bash
python -m image_matchmaker.data_processing.raw_to_n5 --input_path <img> --output_path <out.n5> \
    --output_key input --log_dir <dir> --x_res 1 --y_res 1 --z_res 1
```

| Option | Required | Description |
|--------|----------|-------------|
| `--input_path` (`-in`) | yes | Input image (`.tif` or `.n5`) |
| `--input_key` (`-ink`) | no | Key of the input image in `.n5` format |
| `--output_path` (`-out`) | yes | Output `.n5` path |
| `--output_key` (`-outk`) | yes | Key in the output `.n5` |
| `--log_dir` (`-log`) | yes | Log directory |
| `--x_res` / `--y_res` / `--z_res` | no | Voxel resolution (default `1.0`); image is read as (C)ZYX |

### `registration.prealignment` — SVD pre-alignment

| Option | Required | Description |
|--------|----------|-------------|
| `--fixed_path` (`-fi`) / `--fixed_key` (`-fk`) | yes | Fixed input `.n5` and key |
| `--fixed_spacing` (`-fs`) | yes | Fixed input spacing (3 values) |
| `--moving_path` (`-mi`) / `--moving_key` (`-mk`) | yes | Moving input `.n5` and key |
| `--moving_spacing` (`-ms`) | yes | Moving input spacing (3 values) |
| `--output_dir` (`-o`) / `--output_key` (`-ok`) | yes | Output directory and key |
| `--output_transform_path` (`-trans`) | yes | Path to write the final transform |
| `--axis_orientation` | yes | How to orient the principal axes (`auto`, `IDENTITY`, `X`, `Y`, `Z`) |
| `--save_tif` (`-tif`) | no | Also save a `.tif` (flag) |

### `registration.rigid_alignment_elastix` — rigid Elastix alignment

| Option | Required | Description |
|--------|----------|-------------|
| `--fixed_path` (`-fi`) / `--fixed_key` (`-fk`) | yes | Fixed prealigned input `.n5` and key |
| `--moving_path` (`-mi`) / `--moving_key` (`-mk`) | yes | Moving prealigned input `.n5` and key |
| `--output_dir` (`-o`) / `--output_key` (`-ok`) | yes | Output directory and key |
| `--save_tif` (`-tif`) | no | Also save a `.tif` (flag) |

### `registration.cpd_nonrigid_registration` — Coherent Point Drift

| Option | Required | Description |
|--------|----------|-------------|
| `--fixed_path` (`-fi`) / `--fixed_key` (`-fk`) | yes | Fixed prealigned input `.n5` and key |
| `--moving_path` (`-mi`) / `--moving_key` (`-mk`) | yes | Moving prealigned input `.n5` and key |
| `--output_dir` (`-o`) | yes | Output directory |
| `--w`, `--beta`, `--lmd`, `--maxiter` | yes | CPD parameters (see {doc}`Configuration Reference <config_ref>`) |

### `registration.match_pointclouds` — feature matching

| Option | Required | Description |
|--------|----------|-------------|
| `--fixed_pcd` (`-fcd`) | yes | Fixed point cloud |
| `--moving_pcd` (`-mpcd`) | yes | Registered moving point cloud |
| `--output_dir` (`-o`) | yes | Output directory |
| `--method` | no | Matching algorithm: `hungarian` (default), `ilp`, or `sinkhorn` |
| `--min_neighbours` | `ilp` only | Minimum neighbours to consider for matching (required for `--method ilp`, ignored otherwise) |
| `--max_dist` | no | Maximum distance between neighbours to consider (default `30`) |
| `--tau` | no | Sinkhorn entropy regularization (default `1.0`, `sinkhorn` only) |
| `--sinkhorn_max_iter` | no | Maximum Sinkhorn iterations (default `500`, `sinkhorn` only) |

The matching plot (`point_matching`, with the xy, xz and yz projections as three panels)
is written to a `plots/` subfolder of the output directory.

### `registration.elastix_deformable_pointset_registration` — deformable B-spline

| Option | Required | Description |
|--------|----------|-------------|
| `--fixed_path` (`-fi`) / `--fixed_key` (`-fk`) | yes | Fixed input `.n5` and key |
| `--moving_path` (`-mi`) / `--moving_key` (`-mk`) | yes | Moving input `.n5` and key |
| `--output_dir` (`-o`) / `--output_key` (`-ok`) | yes | Output directory and key |
| `--match_path` (`-match`) | yes | Correspondence table between fixed and moving instances |
| `--prealigned_output_key` (`-pok`) | yes | Output key for the result after applying the pre-alignment transform |
| `--prealignment_transform` (`-transform`) | yes | Pre-alignment transform path |

### `registration.apply_transform` — apply transforms to other images

| Option | Required | Description |
|--------|----------|-------------|
| `--moving_path` (`-mp`) / `--moving_key` (`-mk`) | yes | Moving input path and key |
| `--input_resolution` (`-ir`) | yes | Resolution of the moving input |
| `--output_path` (`-op`) / `--output_key` (`-ok`) | yes | Output path and key |
| `--output_resolution` (`-or`) | no | Resolution of the moving output |
| `--interpolation_order` (`-io`) | yes | Interpolation order (`0` for masks) |
| `--log_dir` (`-ld`) | yes | Log directory |
| `--parameter_map_path` (`-pm`) | yes | Elastix parameter map |
| `--prealignment_transform_path` (`-pt`) | no | Pre-alignment transform |
| `--fixed_path` (`-fi`) / `--fixed_key` (`-fk`) | no | Fixed input for overlay plots |
| `--verbose` (`-vb`) | no | Show verbose logs (flag) |

## 3. Python API

The package namespace is empty, so import from the submodules directly. Reusable
helpers are re-exported from `image_matchmaker.utils`:

```python
from image_matchmaker.utils import (
    read_volume, write_volume, get_attrs,   # .n5 / zarr I/O
    load_config,                            # parse a YAML config
    plot_three_slices, plot_overlay,        # visualization
    rotate_img, resample_volume,            # transforms
)
```

The point-cloud QC plots (`plot_pcd_overlay`, `plot_displacement_field`,
`plot_matching_qc`, `plot_landmark_overlay`) each draw the xy, xz and yz projections
as three panels in one figure. All but `plot_landmark_overlay` take a `projections` argument,
so `projections=("xz",)` gives a single view. All of them write through `PLOT_FORMAT` — see
the note below.

Higher-level stage functions are available from their modules, e.g.
`prealign_samples` / `run_prealignment` (`image_matchmaker.registration.prealignment`),
`run_cpd` (`image_matchmaker.registration.cpd_nonrigid_registration`), and `run_matching`
(`image_matchmaker.registration.match_pointclouds`).

A minimal example — pre-align two masks and save an overlay:

```python
import numpy as np

from image_matchmaker.registration.prealignment import prealign_samples
from image_matchmaker.utils import read_volume, plot_overlay, prealignment_spacing

seg_fixed = read_volume("fixed_image.n5", key="seg")
seg_moving = read_volume("moving_image.n5", key="seg")

# the spacings must be numpy arrays, in (z, y, x) order and µm
fixed_spacing = np.asarray([0.4, 0.4, 0.4], dtype=np.float32)
moving_spacing = np.asarray([0.5, 0.4, 0.4], dtype=np.float32)
new_spacing = np.asarray(prealignment_spacing(fixed_spacing), dtype=np.float32)

results = prealign_samples(
    seg_fixed, seg_moving, fixed_spacing, moving_spacing, new_spacing
)
fixed_prealigned = results["fixed"][0]
moving_prealigned = results["moving"][0]

plot_overlay(fixed_prealigned, moving_prealigned, save_path="overlay.pdf")
```

**Note on the output format:** every plotting helper saves through
`image_matchmaker.utils.vis.PLOT_FORMAT`, which is `"pdf"`. The suffix you pass in
`save_path` is *rewritten* to it, so asking for `overlay.png` still writes `overlay.pdf`.
Set `PLOT_FORMAT = None` to keep the suffix you pass.
