# Usage

Image-Matchmaker can be used in three ways:

1. **The Snakemake workflow** — run the whole pipeline from a config file
   (recommended).
2. **Individual scripts** — run a single pipeline stage from the command line.
3. **Python API** — import and call functions directly.

## 1. Snakemake workflow (recommended)

This is the recommended way to run image-matchmaker and is covered in the
{doc}`Quick Start <quickstart>`. Activate the conda environment first
(e.g. `conda activate imm_env`); the workflows do not manage the environment
for you. `--configfile` is required — the workflows no longer ship with a default
config. The two workflows are:

```bash
# full registration
snakemake -s workflows/registration.smk --configfile <config.yaml> --cores 8

# apply the resulting transforms to other images
snakemake -s workflows/apply_transform.smk --configfile <config.yaml> --cores 8
```

See the {doc}`Configuration Reference <config_ref>` for the config fields and
{doc}`Understanding the Outputs <outputs>` for what each run produces.

## 2. Running individual scripts

Each pipeline stage is a standalone command-line script under `image_matchmaker/`. The
Snakemake workflow simply chains them together, but you can also run a single stage
yourself. Every script accepts `--help` for the full list of options.

The stages, in pipeline order:

### `raw_to_n5.py` — convert inputs to `.n5`

```bash
python image_matchmaker/raw_to_n5.py --input_path <img> --output_path <out.n5> \
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

### `prealignment.py` — SVD pre-alignment

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

### `rigid_alignment_elastix.py` — rigid Elastix alignment

| Option | Required | Description |
|--------|----------|-------------|
| `--fixed_path` (`-fi`) / `--fixed_key` (`-fk`) | yes | Fixed prealigned input `.n5` and key |
| `--moving_path` (`-mi`) / `--moving_key` (`-mk`) | yes | Moving prealigned input `.n5` and key |
| `--output_dir` (`-o`) / `--output_key` (`-ok`) | yes | Output directory and key |
| `--save_tif` (`-tif`) | no | Also save a `.tif` (flag) |

### `cpd_nonrigid_registration.py` — Coherent Point Drift

| Option | Required | Description |
|--------|----------|-------------|
| `--fixed_path` (`-fi`) / `--fixed_key` (`-fk`) | yes | Fixed prealigned input `.n5` and key |
| `--moving_path` (`-mi`) / `--moving_key` (`-mk`) | yes | Moving prealigned input `.n5` and key |
| `--output_dir` (`-o`) | yes | Output directory |
| `--w`, `--beta`, `--lmd`, `--maxiter` | yes | CPD parameters (see {doc}`Configuration Reference <config_ref>`) |

### `match_pointclouds.py` — feature matching

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

The three matching plots (`point_matching_{xz,yz,xy}`) are written to a `plots/`
subfolder of the output directory.

### `elastix_deformable_pointset_registration.py` — deformable B-spline

| Option | Required | Description |
|--------|----------|-------------|
| `--fixed_path` (`-fi`) / `--fixed_key` (`-fk`) | yes | Fixed input `.n5` and key |
| `--moving_path` (`-mi`) / `--moving_key` (`-mk`) | yes | Moving input `.n5` and key |
| `--output_dir` (`-o`) / `--output_key` (`-ok`) | yes | Output directory and key |
| `--match_path` (`-match`) | yes | Correspondence table between fixed and moving instances |
| `--prealigned_output_key` (`-pok`) | yes | Output key for the result after applying the pre-alignment transform |
| `--prealignment_transform` (`-transform`) | yes | Pre-alignment transform path |

### `apply_transform.py` — apply transforms to other images

| Option | Required | Description |
|--------|----------|-------------|
| `--moving_path` (`-mp`) / `--moving_key` (`-mk`) | yes | Moving input path and key |
| `--moving_resolution` (`-mr`) | yes | Resolution of the moving input |
| `--output_path` (`-op`) / `--output_key` (`-ok`) | yes | Output path and key |
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

Higher-level stage functions are available from their modules, e.g.
`prealign_samples` / `run_prealignment` (`image_matchmaker.prealignment`),
`run_cpd` (`image_matchmaker.cpd_nonrigid_registration`), and `run_matching`
(`image_matchmaker.match_pointclouds`).

A minimal example — pre-align two masks and save an overlay:

```python
from image_matchmaker.prealignment import prealign_samples
from image_matchmaker.utils import read_volume, plot_overlay

seg_fixed = read_volume("fixed_image.n5", key="seg")
seg_moving = read_volume("moving_image.n5", key="seg")

results = prealign_samples(seg_fixed, seg_moving)
fixed_prealigned = results["fixed"][0]
moving_prealigned = results["moving"][0]

plot_overlay(fixed_prealigned, moving_prealigned, save_path="overlay.png")
```

For complete, runnable examples of the Python API see `examples/registration_test.py`
(pre-alignment). Refer to each function's source for its full signature.
