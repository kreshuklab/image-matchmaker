# Configuration Reference

## Registration config

Configuration file passed to `workflows/registration.smk` via `--configfile`.
See `examples/register_config_test_rigid.yaml` and
`examples/register_config_test_elastic.yaml` for complete working examples.

### `fixed_image` / `moving_image`

Registration input segmentation masks.

#### `path`

Path to the input segmentation mask (`.tif`/`.tiff` or `.n5`).

#### `source_path` (optional, only for tests)

Path to the original, undeformed segmentation mask. Only used by the example
data generation script (`examples/deform_test_data.py`) to synthesize deformed
moving data; not required for a normal registration run.

#### `input_key` (optional)

Dataset key to read from when the input `path` is an `.n5` file.

#### `name` (optional)

Base name for the output `.n5` container written under `log_dir`. Defaults to
`fixed_image` / `moving_image` (producing `fixed_image.n5` / `moving_image.n5`).

#### `x_res`, `y_res`, `z_res`

Voxel resolution along each axis.

#### `ref_path` / `ref_url` (optional, only for tests, `moving_image` only)

Reference result used by the automated tests: `ref_path` is the local path to a
known-good registered output, and `ref_url` is the URL it is downloaded from if
missing. Leave empty (or omit) for usual runs.

---

### `log_dir`

Directory where logs, plots, intermediate files, and registration outputs are
saved.

### `final_transform_path`

Path where the combined final transform is written.

---

### `prealignment`

Settings for the SVD pre-alignment step.

#### `axis_orientation`

How the moving volume's principal axes are oriented to match the fixed volume.
One of:

- `auto` — estimate the orientation automatically from the samples' intensity
  profiles (rotation overlays are also saved so the estimate can be checked).
- `IDENTITY` — apply no additional axis rotation.
- `X`, `Y`, `Z` — rotate the moving volume 180° around the given axis.

---

### `coherent_point_drift`

Parameters for the non-rigid Coherent Point Drift (CPD) registration
(via *probreg*).

#### `w`

Outlier weight — the assumed fraction of points with no correspondence
(noise/outliers). Range `0`–`1`. Default: `0.00001`.

#### `beta`

Width of the Gaussian smoothing kernel; controls how strongly neighbouring
points move together (larger = smoother, more rigid deformation).
Default: `100`.

#### `lmd`

Regularization weight (`lambda`); trades off goodness of fit against the
smoothness of the deformation. Default: `0.1`.

#### `maxiter`

Maximum number of EM iterations. Typically `100`–`150`; can be lowered
(e.g. `10`–`20`) to speed up debugging runs. Default: `100`.

---

### `matching`

Parameters for the feature-matching step that establishes correspondences
between instances.

#### `method` (optional)

Matching algorithm to use. One of:

- `hungarian` — optimal one-to-one assignment over the full cost matrix
  (uses `max_dist`; `min_neighbours` is ignored).
- `ilp` — sparse integer linear program; a one-to-many candidate matching built
  from each point's nearest neighbours (uses `min_neighbours` and `max_dist`).
- `sinkhorn` — soft (entropy-regularized) assignment, then discretized to a
  one-to-one matching (uses `max_dist`, `tau`, and `max_iter`).

Defaults to `hungarian` if the key is omitted, both when run through the Snakemake
workflow and via `match_pointclouds.py` directly.

#### `max_dist` (optional)

Maximum distance between neighbours considered for matching. Used by all methods.
Default: `30`. Note this is a distance in the data's physical units, so the right
value depends on your voxel resolution and object spacing — the `30` default suits
the example data and is not universally appropriate.

#### `min_neighbours` (optional, `ilp` only)

Minimum number of neighbours considered when matching a point. Required when
`method` is `ilp`; ignored (and not needed) by `hungarian` and `sinkhorn`.
Example: `10`.

#### `tau` (optional, `sinkhorn` only)

Sinkhorn entropy-regularization parameter scaling distances into similarities.
Default: `1.0`.

#### `max_iter` (optional, `sinkhorn` only)

Maximum number of Sinkhorn iterations. Default: `500`. Note the config key is
`max_iter`, while the corresponding CLI flag on `match_pointclouds.py` is
`--sinkhorn_max_iter`.

---

### MoBIE export

Options controlling whether results are exported to a
[MoBIE](https://mobie.github.io/) project.

#### `mobie_export`

If `True`, exports the raw data and each registration stage to a MoBIE project
under `{log_dir}/mobie_project/`. Set to `False` to skip.

#### `semantic_seg` (optional)

If `True`, treats the input as a semantic segmentation during MoBIE export.
Defaults to `False` when omitted.

#### `mobie_dataset_name`

Name of the MoBIE dataset to create.

---

## Transform config

Configuration file passed to `workflows/apply_transform.smk` via `--configfile`.

### `fixed_image` (optional)

Optional fixed image used to generate overlay visualizations of the warped result.
If omitted, the overlay is skipped.

#### `input_path`

Path to the fixed image.

#### `input_key`

Dataset key if the input is an `.n5` file.

### `moving_images` (required)

List of moving images to transform. Multiple moving inputs can be processed in
a single run.

#### `input_path` / `output_path`

Input and output image paths.

#### `input_key` / `output_key`

Dataset key used for `.n5` files.

#### `x_res`, `y_res`, `z_res`

Voxel resolution along each axis.

#### `input_resolution` and `output_resolution`

Voxel resolution along each axis using:

* `x_res`: Voxel resolution along the x-axis.
* `y_res`: Voxel resolution along the y-axis.
* `z_res`: Voxel resolution along the z-axis.

`input_resolution` is required. `output_resolution` is optional and can either be omitted or left empty. If it is omitted or left empty, the output resolution is determined using the registration (fixed) image resolution.

#### `interpolation_order`

B-spline interpolation order used during resampling.

For segmentation masks, use:

```yaml
interpolation_order: 0
```

For intensity images, higher interpolation orders are recommended for smoother
results, at the cost of increased runtime.

### `parameter_map_path`

Path to the final Elastix B-spline transform parameters produced by the
registration run. This is the fine B-spline transform,
typically:

```text
{log_dir}/05_elastix_deformable_pointset_registration/TransformParameters.2.txt
```

### `prealignment_transform_path` (optional)

Path to the SVD pre-alignment transform produced by the registration run,
typically:

```text
{log_dir}/01_svd_prealignment/svd_prealignment_transform.json
```

If provided, the pre-alignment transform is applied before the Elastix transforms.

See {doc}`Understanding the Outputs <outputs>` for exactly where both files are
located within `log_dir`.
