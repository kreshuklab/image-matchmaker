# Quick Start

## 1. Prepare the inputs

The registration pipeline expects:

- a fixed 3D instance segmentation mask
- a moving 3D instance segmentation mask

Currently supported input formats:

* `.tif`/`.tiff`
* `.n5`

You can either:

1. Prepare your own segmentation masks.
2. Use the example datasets provided with the repository (See {doc}`Example Data <example_data>`).

*Note:*
- Input volumes are expected in **ZYX** axis order.
- Voxel resolution must be provided for both fixed and moving images through the registration config.

---

## 2. Run registration

With the conda environment activated (e.g. `conda activate imm_env`), run the
registration workflow. `--configfile` is required — there is no default config:

```bash
snakemake -s workflows/registration.smk \
    --configfile /path/to/the/registration/config.yaml \
    --cores 8
```

To try it immediately on the bundled example data, first generate the example data
(see {doc}`Example Data <example_data>`) and then run the shipped config:

```bash
snakemake -s workflows/registration.smk \
    --configfile examples/register_config_test_rigid.yaml \
    --cores 8
```

### Registration config (example)

```yaml
fixed_image:
  path: /path/to/the/fixed/image
  x_res: 1
  y_res: 1
  z_res: 1

moving_image:
  path: /path/to/the/moving/image
  x_res: 1
  y_res: 1
  z_res: 1

log_dir: /path/to/the/log/directory
final_transform_path: /path/to/the/final_transform.json

prealignment:
  axis_orientation: "auto"

coherent_point_drift:
  w: 0.00001
  lmd: 0.1
  beta: 100
  maxiter: 100

matching:
  method: ilp        # ilp | hungarian | sinkhorn
  min_neighbours: 10
  max_dist: 30

mobie_export: False
mobie_dataset_name: "my_dataset"
```

See the {doc}`Configuration Reference <config_ref>` for the meaning of each field
and all available options.

### Registration outputs

All results are written under `log_dir`: transformed moving masks, the pre-alignment
and Elastix transform files, the table of correspondences between instances,
quality-control plots, and per-stage log files.

The final deformable transformation is stored as three sequential Elastix
transforms (rigid, rough B-spline, fine B-spline) in
`{log_dir}/05_elastix_deformable_pointset_registration/`.

For a full description of the output folder structure, the QC plots, and how to
judge whether the registration succeeded, see
{doc}`Understanding the Outputs <outputs>`.

---

## 3. Apply transforms to other images

After registration finishes successfully, the resulting transforms can be applied to other datasets (for example EM image or additional LM channels).

Run:

```bash
snakemake -s workflows/apply_transform.smk \
    --configfile /path/to/the/transform/config.yaml \
    --cores 8
```

### Transform config (example)

```yaml
fixed_image:
  input_path: /path/to/the/fixed/image
  input_key: "input"

moving_images:
  -
    input_path: /path/to/the/moving/image1
    input_key: "input"
    input_resolution:
      x_res: 1
      y_res: 1
      z_res: 1
    output_path: /path/to/the/output/image1
    output_key: "pointset_alignment_transform"
    output_resolution:
      x_res: 1
      y_res: 1
      z_res: 1
    interpolation_order: 0

  -
    input_path: /path/to/the/moving/image2
    input_key: "input"
    input_resolution:
      x_res: 1
      y_res: 1
      z_res: 1
    output_path: /path/to/the/output/image2
    output_key: "pointset_alignment_transform"
    output_resolution:
    interpolation_order: 0

log_dir: /path/to/the/log/directory
parameter_map_path: /path/to/the/output/parameter/map
prealignment_transform_path: /path/to/the/pre-alignment/transform
```

Both `parameter_map_path` and `prealignment_transform_path` point to files that the
registration run in step 2 wrote inside its `log_dir`:

- `parameter_map_path` — the final Elastix B-spline transform parameters.
- `prealignment_transform_path` — the SVD pre-alignment transform.

See {doc}`Understanding the Outputs <outputs>` for exactly where these files are
located, and the {doc}`Configuration Reference <config_ref>` for a description of
every transform config field.
