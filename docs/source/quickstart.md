# Quick Start

## 1. Prepare the inputs

The registration pipeline expects:

- a fixed 3D instance segmentation mask
- a moving 3D instance segmentation mask

Currently supported input formats:

* `.tif`/`.tiff`
* `.n5`

*Note:*
- Input volumes are expected in **ZYX** axis order.
- Voxel resolution must be provided for both fixed and moving images through the registration config.

---

## 2. Run registration

Run the registration workflow with:

```bash
snakemake -s workflows/registration.smk \
    --configfile /path/to/the/registration/config.yaml \
    --cores 8
```

### Registration config (example)

```yaml
fixed_image:
  path: /path/to/the/fixed/image
  output_name: "fixed_image"
  x_res: 1
  y_res: 1
  z_res: 1

moving_image:
  path: /path/to/the/moving/image
  output_name: "moving_image"
  x_res: 1
  y_res: 1
  z_res: 1

log_dir: /path/to/the/log/directory
```

### Registration outputs

The registration workflow generates:

* transformed moving segmentation masks
* pre-alignment transforms
* Elastix transform parameter files
* Table of correspondence between instances in moving and fixed masks
* quality-control plots
* log files for each registration stage

The final deformable transformation is typically stored in:

```text
{log_dir}/elastix_deformable_pointset_registration/TransformParameters.0.txt
{log_dir}/elastix_deformable_pointset_registration/TransformParameters.1.txt
{log_dir}/elastix_deformable_pointset_registration/TransformParameters.2.txt
```

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
    output_path: /path/to/the/output/image1
    output_key: "pointset_alignment_transform"
    x_res: 1
    y_res: 1
    z_res: 1
    interpolation_order: 0

  -
    input_path: /path/to/the/moving/image2
    input_key: "input"
    output_path: /path/to/the/output/image2
    output_key: "pointset_alignment_transform"
    x_res: 1
    y_res: 1
    z_res: 1
    interpolation_order: 0

log_dir: /path/to/the/log/directory
parameter_map_path: /path/to/the/output/parameter/map
prealignment_transform_path: /path/to/the/pre-alignment/transform
```
