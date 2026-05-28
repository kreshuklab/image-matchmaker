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

### Registration config

Example configuration:

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

#### Parameters

##### `fixed_image` / `moving_image`

Registration input segmentation masks.

##### `path`

Path to the input segmentation mask.

##### `output_name`

Output key used within the generated `.n5` folder.

##### `x_res`, `y_res`, `z_res`

Voxel resolution along each axis.

##### `log_dir`

Directory where logs, plots, intermediate files, and registration outputs are saved.

---

## 3. Apply transforms to other images

After registration finishes successfully, the resulting transforms can be applied to other datasets (for example EM image or additional LM channels).

Run:

```bash
snakemake -s workflows/apply_transform.smk \
    --configfile /path/to/the/transform/config.yaml \
    --cores 8
```

### Transform config

Example configuration:

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

#### Parameters

##### `fixed_image` (optional)

Optional fixed image used for overlay visualization.

##### `input_path`

Path to the fixed image.

##### `input_key`

Dataset key if the input is an `.n5` file.

##### `moving_images` (required)

List of moving images to transform.

Multiple moving inputs can be processed in a single run.

##### `input_path` / `output_path`

Input and output image paths.

##### `input_key` / `output_key`

Dataset key used for `.n5` files.

##### `x_res`, `y_res`, `z_res`

Voxel resolution along each axis.

##### `interpolation_order`

B-spline interpolation order used during resampling.

For segmentation masks, use:

```yaml
interpolation_order: 0
```

For intensity images, higher interpolation orders are recommended for smoother results, at the cost of increased runtime.

##### `parameter_map_path`

Path to the Elastix transform parameter file.

Typically:

```text
TransformParameters.2.txt
```

##### `prealignment_transform_path` (optional)

Path to the SVD pre-alignment transform:

```text
svd_prealignment_transform.json
```

If provided, the pre-alignment transform is applied before the Elastix transforms.
