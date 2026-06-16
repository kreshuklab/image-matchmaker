# Configuration Reference

### Registration config

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

### Transform config

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
