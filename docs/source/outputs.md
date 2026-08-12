# Understanding the Outputs

When the registration workflow runs, all results are written under the directory
set by `log_dir` in the registration config. This page explains the folder layout
and, for each registration step, the files it produces and the quality-control
(QC) plots it generates.

Throughout the plots, the **fixed** image is shown in **pink** and the **moving**
image in **cyan** (semantic coloring); instance segmentations are rendered with a
distinct color per label. Plots are saved in a `plots/` subfolder of each stage.
The output format is configurable (all PDF or PNG or individually per plot).

## Output directory layout

Each registration step writes to its own subfolder of `log_dir`. These subfolders
are prefixed with a two-digit ordinal (`01_`, `02_`, …) so they list in pipeline
order.

```text
{log_dir}/
├── fixed_image.n5                     # fixed image, one key per stage (+ *_binary keys)
├── moving_image.n5                    # moving image, one key per stage (+ *_binary keys)
├── input_image_<name>.pdf             # slices of each raw input
├── image_matchmaker.log                     # main Snakemake log
├── raw_to_n5.log
├── mobie_export.log
├── 01_svd_prealignment/
│   ├── svd_prealignment_transform.json
│   ├── prealignment.log
│   ├── manual_prealignment_options/   # 180° rotation overlays (IDENTITY/X/Y/Z)
│   └── plots/
├── 02_rigid_alignment/
│   ├── TransformParameters.0.txt
│   ├── result.0.{mhd,raw}             # Elastix warped result (intermediate)
│   ├── rigid_alignment.log
│   ├── elastix_log_rigid.log
│   └── plots/
├── 03_cpd_nonrigid_registration/
│   ├── fixed_pcd.pcd
│   ├── moving_pcd.pcd
│   ├── registered_pcd.pcd
│   ├── cpd_nonrigid_registration.log
│   └── plots/
├── 04_match_pointclouds/
│   ├── matched_labels.csv
│   ├── matched_idx_pairs.txt
│   ├── match_pointclouds.log
│   └── plots/
├── 05_elastix_deformable_pointset_registration/
│   ├── TransformParameters.0.txt      # rigid
│   ├── TransformParameters.1.txt      # rough B-spline
│   ├── TransformParameters.2.txt      # fine B-spline
│   ├── fixed_pointset.{csv,txt}
│   ├── moving_pointset.{csv,txt}
│   ├── fixed_pcd.pcd
│   ├── moving_pcd.pcd
│   ├── result.{0,1,2}.*               # Elastix warped results (intermediate)
│   ├── IterationInfo.*.txt            # Elastix optimization logs
│   ├── elastix_deformable_pointset_registration.log
│   ├── elastix_log_deformable.log
│   └── plots/
└── mobie_project/                     # only if mobie_export: True
```

Each stage writes its own `<stage>.log` (and the Elastix stages an additional
`elastix_log_*.log`); the main Snakemake log is `image_matchmaker.log`. The `*_binary`
n5 keys are binarized copies of each stage used for the MoBIE export.

## Outputs by registration step

Steps run in order; each brings the moving volume progressively closer to the
fixed volume. For every step below, the **output files** it produces are listed
first, followed by the **QC plots** you can use to check it.

### Input preparation

Converts the fixed and moving inputs into the internal `.n5` format.

**Output files**

- `fixed_image.n5`, `moving_image.n5` — the inputs stored under the `input` key.

**QC plots** (in `{log_dir}/`)

- `input_image_<name>.pdf` — orthogonal slices of each raw input (named after the
  input file), to confirm the data loaded with the expected dimensions.

### Pre-alignment (SVD)

Global alignment of centroids and principal axes.

**Output files**

- `01_svd_prealignment/svd_prealignment_transform.json` — the pre-alignment transform
  (see structure below). The prealigned volumes are stored under the
  `svd_prealignment` key of both `.n5` files.

**QC plots** (in `01_svd_prealignment/plots/`, except where noted)

- `fixed_input` / `moving_input` (and `*_semantic` variants) — slices of each input
  with the PCA centre of mass and principal axes overlaid.
- `overlay_input` — fixed and moving overlaid **before** any alignment; expect a
  clear mismatch.
- `overlay_after_prealignment` — overlay after pre-alignment and axis orientation;
  centroids and principal axes should now roughly coincide.
- `overlay_after_prealignment_before_axis_orient` — intermediate overlay before the
  axis-orientation correction, useful for diagnosing flips.
- `fixed_prealigned` / `moving_prealigned` — each volume after the pre-alignment
  transform.
- `axis_int_profile_X/Y/Z` — intensity profiles along each axis, used to auto-detect
  whether the moving volume needs to be flipped.
- `manual_prealignment_options/` (`IDENTITY`, `X`, `Y`, `Z`) — overlays showing the
  effect of a 180° rotation around each axis. If `auto` picks the wrong orientation,
  use these to choose the correct `axis_orientation` value (see
  {doc}`Configuration Reference <config_ref>`). Saved directly in
  `01_svd_prealignment/manual_prealignment_options/`, not in `plots/`.

The transform JSON holds a 4×4 affine matrix and output shape for both volumes:

```text
{
  "fixed_prealignment":  { "matrix": [[...]], "output_shape": [z, y, x] },
  "moving_prealignment": { "matrix": [[...]], "output_shape": [z, y, x] }
}
```

It is the file passed as `prealignment_transform_path` when applying transforms to
other images.

### Rigid alignment (Elastix)

Corrects remaining rotation and translation differences.

**Output files**

- `02_rigid_alignment/TransformParameters.0.txt` — Elastix rigid transform
  (ITK/Elastix format). The aligned moving volume is stored under the
  `rigid_alignment` key of `moving_image.n5`. Elastix also writes an intermediate
  warped image (`result.0.mhd`/`result.0.raw`).

**QC plots** (in `02_rigid_alignment/plots/`)

- `overlay_after_rigid_alignment` — overlay after the rigid step; offsets remaining
  after pre-alignment should be corrected here.

### Coherent Point Drift (CPD)

Non-rigid alignment of the point clouds.

**Output files** (in `03_cpd_nonrigid_registration/`)

- `fixed_pcd.pcd`, `moving_pcd.pcd`, `registered_pcd.pcd` — point clouds in Open3D
  ASCII PCD format (`x y z label`).

**QC plots** (in `03_cpd_nonrigid_registration/plots/`)

- `pcds_before_registration` / `pcds_after_registration` — fixed and moving point clouds
  projected onto each plane before and after CPD, with the xy, xz and yz projections as
  three panels in one figure; the two clouds should overlap more closely afterwards.
- `displacement_field` — vectors showing how each point moved during CPD, with the xy, xz
  and yz projections as three panels in one figure. A smooth, coherent field is good;
  **large or discontinuous displacements usually indicate a problem in an earlier step**.

### Feature matching

Finds correspondences between instances.

**Output files** (in `04_match_pointclouds/`)

- `matched_labels.csv` — correspondence table with columns `fixed_label_id`,
  `moving_label_id`; the central result linking instances across the two volumes.
- `matched_idx_pairs.txt` — the same matches as index pairs, one per line.

**QC plots** (in `04_match_pointclouds/plots/`)

- `point_matching` — fixed and moving point clouds with lines drawn between matched
  instances, with the xy, xz and yz projections as three panels in one figure. Lines
  should connect nearby, corresponding structures;
  long crossing lines suggest incorrect matches (consider tuning the `matching`
  parameters or switching `matching.method` — see
  {doc}`Configuration Reference <config_ref>`).

### Deformable B-spline registration (Elastix)

Final deformable alignment using the matched landmarks.

**Output files** (in `05_elastix_deformable_pointset_registration/`)

- `TransformParameters.0.txt`, `TransformParameters.1.txt`,
  `TransformParameters.2.txt` — sequential transforms: `0` rigid, `1` rough
  B-spline, `2` fine B-spline. The aligned moving volume is stored under two keys of
  `moving_image.n5`: `pointset_alignment` — the final result, back in the original
  input space (this is the key exported to MoBIE) — and
  `pointset_alignment_prealignment_space` (the same result in pre-alignment space).
- `fixed_pointset.{csv,txt}`, `moving_pointset.{csv,txt}` and
  `fixed_pcd.pcd`, `moving_pcd.pcd` — the matched instance centroids used as Elastix
  corresponding points. Elastix also writes intermediate warped images
  (`result.*`) and optimization logs (`IterationInfo.*.txt`).

**QC plots** (in `05_elastix_deformable_pointset_registration/plots/`)

- `deformable_pointset_alignment_before` / `deformable_pointset_alignment_final` —
  overlays before and after the final deformable step (a `*_semantic` and a
  `*_prealigned` variant are also written).
- `grid_before` / `grid_after` — a regular grid warped by the deformation. The grid
  should deform smoothly and **must not fold over itself** (folding indicates an
  implausible, unstable deformation).

The combined final transform is also written to the path set by
`final_transform_path` in the registration config.

## Applying the transforms

To warp additional images (e.g. raw EM or extra LM channels) with these results,
use the transform workflow described in the {doc}`Quick Start <quickstart>`. The
fine B-spline `TransformParameters.2.txt` is typically the `parameter_map_path`,
and `01_svd_prealignment/svd_prealignment_transform.json` the
`prealignment_transform_path` (see {doc}`Configuration Reference <config_ref>`).

## Quality checklist

A registration has likely succeeded when:

- the `overlay_*` plots show fixed and moving converging from stage to stage, with
  the final overlay closely aligned;
- the CPD `displacement_field` is smooth and coherent (no large, erratic vectors);
- the `point_matching_*` lines connect nearby corresponding structures rather than
  crossing over long distances;
- the deformable `grid_after` deforms smoothly without folding.

If any of these look wrong, revisit the relevant stage's parameters in the
{doc}`Configuration Reference <config_ref>` — most commonly `axis_orientation`
(pre-alignment), the `coherent_point_drift` parameters (CPD), or the `matching`
parameters (feature matching).
