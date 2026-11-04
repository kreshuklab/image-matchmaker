# Registration limit experiments: test setup

## Goal

We want to find where the image-matchmaker (IMM) registration of an elastically deformed instance segmentation breaks down. To do this, we perturb the moving segmentation more and more and measure the landmark registration error (LRE) between the fixed and the registered moving segmentation.

All scripts are run from `registration_limit_experiments/`.

## Folder layout

```
registration_limit_experiments/
├── data/
│   ├── platy1_muscles_stardist_fixed_rotated.tif     # fixed image, shared by all settings
│   ├── removal_stats.tsv                             # instances removed per setting
│   └── deformed_data_<tag>/
│       ├── platy1_muscles_stardist_elastic.tif       # moving image that is registered
│       ├── platy1_muscles_stardist_elastic_full.tif  # moving image with all instances, used for the LRE
│       ├── deform_test_data.log
│       └── plots/
├── results/
│   ├── <tag>/                                        # registration and apply-transform output
│   ├── lre_imm.tsv                                   # IMM LRE summary of all settings
│   └── lre_nucleisky.tsv                             # NucleiSky3D LRE summary of all settings
└── metric_ids.txt                                    # landmark IDs
```

## Step 01: create the data (`01_create_deform_data.py`)

Source: `../example_data/platy1_muscles_stardist_fixed.tif`, a StarDist instance segmentation of platy muscle nuclei with 13,756 instances and isotropic spacing of 1 (so voxels = µm).

Data is generated with `deform_test_data()` from `image_matchmaker/data_processing/deform_test_data.py`, with seed 42.

- **Fixed image:** source rotated by `(20, 345, 30)`° and cropped to its bounding box. It is the same for all settings.
- **Moving image:** source run through this sequence:
  1. Elastic deformation on a control grid (`alpha=0.9`, `sigma=2`, `grid_spacing=16`). Because the seed is reset for every setting, all settings have the same elastic field. It is identical to the one of `example_data/deformed_data/platy1_muscles_stardist_elastic.tif`.
  2. Rotation by `(155, 30, 65)`°.
  3. Jitter of instances (only `ji` settings).
  4. Removal of instances (only `ri` and `sphere` settings).
  5. Cropping to the bounding box.
- **Full moving image (`_full`):** the moving image after steps 1 and 2, so without jitter and without removal. It is cropped with the same bounding box as the moving image, so both have the same voxel frame. Instances that do not lie completely inside this bounding box are removed, so all kept instances have their true centroid.

Instance IDs stay the same between fixed and moving, so the IDs give the ground-truth correspondence.

### Settings (22 in total)

| Method | Tag | What it does | Values |
|---|---|---|---|
| Random instance removal | `ri-<f>` | Removes a random fraction `f` of all instances. | f = 0.1, 0.3, 0.5, 0.7, 0.9 |
| Sphere removal, center 1 (head) | `sphere-center1-<f>` | Sets everything inside an ellipsoid to background. Its diameter is `f` times each side of the foreground bounding box. `f` is not the fraction of removed instances. Center `[159, 203, 228]` (zyx, in the rotated, uncropped moving volume). | f = 0.1, 0.2, 0.3, 0.4, 0.5 |
| Sphere removal, center 2 (tail) | `sphere-center2-<f>` | Same, with center `[139, 118, 90]`. | f = 0.1, 0.2, 0.3, 0.4, 0.5 |
| Instance jitter | `ji-<f>` | Shifts each instance as a whole by a random integer offset in `[-d, d]` per axis, with `d = round(f * mean instance distance)`. | f = 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5 → d = 1, 2, 3, 4, 6, 7, 9 voxels |

- The mean instance distance is the mean distance from each instance centroid to its nearest neighbour centroid in the source. It is 3.74 voxels.
- No setting removes a random 5 % of instances on top of its own perturbation.

### Removal statistics

Step 01 also writes `data/removal_stats.tsv`, with one row per setting: the number of moving instances after the elastic deformation and rotation, the number left in the saved moving image, and the number and percentage removed. For the sphere settings this is the real fraction of removed instances. For the `ji` settings it shows instances lost by jitter collisions.

## Step 02: select the landmarks (`02_set_metric_ids.py`)

Picks a random 5 % of the instance IDs of the fixed image (about 690, seed 42) and saves them to `metric_ids.txt`. The same landmarks are used for all settings.

## Step 03: run the registration (`03_run_pipeline.py`)

Runs the IMM `register` Snakemake workflow with `register_config_test_elastic.yaml` on each `data/deformed_data_<tag>/`. The fixed image is the shared one, the moving image is `platy1_muscles_stardist_elastic.tif` of the setting. Output goes to `results/<tag>/`. If a setting fails, the error is printed and the setting is skipped.

Workflow steps and parameters:

| Step | Parameters |
|---|---|
| SVD pre-alignment | `axis_orientation: auto` |
| Rigid alignment (elastix) | from the code |
| CPD non-rigid registration | `w=1e-5`, `lmd=0.1`, `beta=100`, `maxiter=100` |
| Point matching | `hungarian`, `max_dist=30` (`min_neighbours` is only used by `ilp`) |
| Elastix deformable point-set registration | from the code. Writes the warped moving image to `moving_image.n5`, key `pointset_alignment`, and the final transform `TransformParameters.2.txt`. |

## Step 04: apply the transform to the full moving image (`04_run_transform.py`)

Runs the IMM `apply` workflow with `register_config_test_elastic_apply_transform.yaml` for each `results/<tag>/` that has a final `TransformParameters.2.txt`.

- **Input:** `data/deformed_data_<tag>/platy1_muscles_stardist_elastic_full.tif`.
- **Transform:** `TransformParameters.2.txt`, with nearest-neighbour interpolation. It maps the moving image onto the original fixed image.
- **Output:** `results/<tag>/moving_image.n5`:
  - key `full_pointset_alignment_transform`: in the fixed space. This is used for the LRE.
  - key `full_pointset_alignment_transform_prealigned`: additionally rotated into the pre-alignment space with `svd_prealignment_transform.json`.

## Step 05: compute the LRE (`05_compute_dis_imm.py`)

The LRE functions are in `lre.py`, which is shared with step 07, so IMM and NucleiSky3D are evaluated in the same way.

For each `results/<tag>/`, computes the Euclidean distance between the centroid of each landmark in the fixed image and in the warped moving image, in voxels (= µm).

Variants:

| Variant | Warped moving image | Settings | Meaning |
|---|---|---|---|
| `full` | `full_pointset_alignment_transform` | all | How well the transform maps the instances to their true positions. |
| `jitter` | `pointset_alignment` | `ji` only | Where the jittered instances land. The error includes the jitter. |

For the removal settings, a `jitter`-like variant on `pointset_alignment` would give the same distances as `full`, just on fewer landmarks. Instances on the border of a removal sphere are only partly removed there, so their centroid would be wrong.

A landmark is skipped if:
- it is missing in the warped image. This happens if it was lost in the elastic deformation or rotation (4 instances), if it was cut by the bounding box in step 01, or if it was mapped outside the fixed volume.
- it touches an outer face of the warped image, because then it may be cut and its centroid is wrong. This also drops the few uncut instances that define the edges of the fixed volume.

Metrics:
- **Mean**: mean over all distances.
- **P95 / P80**: mean over the distances at or below the 95th / 80th percentile.

Output:
- `results/lre_imm.tsv`: one row per setting and variant, with the number of used, missing and cut landmarks and the metrics. The file is rewritten on every run.
- `results/<tag>/lre_imm_<variant>.txt`: the distance for each landmark.

## Step 06: register with NucleiSky3D (`06_run_nucleisky.py`)

NucleiSky3D is the baseline. It runs in its own environment, because the `nucleisky` package needs Python >= 3.12 and SimpleITK (`pip install "nucleisky[simpleitk]"`). The script does not import `image_matchmaker`.

- Computes the NucleiSky features and centroids (in µm, spacing 1) of the shared fixed image once.
- For each `data/deformed_data_<tag>/`, registers the moving image to the fixed image with the `pyramid` matcher. NucleiSky3D finds a similarity transform (scale, rotation, translation), so unlike IMM it cannot recover the elastic deformation.
- Applies the transform with nearest-neighbour interpolation to the same images as in step 04, and writes them in the fixed space:
  - `results/<tag>/nucleisky_full.tif`: the full moving image, for all settings.
  - `results/<tag>/nucleisky_jitter.tif`: the jittered moving image, for `ji` settings only.
- If NucleiSky3D finds no match or fails, the setting is printed and skipped.

## Step 07: compute the NucleiSky3D LRE (`07_compute_dis_nucleisky.py`)

Runs in `imm_env`. Computes the LRE of `nucleisky_full.tif` and `nucleisky_jitter.tif` with `lre.py`, so with the same landmarks, skip rules, variants and metrics as step 05.

Output:
- `results/lre_nucleisky.tsv`: one row per setting and variant, with the same columns as `lre_imm.tsv`.
- `results/<tag>/lre_nucleisky_<variant>.txt`: the distance for each landmark.
