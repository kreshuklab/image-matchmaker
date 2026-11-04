# Handoff: registration limit experiments

Read `test_setup.md` first. It describes the experiment design, the data layout and every step. This file adds what is needed to continue the work: the current state, how to run the steps, what to check after each step, and the decisions that are already made.

## Current state (2026-11-04)

- All scripts (steps 01–07) are written and match `test_setup.md`. **None of them has been run on the real data yet.** There are no results.
- Tested so far: only on small synthetic volumes (instance jitter, full moving image and cut instances, removal stats, logging, LRE functions, steps 05 and 07 end to end). Step 06 is only syntax-checked, because `nucleisky` was not installed. The removal plot was checked on real data for `ri-0.3` and `sphere-center1-0.3`.
- The old results (`test_summary.md`, old `example_data/deform_data_*` and `data/test_elastic_registration_*` folders) are from an earlier setup with a different elastic field and are not comparable. Do not mix them with the new results.
- `data/platy1_muscles_stardist_elastic.tif` is a leftover and not used by any step.

## Code changes that the scripts depend on

The scripts need changes in the `image_matchmaker` package that are **not committed yet** (branch `registration-tests`, base commit `663660e`). Commit them, or copy them to the cluster, before running anything:

| File | Change |
|---|---|
| `image_matchmaker/data_processing/deform_test_data.py` | Mode `jitter_instances` (old names `translate_instances`/`move_voxels` are gone); `save_full_moving` option that writes `<moving>_full.tif`; logs and plots go to the moving image folder, the fixed plot to the fixed image folder; returns `{"num_before", "num_left"}`. |
| `image_matchmaker/utils/log_utils.py` | `force=True` in `logging.basicConfig`, so each `deform_test_data` call in one process logs to its own folder. |
| `image_matchmaker/utils/vis.py` | `plot_removal`: same color per instance in the original and degraded row, removed voxels in red; the `mode` argument is removed. |
| `image_matchmaker/utils/transform_utils.py` | `pad_to_common_bbox` is removed (no longer used). |
| `environment.yml` | `cupy` is commented out. Check if the cluster needs it. |

`registration_limit_experiments/` itself is untracked.

## Environments

- `imm_env` (Python 3.11, `image_matchmaker` installed): steps 01–05 and 07.
- A separate NucleiSky environment (Python >= 3.12, `pip install "nucleisky[simpleitk]" tqdm`): step 06 only. Step 06 does not import `image_matchmaker`.

## Inputs

- Source segmentation: `../example_data/platy1_muscles_stardist_fixed.tif` (relative to `registration_limit_experiments/`, as set in `register_config_test_elastic.yaml`). Copy it to the cluster at the same relative path, or change `source_path`.

## How to run

Run every script from `registration_limit_experiments/`, in this order:

| Step | Command | Env | Notes |
|---|---|---|---|
| 01 | `python 01_create_deform_data.py` | imm | Writes the shared fixed image, 22 `data/deformed_data_<tag>/` folders and `data/removal_stats.tsv`. Runs all settings in one process. |
| 02 | `python 02_set_metric_ids.py` | imm | Needs the fixed image from step 01. Writes `metric_ids.txt`. Run it once and keep the file: all results must use the same landmarks. |
| 03 | `python 03_run_pipeline.py` | imm | Longest step: the full IMM pipeline for 22 settings, `maxiter=100`. Output in `results/<tag>/`. |
| 04 | `python 04_run_transform.py` | imm | Needs step 03. |
| 05 | `python 05_compute_dis_imm.py` | imm | Needs steps 02 and 04. |
| 06 | `python 06_run_nucleisky.py` | nucleisky | Needs step 01 only, so it can run in parallel to steps 03–05. |
| 07 | `python 07_compute_dis_nucleisky.py` | imm | Needs steps 02 and 06. |

Cluster notes:
- Steps 03 and 04 delete `.snakemake/` in the working directory before each setting. Do not run two instances of step 03 or 04 in the same directory at the same time.
- To run settings of step 03 in separate jobs, use `select_folders` in `03_run_pipeline.py` (a list of keywords matched against the folder name) and give each job its own working directory copy, because of `.snakemake/`.
- Steps 03 and 04 use 8 cores (`cores=8` in the scripts).
- Failed settings in steps 03, 04 and 06 are printed and skipped, not raised. Check the job output for "Error encountered" / "skipping".

## What to check after each step

- **01:** 22 folders, each with `platy1_muscles_stardist_elastic.tif`, `..._elastic_full.tif`, its own non-empty `deform_test_data.log` and `plots/`. `data/plots/seg_fixed.*` exists once. In `data/removal_stats.tsv`: `ri` removes about `f` of the instances, `ji` removes about 0, `sphere` grows with `f`. The plots are saved as PDF (global plot format), not PNG.
- **02:** about 690 IDs in `metric_ids.txt`.
- **03:** every `results/<tag>/05_elastix_deformable_pointset_registration/TransformParameters.2.txt` exists.
- **04:** `results/<tag>/moving_image.n5` has the key `full_pointset_alignment_transform`.
- **05 / 07:** in `results/lre_*.tsv`, `num_missing` should be close to the removal for `ri` and `sphere`, `num_cut` small. A mean LRE of the size of the specimen means the global alignment failed (seen before for large sphere removals).

## Decisions already made (do not revisit without asking the user)

- 22 settings exactly as in `test_setup.md`; jitter displacement `round(f * 3.74)` voxels, up to `f = 2.5`. No extra random 5 % removal in the sphere and jitter settings.
- One shared fixed image for all settings; landmarks are a 5 % subset of the fixed image IDs.
- The LRE is computed on the **full moving image** (all instances, no jitter), warped with the final transform, not on the registered moving image. For `ji` settings, the registered (jittered) moving image is evaluated as a second variant `jitter`.
- The full moving image is cropped with the bounding box of the moving image (option B). Instances cut by this box are removed from the full image.
- Landmarks are skipped if they are missing in the warped image or touch its border.
- The LRE is computed in the fixed space only; the pre-alignment-space output of step 04 is kept but not evaluated.
- Removal statistics are written by step 01; there is no separate step for them.
- Steps 06 and 07 stay separate, because they need different environments.
- Do not change shared pipeline code (`image_matchmaker/`) unless it is needed; ask the user first.

## Open items

- `test_setup.md` has no results section yet. After the runs, add the result tables (IMM and NucleiSky3D, both variants) and the observations.
- The sphere centers `[159, 203, 228]` (head) and `[139, 118, 90]` (tail) were chosen in the old setup. Check in `elastic_removal.pdf` of the sphere settings that they are still at head and tail.
