# Handoff: registration limit experiments

Read `test_setup.md` first. It describes the experiment design, the data layout and every step. This file adds what is needed to continue the work: the current state, how to run the steps, what to check after each step, and the decisions that are already made.

## Current state (2026-11-05)

- Steps 01–07 ran on the cluster for all 22 settings (jobs and checks: `monitor.md`). No setting failed or was skipped. Results are in `$OUT/results/` and summarized in `results.md` and the results section of `test_setup.md`.
- Measured runtimes: step 01+02 5 min (3.6 GB); one `02_register.sh` task 7 min to 1:11 h (7.6 GB for `ji-0.25`); `02_nucleisky.sh` 1:30 h; `03_evaluate.sh` 1 min. The limits in the sbatch scripts are large enough.
- Step 08 (`08_check_similarity_fit.py`) was added after the run to explain the NucleiSky3D results. It is not in the sbatch scripts; it was run on the interactive node.
- NucleiSky3D results are not a fair baseline: its scale/rotation estimates are wrong (scale often at its search bound 0.8). See `results.md`.
- Separate run, not part of the 22 settings: `cluster/register_elastic_hungarian.sh` (job 17101748) creates new data with the elastic field and the default 5 % random removal, and registers it with `image_matchmaker/configs/register_config_test_elastic.yaml` changed to `hungarian`, `maxiter: 100`, `max_dist: 30`. Output: `/mnt/lustre-grete/usr/u19382/matchmaker/test_elastic_registration_hungarian`. Not evaluated yet.
- The old results (`test_summary.md`, old `example_data/deform_data_*` and `data/test_elastic_registration_*` folders) are from an earlier setup with a different elastic field and are not comparable. Do not mix them with the new results. This does **not** apply to `example_data/deformed_data/`: checked on 2026-11-05, its fixed image is voxelwise identical to `$OUT/data/platy1_muscles_stardist_fixed_rotated.tif`, and its moving image has the same elastic field (all 13,064 shared instance centroids equal after a 1-voxel crop offset), with the default 5 % random removal on top.
- `data/platy1_muscles_stardist_elastic.tif` is a leftover and not used by any step.

## Code changes that the scripts depend on

The scripts need these changes in the `image_matchmaker` package. They are committed in `c865920` (branch `registration-tests`):

| File | Change |
|---|---|
| `image_matchmaker/data_processing/deform_test_data.py` | Mode `jitter_instances` (old names `translate_instances`/`move_voxels` are gone); `save_full_moving` option that writes `<moving>_full.tif`; logs and plots go to the moving image folder, the fixed plot to the fixed image folder; returns `{"num_before", "num_left"}`. |
| `image_matchmaker/utils/log_utils.py` | `force=True` in `logging.basicConfig`, so each `deform_test_data` call in one process logs to its own folder. |
| `image_matchmaker/utils/vis.py` | `plot_removal`: same color per instance in the original and degraded row, removed voxels in red; the `mode` argument is removed. |
| `image_matchmaker/utils/transform_utils.py` | `pad_to_common_bbox` is removed (no longer used). |
| `environment.yml` | `cupy` is commented out. Check if the cluster needs it. |

`registration_limit_experiments/` is tracked in the same commit.

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
- To run only some settings, pass their tags to steps 03 and 04, e.g. `python 03_run_pipeline.py ri-0.1 ji-1` (exact match; no tags = all settings). Separate jobs need their own working directory, because of `.snakemake/` (see "Run on the cluster").
- Steps 03 and 04 use 8 cores (`cores=8` in the scripts).
- Failed settings in steps 03, 04 and 06 are printed and skipped, not raised. Check the job output for "Error encountered" / "skipping".

## Run on the cluster

The sbatch scripts are in `cluster/`. All outputs go to `OUT=/mnt/lustre-grete/usr/u19382/matchmaker/registration_tests`, not into the repo. The scripts are called by absolute path and run with `$OUT` as working directory, so `data/`, `results/` and `metric_ids.txt` are written there.

| Script | Steps | Env | Notes |
|---|---|---|---|
| `01_create_data.sh` | 01, 02 | imm | Copies both configs to `$OUT`, with an absolute `source_path`. The runs use these copies. |
| `02_register.sh` | 03, 04 | imm | Array `0-21`, one setting per task (tags in sorted order). Each task runs in `$OUT/work/<tag>/` with symlinks to `$OUT/data`, `$OUT/results` and the configs, so it has its own `.snakemake/`. On a rerun, incomplete pipeline outputs of the setting are deleted first; NucleiSky outputs are kept. |
| `02_nucleisky.sh` | 06, 07 | nucleisky_env, then imm | Needs step 01 only; runs in parallel to `02_register.sh`. |
| `03_evaluate.sh` | 05 | imm | |

Submit from `$OUT` (the job logs go to `$OUT/logs/`, which must exist):

```bash
mkdir -p /mnt/lustre-grete/usr/u19382/matchmaker/registration_tests/logs
cd /mnt/lustre-grete/usr/u19382/matchmaker/registration_tests
C=/mnt/vast-nhr/home/mfreita/u19382/image-matchmaker/registration_limit_experiments/cluster
j1=$(sbatch --parsable $C/01_create_data.sh)
sbatch --dependency=afterok:$j1 $C/02_nucleisky.sh
# Test run with one setting; check time and memory with: sacct -j $j2 --format=JobID,Elapsed,MaxRSS
j2=$(sbatch --parsable --dependency=afterok:$j1 --array=0 $C/02_register.sh)
# After the test (adjust --time/--mem if needed): the other settings, then the evaluation
j3=$(sbatch --parsable --array=1-21 $C/02_register.sh)
sbatch --dependency=afterany:$j3 $C/03_evaluate.sh
```

Failed settings do not fail the job (steps 03, 04 and 06 print the error and skip), so check the logs for "Error encountered" / "skipping". For measured runtimes, see "Current state".

`$j1`, `$j2`, `$j3` only exist in the shell that submitted the jobs. In another shell, get the job IDs from `squeue -u $USER` or `sacct`.

Step 08 (no sbatch script; run on a compute node, e.g. `srun --jobid=<interactive job> --overlap ...`), from `$OUT` in `imm_env`:

```bash
python $C/../08_check_similarity_fit.py overlay_ji-0.25.png > results/similarity_fit.tsv
```

## What to check after each step

- **01:** 22 folders, each with `platy1_muscles_stardist_elastic.tif`, `..._elastic_full.tif`, its own non-empty `deform_test_data.log` and `plots/`. `data/plots/seg_fixed.*` exists once. In `data/removal_stats.tsv`: `ri` removes about `f` of the instances, `ji` removes about 0, `sphere` grows with `f`. The plots are saved as PDF (global plot format), not PNG.
- **02:** about 690 IDs in `metric_ids.txt`.
- **03:** every `results/<tag>/05_elastix_deformable_pointset_registration/TransformParameters.2.txt` exists.
- **04:** `results/<tag>/moving_image.n5` has the key `full_pointset_alignment_transform`.
- **05 / 07:** in `results/lre_*.tsv`, `num_missing` and `num_cut` should be small (0–5) in all settings, because the LRE is computed on the full moving image, which keeps all instances. Many missing landmarks mean that parts of the warped image are mapped outside the fixed volume (seen for IMM `sphere-center2-0.5`: 22 missing). A mean LRE of the size of the specimen means the global alignment failed. The `P95`/`P80` columns are trimmed means (mean of the distances up to that percentile), not percentiles, so they are smaller than the mean.
- **08:** after removing the fitted similarity, the NucleiSky3D LRE should equal the oracle value (4.90 in the 2026-11 run). If not, NucleiSky's output is not a pure similarity transform of the moving image, and the export or the evaluation must be checked.

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

- Find which IMM step causes the global misalignment for large sphere removals (SVD pre-alignment, rigid alignment or CPD), e.g. with the LRE after `01_svd_prealignment` and `02_rigid_alignment` for `sphere-center2-0.4/0.5`.
- NucleiSky3D: decide with the user whether to rerun with a narrower scale range around 1 (`matcher_config`, default `scale_min=0.8, scale_max=1.2`) or to report the similarity oracle (4.90) as the similarity baseline instead.
- The sphere centers `[159, 203, 228]` (head) and `[139, 118, 90]` (tail) were chosen in the old setup. Check in `elastic_removal.pdf` of the sphere settings that they are still at head and tail.
