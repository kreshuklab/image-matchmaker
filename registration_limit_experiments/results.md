# Registration limit experiments: results

Run on 2026-11-04/05 on the cluster, output in `/mnt/lustre-grete/usr/u19382/matchmaker/registration_tests` (`$OUT`). Setup: see `test_setup.md`. All 22 settings ran for IMM (steps 03–05) and NucleiSky3D (steps 06–07); no setting failed or was skipped.

All values are in voxels (= µm). The mean distance between neighbouring instances is 3.74 voxels. "Mean" is the mean LRE over all evaluated landmarks (about 686 per setting).

## Summary

1. **IMM is accurate below one instance spacing for random removal and jitter.** Baseline LRE is about 2.1 voxels (median 1.9). Removing up to 70 % of the instances at random barely changes it (2.4 → 2.9). Only 90 % removal gives a clear increase (4.5).
2. **IMM breaks down for a removed connected region (`sphere`), already at small removed fractions.** Removing 13–15 % of the instances in one region gives an LRE of 10.8 (head) and 18.4 (tail), compared with 2.4 for 10 % random removal. The tail is more sensitive than the head. The error is not local to the hole: most of it is a global misalignment (rotation up to 17°, shift up to 8 voxels, scale 0.96).
3. **Jitter mainly affects where the jittered instances end up, not the transform.** On the full (unjittered) image, the LRE rises slowly from 2.2 to 4.4 and levels off from `ji-2` on. On the jittered image, it grows roughly with the jitter size (2.6 → 10.1), as expected.
4. **NucleiSky3D fails on this data (LRE 6.6–36.9) because its scale and rotation estimates are wrong.** The cause is its matching, not our scripts: its transforms are clean similarity transforms with a wrong scale (0.79–1.0 instead of about 1) and rotation (3–33°). For `ji-0.25`, NucleiSky itself reports `best_scale = 0.803`, close to the lower bound of its scale search (0.8).
5. **Any similarity-only method has a floor of 4.90 on this data.** This is the LRE of the best possible similarity transform from the full moving image to the fixed image (oracle). IMM is below this floor in all settings except the five largest sphere removals.

## IMM

| Setting | Removed % | Mean `full` | Mean95 `full` | Mean `jitter` |
|---|---|---|---|---|
| ji-0.25 (d=1) | 0.0 | 2.20 | 1.99 | 2.64 |
| ji-0.5 (d=2) | 0.0 | 2.77 | 2.50 | 3.67 |
| ji-0.75 (d=3) | 0.0 | 2.97 | 2.74 | 4.55 |
| ji-1 (d=4) | 0.0 | 3.42 | 3.19 | 5.59 |
| ji-1.5 (d=6) | 0.0 | 3.88 | 3.68 | 7.42 |
| ji-2 (d=7) | 0.0 | 4.41 | 4.17 | 8.69 |
| ji-2.5 (d=9) | 0.0 | 4.39 | 4.18 | 10.06 |
| ri-0.1 | 10.0 | 2.38 | 2.10 | |
| ri-0.3 | 30.0 | 2.48 | 2.17 | |
| ri-0.5 | 50.0 | 2.45 | 2.18 | |
| ri-0.7 | 70.0 | 2.86 | 2.57 | |
| ri-0.9 | 90.0 | 4.50 | 4.25 | |
| sphere-center1-0.1 | 0.07 | 2.06 | 1.86 | |
| sphere-center1-0.2 | 1.23 | 2.51 | 2.14 | |
| sphere-center1-0.3 | 4.32 | 3.99 | 3.38 | |
| sphere-center1-0.4 | 8.63 | **6.73** | 6.09 | |
| sphere-center1-0.5 | 14.66 | **10.79** | 10.08 | |
| sphere-center2-0.1 | 0.07 | 2.06 | 1.87 | |
| sphere-center2-0.2 | 1.01 | 3.03 | 2.77 | |
| sphere-center2-0.3 | 3.66 | **5.82** | 5.28 | |
| sphere-center2-0.4 | 7.82 | **10.50** | 9.57 | |
| sphere-center2-0.5 | 12.82 | **18.42** | 16.96 | |

Bold: above the similarity floor of 4.90. Mean95 = mean of the 95 % smallest distances (column `P95` in `lre_imm.tsv`; it is a trimmed mean, not a percentile).

Notes:
- `sphere-center2-0.5` is the only setting with many lost landmarks (22 missing, 15 touching the border), so parts of the warped image are mapped outside the fixed volume.
- For the large sphere removals, the median is close to the mean (`sphere-center1-0.5`: median 12.1, `sphere-center2-0.5`: median 15.3). So most landmarks are off, not only those near the hole.

## NucleiSky3D

| Setting | Mean `full` | Fitted scale | Implied NucleiSky scale | Fitted rotation (°) |
|---|---|---|---|---|
| ji-0.25 | 32.07 | 1.246 | 0.80 | 19.3 |
| ji-0.5 | 15.45 | 1.142 | 0.88 | 8.5 |
| ji-0.75 | 15.34 | 1.185 | 0.84 | 11.1 |
| ji-1 | 36.45 | 1.242 | 0.81 | 25.2 |
| ji-1.5 | 36.40 | 1.255 | 0.80 | 32.5 |
| ji-2 | 28.15 | 1.248 | 0.80 | 27.1 |
| ji-2.5 | 26.47 | 1.247 | 0.80 | 14.8 |
| ri-0.1 | 19.36 | 1.207 | 0.83 | 8.1 |
| ri-0.3 | 30.33 | 1.205 | 0.83 | 24.3 |
| ri-0.5 | 36.93 | 1.180 | 0.85 | 32.0 |
| ri-0.7 | 17.79 | 1.134 | 0.88 | 8.7 |
| ri-0.9 | 18.06 | 1.185 | 0.84 | 5.8 |
| sphere-center1-0.1 | 15.20 | 1.207 | 0.83 | 5.7 |
| sphere-center1-0.2 | 10.03 | 1.090 | 0.92 | 4.4 |
| sphere-center1-0.3 | 28.55 | 1.246 | 0.80 | 19.8 |
| sphere-center1-0.4 | 16.59 | 1.155 | 0.87 | 5.8 |
| sphere-center1-0.5 | 6.56 | 1.035 | 0.97 | 2.7 |
| sphere-center2-0.1 | 11.26 | 1.082 | 0.92 | 3.9 |
| sphere-center2-0.2 | 19.71 | 1.197 | 0.84 | 7.5 |
| sphere-center2-0.3 | 10.51 | 1.110 | 0.90 | 5.4 |
| sphere-center2-0.4 | 17.32 | 1.198 | 0.83 | 7.9 |
| sphere-center2-0.5 | 6.95 | 0.997 | 1.00 | 4.1 |

`jitter` variant values are almost the same as `full` (see `lre_nucleisky.tsv`).

How the error was checked (`08_check_similarity_fit.py`, output `$OUT/results/similarity_fit.tsv`):
- For each warped image, a similarity transform (Umeyama) is fitted from the warped to the fixed landmark centroids. "Fitted scale/rotation" is what is left of a global misalignment; "implied NucleiSky scale" = 1 / fitted scale.
- After removing this fitted transform, the NucleiSky LRE drops to 4.89–4.93 in every setting, equal to the similarity floor (4.90). So the NucleiSky output has the right shape, only the wrong scale, rotation and position. There is no mirroring and no swapped axes.
- For `ji-0.25`, NucleiSky was rerun and reports `best_scale = 0.803`; 1 / 0.803 = 1.245 matches the fitted 1.246. So the export and our evaluation are consistent with NucleiSky's own estimate.
- Overlay of fixed and warped images for `ji-0.25`: `$OUT/results/ji-0.25/overlay_fixed_warped.pdf`. NucleiSky's result is visibly shrunk and rotated, IMM's lies on the fixed image.

The scale search of NucleiSky's `pyramid` matcher is limited to [0.8, 1.2] (`nucleisky3d/config.py`). In 7 settings the estimate is at about 0.80, so the search ends at its lower bound. A likely reason is that NucleiSky is built to find a small crop inside a large image, while here the moving image covers the whole specimen and is elastically deformed. The NucleiSky numbers therefore do not show how well a similarity method can do on this data; the oracle floor of 4.90 does.

## IMM error split into global and residual parts

The same fit on the IMM output (`similarity_fit.tsv`):
- Random removal and jitter: fitted scale 1.00 ± 0.004, shift ≤ 1.6, rotation ≤ 4.5°. The error is local (residual after fit 2.0–3.3).
- Large sphere removals: scale 0.96–0.98, shift 4.3–8.4, rotation 4.9–17.3°. After removing this global part, the residual is 4.6–7.1. So the main failure is a global misalignment caused by the missing region; the local non-rigid part is also worse, but less.

## Open questions and next steps

- Find out which IMM step introduces the global misalignment for sphere removals (SVD pre-alignment, rigid alignment or CPD), e.g. by computing the LRE after `01_svd_prealignment` and `02_rigid_alignment` for `sphere-center2-0.4/0.5`.
- Check in `elastic_removal.pdf` of the sphere settings that the centers are still at head and tail (open item from `HANDOFF.md`).
- NucleiSky: try a narrower scale range around the known scale (e.g. `scale_min=0.9, scale_max=1.1`) via `matcher_config`, to get a fair similarity baseline. Otherwise, report the oracle floor of 4.90 as the similarity baseline.
