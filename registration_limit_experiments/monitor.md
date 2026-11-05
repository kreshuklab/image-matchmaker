# Monitor: registration limit experiments on the cluster

Plan: check every hour; when the test job j2 (setting index 0) finishes successfully, check time/memory, adjust `02_register.sh` if needed, then submit settings 1-21 (j3) and `03_evaluate.sh` (afterany:j3).

OUT=/mnt/lustre-grete/usr/u19382/matchmaker/registration_tests

| Job | ID | Script |
|---|---|---|
| j1 | 17093228 | 01_create_data.sh |
| nucleisky | 17093232 | 02_nucleisky.sh |
| j2 | 17093246 | 02_register.sh --array=0 |

## Log

### 2026-11-04 21:58
- j1 COMPLETED (5:17 min, MaxRSS 3.6 GB). No errors in log. 22 `deformed_data_*` folders, 688 IDs in `metric_ids.txt`.
- nucleisky RUNNING (c0361).
- j2 RUNNING since ~21:53 (c0254), in step 03 point cloud registration (maxiter=100, ~35 s/iteration).

### 2026-11-04 22:15
- j2 RUNNING (21 min), point cloud registration at iteration 32/100 (~33 s/iteration, so ~37 min left for this part).
- nucleisky RUNNING (23 min).

### 2026-11-04 23:15
- j2 (17093246, setting 0 = `ji-0.25`) COMPLETED: Elapsed 01:09:54, MaxRSS 7.6 GB (7609316K).
- `grep -i "error\|skipping"` in `logs/rle-register_17093246_0.out`: no hits.
- `results/ji-0.25/05_elastix_deformable_pointset_registration/TransformParameters.2.txt` exists; `results/ji-0.25/moving_image.n5` has `full_pointset_alignment_transform`.
- Limits unchanged (19 % of 6 h, 12 % of 64G).
- Submitted from `$OUT`: j3 = 17095243 (`02_register.sh --array=1-21`), evaluate = 17095244 (`03_evaluate.sh`, afterany:17095243).
- nucleisky (17093232) RUNNING (1:23 h), at 20/22 settings.
- Hourly cron deleted.

### 2026-11-05
- j3 (17095243, settings 1-21), evaluate (17095244) and nucleisky (17093232) all COMPLETED. Register tasks took 7 min to 1:11 h.
- No "Error encountered" / "skipping" in any register, nucleisky or evaluate log. `results/lre_imm.tsv` and `results/lre_nucleisky.tsv` exist.
- Separate run: `cluster/register_elastic_hungarian.sh` submitted as 17101748 (output in `/mnt/lustre-grete/usr/u19382/matchmaker/test_elastic_registration_hungarian`).
