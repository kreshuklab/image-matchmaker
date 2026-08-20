# Troubleshooting

Common problems and how to resolve them. For how to read the diagnostic plots, see
{doc}`Understanding the Outputs <outputs>`; for field meanings, see the
{doc}`Configuration Reference <config_ref>`.

## Running the workflow

### `WorkflowError: No config file` / missing config values

The workflows no longer ship with a default config file, so `--configfile` must be
passed on every run. Point it at your registration or transform config (see the
{doc}`Quick Start <quickstart>`).

### `command not found` / import errors when a rule runs

The workflows no longer manage a conda environment per rule. Activate the
environment yourself before invoking Snakemake:

```bash
conda activate imm_env
```

### `KeyError` when starting a run

The config is missing a key the workflow requires. Compare your config against the
{doc}`Configuration Reference <config_ref>` and the complete example in the
{doc}`Quick Start <quickstart>` — every registration run needs `prealignment`,
`coherent_point_drift`, `matching`, `mobie_export`, and
`mobie_dataset_name`, in addition to the image blocks and `log_dir`.

### "Directory cannot be locked"

A previous Snakemake run was interrupted and left the working directory locked.
Unlock it and re-run:

```bash
snakemake -s workflows/registration.smk --configfile <config.yaml> --unlock
```

### Run stops with incomplete output files

If a run was interrupted, Snakemake may flag partial outputs. Re-run with:

```bash
snakemake -s workflows/registration.smk --configfile <config.yaml> \
    --cores 8 --rerun-incomplete
```

### `pytest -s` fails to download reference data

The test downloads reference data from the project's GitHub release on first run. If
the download fails (e.g. no internet), download it
manually from the
[release page](https://github.com/kreshuklab/image-matchmaker/releases/tag/test_data-v1.0)
and place it under `examples/data/test_data/` (see {doc}`Installation <installation>`).

### Feature matching runs out of memory

The `ilp` matching method builds candidate matches from each point's neighbours, so
its memory use grows with how many candidates are considered. If the
`match_pointclouds` step runs out of memory, lower `matching.max_dist` (smaller
search radius) and/or `matching.min_neighbours` to reduce the number of candidates
(see {doc}`Configuration Reference <config_ref>`).

## Alignment quality

### The moving volume is flipped after pre-alignment

Check `overlay_after_prealignment` in `01_svd_prealignment/plots/`. If the automatic
axis orientation chose the wrong flip, inspect the overlays in
`01_svd_prealignment/manual_prealignment_options/` (`IDENTITY`, `X`, `Y`, `Z`) to see
which rotation lines the volumes up, then set `prealignment.axis_orientation` to that
value instead of `auto` (see {doc}`Configuration Reference <config_ref>`).

### Point matches look wrong (long, crossing lines)

Inspect `point_matching` in `04_match_pointclouds/plots/`. Adjust the `matching`
parameters — lower `max_dist` to reject distant matches, or change `min_neighbours`
— or try a different `matching.method` (`ilp`, `hungarian`, `sinkhorn`) (see
{doc}`Configuration Reference <config_ref>`).

### CPD does not converge / the displacement field is erratic

Inspect `displacement_field` in `03_cpd_nonrigid_registration/plots/`. Increase
`coherent_point_drift.maxiter` (100–150 is typical), and tune `w`, `beta`, and `lmd`.
Large, discontinuous displacements often mean an earlier step (pre-alignment or rigid)
did not align well — fix that first. If you have corresponding landmarks, the
{doc}`CPD Parameter Tuning <cpd_tuning>` workflow can search for good `w`, `beta`,
`lmd`, and `maxiter` values automatically.

### Volumes are at the wrong scale or orientation

Confirm the voxel resolution (`x_res`, `y_res`, `z_res`) is correct for each image and
that inputs are in **ZYX** axis order (see {doc}`Quick Start <quickstart>`).

### Nothing matches / empty results

The inputs must be **instance** segmentation masks (a distinct label per object), not
binary masks — matching operates on per-instance centroids.

## Checking whether it worked

See the quality checklist in {doc}`Understanding the Outputs <outputs>`: overlays
should converge stage to stage, the displacement field should be smooth, matching
lines should connect nearby structures, and the deformation grid should not fold.
