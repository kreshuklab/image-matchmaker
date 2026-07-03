# Example data

The repository ships a small example dataset and ready-made configuration files
under `examples/`, so you can try the pipeline without preparing your own data.

## What is included

- `examples/data/platy1_muscles_stardist_fixed.tif` — a base 3D instance
  segmentation mask (also provided as `.n5`).
- Ready-to-use registration configs in `examples/`:
  - `register_config_test_rigid.yaml`
  - `register_config_test_elastic.yaml`
  - `register_config_test_aniso_rigid.yaml`

These configs are referenced directly by the {doc}`Quick Start <quickstart>` and by
the test suite (see {doc}`Installation <installation>`).

---

## Generate synthetic moving data

The moving datasets are created by deforming the provided fixed segmentation mask. Run, from the
repository root:

```bash
python examples/deform_test_data.py
```

The script generates the following under `examples/data/deformed_data/`, each as
both `.tif` and `.n5` (plus overlay plots):

- a rotated fixed image (`platy1_muscles_stardist_fixed_rotated`)
- a rigidly transformed moving mask (`platy1_muscles_stardist_rigid`)
- an elastically deformed moving mask (`platy1_muscles_stardist_elastic`)

Once the data exists, run the pipeline on it as shown in the {doc}`Quick Start <quickstart>`.
