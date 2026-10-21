# Example data

Image Matchmaker provides example data and configuration files to help you try the registration pipeline without preparing your own data.

The availability of the example data depends on how you install the package:

- **pip installation:** Download the example data and configuration files from the [GitHub Releases](https://github.com/kreshuklab/image-matchmaker/releases/tag/example_data) page and extract them locally.
- **Installation from source:** The example data is included in the repository under `example_data/`, so a plain `git clone` is sufficient.

## What is included

- `platy1_muscles_stardist_fixed.tif` — a base 3D instance segmentation mask and should be placed under `example_data/`. The moving datasets and `.n5` copies are generated locally by the deformation script below.
- Example registration configurations:
  - `register_config_test_rigid.yaml`
  - `register_config_test_elastic.yaml`
  - `register_config_test_aniso_rigid.yaml`

The configuration files are included in the Python package under `image_matchmaker/configs/`. For convenient access when installing via pip, download the example configs from the [GitHub Releases](https://github.com/kreshuklab/image-matchmaker/releases/tag/example_data) page.

These configurations are referenced by the {doc}`Quick Start <quickstart>` and used by the test suite (see {doc}`Installation <installation>`).

---

## Generate synthetic moving data

The moving datasets are created by deforming the provided fixed segmentation mask. Run, from the
repository root:

```bash
python -m image_matchmaker.data_processing.deform_test_data
```

The script generates the following under `example_data/deformed_data/`, each as
both `.tif` and `.n5` (plus overlay plots):

- a rotated fixed image (`platy1_muscles_stardist_fixed_rotated`)
- a rigidly transformed moving mask (`platy1_muscles_stardist_rigid`)
- an elastically deformed moving mask (`platy1_muscles_stardist_elastic`)

Once the data exists, run the pipeline on it as shown in the {doc}`Quick Start <quickstart>`.
