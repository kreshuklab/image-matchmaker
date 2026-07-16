# Installation

## Clone the repository

```bash
git clone https://github.com/kreshuklab/image-matchmaker.git
cd image-matchmaker
```
---

## Create the environment

```bash
conda env create -f environment.yml
conda activate matchmaker_env
```
---

## Verify the installation
Run:

```bash
pytest -s
```

This runs the full registration and apply-transform pipeline on test data using the
rigid example configuration (`examples/register_config_test_rigid.yaml`), so it
takes a few minutes rather than finishing instantly. On the first run it also
downloads reference data from the project's GitHub release, so an internet
connection is required.

If the download fails, download the
reference data manually from the
[release page](https://github.com/kreshuklab/image-matchmaker/releases/tag/test_data-v1.0)
and place it under:

```text
examples/data/test_data/
```

To silence warnings during the test, append `-p no:warnings`:

```bash
pytest -s -p no:warnings
```

If the command finishes without errors, the installation was successful.
