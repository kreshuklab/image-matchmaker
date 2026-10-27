# Installation

## Install with pip

The recommended way to install Image-Matchmaker is via pip:

```bash
pip install image-matchmaker
```

## Install from source

For development, clone the repository and create the Conda environment:

```bash
git clone https://github.com/kreshuklab/image-matchmaker.git
cd image-matchmaker

conda env create -f environment.yml
conda activate imm_env
```

### Verify the installation

Check that the package and CLI are available:

```bash
python -c "import image_matchmaker"
image-matchmaker --help
```

### Run the test suite

To verify the full registration and apply-transform pipeline, run:

```bash
pytest -s
```

This runs the full registration and apply-transform pipeline on test data using the
rigid example configuration (`image_matchmaker/configs/register_config_test_rigid.yaml`), so it
takes a few minutes rather than finishing instantly. On the first run it also
downloads reference data from the project's GitHub release, so an internet
connection is required.

If the download fails, download the reference data manually from the
[release page](https://github.com/kreshuklab/image-matchmaker/releases/tag/test_data-v1.2)
and place it under:

```text
example_data/test_data/
```

To silence warnings during the test, use:

```bash
pytest -s -p no:warnings
```

If the test suite finishes without errors, the development installation is working correctly.
