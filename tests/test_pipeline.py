import subprocess
import numpy as np
import tifffile as tiff
from pathlib import Path

from examples.deform_test_data import deform_test_data
from tests.compare_results import assert_arrays_equal
from matchmaker.utils import (load_config, read_volume, download_file)


def run_pipline(config_path, snakefile, cores=8):
    config = load_config(config_path)

    # Generate deformed data
    deform_test_data(config=config)

    # Run snakemake
    result = subprocess.run(
        [
            "snakemake",
            "--snakefile", snakefile,
            "--configfile", config_path,
            "--cores", str(cores),
        ],
        check=True,
    )
    assert result.returncode == 0, result.stderr

    # Compare results
    output_key = config["keys"]["rigid_alignment"]
    moving_path = f"{config['log_dir']}/{config['moving_image']['output_name']}.n5"

    test_img = read_volume(moving_path, output_key)

    ref_path = config["moving_image"]["ref_path"]
    if not Path(ref_path).exists():
        ref_url = config["moving_image"]["ref_url"]
        download_file(ref_path, ref_url)

    ref_img = tiff.imread(ref_path)

    assert_arrays_equal(test_img, ref_img)
    print("✅ compare_results finished.")


def test_workflow():

    run_pipline("examples/register_config_test_rigid.yaml", "workflows/rigid_registration.smk")
