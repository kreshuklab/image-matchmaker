import sys
import subprocess
import yaml
import shutil
import numpy as np
import tifffile as tiff
from pathlib import Path

from examples.deform_test_data import deform_test_data
from tests.compare_results import assert_arrays_equal
from matchmaker.utils import (load_config, read_volume, download_file, check_no_new_ids,
                            compute_centroid_distances,)


def run_pipline(registration_config_path, registration_snakefile, transform_config_path,
                transform_snakefile, cores=8, test_dir="tmp_pytest", enable_aniso=False,
                enable_elastic=False):
    test_dir = Path(test_dir)
    final_transform_path = test_dir / "final_transform.json"

    if test_dir.exists() and test_dir.is_dir():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    registration_config = load_config(registration_config_path)
    transform_config = load_config(transform_config_path)

    # Generate deformed data
    deform_test_data(config=registration_config, enable_aniso=enable_aniso, enable_elastic=enable_elastic)

    # Run registration snakemake workflow
    result = subprocess.run(
        [
            "snakemake",
            "--snakefile", registration_snakefile,
            "--configfile", registration_config_path,
            "--config", f"log_dir={test_dir}", f"final_transform_path={final_transform_path}",
            "--cores", str(cores),
        ],
        check=True,
    )

    log_dir = transform_config["log_dir"]
    for moving_img in transform_config["moving_images"]:
        moving_img["input_path"] = moving_img["input_path"].replace("data/test_rigid_registration", str(test_dir))
        moving_img["output_path"] = moving_img["output_path"].replace(log_dir, str(test_dir))
    transform_config["log_dir"] = str(test_dir)
    transform_config["final_transform_path"] = str(final_transform_path)
    transform_config["parameter_map_path"] = transform_config["parameter_map_path"].replace("data/test_rigid_registration", str(test_dir))
    transform_config["prealignment_transform_path"] = transform_config["prealignment_transform_path"].replace("data/test_rigid_registration", str(test_dir))

    tmp_config_path = test_dir / "apply_transform.yaml"
    with open(tmp_config_path, "w") as f:
        yaml.dump(transform_config, f)

    # Run apply_transform snakemake workflow
    result = subprocess.run(
        [
            "snakemake",
            "--snakefile", transform_snakefile,
            "--configfile", tmp_config_path,
            "--cores", str(cores),
        ],
        check=True,
    )

    # Compare results
    moving_path = test_dir / f"{registration_config['moving_image']['output_name']}.n5"

    result_img = read_volume(moving_path, "pointset_alignment")
    warped_img = read_volume(moving_path, "pointset_alignment_transform")

    assert_arrays_equal(result_img, warped_img)

    ref_path = registration_config["moving_image"]["ref_path"]
    if not Path(ref_path).exists():
        ref_url = registration_config["moving_image"]["ref_url"]
        download_file(ref_path, ref_url)

    ref_img = tiff.imread(ref_path)

    assert_arrays_equal(result_img, ref_img)

    print("✅ compare_results finished.")
    shutil.rmtree(test_dir)


def test_workflow():

    run_pipline("examples/register_config_test_rigid.yaml", "workflows/registration.smk",
                "examples/register_config_test_rigid_apply_transform.yaml", "workflows/apply_transform.smk")
