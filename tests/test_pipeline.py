import subprocess
import yaml
import shutil
import numpy as np
import tifffile as tiff
from pathlib import Path

from examples.deform_test_data import deform_test_data
from tests.compare_results import assert_arrays_equal
from image_matchmaker.utils import (
    load_config,
    read_volume,
    download_file,
    check_no_new_ids,
    compute_centroid_distances,
)


def run_pipline(
    registration_config_path,
    registration_snakefile,
    transform_config_path,
    transform_snakefile,
    cores=8,
    test_dir="tmp_pytest",
    enable_aniso=False,
    enable_elastic=False,
):
    test_dir = Path(test_dir)
    if test_dir.exists() and test_dir.is_dir():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    registration_config = load_config(registration_config_path)
    transform_config = load_config(transform_config_path)

    # Generate deformed data
    deform_test_data(config=registration_config, enable_aniso=enable_aniso, enable_elastic=enable_elastic)

    registration_config["log_dir"] = str(test_dir)
    registration_config["matching"]["max_dist"] = 10

    tmp_config_path = test_dir / "registration.yaml"
    with open(tmp_config_path, "w") as f:
        yaml.dump(registration_config, f)

    # Run registration snakemake workflow
    result = subprocess.run(
        [
            "snakemake",
            "--snakefile", registration_snakefile,
            "--configfile", tmp_config_path,
            "--cores", str(cores),
        ],
        check=True,
    )

    log_dir = Path(transform_config["log_dir"].replace("data/test_apply_transform", str(test_dir)))
    log_dir.mkdir(parents=True, exist_ok=True)

    for moving_img in transform_config["moving_images"]:
        moving_img["input_path"] = moving_img["input_path"].replace("data/test_rigid_registration", str(test_dir))
        moving_img["output_path"] = moving_img["output_path"].replace("data/test_apply_transform", str(test_dir))
    transform_config["log_dir"] = str(log_dir)
    transform_config["parameter_map_path"] = transform_config[
        "parameter_map_path"
    ].replace("data/test_rigid_registration", str(test_dir))
    prealignment_transform_path = transform_config.get("prealignment_transform_path")
    if prealignment_transform_path:
        transform_config["prealignment_transform_path"] = prealignment_transform_path.replace("data/test_rigid_registration", str(test_dir))

    tmp_config_path = log_dir / "apply_transform.yaml"
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
    moving_name = registration_config["moving_image"].get("name", "moving_image")
    moving_path = test_dir / f"{moving_name}.n5"

    result_img = read_volume(moving_path, "pointset_alignment_prealignment_space")
    warped_img = read_volume(moving_path, "pointset_alignment_transform_prealigned")

    assert_arrays_equal(result_img, warped_img)

    ref_path = registration_config["moving_image"]["ref_path"]
    if not Path(ref_path).exists():
        Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
        ref_url = registration_config["moving_image"]["ref_url"]
        download_file(ref_path, ref_url)

    ref_img = tiff.imread(ref_path)

    # NOTE:
    # Numerical results are not bitwise deterministic across NumPy versions.
    # In practice, small voxel-level differences may occur between releases
    # (e.g. NumPy 1.x vs 2.x), so we validate structural consistency instead
    # of strict array equality.

    no_new_id, _ = check_no_new_ids(result_img, ref_img)
    assert no_new_id

    centroid_distances = compute_centroid_distances(result_img, ref_img, exclude_id=0)
    max_distance = np.max(centroid_distances)
    print("Max centroid distance:", max_distance)
    assert max_distance < 1

    print("✅ compare_results finished.")
    shutil.rmtree(test_dir)


def test_workflow():

    run_pipline("examples/register_config_test_rigid.yaml", "workflows/registration.smk",
                "examples/register_config_test_rigid_apply_transform.yaml", "workflows/apply_transform.smk")
