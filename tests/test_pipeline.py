import sys
import subprocess
import shutil
import numpy as np
import tifffile as tiff
from pathlib import Path

from examples.deform_test_data import deform_test_data
from tests.compare_results import assert_arrays_equal
from matchmaker.utils import (load_config, read_volume, download_file, check_no_new_ids,
                            compute_centroid_distances,)


def run_pipline(config_path, snakefile, cores=8, test_dir="tmp_pytest"):
    test_dir = Path(test_dir)
    final_transform_path = test_dir / "final_transform.json"

    if test_dir.exists() and test_dir.is_dir():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    config = load_config(config_path)

    # Generate deformed data
    deform_test_data(config=config)

    # Run snakemake
    result = subprocess.run(
        [
            "snakemake",
            "--snakefile", snakefile,
            "--configfile", config_path,
            "--config", f"log_dir={test_dir}", f"final_transform_path={final_transform_path}",
            "--cores", str(cores),
        ],
        check=True,
    )
    assert result.returncode == 0, result.stderr

    # Compare results
    output_key = config["keys"]["rigid_alignment"]
    moving_path = test_dir / f"{config['moving_image']['output_name']}.n5"

    test_img = read_volume(moving_path, output_key)

    ref_path = config["moving_image"]["ref_path"]
    if not Path(ref_path).exists():
        ref_url = config["moving_image"]["ref_url"]
        download_file(ref_path, ref_url)

    ref_img = tiff.imread(ref_path)

    if sys.platform.startswith("linux"):
        assert_arrays_equal(test_img, ref_img)
    else:
        # NOTE:
        # scipy.ndimage.affine_transform is not bitwise deterministic across platforms.
        # In practice, small voxel-level differences may occur between operating systems.
        # These differences are geometrically negligible (≤ 1 voxel shift), so we
        # validate structural consistency instead of strict array equality.
        no_new_id, _ = check_no_new_ids(test_img, ref_img)
        assert no_new_id

        centroid_distances = compute_centroid_distances(test_img, ref_img, exclude_id=0)
        assert np.max(centroid_distances) < 1
    print("✅ compare_results finished.")
    shutil.rmtree(test_dir)


def test_workflow():

    run_pipline("examples/register_config_test_rigid.yaml", "workflows/registration.smk")
