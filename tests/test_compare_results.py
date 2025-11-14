import os
import numpy as np
import tifffile as tiff

from matchmaker.utils import (load_test_config, get_n5_path, read_volume, download_file)


def test_compare_results(ref_path="examples/data/moving_rigid_aligned.tif"):
    config = load_test_config()
    output_key = config["keys"]["rigid_alignment"]
    _, moving_path = get_n5_path()

    test_img = read_volume(moving_path, output_key)

    if not os.path.exists(ref_path):
        ref_url = config["moving_image"]["url"]
        download_file(ref_path, ref_url)

    ref_img = tiff.imread(ref_path)

    assert np.array_equal(test_img, ref_img)
