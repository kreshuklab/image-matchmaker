import os
import numpy as np
import tifffile as tiff

from matchmaker.utils import (load_test_config, read_volume, download_file)


def compare_results(config_path):
    config = load_test_config(config_path)
    output_key = config["keys"]["rigid_alignment"]
    moving_path = f"{config['log_dir']}/{config['moving_image']['output_name']}.n5"
    ref_path = config["moving_image"]["ref_path"]

    test_img = read_volume(moving_path, output_key)

    if not os.path.exists(ref_path):
        ref_url = config["moving_image"]["ref_url"]
        download_file(ref_path, ref_url)

    ref_img = tiff.imread(ref_path)

    assert np.array_equal(test_img, ref_img)
    print("✅ compare_results finished.")
