import z5py
import numpy as np

from image_matchmaker.prealignment import prealign_samples
from image_matchmaker.utils import (rotate_img, plot_three_slices, plot_overlay)


def main():
    output_dir = "./data/test"
    Path(f"{output_dir}/plots").mkdir(parents=True, exist_ok=True)

    #######################
    fixed_input = "./data/deformed_data/platy1_muscles_stardist_fixed_rotated.n5"
    with z5py.File(fixed_input, "r") as f:
        seg_fixed = f["seg"][:]

    moving_input = "./data/deformed_data/platy1_muscles_stardist_moving.n5"
    with z5py.File(moving_input, "r") as f:
        seg_moving = f["seg"][:]

    plot_three_slices(seg_fixed, save_path=f"{output_dir}/plots/fixed.png")
    plot_three_slices(seg_moving, save_path=f"{output_dir}/plots/moving.png")
    #######################

    prealigned_results = prealign_samples(seg_fixed, seg_moving)

    seg_fixed_prealigned, T, _, _, _ = prealigned_results["fixed"]
    seg_moving_prealigned, _, _, _, _ = prealigned_results["moving"]

    with z5py.File(f"{output_dir}/fixed_prealigned.n5", "w") as f:
        f.create_dataset("seg", data=seg_fixed_prealigned, compression="gzip")

    with z5py.File(f"{output_dir}/moving_prealigned.n5", "w") as f:
        f.create_dataset("seg", data=seg_moving_prealigned, compression="gzip")

    #######################
    # test backtransform
    seg_fixed_inv = rotate_img(
        seg_fixed_prealigned, np.linalg.inv(T), output_shape=seg_fixed.shape
    )

    plot_three_slices(seg_fixed_inv, save_path=f"{output_dir}/plots/fixed_prealigned_inv.png")

    # plot overlay of prealigned volumes
    plot_overlay(
        seg_fixed_prealigned,
        seg_moving_prealigned,
        save_path=f"{output_dir}/plots/overlay_prealignment.png",
    )


if __name__ == "__main__":
    main()
