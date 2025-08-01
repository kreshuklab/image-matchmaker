import os
import z5py
import numpy as np

from matchmaker.prealignment import prealign_sample
from matchmaker.transform_utils import rotate_img
from matchmaker.vis import plot_three_slices, plot_overlay


def main():
    output_dir = "./data/test"
    if not os.path.exists(f"{output_dir}/plots"):
        os.makedirs(f"{output_dir}/plots")

    #######################
    # prealign moving image
    moving_input = "./data/platy1_muscles_stardist_moving.n5"
    with z5py.File(moving_input, "r") as f:
        seg_moving = f["seg"][:]

    plot_three_slices(seg_moving, save_path=f"{output_dir}/plots/moving.png")

    seg_moving_prealigned = prealign_sample(seg_moving, file_name="moving", save_path=output_dir)

    with z5py.File(f"{output_dir}/moving_prealigned.n5", "w") as f:
        f.create_dataset("seg", data=seg_moving_prealigned, compression="gzip")

    #######################
    # prealign fixed image
    fixed_input = "./data/platy1_muscles_stardist_fixed.n5"
    with z5py.File(fixed_input, "r") as f:
        seg_fixed = f["seg"][:]

    plot_three_slices(seg_fixed, save_path=f"{output_dir}/plots/fixed.png")

    seg_fixed_prealigned = prealign_sample(seg_fixed, file_name="fixed", save_path=output_dir)

    with z5py.File(f"{output_dir}/fixed_prealigned.n5", "w") as f:
        f.create_dataset("seg", data=seg_fixed_prealigned, compression="gzip")

    #######################
    # test backtransform
    T = np.loadtxt(f"{output_dir}/fixed_T_prealignment.txt")
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
