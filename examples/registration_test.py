import z5py
import numpy as np

from matchmaker.prealignment import get_SVD_transform, orient_head
from matchmaker.transform_utils import get_transformation_matrix, rotate_img
from matchmaker.vis import plot_three_slices, plot_overlay


def main():
    with z5py.File("./data/platy1_muscles_stardist_moving.n5", "r") as f:
        seg_moving = f["seg"][:]

    plot_three_slices(seg_moving, save_path="./data/plots/moving.png")

    with z5py.File("./data/platy1_muscles_stardist_fixed.n5", "r") as f:
        seg_fixed = f["seg"][:]

    # 1) align moving image with PCs
    gc, Vt = get_SVD_transform(seg_moving)
    T, new_shape = get_transformation_matrix(seg_moving, gc, Vt)
    seg_moving_rotated = rotate_img(seg_moving, T, output_shape=new_shape)

    # Check if head is oriented correctly, else rotate 180 degrees
    rotate_head = orient_head(seg_moving_rotated, save_path="./data/plots/moving_orient_head.png")
    # NOTE: already enough?
    if rotate_head:
        print("Rotate head 180 degrees ...")
        seg_moving_rotated = np.rot90(seg_moving_rotated, k=2)

    plot_three_slices(
        seg_moving_rotated,
        save_path="./data/plots/moving_rotated.png",
    )

    # 2) align fixed image with PCs
    gc, Vt = get_SVD_transform(seg_fixed)
    T, new_shape = get_transformation_matrix(seg_fixed, gc, Vt)
    seg_fixed_rotated = rotate_img(seg_fixed, T, output_shape=new_shape)

    rotate_head = orient_head(seg_fixed_rotated)
    if rotate_head:
        print("Rotate head 180 degrees ...")
        seg_fixed_rotated = np.rot90(seg_fixed_rotated, k=2)

    plot_three_slices(
        seg_fixed_rotated,
        save_path="./data/plots/fixed_rotated.png",
    )

    # 3) test backtransform
    seg_fixed_inv = rotate_img(seg_fixed_rotated, np.linalg.inv(T), output_shape=seg_fixed.shape)

    plot_three_slices(seg_fixed_inv, save_path="./data/plots/fixed_rotated_inv.png")

    # 4) plot overlay of prealigned volumes
    plot_overlay(
        seg_fixed_rotated,
        seg_moving_rotated,
        save_path="./data/plots/overlay_prealignment.png",
    )


if __name__ == "__main__":
    main()
