import z5py
import numpy as np

from matchmaker.prealignment import get_SVD_transform
from matchmaker.vis import plot_three_slices, plot_overlay
from matchmaker.transform_utils import rotate


def main():
    with z5py.File("./data/platy1_muscles_stardist_moving.n5", "r") as f:
        seg_moving = f["seg"][:]

    with z5py.File("./data/platy1_muscles_stardist_fixed.n5", "r") as f:
        seg_fixed = f["seg"][:]

    gc, Vt, pos_c = get_SVD_transform(seg_moving)

    seg_moving_rotated = rotate(seg_moving, np.linalg.inv(Vt))  # TODO: add gc as rotation center?

    plot_three_slices(seg_moving_rotated)

    plot_overlay(
        seg_moving_rotated,
        seg_fixed,
    )
    # y sieht aus wie z: kacke


if __name__ == "__main__":
    main()
