import numpy as np
import z5py
from matchmaker.vis import plot_three_slices, plot_overlay
from matchmaker.transform_utils import rotate_img


def main():
    '''
    Reverse the deformation of a sample by applying the inverse transformation matrix.
    '''
    with z5py.File("./data/platy1_muscles_stardist_moving.n5", "r") as f:
        seg_moving = f["seg"][:]

    # compare with original image
    with z5py.File("./data/platy1_muscles_stardist_fixed.n5", "r") as f:
        seg_fixed = f["seg"][:]

    T = np.loadtxt("./data/transformation_matrix.txt")
    seg_moving_reverse = rotate_img(seg_moving, np.linalg.inv(T), output_shape=seg_fixed.shape)

    plot_three_slices(seg_moving_reverse)

    # compare with original image
    with z5py.File("./data/platy1_muscles_stardist_fixed.n5", "r") as f:
        seg_fixed = f["seg"][:]

    plot_three_slices(seg_fixed)
    plot_overlay(seg_fixed, seg_moving_reverse)


if __name__ == "__main__":
    main()
