import numpy as np
from matchmaker.vis import plot_three_slices, plot_overlay
from matchmaker.transform_utils import rotate_img
from matchmaker.n5_utils import read_volume


def main():
    '''
    Reverse the deformation of a sample by applying the inverse transformation matrix.
    '''
    seg_moving = read_volume(
        f="./data/platy1_muscles_stardist_moving.n5",
        key="seg",
    )

    # compare with original image
    seg_fixed = read_volume(
        f="./data/platy1_muscles_stardist_fixed.n5",
        key="seg",
    )

    T = np.loadtxt("./data/transformation_matrix.txt")
    seg_moving_reverse = rotate_img(seg_moving, np.linalg.inv(T), output_shape=seg_fixed.shape)

    plot_three_slices(seg_moving_reverse)
    plot_three_slices(seg_fixed)
    plot_overlay(seg_fixed, seg_moving_reverse)


if __name__ == "__main__":
    main()
