import z5py
import numpy as np

from matchmaker.prealignment import get_SVD_transform
from matchmaker.vis import plot_three_slices, plot_overlay
from matchmaker.transform_utils import rotate_with_padding, rotate_with_shape
from scipy.ndimage import affine_transform



def main():
    with z5py.File("./data/platy1_muscles_stardist_moving.n5", "r") as f:
        seg_moving = f["seg"][:]

    with z5py.File("./data/platy1_muscles_stardist_fixed.n5", "r") as f:
        seg_fixed = f["seg"][:]

    gc, Vt, pos_c = get_SVD_transform(seg_moving)

    

    # seg_moving_rotated = rotate_with_padding(seg_moving, Vt.T)  # TODO: add gc as rotation center?
    # seg_moving_rotated2 = rotate_with_shape(seg_moving, Vt.T)

    import napari
    v = napari.Viewer()
    v.add_image(seg_moving_rotated2, name="fixed")
    napari.run()

    # plot_three_slices(seg_moving_rotated)

    # plot_overlay(
    #     seg_fixed,
    #     seg_moving_rotated2,
    # )


if __name__ == "__main__":
    main()
