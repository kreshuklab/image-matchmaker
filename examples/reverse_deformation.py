import numpy as np
import z5py
from matchmaker.vis import plot_three_slices, plot_overlay
from deform_test_data import pad_seg, crop_to_bbox, rotate_seg


def main():
    with z5py.File("./data/platy1_muscles_stardist_moving.n5", "r") as f:
        seg = f["seg"][:]

    seg_padded = pad_seg(seg)
    rotation_matrix = np.loadtxt("./data/rotation_matrix.txt")
    rotated_seg = rotate_seg(seg_padded, np.linalg.inv(rotation_matrix))
    seg_cropped = crop_to_bbox(rotated_seg)
    print("Cropped shape", seg_cropped.shape)

    plot_three_slices(seg_cropped)

    # compare with original image
    with z5py.File("./data/platy1_muscles_stardist_fixed.n5", "r") as f:
        seg_fixed = f["seg"][:]

    plot_overlay(seg_cropped, seg_fixed)


if __name__ == "__main__":
    main()
