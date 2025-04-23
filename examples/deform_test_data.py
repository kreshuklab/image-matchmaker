import numpy as np
import tifffile as tif
import z5py
from elf.wrapper.resized_volume import ResizedVolume

from matchmaker.transform_utils import pad_img, get_rotation_matrix, rotate_img, crop_to_bbox
from matchmaker.vis import plot_three_slices, plot_overlay


def downscale_seg(seg, factor):
    new_shape = np.array(seg.shape) // 2
    downsampled_seg = ResizedVolume(seg, shape=new_shape)[:]
    print("Downsampled shape", downsampled_seg.shape)

    return downsampled_seg


def remove_instances(seg, prob=0.05):
    # Get all unique instance IDs, excluding background (assumed to be 0)
    instance_ids = np.unique(seg)
    instance_ids = instance_ids[instance_ids != 0]

    # Randomly select 10% of the instance IDs
    num_to_remove = int(len(instance_ids) * prob)
    print(f"Number of instances to remove: {num_to_remove}")
    selected_ids = np.random.choice(instance_ids, size=num_to_remove, replace=False)

    # Create a mask for the selected IDs and set them to 0
    mask = np.isin(seg, selected_ids)
    seg[mask] = 0

    print(f"Number of instances left: {len(np.unique(seg))}")
    return seg


def main():
    seg = tif.imread("data/platy1_muscles_stardist.tif")

    # downsample image
    factor = 4
    seg = downscale_seg(seg, factor)
    seg_fixed = crop_to_bbox(seg)
    print("Cropped shape", seg_fixed.shape)
    # save downsampled image
    with z5py.File("./data/platy1_muscles_stardist_fixed.n5", "w") as f:
        f.create_dataset("seg", data=seg_fixed)

    # rotate image
    seg_padded = pad_img(seg)
    angles = [np.deg2rad(155), np.deg2rad(30), np.deg2rad(65)]  # z,y,x
    rotation_matrix = get_rotation_matrix(angles, save_path="./data/rotation_matrix.txt")
    rotated_seg = rotate_img(seg_padded, rotation_matrix)
    seg_cropped = crop_to_bbox(rotated_seg)
    print("Cropped shape", seg_cropped.shape)

    # randomly remove instances
    probability = 0.05
    seg_moving = remove_instances(seg_cropped, prob=probability)

    with z5py.File("./data/platy1_muscles_stardist_moving.n5", "w") as f:
        f.create_dataset("seg", data=seg_moving)

    # visualize
    plot_three_slices(seg_moving)
    plot_overlay(seg_fixed, seg_moving)


if __name__ == "__main__":
    main()
