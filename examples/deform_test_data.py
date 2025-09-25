import numpy as np
import tifffile as tif
import transforms3d as tf3d

from matchmaker.transform_utils import (
    downscale_seg,
    get_transformation_matrix,
    rotate_img,
    crop_to_bbox,
)
from matchmaker.vis import plot_three_slices, plot_overlay
from matchmaker.n5_utils import write_volume


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

    # save downsampled, fixed image
    attributes = {"resolution": [1, 1, 1]}
    write_volume(
        f="./data/platy1_muscles_stardist_fixed.n5",
        arr=seg_fixed,
        key="seg",
        chunks=(128, 512, 512),
        attrs=attributes,
    )

    # save also as tiff
    tif.imwrite("./data/platy1_muscles_stardist_fixed.tif", seg_fixed)

    # rotate image
    center = np.array(seg_fixed.shape) // 2
    rotation = tf3d.euler.euler2mat(
        *[np.deg2rad(155), np.deg2rad(30), np.deg2rad(65)], axes="szyx"
    )

    T, new_shape = get_transformation_matrix(
        seg_fixed, center, rotation, save_path="./data/transformation_matrix.txt"
    )
    seg_moving = rotate_img(seg_fixed, T, output_shape=new_shape)

    # randomly remove instances
    probability = 0.05
    seg_moving = remove_instances(seg_moving, prob=probability)

    # save moving image
    attributes = {"resolution": [1, 1, 1]}
    write_volume(
        f="./data/platy1_muscles_stardist_moving.n5",
        arr=seg_moving,
        key="seg",
        chunks=(128, 512, 512),
        attrs=attributes,
    )

    # save also as tiff
    tif.imwrite("./data/platy1_muscles_stardist_moving.tif", seg_moving)

    # visualize
    plot_three_slices(seg_moving, save_path="./data/plots/seg_moving.png")
    plot_three_slices(seg_fixed, save_path="./data/plots/seg_fixed.png")
    plot_overlay(seg_fixed, seg_moving, save_path="./data/plots/seg_overlay.png")


if __name__ == "__main__":
    main()
