import numpy as np
import tifffile as tif
import transforms3d as tf3d
import z5py
from scipy.ndimage import affine_transform
import napari
from elf.wrapper.resized_volume import ResizedVolume

import sys
sys.path.append("..")
from matchmaker.vis import plot_overlay


def downscale_seg(seg, factor):
    new_shape = np.array(seg.shape) // 2
    downsampled_seg = ResizedVolume(seg, shape=new_shape)[:]
    print("Downsampled shape", downsampled_seg.shape)

    return downsampled_seg


def rotate_seg(seg, angles):
    # pad zeros so segmentation is not cut out off the image
    shape = seg.shape
    diagonal = int(np.ceil(np.linalg.norm(shape)))
    print("Diagonal", diagonal)

    # Calculate how much padding is needed on each axis
    pad_widths = []
    for dim in shape:
        total_pad = diagonal - dim
        before = total_pad // 2
        after = total_pad - before
        pad_widths.append((before, after))

    # Apply zero padding
    padded = np.pad(seg, pad_width=pad_widths, mode='constant', constant_values=0)
    print("New shape after padding:", padded.shape)

    rotation_matrix = tf3d.euler.euler2mat(*angles, axes='szyx')
    print(rotation_matrix)

    # Compute center
    center = np.array(padded.shape) / 2
    offset = center - rotation_matrix @ center

    # rotate the image around the center
    rotated_seg = affine_transform(
        padded,
        matrix=rotation_matrix,
        offset=offset,  # offset to ensure the image fits within the new bounding box
        order=0,  # interpolation (use 0 for discrete/label data)
        mode='constant',  # fill mode
        cval=0.0  # fill value (if constant mode)
    )

    print(f"New shape after rotation: {rotated_seg.shape}")
    return rotated_seg


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


def crop_to_bbox(seg):
    """Crops a 3D volume to the minimal bounding box around all nonzero voxels."""
    # Find where the volume is nonzero (i.e., contains instances)
    nonzero = np.argwhere(seg)

    # Compute bounding box from min to max index along each axis
    z_min, y_min, x_min = nonzero.min(axis=0)
    z_max, y_max, x_max = nonzero.max(axis=0) + 1  # +1 to include the max index

    # Crop the volume
    cropped = seg[z_min:z_max, y_min:y_max, x_min:x_max]
    return cropped


def main():
    seg = tif.imread("data/platy1_muscles_stardist.tif")

    # downsample image
    factor = 4
    seg = downscale_seg(seg, factor)
    with z5py.File("./data/platy1_muscles_stardist_fixed.n5", "w") as f:
        f.create_dataset("seg", data=seg)

    # rotate image
    angles = [np.deg2rad(155), np.deg2rad(30), np.deg2rad(65)]  # z,y,x
    rotated_seg = rotate_seg(seg, angles)

    # randomly remove instances
    probability = 0.05
    seg_rm = remove_instances(rotated_seg, prob=probability)
    seg_cropped = crop_to_bbox(seg_rm)
    print("Cropped shape", seg_cropped.shape)

    with z5py.File("./data/platy1_muscles_stardist_moving.n5", "w") as f:
        f.create_dataset("seg", data=seg_cropped)

    plot_overlay(seg, seg_cropped)
    # visualize
    # v = napari.Viewer()
    # v.add_labels(seg_cropped, name="moving")
    # napari.run()


if __name__ == "__main__":
    main()
