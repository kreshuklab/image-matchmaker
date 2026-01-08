import os
import numpy as np
import tifffile as tif
import transforms3d as tf3d

from matchmaker.utils import (get_transformation_matrix, rotate_img, write_volume,
                                plot_three_slices, plot_overlay, to_pair, get_gaussian_kernel2d,
                                grid_sample3d)


def remove_instances(seg, prob=0.05, seed=None):
    if seed is not None:
        np.random.seed(seed)

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


def save_volume(path, array, key="seg", chunks=(128, 512, 512), attributes={"resolution":[1,1,1]},
                save_tif=True):
    assert path.endswith(".n5")
    write_volume(f=path, arr=array, key=key, chunks=chunks, attrs=attributes,)

    if save_tif:
        tif.imwrite(path.replace(".n5", ".tif"), array)


def rigid_deform(fixed, angles):
    if not isinstance(angles, (list, tuple)):
        raise TypeError()

    assert len(angles) == 3

    center = np.array(fixed.shape) // 2
    rotation = tf3d.euler.euler2mat(
        *[np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])], axes="szyx"
    )

    T, new_shape = get_transformation_matrix(fixed, center, rotation)
    moving = rotate_img(fixed, T, output_shape=new_shape)
    return moving


def elastic_deform(volume, noise=None, kernel_size=(63, 63), sigma=(32.0, 32.0),
                    alpha=(1.0,1.0,1.0), align_corners=False, mode="trilinear",
                    seed=None, z_variation=0.05):
    """
    Apply elastic deformation to a 3D volume.

    Args:
        volume (np.ndarray): Input volume of shape (D, H, W).
        noise (np.ndarray): Noise field of shape (3, H, W).
        kernel_size (tuple[int, int]): Gaussian kernel size.
        sigma (tuple[float, float]): Gaussian smoothing std.
        alpha (tuple[float, float, float]): Displacement scaling factors.
        align_corners (bool): Grid sampling alignment flag.
        mode (str): Interpolation mode, "nearest" or "trilinear".
        seed (int | None): Random seed.
        z_variation (float): Per-slice z-axis displacement scaling factor.

    Returns:
        np.ndarray: Deformed volume of shape (D, H, W).
    """
    import cv2
    assert volume.ndim == 3, "Input volume must be (D,H,W)"
    vol = volume.astype(np.float32)
    D, H, W = vol.shape
    assert D > 1

    kernel_size, sigma = to_pair(kernel_size), to_pair(sigma)

    if isinstance(alpha, (int, float)):
        alpha_xyz = np.array([alpha, alpha, alpha], dtype=np.float32)
    elif isinstance(alpha, (list, tuple)):
        assert len(alpha) == 3
        alpha_xyz = np.array(alpha, dtype=np.float32)
    else:
        raise ValueError

    if seed is not None:
        np.random.seed(seed)

    if noise is None:
        noise = np.random.randn(3, H, W).astype(np.float32)
    else:
        noise = np.asarray(noise, dtype=np.float32)
        assert noise.shape == (3, H, W)

    z_noise = np.random.randn(D).astype(np.float32)
    scale = 1. + z_variation * z_noise

    kernel = get_gaussian_kernel2d(kernel_size, sigma)
    disp_hw = np.stack([cv2.filter2D(noise[c], -1, kernel, borderType=cv2.BORDER_CONSTANT)
                        for c in range(3)], axis=-1)

    disp_hw *= alpha_xyz
    disp = np.repeat(disp_hw[None, ...], D, axis=0)
    disp *= scale[:, None, None, None]

    def normalize_axis(d):
        x = np.linspace(0, d - 1, d, dtype=np.float32)
        return (x / (d - 1) - 0.5) * 2

    xs_n, ys_n, zs_n = normalize_axis(W), normalize_axis(H), normalize_axis(D)
    grid_z, grid_y, grid_x = np.meshgrid(zs_n, ys_n, xs_n, indexing="ij")
    grid = np.stack((grid_x, grid_y, grid_z), axis=-1)
    new_grid = np.clip(grid + disp, -1., 1.)

    sampled = grid_sample3d(vol, new_grid, align_corners=align_corners, mode=mode)

    return sampled


def deform_test_data(apply_rigid=True, apply_elastic=True, remove_p=0.05,
                        rotate_angles=[155,30,65], kernel_size=63, sigma=6, alpha=0.2,
                        align_corners=False, mode="nearest", seed=42, visualize=True):
    assert apply_rigid or apply_elastic

    seg_fixed = tif.imread("data/platy1_muscles_stardist_fixed.tif")
    print("Cropped shape", seg_fixed.shape)

    seg_moving = seg_fixed.copy()
    if apply_rigid:
        seg_moving = rigid_deform(seg_moving, angles=rotate_angles)

    if apply_elastic:
        seg_moving = elastic_deform(seg_moving, kernel_size=kernel_size, sigma=sigma, alpha=alpha,
                                        align_corners=align_corners, mode=mode, seed=seed)

    seg_moving = remove_instances(seg_moving, prob=remove_p, seed=seed)

    save_volume("./data/platy1_muscles_stardist_fixed.n5", seg_fixed, save_tif=False)

    save_volume("./data/platy1_muscles_stardist_moving.n5", seg_moving)

    if visualize:
        os.makedirs("./data/plots", exist_ok=True)
        plot_three_slices(seg_moving, save_path="./data/plots/seg_moving.png")
        plot_three_slices(seg_fixed, save_path="./data/plots/seg_fixed.png")
        plot_overlay(seg_fixed, seg_moving, save_path="./data/plots/seg_overlay.png")


if __name__ == "__main__":
    deform_test_data()
