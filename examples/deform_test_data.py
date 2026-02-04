import os
import yaml
import numpy as np
import tifffile as tif
import transforms3d as tf3d
from scipy.ndimage import zoom
from skimage.filters import gaussian

from matchmaker.utils import (get_transformation_matrix, rotate_img, write_volume,
                                plot_three_slices, plot_overlay, grid_sample3d)


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
    dir = os.path.dirname(path)
    os.makedirs(dir, exist_ok=True)

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


def elastic_deform(volume, alpha=(1.,1.,1.), sigma=None, spacing=16, mode="nearest",
                    align_corners=False, seed=None,):
    """
    Apply elastic deformation to a 3D volume.

    Args:
        volume (np.ndarray): Input volume of shape (D, H, W).
        alpha (tuple[float, float, float]): Displacement amplitude scaling factors.
        sigma (float): Gaussian smoothing std.
        spacing (int): Control point spacing (in voxels).
        mode (str): Interpolation mode, "nearest" or "trilinear".
        align_corners (bool): Grid sampling alignment flag.
        seed (int | None): Random seed.

    Returns:
        np.ndarray: Deformed volume of shape (D, H, W).
    """
    assert volume.ndim == 3, "Input volume must be (D,H,W)"
    vol = volume.astype(np.float32)
    D, H, W = vol.shape
    assert D > 1

    if seed is not None:
        np.random.seed(seed)

    if isinstance(alpha, (int, float)):
        alpha = np.array([alpha] * 3, dtype=np.float32)
    elif isinstance(alpha, (list, tuple)):
        assert len(alpha) == 3
        alpha = np.array(alpha, dtype=np.float32)
    else:
        raise ValueError

    if sigma is None:
        sigma = (spacing / 2, spacing / 2, spacing / 2)
    elif isinstance(sigma, (int, float)):
        sigma = (sigma, sigma, sigma)
    elif isinstance(sigma, (list, tuple)):
        assert len(sigma) == 3
        sigma = tuple(sigma)
    else:
        raise ValueError

    shape = (int(np.ceil(D/spacing)), int(np.ceil(H/spacing)), int(np.ceil(W/spacing)))
    disp = np.random.randn(*shape, 3).astype(np.float32)

    disp = gaussian(disp, sigma=(*sigma, 0), mode="constant", preserve_range=True,)
    disp *= alpha

    if spacing > 1:
        zoom_factors = (D / shape[0], H / shape[1], W / shape[2], 1.,)
        disp = zoom(disp, zoom_factors, order=1)

    def normalize_axis(d):
        x = np.linspace(0, d - 1, d, dtype=np.float32)
        return (x / (d - 1) - 0.5) * 2

    xs_n, ys_n, zs_n = normalize_axis(W), normalize_axis(H), normalize_axis(D)
    grid_z, grid_y, grid_x = np.meshgrid(zs_n, ys_n, xs_n, indexing="ij")
    grid = np.stack((grid_x, grid_y, grid_z), axis=-1)
    new_grid = np.clip(grid + disp, -1., 1.)

    sampled = grid_sample3d(vol, new_grid, align_corners=align_corners, mode=mode)

    return sampled


def deform_test_data(cfg_path="examples/register_config_test.yaml", alpha=0.9, sigma=2,
                        spacing=16, rotate_angles=[155,30,65], remove_p=0.05, seed=42,
                        visualize=True):
    assert os.path.exists(cfg_path)

    with open(cfg_path) as f:
        configs = yaml.safe_load(f)

    seg_fixed = tif.imread(configs["fixed_image"]["path"])
    print("Cropped shape", seg_fixed.shape)

    seg_elastic = seg_fixed.copy()
    seg_rigid = seg_fixed.copy()

    seg_elastic = elastic_deform(seg_elastic, alpha=alpha, sigma=sigma, spacing=spacing, seed=seed,)

    seg_elastic = rigid_deform(seg_elastic, angles=rotate_angles)
    seg_rigid = rigid_deform(seg_rigid, angles=rotate_angles)

    seg_elastic = remove_instances(seg_elastic, prob=remove_p, seed=seed)
    seg_rigid = remove_instances(seg_rigid, prob=remove_p, seed=seed)

    save_volume(configs["fixed_image"]["path"].replace(".tif", ".n5"), seg_fixed, save_tif=False)

    save_volume(configs["moving_elastic"]["path"].replace(".tif", ".n5"), seg_elastic)
    save_volume(configs["moving_image"]["path"].replace(".tif", ".n5"), seg_rigid)

    if visualize:
        os.makedirs("./data/plots", exist_ok=True)
        plot_three_slices(seg_fixed, save_path="./data/plots/seg_fixed.png")

        plot_three_slices(seg_elastic, save_path="./data/plots/seg_elastic.png")
        plot_overlay(seg_fixed, seg_elastic, save_path="./data/plots/elastic_overlay.png")

        plot_three_slices(seg_rigid, save_path="./data/plots/seg_rigid.png")
        plot_overlay(seg_fixed, seg_rigid, save_path="./data/plots/rigid_overlay.png")


if __name__ == "__main__":
    deform_test_data()
