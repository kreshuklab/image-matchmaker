import yaml
import numpy as np
import tifffile as tif
from pathlib import Path
import transforms3d as tf3d
from scipy.ndimage import zoom
from skimage.filters import gaussian

from matchmaker.utils import (get_transformation_matrix, rotate_img, write_volume,
                                plot_three_slices, plot_overlay, grid_sample3d, load_config,
                                crop_to_bbox)


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


def save_volume(path, array, key="seg", chunks=(128, 512, 512), resolution=[1,1,1],
                save_tif=True):
    assert path.endswith(".n5")

    write_volume(f=path, arr=array, key=key, chunks=chunks, attrs={"resolution":resolution,},)

    if save_tif:
        tif.imwrite(path.replace(".n5", ".tif"), array)


def rigid_deform(fixed, angles, voxel_spacing):
    if not isinstance(angles, (list, tuple)):
        raise TypeError()

    assert len(angles) == 3
    iso_spacing = np.asarray([1, 1, 1], dtype=np.float32)

    center = np.array(fixed.shape) // 2
    rotation = tf3d.euler.euler2mat(
        *[np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])], axes="szyx"
    )

    T, new_shape = get_transformation_matrix(fixed, center, rotation, iso_spacing,
                                                spacing_out=voxel_spacing)
    moving = rotate_img(fixed, T, output_shape=new_shape)
    return moving


def elastic_deform(volume, alpha=(1.,1.,1.), sigma=None, grid_spacing=16, mode="nearest",
                    align_corners=False, seed=None,):
    """
    Apply elastic deformation to a 3D volume.

    Args:
        volume (np.ndarray): Input volume of shape (D, H, W).
        alpha (tuple[float, float, float]): Displacement amplitude scaling factors.
        sigma (float): Gaussian smoothing std.
        grid_spacing (int | tuple[int, int, int]): Control point spacing (in voxels).
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

    if isinstance(grid_spacing, int):
        grid_spacing = [grid_spacing] * 3
    elif isinstance(grid_spacing, (list, tuple)):
        assert len(grid_spacing) == 3
    else:
        raise ValueError

    if sigma is None:
        sigma = (grid_spacing[0] / 2, grid_spacing[1] / 2, grid_spacing[2] / 2)
    elif isinstance(sigma, (int, float)):
        sigma = (sigma, sigma, sigma)
    elif isinstance(sigma, (list, tuple)):
        assert len(sigma) == 3
        sigma = tuple(sigma)
    else:
        raise ValueError

    shape = (int(np.ceil(D/grid_spacing[0])), int(np.ceil(H/grid_spacing[1])), int(np.ceil(W/grid_spacing[2])))
    disp = np.random.randn(*shape, 3).astype(np.float32)

    disp = gaussian(disp, sigma=(*sigma, 0), mode="constant", preserve_range=True,)
    disp *= alpha

    if (grid_spacing[0] > 1) or (grid_spacing[1] > 1) or (grid_spacing[2] > 1):
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

    return sampled.astype(volume.dtype)


def deform_test_data(cfg_path="", config=None, enable_aniso=False, enable_elastic=False,
                        alpha=0.9, sigma=2, grid_spacing=16, rotate_angles_fixed=[20,345,30],
                        rotate_angles_moving=[155,30,65], remove_p=0.05, seed=42, visualize=True):
    if config is None:
        config = load_config(cfg_path)

    data_dir = Path(config["fixed_image"]["path"]).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    fixed_spacing = [config["fixed_image"]["z_res"], config["fixed_image"]["y_res"], config["fixed_image"]["x_res"]]
    moving_spacing = [config["moving_image"]["z_res"], config["moving_image"]["y_res"], config["moving_image"]["x_res"]]
    fixed_spacing = np.asarray(fixed_spacing, dtype=np.float32)
    moving_spacing = np.asarray(moving_spacing, dtype=np.float32)

    if enable_aniso:
        assert not np.all(fixed_spacing == 1)
        assert not np.all(moving_spacing == 1)

    seg_fixed = tif.imread(config["fixed_image"]["source_path"])
    seg_moving = seg_fixed.copy()

    seg_fixed = rigid_deform(seg_fixed, rotate_angles_fixed, fixed_spacing)
    seg_fixed = crop_to_bbox(seg_fixed)
    print("Fixed volume shape", seg_fixed.shape)

    if enable_elastic:
        seg_moving = elastic_deform(seg_moving, alpha=alpha, sigma=sigma, grid_spacing=grid_spacing, seed=seed,)

    seg_moving = rigid_deform(seg_moving, rotate_angles_moving, moving_spacing)
    seg_moving = remove_instances(seg_moving, prob=remove_p, seed=seed)
    seg_moving = crop_to_bbox(seg_moving)
    print("Moving volume shape", seg_moving.shape)

    save_volume(config["fixed_image"]["path"].replace(".tif", ".n5"), seg_fixed, resolution=fixed_spacing.tolist())
    save_volume(config["moving_image"]["path"].replace(".tif", ".n5"), seg_moving, resolution=moving_spacing.tolist())

    if visualize:
        plot_dir = data_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        iso_name = "aniso_" if enable_aniso else ""
        elastic_name = "elastic" if enable_elastic else "rigid"

        plot_three_slices(seg_fixed, save_path=plot_dir/f"seg_{iso_name}fixed.png")
        plot_three_slices(seg_moving, save_path=plot_dir/f"seg_{iso_name}{elastic_name}.png")
        plot_overlay(seg_fixed, seg_moving, save_path=plot_dir/f"{iso_name}{elastic_name}_overlay.png")


if __name__ == "__main__":
    deform_test_data(cfg_path="examples/register_config_test_rigid.yaml")
    deform_test_data(cfg_path="examples/register_config_test_elastic.yaml", enable_elastic=True)
