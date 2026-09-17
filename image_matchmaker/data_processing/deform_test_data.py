import numpy as np
from pathlib import Path
import transforms3d as tf3d
from scipy.ndimage import zoom
from skimage.filters import gaussian

from image_matchmaker.utils import (get_transformation_matrix, rotate_img, load_data, save_data,
                                plot_three_slices, plot_overlay, grid_sample3d, load_config,
                                crop_to_bbox, resample_volume, get_spacings, get_paths)


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
    result = seg.copy()
    result[mask] = 0
    print(f"Number of instances left: {len(np.unique(result))}")

    return result


def rigid_deform(volume, angles, input_spacing):
    if not isinstance(angles, (list, tuple)):
        raise TypeError()

    if len(angles) != 3:
        raise ValueError("rotation must contain 3 angles")

    center = np.array(volume.shape) // 2
    rotation = tf3d.euler.euler2mat(*np.deg2rad(angles), axes="szyx")

    T, new_shape = get_transformation_matrix(volume, center, rotation, input_spacing)
    result = rotate_img(volume, T, output_shape=new_shape)

    return result


def elastic_deform(volume, alpha=(1.,1.,1.), sigma=None, grid_spacing=16, mode="nearest",
                    align_corners=False,):
    """
    Apply elastic deformation to a 3D volume.

    Args:
        volume (np.ndarray): Input volume of shape (D, H, W).
        alpha (tuple[float, float, float]): Displacement amplitude scaling factors.
        sigma (float): Gaussian smoothing std.
        grid_spacing (int | tuple[int, int, int]): Control point spacing (in voxels).
        mode (str): Interpolation mode, "nearest" or "trilinear".
        align_corners (bool): Grid sampling alignment flag.

    Returns:
        np.ndarray: Deformed volume of shape (D, H, W).
    """
    assert volume.ndim == 3, "Input volume must be (D,H,W)"
    vol = volume.astype(np.float32)
    D, H, W = vol.shape
    assert D > 1

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


def deform_data(volume, elastic=False, alpha=None, sigma=None, grid_spacing=None,
                rotation=[0,0,0], remove_p=0., input_spacing=(1,1,1), output_spacing=(1,1,1)):
    """
    Deform a 3D volume(ZYX) with the following sequential operations (if the corresponding
    parameters are given):

    Input -> Elastic deformation (control grid based) -> Rigid deformation (rotation)
        -> Resample -> Remove Instances -> Crop to minimized bbox
    """
    input_spacing = np.asarray(input_spacing, dtype=np.float32)
    output_spacing = np.asarray(output_spacing, dtype=np.float32)

    if elastic:
        assert (alpha is not None) and (sigma is not None) and (grid_spacing is not None)
        result = elastic_deform(volume, alpha=alpha, sigma=sigma, grid_spacing=grid_spacing)
    else:
        result = volume.copy()

    if any(rotation):
        result = rigid_deform(result, rotation, input_spacing)

    if not np.array_equal(output_spacing, input_spacing):
        result = resample_volume(result, input_spacing, output_spacing)

    if remove_p > 0:
        result = remove_instances(result, prob=remove_p)

    result = crop_to_bbox(result)

    return result


def deform_test_data(cfg_path="", config=None, enable_elastic=False,
                        alpha=0.9, sigma=2, grid_spacing=16, rotate_angles_fixed=[20,345,30],
                        rotate_angles_moving=[155,30,65], remove_p=0.05, seed=42,
                        save_as_n5=False, chunks=(128,512,512), visualize=True):
    if config is None:
        config = load_config(cfg_path)

    if seed is not None:
        np.random.seed(seed)

    data_dir = Path(config["fixed_image"]["path"]).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    fixed_spacing, moving_spacing = get_spacings(config)
    fixed_attrs = {"resolution": fixed_spacing,}
    moving_attrs = {"resolution": moving_spacing,}

    seg_fixed = load_data(config["fixed_image"]["source_path"])
    seg_moving = seg_fixed.copy()

    seg_fixed = deform_data(seg_fixed, rotation=rotate_angles_fixed, output_spacing=fixed_spacing)
    print("Fixed volume shape", seg_fixed.shape)

    seg_moving = deform_data(seg_moving, elastic=enable_elastic, alpha=alpha, sigma=sigma,
                            grid_spacing=grid_spacing, rotation=rotate_angles_moving,
                            remove_p=remove_p, output_spacing=moving_spacing)
    print("Moving volume shape", seg_moving.shape)

    fixed_path, moving_path = get_paths(config)
    if save_as_n5:
        fixed_path = fixed_path.replace(".tif", ".n5")
        moving_path = moving_path.replace(".tif", ".n5")

    save_data(seg_fixed, fixed_path, output_key="seg", n5_exists=False, chunks=chunks, attrs=fixed_attrs)
    save_data(seg_moving, moving_path, output_key="seg", n5_exists=False, chunks=chunks, attrs=moving_attrs)

    if visualize:
        plot_dir = data_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        iso_name = "" if np.all(moving_spacing == 1) else "aniso_"
        elastic_name = "elastic" if enable_elastic else "rigid"

        plot_three_slices(seg_fixed, save_path=plot_dir/f"seg_{iso_name}fixed.png")
        plot_three_slices(seg_moving, save_path=plot_dir/f"seg_{iso_name}{elastic_name}.png")
        plot_overlay(seg_fixed, seg_moving, save_path=plot_dir/f"{iso_name}{elastic_name}_overlay.png")


if __name__ == "__main__":
    deform_test_data(cfg_path="examples/register_config_test_rigid.yaml")
    deform_test_data(cfg_path="examples/register_config_test_elastic.yaml", enable_elastic=True)
