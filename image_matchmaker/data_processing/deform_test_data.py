import logging
import numpy as np
from pathlib import Path
import transforms3d as tf3d
from scipy.ndimage import zoom, center_of_mass
from skimage.filters import gaussian

from image_matchmaker.utils import (
    get_transformation_matrix,
    rotate_img,
    load_data,
    save_data,
    plot_three_slices,
    plot_overlay,
    grid_sample3d,
    load_config,
    crop_to_bbox,
    resample_volume,
    get_spacings,
    get_paths,
    plot_removal,
    plot_jitter,
    setup_logging,
)

INSTANCE_REMOVAL_MODES = ("random_instances", "clustered_instances")
REGION_REMOVAL_MODES = ("box", "sphere")


def _to3(value, name):
    """Return a scalar or 3-element sequence as an array of 3 floats."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim > 1 or arr.size not in (1, 3):
        raise ValueError(f"{name} must be a scalar or have 3 elements, got {value}")
    return np.broadcast_to(arr, 3).copy()


def _count_instances(seg):
    return np.count_nonzero(np.unique(seg))


def _log_removal(num_before, result):
    num_left = _count_instances(result)
    logging.info(f"Number of instances left: {num_left}")
    logging.info(f"Number of instances removed: {num_before - num_left}")
    logging.info(f"Fraction of instances that are removed: {1 - num_left / num_before:.2%}")


def remove_instances(seg, fraction=0.05, mode="random_instances", exclude_ids=(), center=None):
    """Remove a fraction of all instances from a 3D instance segmentation.

    Args:
        seg: Input instance segmentation with background label 0.
        fraction: Fraction of instances to remove, in [0, 1).
        mode: "random_instances" removes randomly chosen instances.
            "clustered_instances" removes the instances whose centroids are
            closest to `center`.
        exclude_ids: Instance IDs that must not be removed.
        center: (z, y, x) center for "clustered_instances". If None, a random
            foreground voxel is used.

    Returns:
        Segmentation with the selected instances set to 0.
    """
    if not 0 <= fraction < 1:
        raise ValueError("fraction must be in [0, 1)")
    if mode not in INSTANCE_REMOVAL_MODES:
        raise ValueError(f"Unsupported instance removal mode: {mode}")

    instance_ids = np.unique(seg)
    instance_ids = instance_ids[instance_ids != 0]
    num_before = len(instance_ids)
    logging.info(f"Number of instances before removal: {num_before}")

    num_to_remove = int(num_before * fraction)
    logging.info(f"Number of instances to remove: {num_to_remove}")
    if num_to_remove == 0:
        return seg.copy()

    instance_ids = instance_ids[~np.isin(instance_ids, exclude_ids)]
    if num_to_remove > len(instance_ids):
        raise ValueError("Not enough instances available after applying exclude_ids")

    if mode == "random_instances":
        selected_ids = np.random.choice(instance_ids, size=num_to_remove, replace=False)
    else:
        if center is None:
            coords = np.argwhere(seg > 0)
            center = coords[np.random.randint(len(coords))]
        logging.info(f"Center of removal: {center}")

        instance_centers = np.asarray(center_of_mass(seg > 0, seg, instance_ids))
        distances = np.linalg.norm(instance_centers - center, axis=1)
        selected_ids = instance_ids[np.argsort(distances)[:num_to_remove]]

    result = seg.copy()
    result[np.isin(seg, selected_ids)] = 0
    _log_removal(num_before, result)
    return result


def remove_region(seg, size_fraction=0.05, shape="sphere", exclude_ids=(), center=None,
                  max_tries=1000):
    """Set a box or ellipsoid region of a 3D instance segmentation to background.

    The region size is given relative to the foreground bounding box, not as
    a fraction of instances. The number of removed instances depends on how
    dense the segmentation is inside the region.

    Args:
        seg: Input instance segmentation with background label 0.
        size_fraction: Extent of the region as a fraction of each dimension of
            the foreground bounding box, in [0, 1). For "sphere" this is the
            diameter of an ellipsoid with radii `size_fraction / 2 * extent`.
        shape: "box" or "sphere".
        exclude_ids: Instance IDs that must not be touched by the region.
        center: (z, y, x) center of the region. If None, random centers are
            tried until the region does not overlap `exclude_ids`.
        max_tries: Maximum number of random centers to try.

    Returns:
        Segmentation with the region set to 0.
    """
    if not 0 <= size_fraction < 1:
        raise ValueError("size_fraction must be in [0, 1)")
    if shape not in REGION_REMOVAL_MODES:
        raise ValueError(f"Unsupported region shape: {shape}")
    if max_tries < 1:
        raise ValueError("max_tries must be at least 1")

    num_before = _count_instances(seg)
    logging.info(f"Number of instances before removal: {num_before}")

    _, bbox = crop_to_bbox(seg, return_bbox=True)
    extent = np.diff(np.reshape(bbox, (3, 2)), axis=1).ravel()
    logging.info(f"Foreground bbox extent: {extent.tolist()}")
    half_size = size_fraction * extent / 2
    D, H, W = seg.shape
    z, y, x = np.ogrid[:D, :H, :W]
    excluded_mask = np.isin(seg, exclude_ids)

    def region_mask(c):
        if shape == "box":
            return ((np.abs(z - c[0]) <= half_size[0])
                    & (np.abs(y - c[1]) <= half_size[1])
                    & (np.abs(x - c[2]) <= half_size[2]))
        return (((z - c[0]) / half_size[0]) ** 2
                + ((y - c[1]) / half_size[1]) ** 2
                + ((x - c[2]) / half_size[2]) ** 2 <= 1)

    if center is not None:
        center = np.asarray(center)
        mask = region_mask(center)
        if np.any(mask & excluded_mask):
            raise ValueError(f"The {shape} region at {center.tolist()} overlaps excluded instances.")
    else:
        for _ in range(max_tries):
            center = np.array([np.random.randint(D), np.random.randint(H), np.random.randint(W)])
            mask = region_mask(center)
            if not np.any(mask & excluded_mask):
                break
        else:
            raise RuntimeError(f"Could not find a {shape} region without excluded instances "
                               f"after {max_tries} attempts.")

    logging.info(f"Center of removal: {center}")
    logging.info(f"Removed {np.sum(mask):,} voxels")

    result = seg.copy()
    result[mask] = 0
    _log_removal(num_before, result)
    return result


def move_instances(seg, max_displacement, mode="jitter_instances"):
    """Randomly shift each instance as a whole in a 3D segmentation.

    Args:
        seg: Input instance segmentation with background label 0.
        max_displacement: Maximum absolute displacement in voxels, as a scalar
            or (z, y, x) tuple.
        mode: Only "jitter_instances" is supported.

    Returns:
        Segmentation with moved foreground voxels.
    """
    logging.info("Movement mode: %s", mode)

    max_displacement = np.asarray(max_displacement)
    if max_displacement.ndim == 0:
        max_displacement = np.repeat(max_displacement, 3)
    if (max_displacement.shape != (3,) or
            np.any(max_displacement < 0) or
            np.any(max_displacement != max_displacement.astype(int))):
        raise ValueError("max_displacement must be a non-negative integer or 3-tuple")

    max_displacement = max_displacement.astype(int)
    if mode != "jitter_instances":
        raise ValueError(f"Unsupported movement mode: {mode}")
    if not np.any(max_displacement):
        return seg.copy()

    coords = np.argwhere(seg > 0)
    logging.info("Foreground voxels before movement: %d", len(coords))
    if len(coords) == 0:
        return seg.copy()

    labels = seg[tuple(coords.T)]

    _, inverse = np.unique(labels, return_inverse=True)
    shifts = np.random.randint(
        -max_displacement, max_displacement + 1, size=(inverse.max() + 1, 3)
    )
    displacements = shifts[inverse]

    moved_coords = coords + displacements
    valid = np.all((moved_coords >= 0) & (moved_coords < seg.shape), axis=1)
    moved_coords, labels = moved_coords[valid], labels[valid]

    # Resolve collisions by randomly selecting one voxel per destination.
    order = np.random.permutation(len(moved_coords))
    flat_indices = np.ravel_multi_index(moved_coords[order].T, seg.shape)
    unique_indices, first = np.unique(flat_indices, return_index=True)

    result = np.zeros_like(seg)
    result.ravel()[unique_indices] = labels[order[first]]
    logging.info("Foreground voxels after movement: %d", np.count_nonzero(result))

    return result


def rigid_deform(volume, angles, input_spacing):
    """Rotate a 3D volume around its center.

    Args:
        volume: Input volume (z, y, x).
        angles: Three rotation angles in degrees, applied in "szyx" Euler order.
        input_spacing: Voxel spacing (z, y, x) of the volume.

    Returns:
        Rotated volume. The output shape is enlarged to contain the whole
        rotated volume.
    """
    angles = _to3(angles, "angles")
    center = np.array(volume.shape) // 2
    rotation = tf3d.euler.euler2mat(*np.deg2rad(angles), axes="szyx")

    T, new_shape = get_transformation_matrix(volume, center, rotation, input_spacing)
    return rotate_img(volume, T, output_shape=new_shape)


def elastic_deform(volume, alpha=1., sigma=None, grid_spacing=16, mode="nearest",
                   align_corners=False):
    """
    Apply elastic deformation to a 3D volume.

    A random displacement field is drawn on a coarse control grid, smoothed,
    scaled and upsampled to the volume size. The displacements are in the
    normalized grid coordinates [-1, 1] of `grid_sample3d`, so the shift in
    voxels for the same `alpha` grows with the volume size along each axis.

    Args:
        volume (np.ndarray): Input volume of shape (D, H, W).
        alpha (float | tuple[float, float, float]): Displacement amplitude
            scaling factors, in (x, y, z) order of the sampling grid.
        sigma (float | tuple[float, float, float] | None): Gaussian smoothing
            std on the control grid. Defaults to grid_spacing / 2.
        grid_spacing (int | tuple[int, int, int]): Control point spacing (in voxels).
        mode (str): Interpolation mode, "nearest" or "trilinear".
        align_corners (bool): Grid sampling alignment flag.

    Returns:
        np.ndarray: Deformed volume of shape (D, H, W).
    """
    if volume.ndim != 3:
        raise ValueError(f"Input volume must be (D, H, W), got shape {volume.shape}")
    vol = volume.astype(np.float32)
    D, H, W = vol.shape
    if min(D, H, W) < 2:
        raise ValueError(f"All volume dimensions must be > 1, got shape {volume.shape}")

    alpha = _to3(alpha, "alpha").astype(np.float32)
    grid_spacing = _to3(grid_spacing, "grid_spacing")
    sigma = grid_spacing / 2 if sigma is None else _to3(sigma, "sigma")

    shape = tuple(int(np.ceil(n / g)) for n, g in zip((D, H, W), grid_spacing))
    disp = np.random.randn(*shape, 3).astype(np.float32)

    disp = gaussian(disp, sigma=(*sigma, 0), mode="constant", preserve_range=True,)
    disp *= alpha

    if np.any(grid_spacing > 1):
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


def _rotate_resample(volume, angles, output_spacing):
    """Rotate a volume with unit spacing and resample it to `output_spacing`."""
    input_spacing = np.ones(3, dtype=np.float32)
    output_spacing = np.asarray(output_spacing, dtype=np.float32)

    if any(angles):
        volume = rigid_deform(volume, angles, input_spacing)
    if not np.array_equal(output_spacing, input_spacing):
        volume = resample_volume(volume, input_spacing, output_spacing)
    return volume


def deform_test_data(
    cfg_path="",
    config=None,
    enable_elastic=False,
    alpha=0.9,
    sigma=2,
    grid_spacing=16,
    rotate_angles_fixed=(20, 345, 30),
    rotate_angles_moving=(155, 30, 65),
    move_max_displacement=None,
    move_mode="jitter_instances",
    remove_fraction=0.05,
    remove_mode="random_instances",
    exclude_ids=(),
    remove_center=None,
    seed=42,
    save_as_n5=False,
    chunks=(128, 512, 512),
    visualize=True,
    save_full_moving=False,
):
    """Create a fixed/moving segmentation pair from the config's source segmentation.

    Fixed: source -> rotation -> resampling -> crop to bbox.
    Moving: source -> elastic deformation (optional) -> rotation -> resampling
        -> jittering instances (optional) -> removal -> crop to bbox.

    If `save_full_moving` is True, the moving volume without jitter and removal is
    also saved as "<moving name>_full", cropped with the bbox of the moving volume,
    so both share the same voxel frame. Instances that do not lie completely inside
    this bbox are removed, so all kept instances have their true centroid.

    Logs and plots are written to the folder of the moving image, the plot of the
    fixed image to the folder of the fixed image.
    Instance IDs are kept, so they give the ground-truth correspondence.
    For `remove_mode` "box"/"sphere", `remove_fraction` is the region size as a
    fraction of each foreground bbox dimension (see `remove_region`), otherwise it is
    the fraction of removed instances (see `remove_instances`).

    Returns:
        dict with "num_before", the number of moving instances after elastic deformation
        and rotation, and "num_left", the number of instances in the saved moving image.
        The difference includes instances lost by jitter collisions and by removal.
    """
    if remove_mode not in INSTANCE_REMOVAL_MODES + REGION_REMOVAL_MODES:
        raise ValueError(f"Unsupported removal mode: {remove_mode}")
    if config is None:
        config = load_config(cfg_path)

    if seed is not None:
        np.random.seed(seed)

    data_dir = Path(config["moving_image"]["path"]).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    Path(config["fixed_image"]["path"]).parent.mkdir(parents=True, exist_ok=True)

    setup_logging(data_dir, "deform_test_data.log")
    logging.info(f"Removal mode: {remove_mode}")

    fixed_spacing, moving_spacing = get_spacings(config)
    logging.info(f"Fixed spacing: {fixed_spacing}")
    logging.info(f"Moving spacing: {moving_spacing}")

    seg_source = load_data(config["fixed_image"]["source_path"])

    seg_fixed = crop_to_bbox(_rotate_resample(seg_source, rotate_angles_fixed, fixed_spacing))
    logging.info(f"Fixed volume shape: {seg_fixed.shape}")

    seg_before_removal = seg_source
    if enable_elastic:
        seg_before_removal = elastic_deform(seg_before_removal, alpha=alpha, sigma=sigma,
                                            grid_spacing=grid_spacing)
    seg_before_removal = _rotate_resample(seg_before_removal, rotate_angles_moving, moving_spacing)
    seg_full = seg_before_removal
    if move_max_displacement is not None:
        seg_before_removal = move_instances(seg_before_removal, move_max_displacement, move_mode)

    if remove_fraction == 0:
        seg_removed = seg_before_removal
    elif remove_mode in REGION_REMOVAL_MODES:
        seg_removed = remove_region(seg_before_removal, size_fraction=remove_fraction,
                                    shape=remove_mode, exclude_ids=exclude_ids,
                                    center=remove_center)
    else:
        seg_removed = remove_instances(seg_before_removal, fraction=remove_fraction,
                                       mode=remove_mode, exclude_ids=exclude_ids,
                                       center=remove_center)

    seg_moving, (z0, z1, y0, y1, x0, x1) = crop_to_bbox(seg_removed, return_bbox=True)
    logging.info(f"Moving volume shape: {seg_moving.shape}")
    stats = {"num_before": _count_instances(seg_full), "num_left": _count_instances(seg_moving)}

    fixed_path, moving_path = get_paths(config)
    if save_as_n5:
        fixed_path = fixed_path.replace(".tif", ".n5")
        moving_path = moving_path.replace(".tif", ".n5")

    save_data(seg_fixed, fixed_path, output_key="seg", n5_exists=False, chunks=chunks,
              attrs={"resolution": fixed_spacing})
    save_data(seg_moving, moving_path, output_key="seg", n5_exists=False, chunks=chunks,
              attrs={"resolution": moving_spacing})
    if save_full_moving:
        full_path = Path(moving_path)
        full_path = str(full_path.with_name(f"{full_path.stem}_full{full_path.suffix}"))
        ids, counts = np.unique(seg_full, return_counts=True)
        seg_full_crop = seg_full[z0:z1, y0:y1, x0:x1].copy()
        cropped_ids, cropped_counts = np.unique(seg_full_crop, return_counts=True)
        cut_ids = cropped_ids[cropped_counts != counts[np.searchsorted(ids, cropped_ids)]]
        cut_ids = cut_ids[cut_ids != 0]
        seg_full_crop[np.isin(seg_full_crop, cut_ids)] = 0
        logging.info(f"Full moving: removed {len(cut_ids)} instances cut by the bbox")
        logging.info(f"Full moving: {_count_instances(seg_full_crop)} instances, saved to {full_path}")
        save_data(seg_full_crop, full_path, output_key="seg", n5_exists=False, chunks=chunks,
                  attrs={"resolution": moving_spacing})

    if visualize:
        plot_dir = data_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        iso_name = "" if (moving_spacing == [moving_spacing[0]] * len(moving_spacing)) else "aniso_"
        elastic_name = "elastic" if enable_elastic else "rigid"

        fixed_plot_dir = Path(fixed_path).parent / "plots"
        fixed_plot_dir.mkdir(parents=True, exist_ok=True)
        plot_three_slices(seg_fixed, save_path=fixed_plot_dir/f"seg_{iso_name}fixed.png")
        plot_three_slices(seg_moving, save_path=plot_dir/f"seg_{iso_name}{elastic_name}.png")
        plot_overlay(seg_fixed, seg_moving, save_path=plot_dir/f"{iso_name}{elastic_name}_overlay.png")

        if move_max_displacement is not None:
            plot_jitter(seg_full[z0:z1, y0:y1, x0:x1], seg_moving,
                        save_path=plot_dir/f"{iso_name}{elastic_name}_jitter.png")

        # Removal only sets voxels to 0, so the pre-removal bbox contains both volumes.
        seg_before_removal, bbox = crop_to_bbox(seg_before_removal, return_bbox=True)
        z0, z1, y0, y1, x0, x1 = bbox
        plot_removal(seg_before_removal, seg_removed[z0:z1, y0:y1, x0:x1],
                     save_path=plot_dir/f"{iso_name}{elastic_name}_removal.png")

    return stats


if __name__ == "__main__":
    deform_test_data(cfg_path="image_matchmaker/configs/register_config_test_rigid.yaml")
    # deform_test_data(cfg_path="image_matchmaker/configs/register_config_test_elastic.yaml", enable_elastic=True)
