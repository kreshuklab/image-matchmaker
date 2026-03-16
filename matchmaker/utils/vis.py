import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import logging
import matplotlib.colors as mcolors

from matchmaker.preprocessing import percentile_norm

import seaborn as sns


def get_pink_cmap():
    pink_colors = [
        '#FFFFFF',  # White (for 0)
        '#FFD1DC',  # Light Pink
        '#FFB6C1',  # Light Pink
        '#FF69B4',  # Hot Pink
        '#FF1493',  # Deep Pink
        '#C71585',  # Medium Violet Red
        '#9B30FF',  # Very Deep Pink / Purple-Pink (optional)
    ]

    # Create the colormap
    custom_pink_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_pink", pink_colors, N=256
    )

    return custom_pink_cmap


def get_cyan_cmap():
    cyan_colors = [
        '#FFFFFF',  # White (for 0)
        '#E0FFFF',  # Light Cyan
        '#B0E0E6',  # Powder Blue
        '#87CEFA',  # Sky Blue
        '#00CED1',  # Dark Turquoise
        '#008B8B',  # Dark Cyan
        '#006666',  # Deep Cyan (almost teal)
    ]

    # Create the colormap
    custom_cyan_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cyan", cyan_colors, N=256
    )
    return custom_cyan_cmap

PINK = get_pink_cmap()
CYAN = get_cyan_cmap()


def plot_three_slices(
    img,
    save_path=None,
    x_pos=None,
    y_pos=None,
    z_pos=None,
    cmap="Greys_r",
    max_pos=False,
    alpha=False,
):
    """
    Plot slices of a 3D image along each axis.

    Args:
        img: _description_
        save_path: _description_
        x_pos: _description_. Defaults to None.
        y_pos: _description_. Defaults to None.
        z_pos: _description_. Defaults to None.
        cmap: _description_. Defaults to "Greys".
        max_pos: _description_. Defaults to False.
    """
    img = img.astype(np.uint16)

    assert img.ndim == 3
    if x_pos is None:
        x_pos = int(img.shape[2] // 2)
    if y_pos is None:
        y_pos = int(img.shape[1] // 2)
    if z_pos is None:
        z_pos = int(img.shape[0] // 2)

    if max_pos:
        z_pos, y_pos, x_pos = np.unravel_index(np.argmax(img), img.shape)

    if alpha:
        alpha = (img > 0).astype(np.float32)
    else:
        alpha = np.ones_like(img)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title(f"z slice at {z_pos}")
    plt.imshow(img[z_pos, :, :], cmap=cmap, alpha=alpha[z_pos, :, :])
    plt.subplot(1, 3, 2)
    plt.title(f"y slice at {y_pos}")
    plt.imshow(img[:, y_pos, :], cmap=cmap, alpha=alpha[:, y_pos, :])
    plt.subplot(1, 3, 3)
    plt.title(f"x slice at {x_pos}")
    plt.imshow(img[:, :, x_pos], cmap=cmap, alpha=alpha[:, :, x_pos])

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_overlay(img1, img2, save_path=None, x_pos=None, y_pos=None, z_pos=None):
    """
    Plot slices of two 3D images along each axis.

    Args:
        img1: (fixed, pink) _description_target_shape = (25, 22, 29)
        img2: (moving, cyan) _description_target_shape = (25, 22, 29)
        save_path: _description_. Defaults to None.
        x_pos: _description_. Defaults to None.
        y_pos: _description_. Defaults to None.
        z_pos: _description_. Defaults to None.
    """
    assert img1.ndim == 3
    assert img2.ndim == 3

    if x_pos is None:
        x_pos = min(int(img1.shape[2] // 2), int(img2.shape[2] // 2))
    if y_pos is None:
        y_pos = min(int(img1.shape[1] // 2), int(img2.shape[1] // 2))
    if z_pos is None:
        z_pos = min(int(img1.shape[0] // 2), int(img2.shape[0] // 2))

    plt.figure(figsize=(30, 10), dpi=300)
    plt.subplot(1, 3, 1)
    plt.title(f"z slice at {z_pos}")
    img1_alpha = (percentile_norm(img1, 0, 100) > 0) * 0.5
    img2_alpha = (percentile_norm(img2, 0, 100) > 0) * 0.5
    plt.imshow(img1[z_pos, :, :], cmap=PINK, alpha=img1_alpha[z_pos, :, :])
    plt.imshow(img2[z_pos, :, :], cmap=CYAN, alpha=img2_alpha[z_pos, :, :])

    plt.subplot(1, 3, 2)
    plt.title(f"y slice at {y_pos}")
    plt.imshow(img1[:, y_pos, :], cmap=PINK, alpha=img1_alpha[:, y_pos, :])
    plt.imshow(img2[:, y_pos, :], cmap=CYAN, alpha=img2_alpha[:, y_pos, :])

    plt.subplot(1, 3, 3)
    plt.title(f"x slice at {x_pos}")
    plt.imshow(img1[:, :, x_pos], cmap=PINK, alpha=img1_alpha[:, :, x_pos])
    plt.imshow(img2[:, :, x_pos], cmap=CYAN, alpha=img2_alpha[:, :, x_pos])

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
    plt.close()


def get_pcd_slice(pcd_np, max_points):
    com = np.mean(pcd_np)
    point_ratio = min(1, max_points / len(pcd_np))
    centered_coord = np.abs(pcd_np - com)
    slice_thickness = np.quantile(centered_coord, point_ratio)
    return com - slice_thickness, com + slice_thickness


def plot_projection(fixed_np, moving_np, projection, center_slice, max_points):
    axis_order = {"x": 0, "y": 1, "z": 2}
    roi_x = np.s_[:, axis_order[projection[0]]]
    roi_y = np.s_[:, axis_order[projection[1]]]
    if "z" not in projection:
        orth_axis = axis_order["z"]
    elif "x" not in projection:
        orth_axis = axis_order["x"]
    else:
        orth_axis = axis_order["y"]

    if center_slice:
        min_range, max_range = get_pcd_slice(fixed_np, max_points)
        fixed_mask = (fixed_np[:, orth_axis] > min_range) & (
            fixed_np[:, orth_axis] < max_range
        )
        moving_mask = (moving_np[:, orth_axis] > min_range) & (
            moving_np[:, orth_axis] < max_range
        )

    else:
        fixed_mask = np.ones(len(fixed_np))
        moving_mask = np.ones(len(moving_np))

    return roi_x, roi_y, fixed_mask, moving_mask, min_range, max_range
    

def overlay_pcds(
    fixed_pcd: o3d.t.geometry.PointCloud,
    moving_pcd: o3d.t.geometry.PointCloud,
    fixed_col="cornflowerblue",
    moving_col="orangered",
    projection="xy",
    save_path=None,
    title="",
    center_slice=True,
    max_points=1000,
):
    """
    Overlay two point clouds. Optionally only plot points around COM slice of the point cloud to make it easier to see/
    """
    assert (
        len(projection) == 2
    ), f"Projection should be xy, yz or something like that of length 2, not {projection}"
    

    fixed_np = fixed_pcd.point.positions.numpy()
    moving_np = moving_pcd.point.positions.numpy()

    roi_x, roi_y, fixed_mask, moving_mask, min_range, max_range = plot_projection(fixed_np, moving_np, projection, center_slice, max_points)

    plt.figure(figsize=(10, 10))
    plt.cla()
    plt.axis("equal")

    plt.title(title)
    plt.scatter(
        fixed_np[roi_x][fixed_mask],
        fixed_np[roi_y][fixed_mask],
        s=0.6,
        c=fixed_col,
        alpha=0.5,
        label="Fixed point cloud",
    )
    plt.scatter(
        moving_np[roi_x][moving_mask],
        moving_np[roi_y][moving_mask],
        s=0.6,
        c=moving_col,
        alpha=0.5,
        label="Moving point cloud",
    )
    plt.legend()
    plt.xlabel(projection[0])
    plt.ylabel(projection[1])
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
    plt.close()


def visualize_displacement_field(
    moving_pcd: o3d.t.geometry.PointCloud,
    registered_pcd: o3d.t.geometry.PointCloud,
    save_path=None,
    projection="xy",
    center_slice=True,
    max_points=1000,
):
    assert (
        len(projection) == 2
    ), f"Projection should be xy, yz or something like that of length 2, not {projection}"

    moving_np = moving_pcd.point.positions.numpy()
    registered_np = registered_pcd.point.positions.numpy()

    roi_x, roi_y, moving_mask, registered_mask, min_range, max_range = plot_projection(moving_np, registered_np, projection, center_slice, max_points)

    for idx in np.nonzero(moving_mask):

        plt.plot(
            [moving_np[roi_x][idx], registered_np[roi_x][idx]],
            [moving_np[roi_y][idx], registered_np[roi_y][idx]],
        )

    plt.axis("equal")
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_matching_qc(
    fixed_np, moving_np, fig_name, pairs=None, projection="xz", center_slice=True, max_points=1000
):
    
    axis_order = {"x": 0, "y": 1, "z": 2}
    d1 = axis_order[projection[0]]
    d2 = axis_order[projection[1]]
    plt.figure()

    roi_x, roi_y, fixed_mask, moving_mask, min_range, max_range = plot_projection(fixed_np, moving_np, projection, center_slice, max_points)

    sns.scatterplot(
        x=fixed_np[fixed_mask, d1],
        y=fixed_np[fixed_mask, d2],
        alpha=0.8,
        label="fixed",
        c="mediumpurple",
    )
    sns.scatterplot(
        x=moving_np[moving_mask, d1],
        y=moving_np[moving_mask, d2],
        alpha=0.8,
        label="moving",
        c="lightseagreen",
    )

    if "z" not in projection:
        orth_axis = axis_order["z"]
    elif "x" not in projection:
        orth_axis = axis_order["x"]
    else:
        orth_axis = axis_order["y"]

    if pairs is not None:
        for idx_1, idx_2 in pairs:
            p1 = fixed_np[idx_1, :]
            p2 = moving_np[idx_2, :]
            if (
                (p1[orth_axis] > min_range)
                & (p1[orth_axis] < max_range)
                & (p2[orth_axis] > min_range)
                & (p2[orth_axis] < max_range)
            ):
                plt.plot([p1[d1], p2[d1]], [p1[d2], p2[d2]], c="green")

    plt.legend()

    plt.savefig(fig_name, dpi=300)
    plt.close()


