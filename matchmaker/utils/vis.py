import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import seaborn as sns
import matplotlib.colors as mcolors
import logging

from matchmaker.preprocessing import percentile_norm


PINK_HEX = '#FF3E96'
CYAN_HEX = '#00CED1'


def get_pink_cmap():
    pink_colors = [
        '#FFFFFF',  # White (for 0)
        '#FF9CCA',  # Mid Pink
        '#FF69B4',  # Hot Pink
        '#FF3E96',  # Strong Pink
        '#FF1493',  # Deep Pink
    ]

    # Create the colormap
    custom_pink_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_pink", pink_colors, N=256
    )

    return custom_pink_cmap


def get_cyan_cmap():
    cyan_colors = [
        '#FFFFFF',  # White (for 0)
        '#9FFBFF',  # Soft Cyan
        '#66F2FF',  # Mid Cyan
        '#33EAF7',  # Bright Cyan
        '#00CED1',  # Dark Turquoise / Cyan
    ]

    # Create the colormap
    custom_cyan_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cyan", cyan_colors, N=256
    )
    return custom_cyan_cmap


PINK = get_pink_cmap()
CYAN = get_cyan_cmap()


def _slice_gc_coords(gc, axis):
    cz, cy, cx = gc
    if axis == 0:
        return cx, cy
    if axis == 1:
        return cx, cz
    if axis == 2:
        return cy, cz


def _draw_axes(px, py, Vt, shape, axis):
    if Vt is None:
        return

    L = max(shape) * 0.15
    colors = ["r", "g", "b"]

    for i, v in enumerate(Vt):
        if axis == 0:
            dx, dy = v[2]*L, v[1]*L
        elif axis == 1:
            dx, dy = v[2]*L, v[0]*L
        else:
            dx, dy = v[1]*L, v[0]*L

        x0 = px - dx
        x1 = px + dx
        y0 = py - dy
        y1 = py + dy

        plt.plot([x0, x1], [y0, y1], color=colors[i], linewidth=2)


def plot_three_slices(
    img,
    save_path=None,
    x_pos=None,
    y_pos=None,
    z_pos=None,
    cmap="Greys_r",
    max_pos=False,
    alpha=False,
    gc=None,
    Vt=None,
):
    """
    Plot slices of a 3D image along each axis.

    Args:
        img: 3D image
        save_path: path to save figure
        x_pos: x slice index
        y_pos: y slice index
        z_pos: z slice index
        cmap: colormap
        max_pos: use max voxel position
        alpha: show only foreground
        gc: center of mass (z,y,x)
        Vt: PCA axes (3x3)
    """
    img = img.astype(np.uint16)

    assert img.ndim == 3
    if x_pos is None:
        x_pos = int(img.shape[2] // 2)
    if y_pos is None:
        y_pos = int(img.shape[1] // 2)
    if z_pos is None:
        z_pos = int(img.shape[0] // 2)
    assert img.ndim == 3

    if max_pos:
        z_pos, y_pos, x_pos = np.unravel_index(np.argmax(img), img.shape)
    else:
        if x_pos is None:
            x_pos = img.shape[2] // 2
        if y_pos is None:
            y_pos = img.shape[1] // 2
        if z_pos is None:
            z_pos = img.shape[0] // 2

    alpha = (img > 0).astype(np.float32) if alpha else np.ones_like(img)

    plt.figure(figsize=(15, 5))

    slices = [
        (0, z_pos, img[z_pos, :, :], alpha[z_pos, :, :]),
        (1, y_pos, img[:, y_pos, :], alpha[:, y_pos, :]),
        (2, x_pos, img[:, :, x_pos], alpha[:, :, x_pos])
    ]

    for i, (axis, pos, s, a) in enumerate(slices, 1):
        plt.subplot(1, 3, i)
        plt.title(f"{'zyx'[axis]} slice at {pos}")
        semantic = cmap is PINK or cmap is CYAN
        display = (s > 0).astype(np.float32) if semantic else s
        plt.imshow(display, cmap=cmap, alpha=a, vmin=0 if semantic else None, vmax=1 if semantic else None)

        if gc is not None:
            px, py = _slice_gc_coords(gc, axis)
            plt.scatter(px, py, c="yellow", s=40)
            _draw_axes(px, py, Vt, img.shape, axis)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)

    plt.close()


def plot_landmark_qc(
    seg_with_lm,
    id_map,
    save_path=None,
    cell_cmap=None,
    landmark_color="red",
):
    """Three-slice QC plot: cells in cell_cmap, each landmark centroid as a labeled scatter dot.

    Landmark positions are projected onto the mid-slice of each axis so all landmarks
    are visible regardless of depth, making it easy to confirm placements visually.

    Args:
        seg_with_lm: ZYX integer array with landmark spheres embedded at label IDs from id_map
        id_map: {landmark_name: label_id}
        save_path: output path; shows interactively if None
        cell_cmap: colormap for regular cells (default: PINK)
        landmark_color: scatter / text color for landmarks (default: "red")
    """
    if cell_cmap is None:
        cell_cmap = PINK

    min_lm_id = min(id_map.values())
    cells = (seg_with_lm > 0) & (seg_with_lm < min_lm_id)

    lm_centroids = {}
    for name, lbl in id_map.items():
        voxels = np.argwhere(seg_with_lm == lbl)
        if len(voxels) == 0:
            logging.warning(f"Landmark {name!r} (id={lbl}) not found in segmentation")
            continue
        lm_centroids[name] = voxels.mean(axis=0)  # [z, y, x]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    proj_specs = [
        (0, "xy projection", cells.max(axis=0)),
        (1, "xz projection", cells.max(axis=1)),
        (2, "yz projection", cells.max(axis=2)),
    ]

    for ax, (axis, title, s) in zip(axes, proj_specs):
        ax.set_title(title)
        ax.imshow(s.astype(np.float32), cmap=cell_cmap, vmin=0, vmax=1, alpha=0.5)
        for name, c in lm_centroids.items():
            col, row = _slice_gc_coords(c, axis)
            ax.scatter(col, row, c=landmark_color, s=15, zorder=5, linewidths=0)
            ax.text(col + 2, row, name, fontsize=4, color=landmark_color, zorder=6, va="center")
        ax.invert_yaxis()

    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_overlay(img1, img2, save_path=None, x_pos=None, y_pos=None, z_pos=None,
                 gc1=None, Vt1=None, gc2=None, Vt2=None):
    """
    Plot slices of two 3D images along each axis.
    """
    assert img1.ndim == 3 and img2.ndim == 3

    if x_pos is None:
        x_pos = min(img1.shape[2] // 2, img2.shape[2] // 2)
    if y_pos is None:
        y_pos = min(img1.shape[1] // 2, img2.shape[1] // 2)
    if z_pos is None:
        z_pos = min(img1.shape[0] // 2, img2.shape[0] // 2)

    plt.figure(figsize=(30, 10), dpi=300)

    img1_alpha = (percentile_norm(img1, 0, 100) > 0) * 0.5
    img2_alpha = (percentile_norm(img2, 0, 100) > 0) * 0.5

    slices = [
        (0, z_pos, img1[z_pos, :, :], img2[z_pos, :, :], img1_alpha[z_pos, :, :], img2_alpha[z_pos, :, :]),
        (1, y_pos, img1[:, y_pos, :], img2[:, y_pos, :], img1_alpha[:, y_pos, :], img2_alpha[:, y_pos, :]),
        (2, x_pos, img1[:, :, x_pos], img2[:, :, x_pos], img1_alpha[:, :, x_pos], img2_alpha[:, :, x_pos])
    ]

    for i, (axis, pos, s1, s2, a1, a2) in enumerate(slices, 1):
        plt.subplot(1, 3, i)
        plt.title(f"{'zyx'[axis]} slice at {pos}")

        plt.imshow((s1 > 0).astype(np.float32), cmap=PINK, alpha=a1, vmin=0, vmax=1)
        plt.imshow((s2 > 0).astype(np.float32), cmap=CYAN, alpha=a2, vmin=0, vmax=1)

        if gc1 is not None:
            px, py = _slice_gc_coords(gc1, axis)
            plt.scatter(px, py, c="red", s=40)
            _draw_axes(px, py, Vt1, img1.shape, axis)

        if gc2 is not None:
            px, py = _slice_gc_coords(gc2, axis)
            plt.scatter(px, py, c="blue", s=40)
            _draw_axes(px, py, Vt2, img2.shape, axis)

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
        min_range, max_range = get_pcd_slice(fixed_np[:, orth_axis], max_points)
        logging.info(f"Min max range {min_range} {max_range}")
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
    fixed_col=PINK_HEX,
    moving_col=CYAN_HEX,
    projection="xy",
    save_path=None,
    title="",
    center_slice=True,
    max_points=2000,
):
    """
    Overlay two point clouds. Optionally only plot points around COM slice of the point cloud to make it easier to see/
    """
    assert (
        len(projection) == 2
    ), f"Projection should be xy, yz or something like that of length 2, not {projection}"

    fixed_np = fixed_pcd.point.positions.numpy()
    moving_np = moving_pcd.point.positions.numpy()

    roi_x, roi_y, fixed_mask, moving_mask, min_range, max_range = plot_projection(
        fixed_np, moving_np, projection, center_slice, max_points
    )

    plt.figure(figsize=(10, 10))
    plt.cla()
    plt.axis("equal")

    plt.title(title)
    plt.scatter(
        fixed_np[roi_x][fixed_mask],
        fixed_np[roi_y][fixed_mask],
        s=2,
        c=fixed_col,
        alpha=0.5,
        label="Fixed point cloud",
    )
    plt.scatter(
        moving_np[roi_x][moving_mask],
        moving_np[roi_y][moving_mask],
        s=2,
        c=moving_col,
        alpha=0.5,
        label="Moving point cloud",
    )
    plt.legend()
    plt.xlabel(projection[0])
    plt.ylabel(projection[1])
    plt.gca().invert_yaxis()
    plt.axis("equal")
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
    max_points=2000,
    title=None,
):
    assert (
        len(projection) == 2
    ), f"Projection should be xy, yz or something like that of length 2, not {projection}"

    moving_np = moving_pcd.point.positions.numpy()
    registered_np = registered_pcd.point.positions.numpy()

    roi_x, roi_y, moving_mask, registered_mask, min_range, max_range = plot_projection(
        moving_np, registered_np, projection, center_slice, max_points
    )

    for idx in np.nonzero(moving_mask):
        plt.plot(
            [moving_np[roi_x][idx], registered_np[roi_x][idx]],
            [moving_np[roi_y][idx], registered_np[roi_y][idx]],
            linewidth=0.5,
        )

    plt.axis("equal")
    plt.gca().invert_yaxis()
    if title is not None:
        plt.title(title, fontsize=8)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_matching_qc(
    fixed_np, moving_np, fig_name, pairs=None, projection="xz", center_slice=True, max_points=500
):

    axis_order = {"x": 0, "y": 1, "z": 2}
    d1 = axis_order[projection[0]]
    d2 = axis_order[projection[1]]
    plt.figure()

    roi_x, roi_y, fixed_mask, moving_mask, min_range, max_range = plot_projection(
        fixed_np, moving_np, projection, center_slice, max_points
    )

    sns.scatterplot(
        x=fixed_np[fixed_mask, d1],
        y=fixed_np[fixed_mask, d2],
        alpha=0.8,
        label="fixed",
        color=PINK_HEX,
    )
    sns.scatterplot(
        x=moving_np[moving_mask, d1],
        y=moving_np[moving_mask, d2],
        alpha=0.8,
        label="moving",
        color=CYAN_HEX,
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
                plt.plot([p1[d1], p2[d1]], [p1[d2], p2[d2]], c="lightseagreen", linewidth=0.5)

    plt.legend()
    plt.axis("equal")
    plt.gca().invert_yaxis()

    plt.savefig(fig_name, dpi=300)
    plt.close()


def transform_axes_vis(Vt, T):
    """
    Transform PCA axes into the target space for visualization.
    Extracts the pure rotation from the affine transform T (removing scaling)
    and applies it to the PCA axes Vt. The resulting axes are normalized.
    """
    U, _, Vt_svd = np.linalg.svd(T[:3, :3])
    R = U @ Vt_svd
    Vt_vis = Vt @ R
    return Vt_vis / np.linalg.norm(Vt_vis, axis=1, keepdims=True)
