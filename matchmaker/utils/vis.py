import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import seaborn as sns
import logging
import matplotlib.colors as mcolors

from matchmaker.preprocessing import percentile_norm


def _slice_gc_coords(gc, axis):
    cz, cy, cx = gc
    if axis==0: return cx, cy
    if axis==1: return cx, cz
    if axis==2: return cy, cz


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


def plot_three_slices(img, save_path=None, x_pos=None, y_pos=None, z_pos=None,
                      cmap="Greys_r", max_pos=False, alpha=False, gc=None, Vt=None):
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
    assert img.ndim==3

    if max_pos:
        z_pos, y_pos, x_pos = np.unravel_index(np.argmax(img), img.shape)
    else:
        if x_pos is None:
            x_pos=img.shape[2]//2
        if y_pos is None:
            y_pos=img.shape[1]//2
        if z_pos is None:
            z_pos=img.shape[0]//2

    alpha=(img>0).astype(np.float32) if alpha else np.ones_like(img)

    plt.figure(figsize=(15,5))

    slices=[
        (0, z_pos, img[z_pos,:,:], alpha[z_pos,:,:]),
        (1, y_pos, img[:,y_pos,:], alpha[:,y_pos,:]),
        (2, x_pos, img[:,:,x_pos], alpha[:,:,x_pos])
    ]

    for i, (axis, pos, s, a) in enumerate(slices, 1):
        plt.subplot(1,3,i)
        plt.title(f"{'zyx'[axis]} slice at {pos}")
        plt.imshow(s, cmap=cmap, alpha=a)

        if gc is not None:
            px, py = _slice_gc_coords(gc, axis)
            plt.scatter(px, py, c="yellow", s=40)
            _draw_axes(px, py, Vt, img.shape, axis)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)

    plt.close()


def plot_overlay(img1, img2, save_path=None, x_pos=None, y_pos=None, z_pos=None,
                 gc1=None, Vt1=None, gc2=None, Vt2=None):
    """
    Plot slices of two 3D images along each axis.
    """
    assert img1.ndim==3 and img2.ndim==3

    if x_pos is None: x_pos=min(img1.shape[2]//2, img2.shape[2]//2)
    if y_pos is None: y_pos=min(img1.shape[1]//2, img2.shape[1]//2)
    if z_pos is None: z_pos=min(img1.shape[0]//2, img2.shape[0]//2)

    plt.figure(figsize=(30,10), dpi=300)

    img1_alpha = (percentile_norm(img1,0,100) > 0) * 0.5
    img2_alpha = (percentile_norm(img2,0,100) > 0) * 0.5

    slices = [
        (0,z_pos,img1[z_pos,:,:],img2[z_pos,:,:],img1_alpha[z_pos,:,:],img2_alpha[z_pos,:,:]),
        (1,y_pos,img1[:,y_pos,:],img2[:,y_pos,:],img1_alpha[:,y_pos,:],img2_alpha[:,y_pos,:]),
        (2,x_pos,img1[:,:,x_pos],img2[:,:,x_pos],img1_alpha[:,:,x_pos],img2_alpha[:,:,x_pos])
    ]

    for i, (axis, pos, s1, s2, a1, a2) in enumerate(slices, 1):
        plt.subplot(1,3,i)
        plt.title(f"{'zyx'[axis]} slice at {pos}")

        plt.imshow(s1, cmap=PINK, alpha=a1)
        plt.imshow(s2, cmap=CYAN, alpha=a2)

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


