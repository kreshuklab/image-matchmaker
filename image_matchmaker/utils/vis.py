import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import seaborn as sns
import matplotlib.colors as mcolors
import logging
from pathlib import Path
from skimage.color import label2rgb
from skimage.util import map_array
from skimage.measure import regionprops_table

from image_matchmaker.preprocessing import percentile_norm

# Change to 'png' or None (infer from path extension) to switch output format
PLOT_FORMAT = 'pdf'


def _savefig(save_path, dpi=300):
    if PLOT_FORMAT is not None and save_path is not None:
        save_path = Path(save_path).with_suffix(f'.{PLOT_FORMAT}')
    plt.savefig(save_path, dpi=dpi)


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


class _LabelCmap:
    pass


LABEL = _LabelCmap()


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
        if cmap is LABEL:
            display = label2rgb(s, bg_label=0, bg_color=(1, 1, 1))
            plt.imshow(display)
        else:
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
        _savefig(save_path)

    plt.close()


def _matched_instance_lut(fixed_seg, matched_label_df, cmap, gradient_axis):
    """Build the rainbow color lookup shared by the slice and projection plots.

    Returns ``(lut, moving_ids, fixed_of_moving, gradient_axis)`` where ``lut`` is an
    RGB lookup table indexed by fixed_label_id (background/unlabeled -> white), and
    ``moving_ids``/``fixed_of_moving`` map each matched moving label to its fixed
    partner.
    """
    # matched-label mapping (dedupe in case of rare duplicate ids).
    # Copy the arrays because map_array rejects read-only buffers.
    m = matched_label_df.drop_duplicates("moving_label_id")
    moving_ids = np.array(m["moving_label_id"])
    fixed_of_moving = np.array(m["fixed_label_id"])

    # LUT keyed by fixed_label_id -> rainbow color from the instance centroid's
    # position along the chosen axis; 0 (background) and unlabeled ids stay white.
    props = regionprops_table(fixed_seg, properties=("label", "centroid"))
    fixed_labels = props["label"]
    centroids = np.column_stack([props[f"centroid-{a}"] for a in range(3)])
    if gradient_axis is None:
        gradient_axis = int(np.argmax(centroids.max(axis=0) - centroids.min(axis=0)))
    coord = centroids[:, gradient_axis]
    span = coord.max() - coord.min()
    norm = (coord - coord.min()) / span if span > 0 else np.zeros_like(coord)
    colors = plt.get_cmap(cmap)(norm)[:, :3]

    max_fixed = int(max(fixed_seg.max(), fixed_of_moving.max(), fixed_labels.max()))
    lut = np.ones((max_fixed + 1, 3))
    lut[fixed_labels] = colors
    lut[0] = (1.0, 1.0, 1.0)
    return lut, moving_ids, fixed_of_moving, gradient_axis


def plot_matched_instances(
    fixed_seg,
    moving_seg,
    matched_label_df,
    save_path=None,
    x_pos=None,
    y_pos=None,
    z_pos=None,
    unmatched_color=(0.6, 0.6, 0.6),
    cmap="gist_rainbow",
    gradient_axis=None,
):
    """Plot matched instances across the fixed and moving segmentations.

    2x3 grid: top row = three orthogonal slices of the fixed segmentation with every
    instance colored; bottom row = the same slices of the moving segmentation with each
    matched instance colored like its fixed partner (unmatched instances shown grey).
    Uses the same slice layout as ``plot_three_slices``.

    Each fixed instance is colored with a rainbow gradient based on its centroid
    position along one axis, so the color varies smoothly through space. If the
    matching is good, the moving image shows the same smooth gradient; mismatches
    stand out as color discontinuities.

    Args:
        fixed_seg: 3D fixed instance-label volume (ZYX)
        moving_seg: 3D moving instance-label volume (ZYX), same grid as fixed_seg
        matched_label_df: DataFrame with columns ``fixed_label_id`` and ``moving_label_id``
        save_path: path to save figure
        x_pos: x slice index (defaults to center)
        y_pos: y slice index (defaults to center)
        z_pos: z slice index (defaults to center)
        unmatched_color: RGB for moving instances without a match
        cmap: matplotlib colormap name used for the gradient
        gradient_axis: axis (0=z, 1=y, 2=x) along which the gradient runs; defaults
            to the axis with the largest spread of instance centroids (the long axis)
    """
    assert fixed_seg.ndim == 3 and moving_seg.ndim == 3
    if z_pos is None:
        z_pos = int(fixed_seg.shape[0] // 2)
    if y_pos is None:
        y_pos = int(fixed_seg.shape[1] // 2)
    if x_pos is None:
        x_pos = int(fixed_seg.shape[2] // 2)

    lut, moving_ids, fixed_of_moving, gradient_axis = _matched_instance_lut(
        fixed_seg, matched_label_df, cmap, gradient_axis
    )

    def fixed_rgb(s):  # s: 2D fixed label slice
        return lut[s]

    def moving_rgb(s):  # s: 2D moving label slice
        # np.array(s) forces a writable, contiguous copy for map_array
        remapped = map_array(np.array(s), moving_ids, fixed_of_moving)  # unmatched -> 0
        rgb = lut[remapped]
        unmatched_fg = (s > 0) & (remapped == 0)
        rgb[unmatched_fg] = unmatched_color
        return rgb

    fixed_slices = [
        (0, z_pos, fixed_seg[z_pos, :, :]),
        (1, y_pos, fixed_seg[:, y_pos, :]),
        (2, x_pos, fixed_seg[:, :, x_pos]),
    ]
    moving_slices = [
        (0, z_pos, moving_seg[z_pos, :, :]),
        (1, y_pos, moving_seg[:, y_pos, :]),
        (2, x_pos, moving_seg[:, :, x_pos]),
    ]

    plt.figure(figsize=(15, 10))
    for col, (axis, pos, s) in enumerate(fixed_slices, 1):
        plt.subplot(2, 3, col)
        plt.title(f"fixed {'zyx'[axis]} slice at {pos}")
        plt.imshow(fixed_rgb(s))
    for col, (axis, pos, s) in enumerate(moving_slices, 1):
        plt.subplot(2, 3, 3 + col)
        plt.title(f"moving {'zyx'[axis]} slice at {pos}")
        plt.imshow(moving_rgb(s))

    if save_path is None:
        plt.show()
    else:
        _savefig(save_path)

    plt.close()


def plot_matched_instances_projection(
    fixed_seg,
    moving_seg,
    matched_label_df,
    save_path=None,
    unmatched_color=(0.6, 0.6, 0.6),
    cmap="gist_rainbow",
    gradient_axis=None,
):
    """Like ``plot_matched_instances`` but projects *all* instances onto each plane.

    Instead of a single central slice per axis, every instance is collapsed onto the
    plane via a maximum-intensity projection along each axis, so all instances are
    visible at once. 2x3 grid: top row = fixed projections onto the (xy, xz, yz)
    planes; bottom row = the same projections of the moving segmentation, each matched
    instance colored like its fixed partner (unmatched instances shown grey). Colors
    use the same rainbow gradient as ``plot_matched_instances``.

    Args:
        fixed_seg: 3D fixed instance-label volume (ZYX)
        moving_seg: 3D moving instance-label volume (ZYX), same grid as fixed_seg
        matched_label_df: DataFrame with columns ``fixed_label_id`` and ``moving_label_id``
        save_path: path to save figure
        unmatched_color: RGB for moving instances without a match
        cmap: matplotlib colormap name used for the gradient
        gradient_axis: axis (0=z, 1=y, 2=x) along which the gradient runs; defaults
            to the axis with the largest spread of instance centroids (the long axis)
    """
    assert fixed_seg.ndim == 3 and moving_seg.ndim == 3

    lut, moving_ids, fixed_of_moving, gradient_axis = _matched_instance_lut(
        fixed_seg, matched_label_df, cmap, gradient_axis
    )

    # Relabel the whole moving volume to fixed-partner ids *before* projecting, so the
    # fixed and moving max-projections select corresponding instances. (Projecting raw
    # moving ids and remapping afterwards picks the largest moving id per column, an
    # instance unrelated to the fixed winner, which makes the two views inconsistent.)
    mov2fix = np.zeros(int(moving_seg.max()) + 1, dtype=np.uint32)
    mov2fix[moving_ids] = fixed_of_moving
    moving_as_fixed = mov2fix[moving_seg]  # matched -> fixed id, unmatched/bg -> 0
    unmatched = (moving_seg > 0) & (moving_as_fixed == 0)

    # Maximum-intensity projection of the (fixed-id) labels onto each plane.
    planes = ["xy", "xz", "yz"]

    plt.figure(figsize=(15, 10))
    for col, a in enumerate(range(3), 1):
        plt.subplot(2, 3, col)
        plt.title(f"fixed {planes[a]} projection")
        plt.imshow(lut[fixed_seg.max(axis=a)])
    for col, a in enumerate(range(3), 1):
        mproj = moving_as_fixed.max(axis=a)
        rgb = lut[mproj]
        rgb[(mproj == 0) & unmatched.max(axis=a)] = unmatched_color
        plt.subplot(2, 3, 3 + col)
        plt.title(f"moving {planes[a]} projection")
        plt.imshow(rgb)

    if save_path is None:
        plt.show()
    else:
        _savefig(save_path)

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
    Plot an overlay of two 3D images, sliced along each axis.

    ``img1`` is drawn in pink and ``img2`` in cyan.

    Parameters
    ----------
    img1, img2 : numpy.ndarray
        The two 3D volumes to overlay (must have the same number of dimensions).
    save_path : str, optional
        If given, the figure is written here; otherwise it is shown.
    x_pos, y_pos, z_pos : int, optional
        Slice indices per axis (default to the centre of mass).
    gc1, Vt1, gc2, Vt2 : optional
        Optional centre of mass and principal axes to overlay for each image.
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
        _savefig(save_path)

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
    Overlay two point clouds in a 2D projection.

    Optionally restricts the plot to points near the centre-of-mass slice to
    make dense clouds easier to read.

    Parameters
    ----------
    fixed_pcd, moving_pcd : open3d.t.geometry.PointCloud
        Point clouds to overlay (drawn in ``fixed_col`` / ``moving_col``).
    fixed_col, moving_col : optional
        Colours for the fixed and moving clouds (default pink / cyan).
    projection : str, optional
        Two-axis projection plane, e.g. ``"xy"``, ``"yz"`` (default ``"xy"``).
    save_path : str, optional
        If given, the figure is written here; otherwise it is shown.
    title : str, optional
        Plot title.
    center_slice : bool, optional
        If ``True``, only plot points near the centre-of-mass slice.
    max_points : int, optional
        Maximum number of points to plot per cloud (default ``2000``).
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
        _savefig(save_path)
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
    """
    Plot the displacement field between a point cloud and its registered version.

    Draws a line from each moving point to its registered position in a 2D
    projection, so the deformation can be inspected for smoothness.

    Parameters
    ----------
    moving_pcd : open3d.t.geometry.PointCloud
        Point cloud before registration.
    registered_pcd : open3d.t.geometry.PointCloud
        The same points after registration.
    save_path : str, optional
        If given, the figure is written here; otherwise it is shown.
    projection : str, optional
        Two-axis projection plane, e.g. ``"xy"``, ``"yz"`` (default ``"xy"``).
    center_slice : bool, optional
        If ``True``, only plot points near the centre-of-mass slice.
    max_points : int, optional
        Maximum number of points to plot (default ``2000``).
    """
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
        _savefig(save_path)
    plt.close()


def plot_matching_qc(
    fixed_np, moving_np, fig_name, pairs=None, projection="xz", center_slice=True, max_points=500
):
    """
    Plot matched point-cloud correspondences for quality control.

    Scatters the fixed and moving points in a 2D projection and draws a line
    between each matched pair, so incorrect (long, crossing) matches are easy to
    spot.

    Parameters
    ----------
    fixed_np : numpy.ndarray
        ``(N, 3)`` coordinates of the fixed point set.
    moving_np : numpy.ndarray
        ``(M, 3)`` coordinates of the moving point set.
    fig_name : str
        Path where the figure is saved.
    pairs : list of tuple of int, optional
        Matched index pairs ``(i, j)`` into ``fixed_np`` and ``moving_np``.
    projection : str, optional
        Two-axis projection plane, e.g. ``"xz"`` (default ``"xz"``).
    center_slice : bool, optional
        If ``True``, only plot points near the centre-of-mass slice.
    max_points : int, optional
        Maximum number of points to plot (default ``500``).
    """
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

    _savefig(fig_name)
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
