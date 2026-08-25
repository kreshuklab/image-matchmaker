import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import seaborn as sns
import matplotlib.colors as mcolors
import logging
from pathlib import Path
from skimage.color import label2rgb

from image_matchmaker.data_processing.preprocessing import percentile_norm

# Change to 'png' or None (infer from path extension) to switch output format
PLOT_FORMAT = 'pdf'


def _savefig(save_path, dpi=300, bbox_inches=None, fig=None):
    """Save the current figure, rewriting the suffix to PLOT_FORMAT.

    bbox_inches="tight" crops the figure to its content; the landmark plots use it because
    their outermost labels otherwise sit in the margin. None is matplotlib's own default.

    Pass ``fig`` from code that may run in several threads at once: the pyplot "current figure"
    is process-global, so concurrent plotting into it interleaves.
    """
    if PLOT_FORMAT is not None and save_path is not None:
        save_path = Path(save_path).with_suffix(f'.{PLOT_FORMAT}')
    (fig or plt).savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)


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

    fig.tight_layout()
    if save_path is None:
        plt.show()
    else:
        _savefig(save_path, bbox_inches="tight", fig=fig)
    plt.close(fig)


def plot_landmark_overlay(
    fixed_pos,
    moving_pos,
    id_map,
    save_path=None,
    title="",
    fixed_bg=None,
    moving_bg=None,
):
    """Overlay fixed and moving landmarks in three orthogonal projections.

    Each landmark is drawn twice - red at its fixed position, blue at its position after
    the registration stage being inspected - joined by a line, so the residual error is
    visible per landmark.

    Positions are (x, y, z) in µm, as returned by extract_centroids. They are reversed to
    (z, y, x) before being passed to _slice_gc_coords, which is shared with
    plot_landmark_qc so both produce identical panel layouts.

    Args:
        fixed_pos: {label_id: (x, y, z)} of the fixed landmarks, in µm
        moving_pos: {label_id: (x, y, z)} of the moving landmarks at this stage, in µm
        id_map: {landmark_name: label_id}
        save_path: output path; shows interactively if None. Saved through _savefig, so the
            suffix is rewritten to PLOT_FORMAT like the rest of the workflow's plots.
        title: figure title, e.g. "after CPD | mean LRE 22.17 µm"
        fixed_bg, moving_bg: optional (N, 3) instance centroids in µm, drawn as a faint
            cloud for anatomical context
    """
    # _slice_gc_coords takes (z, y, x), positions here are (x, y, z), hence the [::-1].
    # Unpacking a (3, N) array yields three (N,) rows, so whole clouds go through it too.
    pairs = []
    for name, lbl in id_map.items():
        if lbl not in fixed_pos or lbl not in moving_pos:
            logging.warning(f"Landmark {name!r} (id={lbl}) missing, not plotted")
            continue
        pairs.append(
            (name, np.asarray(fixed_pos[lbl])[::-1], np.asarray(moving_pos[lbl])[::-1])
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for axis, (ax, proj) in enumerate(zip(axes, ("xy", "xz", "yz"))):
        ax.set_title(f"{proj} projection")

        for bg, color in ((fixed_bg, PINK_HEX), (moving_bg, CYAN_HEX)):
            if bg is not None:
                col, row = _slice_gc_coords(np.asarray(bg).T[::-1], axis)
                ax.scatter(col, row, c=color, s=1, alpha=0.15, linewidths=0, rasterized=True)

        for name, fixed_zyx, moving_zyx in pairs:
            fx, fy = _slice_gc_coords(fixed_zyx, axis)
            mx, my = _slice_gc_coords(moving_zyx, axis)
            ax.plot([fx, mx], [fy, my], c="grey", linewidth=0.5, zorder=4)
            ax.scatter(fx, fy, c="red", s=15, zorder=5, linewidths=0)
            ax.scatter(mx, my, c="blue", s=15, zorder=5, linewidths=0)
            ax.text(fx + 2, fy, name, fontsize=4, color="red", zorder=6, va="center")

        ax.set_aspect("equal")
        ax.invert_yaxis()  # scatter defaults to origin bottom-left; put it top-left

    fig.suptitle(title)
    fig.tight_layout()
    if save_path is None:
        plt.show()
    else:
        _savefig(save_path, bbox_inches="tight", fig=fig)
    plt.close(fig)


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

        # Each imshow resets the limits to its own extent, so the last one drawn would crop
        # the other volume wherever it is larger. Show the union instead.
        plt.xlim(-0.5, max(s1.shape[1], s2.shape[1]) - 0.5)
        plt.ylim(max(s1.shape[0], s2.shape[0]) - 0.5, -0.5)

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


def _draw_pcd_overlay(
    ax, fixed_np, moving_np, projection, fixed_col, moving_col, center_slice, max_points
):
    """Draw one point-cloud overlay projection into ``ax``.

    The caller adds the legend, so a multi-panel figure can show just one.
    """
    assert (
        len(projection) == 2
    ), f"Projection should be xy, yz or something like that of length 2, not {projection}"

    roi_x, roi_y, fixed_mask, moving_mask, min_range, max_range = plot_projection(
        fixed_np, moving_np, projection, center_slice, max_points
    )

    ax.scatter(
        fixed_np[roi_x][fixed_mask],
        fixed_np[roi_y][fixed_mask],
        s=2,
        c=fixed_col,
        alpha=0.5,
        label="Fixed point cloud",
        rasterized=True,
    )
    ax.scatter(
        moving_np[roi_x][moving_mask],
        moving_np[roi_y][moving_mask],
        s=2,
        c=moving_col,
        alpha=0.5,
        label="Moving point cloud",
        rasterized=True,
    )
    ax.set_xlabel(projection[0])
    ax.set_ylabel(projection[1])
    ax.invert_yaxis()
    # ax.axis, not set_aspect: keeps the datalim-adjusting behaviour the old plt.axis call had
    ax.axis("equal")


def plot_pcd_overlay(
    fixed_pcd: o3d.t.geometry.PointCloud,
    moving_pcd: o3d.t.geometry.PointCloud,
    fixed_col=PINK_HEX,
    moving_col=CYAN_HEX,
    save_path=None,
    projections=("xy", "xz", "yz"),
    title="",
    center_slice=True,
    max_points=2000,
):
    """
    Overlay two point clouds in three orthogonal projections, one panel per projection.

    Same figure layout as :func:`plot_landmark_overlay` and
    :func:`plot_displacement_field`, so one file covers all three views.

    Parameters
    ----------
    fixed_pcd, moving_pcd : open3d.t.geometry.PointCloud
        Point clouds to overlay (drawn in ``fixed_col`` / ``moving_col``).
    fixed_col, moving_col : optional
        Colours for the fixed and moving clouds (default pink / cyan).
    save_path : str, optional
        If given, the figure is written here; otherwise it is shown.
    projections : sequence of str, optional
        Two-axis projection planes, one panel each (default ``("xy", "xz", "yz")``).
    title : str, optional
        Figure title.
    center_slice : bool, optional
        If ``True``, only plot points near the centre-of-mass slice.
    max_points : int, optional
        Maximum number of points to plot per cloud per panel (default ``2000``).
    """
    fixed_np = fixed_pcd.point.positions.numpy()
    moving_np = moving_pcd.point.positions.numpy()

    fig, axes = plt.subplots(1, len(projections), figsize=(6 * len(projections), 6))

    for i, (ax, projection) in enumerate(zip(np.atleast_1d(axes), projections)):
        ax.set_title(f"{projection} projection")
        _draw_pcd_overlay(
            ax, fixed_np, moving_np, projection, fixed_col, moving_col,
            center_slice, max_points,
        )
        if i == 0:
            ax.legend()  # one legend is enough; the panels share their colour coding

    fig.suptitle(title)
    fig.tight_layout()
    if save_path is None:
        plt.show()
    else:
        _savefig(save_path, bbox_inches="tight", fig=fig)
    plt.close(fig)


def _draw_displacement_field(
    ax, moving_np, registered_np, projection, center_slice, max_points
):
    """Draw one displacement-field projection into ``ax``."""
    assert (
        len(projection) == 2
    ), f"Projection should be xy, yz or something like that of length 2, not {projection}"

    roi_x, roi_y, moving_mask, registered_mask, min_range, max_range = plot_projection(
        moving_np, registered_np, projection, center_slice, max_points
    )

    for idx in np.nonzero(moving_mask):
        ax.plot(
            [moving_np[roi_x][idx], registered_np[roi_x][idx]],
            [moving_np[roi_y][idx], registered_np[roi_y][idx]],
            linewidth=0.5,
        )

    # ax.axis, not set_aspect: keeps the datalim-adjusting behaviour the old plt.axis call had
    ax.axis("equal")
    ax.invert_yaxis()


def plot_displacement_field(
    moving_pcd: o3d.t.geometry.PointCloud,
    registered_pcd: o3d.t.geometry.PointCloud,
    save_path=None,
    projections=("xy", "xz", "yz"),
    center_slice=True,
    max_points=2000,
    title="",
):
    """
    Plot the displacement field in three orthogonal projections, one panel per projection.

    Same figure layout as :func:`plot_landmark_overlay`, so a trial's deformation and its
    landmark residuals can be read side by side, and one file covers all three views.

    Parameters
    ----------
    moving_pcd : open3d.t.geometry.PointCloud
        Point cloud before registration.
    registered_pcd : open3d.t.geometry.PointCloud
        The same points after registration.
    save_path : str, optional
        If given, the figure is written here; otherwise it is shown.
    projections : sequence of str, optional
        Two-axis projection planes, one panel each (default ``("xy", "xz", "yz")``).
    center_slice : bool, optional
        If ``True``, only plot points near the centre-of-mass slice.
    max_points : int, optional
        Maximum number of points to plot per panel (default ``2000``).
    title : str, optional
        Figure title, e.g. the trial's parameters and LRE.
    """
    moving_np = moving_pcd.point.positions.numpy()
    registered_np = registered_pcd.point.positions.numpy()

    fig, axes = plt.subplots(1, len(projections), figsize=(6 * len(projections), 6))

    for ax, projection in zip(np.atleast_1d(axes), projections):
        ax.set_title(f"{projection} projection")
        _draw_displacement_field(
            ax, moving_np, registered_np, projection, center_slice, max_points
        )

    fig.suptitle(title)
    fig.tight_layout()
    if save_path is None:
        plt.show()
    else:
        _savefig(save_path, bbox_inches="tight", fig=fig)
    plt.close(fig)


def _draw_matching_qc(
    ax, fixed_np, moving_np, projection, pairs, center_slice, max_points
):
    """Draw one matching-QC projection into ``ax``.

    The caller drops the extra legends, so a multi-panel figure can show just one.
    """
    axis_order = {"x": 0, "y": 1, "z": 2}
    d1 = axis_order[projection[0]]
    d2 = axis_order[projection[1]]

    roi_x, roi_y, fixed_mask, moving_mask, min_range, max_range = plot_projection(
        fixed_np, moving_np, projection, center_slice, max_points
    )

    sns.scatterplot(
        x=fixed_np[fixed_mask, d1],
        y=fixed_np[fixed_mask, d2],
        alpha=0.8,
        label="fixed",
        color=PINK_HEX,
        ax=ax,
    )
    sns.scatterplot(
        x=moving_np[moving_mask, d1],
        y=moving_np[moving_mask, d2],
        alpha=0.8,
        label="moving",
        color=CYAN_HEX,
        ax=ax,
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
                ax.plot([p1[d1], p2[d1]], [p1[d2], p2[d2]], c="lightseagreen", linewidth=0.5)

    # ax.axis, not set_aspect: keeps the datalim-adjusting behaviour the old plt.axis call had
    ax.axis("equal")
    ax.invert_yaxis()


def plot_matching_qc(
    fixed_np,
    moving_np,
    save_path=None,
    pairs=None,
    projections=("xy", "xz", "yz"),
    center_slice=True,
    max_points=500,
    title="",
):
    """
    Plot matched correspondences in three orthogonal projections, one panel per projection.

    Same figure layout as :func:`plot_landmark_overlay`, so one file covers all three views.

    Parameters
    ----------
    fixed_np : numpy.ndarray
        ``(N, 3)`` coordinates of the fixed point set.
    moving_np : numpy.ndarray
        ``(M, 3)`` coordinates of the moving point set.
    save_path : str, optional
        If given, the figure is written here; otherwise it is shown.
    pairs : list of tuple of int, optional
        Matched index pairs ``(i, j)`` into ``fixed_np`` and ``moving_np``.
    projections : sequence of str, optional
        Two-axis projection planes, one panel each (default ``("xy", "xz", "yz")``).
    center_slice : bool, optional
        If ``True``, only plot points near the centre-of-mass slice.
    max_points : int, optional
        Maximum number of points to plot per panel (default ``500``).
    title : str, optional
        Figure title.
    """
    fig, axes = plt.subplots(1, len(projections), figsize=(6 * len(projections), 6))

    for i, (ax, projection) in enumerate(zip(np.atleast_1d(axes), projections)):
        ax.set_title(f"{projection} projection")
        _draw_matching_qc(
            ax, fixed_np, moving_np, projection, pairs, center_slice, max_points
        )
        # sns.scatterplot adds a legend to every axes it draws into, so drop all but the
        # first panel's - the panels share their colour coding.
        if i > 0 and ax.get_legend() is not None:
            ax.get_legend().remove()

    fig.suptitle(title)
    fig.tight_layout()
    if save_path is None:
        plt.show()
    else:
        _savefig(save_path, bbox_inches="tight", fig=fig)
    plt.close(fig)


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
