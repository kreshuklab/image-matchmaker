import numpy as np
import open3d as o3d
import copy
import time
from probreg import cpd
import logging
from probreg.transformation import Transformation
from skimage.measure import regionprops_table
import pandas as pd

use_cuda = False
if use_cuda:
    import cupy as cp

    to_cpu = cp.asnumpy
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    asnumpy = cp.asnumpy
else:
    cp = np

    def to_cpu(x):
        return x

    def asnumpy(x):
        return x


class PrintIterationsCallback(object):
    """Print iteration number.

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        save (bool, optional): If this flag is True,
            each iteration image is saved in a sequential number.
    """

    def __init__(self):
        self._cnt = 0

    def __call__(self, transformation: Transformation) -> None:
        logging.info(f"Iteration {self._cnt}")
        self._cnt += 1


def create_matched_pcds(
    fixed_img_np, fixed_resolution, moving_img_np, moving_resolution, matched_label_df
):

    fixed_labels, fixed_center_coords = extract_centroids(
        fixed_img_np, fixed_resolution
    )
    moving_labels, moving_center_coords = extract_centroids(
        moving_img_np, moving_resolution
    )

    fixed_df = pd.DataFrame(
        data=fixed_center_coords, index=fixed_labels, columns=["x", "y", "z"]
    )
    fixed_df = fixed_df.loc[matched_label_df["fixed_label_id"], :]

    moving_df = pd.DataFrame(
        data=moving_center_coords, index=moving_labels, columns=["x", "y", "z"]
    )
    moving_df = moving_df.loc[matched_label_df["moving_label_id"], :]

    fixed_pcd = create_pcd(np.array(fixed_df), np.array(fixed_df.index))
    moving_pcd = create_pcd(np.array(moving_df), np.array(moving_df.index))
    return fixed_pcd, moving_pcd, fixed_df, moving_df


def pcd_to_elastix(pcd_path, elastix_path):
    """_summary_

    Args:
        pcd_path: Point cloud in PCD format
        elastix_path: Point cloud in Elastix format, for example:
            point
            5
            2214.0 282.2 0.0
            2445.0 2013.0 0.0
            795.0 366.0 0.0
            153.0 609.0
            324.0 2322.0
    """

    pcd = o3d.t.io.read_point_cloud(pcd_path)
    points = pcd.point.positions.numpy()
    with open(elastix_path, "w") as f:
        f.write("point\n")
        f.write(f"{len(points)}\n")
        for x, y, z in points:
            f.write(f"{z} {x} {y}\n")


def extract_centroids(segm, resolution):
    """
    Extract per-instance centroids from an instance segmentation.

    Parameters
    ----------
    segm : numpy.ndarray
        3D instance segmentation (one label per object), in ZYX order.
    resolution : sequence of float
        Voxel spacing ``(z, y, x)``; centroids are scaled into physical units.

    Returns
    -------
    labels : numpy.ndarray
        Instance label ids.
    center_coords : numpy.ndarray
        ``(N, 3)`` centroid coordinates in ``(x, y, z)`` order.
    """
    coords_df = pd.DataFrame(regionprops_table(segm, properties=("label", "centroid")))
    center_coords = np.array(
        [
            coords_df["centroid-2"] * resolution[2],
            coords_df["centroid-1"] * resolution[1],
            coords_df["centroid-0"] * resolution[0],
        ]
    ).T.astype(np.float64)
    labels = np.array(coords_df["label"])
    return labels, center_coords


def create_pcd(center_coords, labels):
    """
    Build an Open3D point cloud from centroid coordinates and labels.

    Parameters
    ----------
    center_coords : numpy.ndarray
        ``(N, 3)`` point coordinates.
    labels : numpy.ndarray
        ``(N,)`` instance label ids, stored as a per-point attribute.

    Returns
    -------
    open3d.t.geometry.PointCloud
        Point cloud with ``positions`` and a ``label`` attribute.
    """
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3d.core.Tensor(center_coords)
    pcd.point.label = o3d.core.Tensor(labels[:, None])
    return pcd


def cpd_from_pcds(fixed_pcd, moving_pcd, w, beta, lmd, maxiter):
    """
    Run non-rigid Coherent Point Drift (CPD) to register two point clouds.

    Parameters
    ----------
    fixed_pcd : open3d.t.geometry.PointCloud
        Target (fixed) point cloud.
    moving_pcd : open3d.t.geometry.PointCloud
        Source (moving) point cloud to be deformed onto ``fixed_pcd``.
    w : float
        Outlier weight (fraction of points assumed to be noise).
    beta : float
        Width of the Gaussian smoothing kernel.
    lmd : float
        Regularization weight (trade-off between fit and smoothness).
    maxiter : int
        Maximum number of EM iterations.

    Returns
    -------
    open3d.t.geometry.PointCloud
        The registered (deformed) moving point cloud.
    """
    # source_pt = asnumpy(moving_pcd.point.positions.numpy())
    # target_pt = asnumpy(fixed_pcd.point.positions.numpy())

    source_pt = cp.asarray(moving_pcd.point.positions.numpy(), dtype=cp.float32)
    target_pt = cp.asarray(fixed_pcd.point.positions.numpy(), dtype=cp.float32)

    cbs = [PrintIterationsCallback()]
    start = time.time()

    tf_param, _, _ = cpd.registration_cpd(
        source_pt,
        target_pt,
        use_cuda=use_cuda,
        callbacks=cbs,
        maxiter=maxiter,
        tf_type_name="nonrigid",
        w=w,
        beta=float(beta),
        lmd=lmd,
    )
    elapsed = time.time() - start
    logging.info(f"time: {elapsed}")

    # print("result: ", to_cpu(tf_param.w), to_cpu(tf_param.g))

    result = to_cpu(tf_param.transform(source_pt))
    registered_pcd = copy.deepcopy(moving_pcd)
    registered_pcd.point.positions = o3d.core.Tensor(result)

    return registered_pcd
