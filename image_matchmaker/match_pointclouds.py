import os
import sys
import click
import logging
from pathlib import Path
import pandas as pd

import open3d as o3d

from image_matchmaker.utils import (
    sparse_ilp_matching,
    hungarian_matching,
    sinkhorn_matching,
    write_index_pairs,
    plot_matching_qc_panels,
    setup_logging,
)


def run_matching(
    fixed_pcd,
    registered_pcd,
    output_dir,
    method="hungarian",
    max_dist=30,
    min_neighbours=None,
    tau=1.0,
    sinkhorn_max_iter=500,
):
    """
    Establish instance correspondences between two point clouds.

    Matches the fixed and (CPD-)registered moving point clouds using the selected
    ``method`` and writes ``point_matching_*`` QC plots to ``output_dir``.

    Parameters
    ----------
    fixed_pcd : open3d.t.geometry.PointCloud
        Fixed point cloud (with a ``label`` attribute).
    registered_pcd : open3d.t.geometry.PointCloud
        Registered moving point cloud (with a ``label`` attribute).
    output_dir : pathlib.Path
        Directory where the ``point_matching_*`` plots are written.
    method : {"hungarian", "ilp", "sinkhorn"}, optional
        Matching algorithm (default ``"hungarian"``):

        - ``"hungarian"`` — optimal one-to-one assignment over the full cost
          matrix (uses ``max_dist``).
        - ``"ilp"`` — sparse integer linear program over each point's nearest
          neighbours (uses ``max_dist`` and ``min_neighbours``).
        - ``"sinkhorn"`` — entropy-regularized soft assignment, discretized to a
          one-to-one matching (uses ``max_dist``, ``tau`` and
          ``sinkhorn_max_iter``).
    max_dist : float, optional
        Maximum distance between candidate neighbours. Used by all methods.
        Default ``30``.
    min_neighbours : int, optional
        Minimum number of neighbours considered per point. Only used by (and
        required for) ``method="ilp"``; ignored by the other methods.
    tau : float, optional
        Sinkhorn entropy-regularization parameter (``sinkhorn`` only).
        Default ``1.0``.
    sinkhorn_max_iter : int, optional
        Maximum number of Sinkhorn iterations (``sinkhorn`` only). Default ``500``.

    Returns
    -------
    matched_idx_pairs : list of tuple of int
        Matched index pairs into the two point clouds.
    matched_label_pairs : list of tuple
        The corresponding ``(fixed_label, moving_label)`` id pairs.

    Raises
    ------
    ValueError
        If ``method="ilp"`` and ``min_neighbours`` is not provided, or if
        ``method`` is not one of the supported values.
    """
    logging.info(f"Number of points in fixed pcd: {len(fixed_pcd.point.positions)}")
    logging.info(
        f"Number of points in moving pcd: {len(registered_pcd.point.positions)}"
    )

    # Order matters for adding constraints in the optimization problem
    if len(fixed_pcd.point.positions) <= len(registered_pcd.point.positions):
        pos_1 = fixed_pcd.point.positions.numpy()
        pos_2 = registered_pcd.point.positions.numpy()
        swap_order = False
    else:
        pos_1 = registered_pcd.point.positions.numpy()
        pos_2 = fixed_pcd.point.positions.numpy()
        swap_order = True

    logging.info(f"Matching method: {method}")
    if method == "ilp":
        if min_neighbours is None:
            raise ValueError("min_neighbours is required when method='ilp'")
        matched_idx_pairs = sparse_ilp_matching(
            pos_1, pos_2, max_dist=max_dist, min_neighbours=min_neighbours
        )
    elif method == "hungarian":
        matched_idx_pairs = hungarian_matching(pos_1, pos_2, max_dist=max_dist)
    elif method == "sinkhorn":
        matched_idx_pairs = sinkhorn_matching(
            pos_1, pos_2, max_dist=max_dist, tau=tau, max_iter=sinkhorn_max_iter
        )
    else:
        raise ValueError(f"Unknown matching method: {method}")

    if swap_order:
        matched_idx_pairs = [(p2, p1) for p1, p2 in matched_idx_pairs]

    matched_label_pairs = [
        (
            fixed_pcd.point.label.numpy()[idx1, 0],
            registered_pcd.point.label.numpy()[idx2, 0],
        )
        for idx1, idx2 in matched_idx_pairs
    ]

    # matched_idx_pairs was swapped back above, so it indexes these two in this order
    qc_fixed, qc_moving = (pos_2, pos_1) if swap_order else (pos_1, pos_2)
    plot_matching_qc_panels(
        qc_fixed,
        qc_moving,
        output_dir / "plots/point_matching.pdf",
        pairs=matched_idx_pairs,
    )

    return matched_idx_pairs, matched_label_pairs


@click.command()
@click.option("-fcd", "--fixed_pcd", required=True, help="Fixed point cloud")
@click.option(
    "-mpcd", "--moving_pcd", required=True, help="Registered moving point cloud"
)
@click.option("-o", "--output_dir", required=True, help="Output directory")
@click.option(
    "--method",
    type=click.Choice(["ilp", "hungarian", "sinkhorn"]),
    default="hungarian",
    show_default=True,
    help="Matching algorithm to use",
)
@click.option(
    "-max_d",
    "--max_dist",
    required=False,
    default=30,
    show_default=True,
    type=float,
    help="Maximum distance between neighbours to consider for matching",
)
@click.option(
    "-min_knn",
    "--min_neighbours",
    required=False,
    default=None,
    type=int,
    help="Minimum number of neighbours to consider for matching (required for --method ilp)",
)
@click.option(
    "--tau",
    type=float,
    default=1.0,
    show_default=True,
    help="Sinkhorn entropy regularization parameter (sinkhorn only)",
)
@click.option(
    "--sinkhorn_max_iter",
    type=int,
    default=500,
    show_default=True,
    help="Maximum number of Sinkhorn iterations (sinkhorn only)",
)
def main(
    fixed_pcd,
    moving_pcd,
    output_dir,
    min_neighbours,
    max_dist,
    method,
    tau,
    sinkhorn_max_iter,
):

    setup_logging(output_dir, "match_pointclouds.log")

    output_dir = Path(output_dir)
    os.makedirs(output_dir / "plots", exist_ok=True)

    logging.info("Reading fixed point cloud")
    fixed_pcd = o3d.t.io.read_point_cloud(fixed_pcd)

    logging.info("Reading moving point cloud")
    moving_pcd = o3d.t.io.read_point_cloud(moving_pcd)

    matched_idx_pairs, matched_label_pairs = run_matching(
        fixed_pcd,
        moving_pcd,
        output_dir,
        method=method,
        max_dist=max_dist,
        min_neighbours=min_neighbours,
        tau=tau,
        sinkhorn_max_iter=sinkhorn_max_iter,
    )

    write_index_pairs(matched_idx_pairs, str(output_dir / "matched_idx_pairs.txt"))

    # Convert label paits to csv
    fixed_pcd_labels = [
        fixed_label for fixed_label, moving_label in matched_label_pairs
    ]
    moving_pcd_labels = [
        moving_label for fixed_label, moving_label in matched_label_pairs
    ]
    matched_label_df = pd.DataFrame(
        {"fixed_label_id": fixed_pcd_labels, "moving_label_id": moving_pcd_labels}
    )
    matched_label_df.to_csv(output_dir / "matched_labels.csv", index=False)


if __name__ == "__main__":
    main()
