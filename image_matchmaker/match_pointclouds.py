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
    plot_matching_qc,
)


def match_points(
    fixed_pcd,
    registered_pcd,
    output_dir,
    max_dist,
    min_neighbours,
    method="ilp",
    tau=1.0,
    sinkhorn_max_iter=500,
):

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

    if swap_order:
        plot_matching_qc(pos_2, pos_1, output_dir / "plots/point_matching_xz.png", pairs=matched_idx_pairs, projection="xz")
        plot_matching_qc(pos_2, pos_1, output_dir / "plots/point_matching_yz.png", pairs=matched_idx_pairs, projection="yz")
        plot_matching_qc(pos_2, pos_1, output_dir / "plots/point_matching_xy.png", pairs=matched_idx_pairs, projection="xy")

    else:
        plot_matching_qc(pos_1, pos_2, output_dir / "plots/point_matching_xz.png", pairs=matched_idx_pairs, projection="xz")
        plot_matching_qc(pos_1, pos_2, output_dir / "plots/point_matching_yz.png", pairs=matched_idx_pairs, projection="yz")
        plot_matching_qc(pos_1, pos_2, output_dir / "plots/point_matching_xy.png", pairs=matched_idx_pairs, projection="xy")

    return matched_idx_pairs, matched_label_pairs


@click.command()
@click.option("-fcd", "--fixed_pcd", required=True, help="Fixed point cloud")
@click.option(
    "-mpcd", "--moving_pcd", required=True, help="Registered moving point cloud"
)
@click.option("-o", "--output_dir", required=True, help="Output directory")
@click.option(
    "-min_knn",
    "--min_neighbours",
    required=True,
    type=int,
    help="Minimum number of neighbours to consider for matching",
)
@click.option(
    "-max_d",
    "--max_dist",
    required=True,
    type=float,
    help="Maximum distance between neighbours to consider for matching",
)
@click.option(
    "--method",
    type=click.Choice(["ilp", "hungarian", "sinkhorn"]),
    default="ilp",
    show_default=True,
    help="Matching algorithm to use",
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{output_dir}/match_pointclouds.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_dir = Path(output_dir)
    os.makedirs(output_dir / "plots", exist_ok=True)

    logging.info("Reading fixed point cloud")
    fixed_pcd = o3d.t.io.read_point_cloud(fixed_pcd)

    logging.info("Reading moving point cloud")
    moving_pcd = o3d.t.io.read_point_cloud(moving_pcd)

    matched_idx_pairs, matched_label_pairs = match_points(
        fixed_pcd,
        moving_pcd,
        output_dir,
        max_dist,
        min_neighbours,
        method=method,
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
