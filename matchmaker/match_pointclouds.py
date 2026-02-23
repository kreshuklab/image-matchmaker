import os
import sys
import itk
import click
import logging
import numpy as np
from pathlib import Path
import pandas as pd

import open3d as o3d

from matchmaker.utils import (
    sparse_ilp_matching,
    write_index_pairs,
    plot_matching_qc
)


from matchmaker.utils import overlay_pcds, visualize_displacement_field, extract_centroids, run_cpd, create_pcd



def match_points(fixed_pcd, registered_pcd, output_dir):

    logging.info(f"Number of points in fixed pcd: {len(fixed_pcd.point.positions)}")
    logging.info(f"Number of points in moving pcd: {len(registered_pcd.point.positions)}")

    # Order matters for adding constraints in the optimization problem
    if len(fixed_pcd.point.positions) <= len(registered_pcd.point.positions):
        pos_1 = fixed_pcd.point.positions.numpy()
        pos_2 = registered_pcd.point.positions.numpy()
        swap_order = False
    else:
        pos_1 = registered_pcd.point.positions.numpy()
        pos_2 = fixed_pcd.point.positions.numpy()
        swap_order = True

    matched_pairs = sparse_ilp_matching(pos_1, pos_2, max_dist=8, min_neighbours=30)

    if swap_order:
        matched_pairs = [(p2, p1) for p1, p2 in matched_pairs]


    if swap_order:
        plot_matching_qc(pos_2, pos_1, output_dir / "point_matching.png", pairs=matched_pairs)

    else:
        plot_matching_qc(pos_1, pos_2, output_dir / "point_matching.png", pairs=matched_pairs)

    return matched_pairs


@click.command()
@click.option("-fcd", "--fixed_pcd", required=True, help="Fixed point cloud")
@click.option("-mpcd", "--moving_pcd", required=True, help="Registered moving point cloud")
@click.option("-o", "--output_dir", required=True, help="Output directory")
@click.option("-min_knn", "--min_neighbours", required=True, type=int, help="Minimum number of neighbours to consider for matching")
@click.option("-max_d", "--max_dist", required=True, type=float, help="Maximum distance between neighbours to consider for matching")
def main(fixed_pcd, moving_pcd, output_dir, min_neighbours, max_dist):

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
    logging.info("Reading fixed point cloud")
    fixed_pcd = o3d.t.io.read_point_cloud(fixed_pcd)

    logging.info("Reading moving point cloud")
    moving_pcd = o3d.t.io.read_point_cloud(moving_pcd)

    matched_pairs = match_points(fixed_pcd, moving_pcd, output_dir)

    write_index_pairs(matched_pairs, str(output_dir / "matched_pairs.txt"))


if __name__ == "__main__":
    main()