import os
import sys
import itk
import click
import logging
import numpy as np
from pathlib import Path
import pandas as pd
from skimage.measure import regionprops_table
import open3d as o3d

from matchmaker.utils import (
    read_volume,
    write_volume,
    get_attrs,
    plot_overlay,
    itk_scalar_img,
    run_registration,
    itk_to_np_order,
    apply_transform_chanwise
)


def extract_centroids(segm, resolution):
    coords_df = pd.DataFrame(regionprops_table(segm, properties=("label", 'centroid')))
    center_coords = np.array([coords_df["centroid-2"] * resolution[2], coords_df["centroid-1"] * resolution[1], coords_df["centroid-0"] * resolution[0]]).T.astype(np.float64)
    labels = np.array(coords_df['label'])
    return labels, center_coords


def create_pcd(center_coords, labels):
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3d.core.Tensor(center_coords)
    pcd.point.label = o3d.core.Tensor(labels[:, None])
    return pcd


def run_centroid_matching(fixed_img, fixed_resolution, moving_img, moving_resolution, output_dir):
    fixed_labels, fixed_center_coords = extract_centroids(fixed_img, fixed_resolution)
    moving_labels, moving_center_coord = extract_centroids(moving_img, moving_resolution)

    fixed_pcd = create_pcd(fixed_center_coords, fixed_labels)
    moving_pcd = create_pcd(moving_center_coord, moving_labels)

    

    matched_pairs = pd.DataFrame(columns=["fixed", "moving"])
    return matched_pairs


@click.command()
@click.option("-fi", "--fixed_path", required=True, help="Fixed prealigned input .n5 file")
@click.option("-fk", "--fixed_key", required=True, help="Fixed input key")
@click.option("-mi", "--moving_path", required=True, help="Moving prealigned input .n5 file")
@click.option("-mk", "--moving_key", required=True, help="Moving input key")
@click.option("-o", "--output_dir", required=True, help="Output directory")
@click.option("-match", "--match_path", required=True, help="Path to the list of matched instances")
def main(fixed_path, fixed_key, moving_path, moving_key, output_dir, match_path):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{output_dir}/rigid_alignment.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("Reading fixed image")
    fixed_img = read_volume(fixed_path, fixed_key)
    fixed_resolution = get_attrs(fixed_path, fixed_key)["resolution"]
    logging.info(f"Fixed image shape: {fixed_img.shape}, dtype {fixed_img.dtype}")

    logging.info("Reading moving image")
    moving_img = read_volume(moving_path, moving_key)
    moving_resolution = get_attrs(moving_path, moving_key)["resolution"]
    logging.info(f"Moving image shape: {moving_img.shape}, dtype {moving_img.dtype}")

    matched_pairs = run_centroid_matching(
        fixed_img,
        fixed_resolution,
        moving_img,
        moving_resolution,
        output_dir
    )

    logging.info("Save matched pairs")
    
    matched_pairs.to_csv(match_path, index=False)
    


if __name__ == "__main__":
    main()