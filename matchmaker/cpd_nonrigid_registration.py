import os
import sys
import click
import logging
from pathlib import Path

import open3d as o3d

from matchmaker.utils import (
    read_volume,
    get_attrs
)


from matchmaker.utils import overlay_pcds, visualize_displacement_field, extract_centroids, cpd_from_pcds, create_pcd



def run_cpd(fixed_img, fixed_resolution, moving_img, moving_resolution, output_dir, w, beta, lmd, maxiter):
    """
    Run non-rigid CPD registration starting from two segmentation volumes.

    Extracts instance centroids from both volumes, builds point clouds, runs
    CPD, and writes the fixed, moving, and registered point clouds together with
    before/after and displacement-field QC plots to ``output_dir``.

    Parameters
    ----------
    fixed_img : numpy.ndarray
        Fixed segmentation volume.
    fixed_resolution : sequence of float
        Voxel spacing of the fixed volume.
    moving_img : numpy.ndarray
        Moving segmentation volume.
    moving_resolution : sequence of float
        Voxel spacing of the moving volume.
    output_dir : pathlib.Path
        Directory where point clouds and QC plots are written.
    w : float
        Outlier weight (fraction of points assumed to be noise).
    beta : float
        Width of the Gaussian smoothing kernel.
    lmd : float
        Regularization weight (trade-off between fit and smoothness).
    maxiter : int
        Maximum number of EM iterations.
    """
    fixed_labels, fixed_center_coords = extract_centroids(fixed_img, fixed_resolution)
    moving_labels, moving_center_coord = extract_centroids(moving_img, moving_resolution)

    fixed_pcd = create_pcd(fixed_center_coords, fixed_labels)
    moving_pcd = create_pcd(moving_center_coord, moving_labels)

    overlay_pcds(fixed_pcd, moving_pcd, projection="xz", save_path = output_dir / "pcds_before_registration_xz.png")
    overlay_pcds(fixed_pcd, moving_pcd, projection="yz", save_path = output_dir / "pcds_before_registration_yz.png")
    overlay_pcds(fixed_pcd, moving_pcd, projection="xy", save_path = output_dir / "pcds_before_registration_xy.png")

    logging.info(f"Point cloud registration with parameters w={w}, beta={beta}, lmd={lmd}, maxiter={maxiter}")

    registered_pcd = cpd_from_pcds(fixed_pcd, moving_pcd, w, beta, lmd, maxiter)

    overlay_pcds(fixed_pcd, registered_pcd, projection="xz", save_path = output_dir / "pcds_after_registration_xz.png")
    overlay_pcds(fixed_pcd, registered_pcd, projection="yz", save_path = output_dir / "pcds_after_registration_yz.png")
    overlay_pcds(fixed_pcd, registered_pcd, projection="xy", save_path = output_dir / "pcds_after_registration_xy.png")

    visualize_displacement_field(moving_pcd, registered_pcd, projection="xz", save_path = output_dir / "displacement_field_xz.png")
    visualize_displacement_field(moving_pcd, registered_pcd, projection="yz", save_path = output_dir / "displacement_field_yz.png")
    visualize_displacement_field(moving_pcd, registered_pcd, projection="xy", save_path = output_dir / "displacement_field_xy.png")

    o3d.t.io.write_point_cloud(str(output_dir / "fixed_pcd.pcd"), fixed_pcd, write_ascii=True)
    o3d.t.io.write_point_cloud(str(output_dir / "moving_pcd.pcd"), moving_pcd, write_ascii=True)
    o3d.t.io.write_point_cloud(str(output_dir / "registered_pcd.pcd"), registered_pcd, write_ascii=True)



@click.command()
@click.option("-fi", "--fixed_path", required=True, help="Fixed prealigned input .n5 file")
@click.option("-fk", "--fixed_key", required=True, help="Fixed input key")
@click.option("-mi", "--moving_path", required=True, help="Moving prealigned input .n5 file")
@click.option("-mk", "--moving_key", required=True, help="Moving input key")
@click.option("-o", "--output_dir", required=True, help="Output directory")
@click.option("-w", "--w", required=True, type=float, help="Parameter of nonrigid CPD")
@click.option("-beta", "--beta", required=True, type=float, help="Parameter of nonrigid CPD")
@click.option("-lmd", "--lmd", required=True, type=float, help="Parameter of nonrigid CPD")
@click.option("-maxiter", "--maxiter", required=True, type=int, help="Parameter of nonrigid CPD")
def main(fixed_path, fixed_key, moving_path, moving_key, output_dir, w, beta, lmd, maxiter):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{output_dir}/cpd_nonrigid_registration.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_dir = Path(output_dir)
    os.makedirs(output_dir / "plots", exist_ok=True)

    logging.info("Reading fixed image")
    fixed_img = read_volume(fixed_path, fixed_key)
    fixed_resolution = get_attrs(fixed_path, fixed_key)["resolution"]
    logging.info(f"Fixed image shape: {fixed_img.shape}, dtype {fixed_img.dtype}")

    logging.info("Reading moving image")
    moving_img = read_volume(moving_path, moving_key)
    moving_resolution = get_attrs(moving_path, moving_key)["resolution"]
    logging.info(f"Moving image shape: {moving_img.shape}, dtype {moving_img.dtype}")

    run_cpd(
        fixed_img,
        fixed_resolution,
        moving_img,
        moving_resolution,
        output_dir,
        w, beta, lmd, maxiter
    )


    


if __name__ == "__main__":
    main()