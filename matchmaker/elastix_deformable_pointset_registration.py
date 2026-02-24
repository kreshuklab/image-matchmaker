import os
import sys
import click
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itk
from pathlib import Path

import open3d as o3d


from matchmaker.data import create_point_cloud
from matchmaker.utils import (
    get_transformation_matrix,
    rotate_img,
    read_volume,
    get_attrs,
    write_volume,
    write_transform_dict,
    plot_three_slices,
    plot_overlay,
)

from matchmaker.utils import (
    read_volume,
    write_volume,
    get_attrs,
    plot_overlay,
    itk_scalar_img,
    itk_to_np_order,
    apply_transform_chanwise,
    create_parameter_object,
)

from matchmaker.utils import (
    overlay_pcds,
    visualize_displacement_field,
    extract_centroids,
    run_cpd,
    create_pcd,
)


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


def create_matched_pcds(
    fixed_img_np, fixed_resolution, moving_img_np, moving_resolution, matched_label_df
):

    fixed_labels, fixed_center_coords = extract_centroids(
        fixed_img_np, fixed_resolution
    )
    moving_labels, moving_center_coords = extract_centroids(
        moving_img_np, moving_resolution
    )

    print(matched_label_df)
    print(matched_label_df["fixed_label_id"])
    fixed_df = pd.DataFrame(
        data=fixed_center_coords, index=fixed_labels, columns=["x", "y", "z"]
    )
    fixed_df = fixed_df.loc[matched_label_df["fixed_label_id"], :]
    print(fixed_df)
    moving_df = pd.DataFrame(
        data=moving_center_coords, index=moving_labels, columns=["x", "y", "z"]
    )
    moving_df = moving_df.loc[matched_label_df["moving_label_id"], :]
    print(moving_df)

    fixed_pcd = create_pcd(np.array(fixed_df), np.array(fixed_df.index))
    moving_pcd = create_pcd(np.array(moving_df), np.array(moving_df.index))
    return fixed_pcd, moving_pcd, fixed_df, moving_df


def run_pointset_registration(
    fixed_img,
    moving_img,
    parameter_map_paths,
    fixed_pointset,
    moving_pointset,
    output_dir,
    log_name="elastix.log",
    set_threads=False,
):
    logging.info("Start creating parameter object")
    parameter_object = create_parameter_object(parameter_map_paths)
    # Load Elastix Image Filter Object
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_img, moving_img)
    elastix_object.SetFixedPointSetFileName(fixed_pointset)
    elastix_object.SetMovingPointSetFileName(moving_pointset)
    elastix_object.SetParameterObject(parameter_object)
    if set_threads:
        elastix_object.SetNumberOfThreads(32)
    elastix_object.SetLogToConsole(True)
    elastix_object.SetLogToFile(True)
    elastix_object.SetOutputDirectory(output_dir)
    elastix_object.SetLogFileName(log_name)

    logging.info("Created registration object")
    logging.info(elastix_object)
    logging.info("Set parameter map")

    logging.info("Start registration")
    elastix_object.UpdateLargestPossibleRegion()

    result_image = elastix_object.GetOutput()
    result_transform_parameters = elastix_object.GetTransformParameterObject()

    return result_image, result_transform_parameters


def elastix_deformable_pointset_alignment(
    fixed_img_np,
    fixed_resolution,
    moving_img_np,
    moving_resolution,
    matched_label_df,
    output_dir,
):
    """
    Run deformable alignment using elastix based on a set of key points.
    """
    logging.info(f"Extract point clouds")
    fixed_pcd, moving_pcd, fixed_df, moving_df = create_matched_pcds(
        fixed_img_np,
        fixed_resolution,
        moving_img_np,
        moving_resolution,
        matched_label_df,
    )
    fixed_df.to_csv(output_dir / "fixed_pointset.csv")
    moving_df.to_csv(output_dir / "moving_pointset.csv")
    o3d.t.io.write_point_cloud(
        str(output_dir / "fixed_pcd.pcd"), fixed_pcd, write_ascii=True
    )
    o3d.t.io.write_point_cloud(
        str(output_dir / "moving_pcd.pcd"), moving_pcd, write_ascii=True
    )

    fixed_pointset = str(output_dir / "fixed_pointset.txt")
    moving_pointset = str(output_dir / "moving_pointset.txt")
    pcd_to_elastix(str(output_dir / "fixed_pcd.pcd"), fixed_pointset)
    pcd_to_elastix(str(output_dir / "moving_pcd.pcd"), moving_pointset)

    logging.info(f"Do deformable alignment")

    fixed_img_semantic_np = (fixed_img_np > 0).astype(np.float32)
    moving_img_semantic_np = (moving_img_np > 0).astype(np.float32)

    fixed_img = itk_scalar_img(fixed_img_semantic_np, fixed_resolution)
    moving_img = itk_scalar_img(moving_img_semantic_np, moving_resolution)

    plot_overlay(
        itk_to_np_order(itk.GetArrayFromImage(fixed_img)),
        itk_to_np_order(itk.GetArrayFromImage(moving_img)),
        output_dir / f"deformable_pointset_alignment_before.png",
    )

    SCRIPT_DIR = Path(__file__).resolve().parent
    parameter_map_paths = [
        f"{SCRIPT_DIR}/ParameterMap_rigid_pointset.txt",
        # f"{SCRIPT_DIR}/ParameterMap_bspline_pointset_rough.txt",
        # f"{SCRIPT_DIR}/ParameterMap_bspline_pointset_fine.txt",
    ]

    print("Start registration")
    log_name = f"elastix_log_deformable.log"
    result_image, result_transform_parameters = run_pointset_registration(
        fixed_img,
        moving_img,
        parameter_map_paths,
        fixed_pointset,
        moving_pointset,
        str(output_dir),
        log_name=log_name,
    )

    result_img_np = itk_to_np_order(itk.GetArrayFromImage(result_image))
    plot_overlay(
        itk_to_np_order(itk.GetArrayFromImage(fixed_img)),
        result_img_np,
        output_dir / f"deformable_pointset_alignment_semantic.png",
    )

    logging.info(f"Apply transform to all channels")
    logging.info(f"Moving image shape {moving_img_np.shape}")
    result_img_np = apply_transform_chanwise(
        result_transform_parameters, moving_img_np.astype(np.float32), moving_resolution
    )
    plot_overlay(
        fixed_img_np,
        result_img_np,
        output_dir / f"deformable_pointset_alignment_final.png",
    )
    logging.info(f"Result image shape {result_img_np.shape}")

    # Transform a grid
    grid_img_np = np.zeros_like(moving_img_np)
    grid_img_np[::10] = 1
    grid_img_np[:, ::10, :] = 1
    grid_img_np[:, :, ::10] = 1
    transformed_grid_np = apply_transform_chanwise(
        result_transform_parameters, grid_img_np, moving_resolution
    )
    plot_overlay(fixed_img_np, grid_img_np, output_dir / f"grid_before.png")
    plot_overlay(fixed_img_np, transformed_grid_np, output_dir / f"grid_after.png")

    return result_img_np


@click.command()
@click.option("-fi", "--fixed_path", required=True, help="Fixed input .n5 file")
@click.option("-fk", "--fixed_key", required=True, help="Fixed input key")
@click.option("-mi", "--moving_path", required=True, help="Moving input .n5 file")
@click.option("-mk", "--moving_key", required=True, help="Moving input key")
@click.option("-o", "--output_dir", required=True, help="Output directory")
@click.option("-ok", "--output_key", required=True, help="Output key (same in both n5)")
@click.option(
    "-match",
    "--match_path",
    required=True,
    help="Path to the correspondence table between the instances in fixed and moving images",
)
def main(
    fixed_path, fixed_key, moving_path, moving_key, output_dir, output_key, match_path
):
    """
    Perform alignment of segmentations based on matching instances.

    Args:
        fixed_path (str): Path to the fixed input .n5 file.
        fixed_key (str): Key to the fixed image data in the .n5 file.
        moving_path (str): Path to the moving input .n5 file.
        moving_key (str): Key to the moving image data in the .n5 file.
        output_dir (str): Directory where the results should be saved.

    Returns:
        None
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(
                f"{output_dir}/elastix_deformable_pointset_registration.log", mode="w"
            ),
            logging.StreamHandler(sys.stdout),
        ],
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_dir = Path(output_dir)

    logging.info("Reading fixed image")
    fixed_img = read_volume(fixed_path, fixed_key)
    fixed_resolution = get_attrs(fixed_path, fixed_key)["resolution"]
    logging.info(f"Fixed image shape: {fixed_img.shape}, dtype {fixed_img.dtype}")

    logging.info("Reading moving image")
    moving_img = read_volume(moving_path, moving_key)
    moving_resolution = get_attrs(moving_path, moving_key)["resolution"]
    logging.info(f"Moving image shape: {moving_img.shape}, dtype {moving_img.dtype}")

    matched_label_df = pd.read_csv(match_path)
    print(matched_label_df)

    moving_aligned = elastix_deformable_pointset_alignment(
        fixed_img,
        fixed_resolution,
        moving_img,
        moving_resolution,
        matched_label_df,
        output_dir,
    )

    logging.info("Save registered moving image")
    moving_attributes = dict(get_attrs(moving_path, moving_key))
    write_volume(
        f=moving_path, arr=moving_aligned, key=output_key, attrs=moving_attributes
    )


if __name__ == "__main__":
    main()
