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

from matchmaker.utils import (
    rotate_img,
    read_volume,
    get_attrs,
    write_volume,
    plot_overlay,
    read_volume,
    write_volume,
    get_attrs,
    plot_overlay,
    itk_scalar_img,
    itk_to_np_order,
    apply_transform_chanwise,
    create_parameter_object,
    extract_centroids,
    create_pcd,
    read_transform_dict,
    rotate_img,
    pcd_to_elastix,
    create_matched_pcds,
)


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
        f"{SCRIPT_DIR}/ParameterMap_bspline_pointset_rough.txt",
        f"{SCRIPT_DIR}/ParameterMap_bspline_pointset_fine.txt",
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
        result_transform_parameters, grid_img_np.astype(np.float32), moving_resolution
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
@click.option(
    "-ok",
    "--output_key",
    required=True,
    help="Output key for the regsitered moving image",
)
@click.option(
    "-match",
    "--match_path",
    required=True,
    help="Path to the correspondence table between the instances in fixed and moving images",
)
@click.option(
    "-pok",
    "--prealigned_output_key",
    required=True,
    help="Output key for the registered moving after applying prealignment transform",
)
@click.option(
    "-transform",
    "--prealignment_transform",
    required=True,
    help="Prealignment transform path",
)
def main(
    fixed_path,
    fixed_key,
    moving_path,
    moving_key,
    output_dir,
    output_key,
    match_path,
    prealigned_output_key,
    prealignment_transform,
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

    prealignment_transform = read_transform_dict(prealignment_transform)
    matrix = prealignment_transform["fixed_prealignment"]["matrix"]
    output_shape = prealignment_transform["fixed_prealignment"]["output_shape"]
    prealigned_moving_aligned = rotate_img(
        moving_aligned, matrix, output_shape=output_shape
    )
    prealigned_fixed = rotate_img(fixed_img, matrix, output_shape=output_shape)

    write_volume(
        f=moving_path,
        arr=prealigned_moving_aligned,
        key=prealigned_output_key,
        attrs=moving_attributes,
    )
    plot_overlay(
        prealigned_fixed,
        prealigned_moving_aligned,
        output_dir / f"deformable_pointset_alignment_prealigned.png",
    )


if __name__ == "__main__":
    main()
