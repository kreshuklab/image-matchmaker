import os
import click
import logging
import numpy as np
import pandas as pd
import itk
from pathlib import Path
import open3d as o3d

from image_matchmaker.utils import (
    prealignment_spacing,
    rotate_img,
    read_volume,
    get_attrs,
    write_volume,
    plot_overlay,
    itk_scalar_img,
    apply_transform_chanwise,
    read_transform_dict,
    pcd_to_elastix,
    create_matched_pcds,
    setup_logging,
    elastix_pointset_registration
)


def run_pointset_registration(
    fixed_img_np,
    fixed_resolution,
    moving_img_np,
    moving_resolution,
    matched_label_df,
    output_dir,
):
    """
    Run the deformable B-spline registration stage from segmentation volumes.

    Builds matched fixed/moving point sets from the correspondence table, runs
    the Elastix point-set registration, applies the resulting transform to all
    channels, and writes the alignment and grid-deformation QC plots.

    Parameters
    ----------
    fixed_img_np : numpy.ndarray
        Fixed segmentation volume.
    fixed_resolution : sequence of float
        Voxel spacing of the fixed volume.
    moving_img_np : numpy.ndarray
        Moving segmentation volume.
    moving_resolution : sequence of float
        Voxel spacing of the moving volume.
    matched_label_df : pandas.DataFrame
        Correspondence table (``fixed_label_id`` / ``moving_label_id``).
    output_dir : pathlib.Path
        Directory where point sets, transforms, and QC plots are written.

    Returns
    -------
    numpy.ndarray
        The deformably aligned moving volume.
    """
    logging.info("Extract point clouds")
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

    logging.info("Do deformable alignment")

    fixed_img_semantic_np = (fixed_img_np > 0).astype(np.float32)
    moving_img_semantic_np = (moving_img_np > 0).astype(np.float32)

    fixed_img = itk_scalar_img(fixed_img_semantic_np, fixed_resolution)
    moving_img = itk_scalar_img(moving_img_semantic_np, moving_resolution)

    fixed_img_scalar_np = itk.GetArrayFromImage(fixed_img)
    moving_img_scalar_np = itk.GetArrayFromImage(moving_img)

    plot_overlay(
        fixed_img_scalar_np,
        moving_img_scalar_np,
        f"{output_dir}/plots/deformable_pointset_alignment_before.pdf",
    )
    plot_overlay(
        fixed_img_scalar_np,
        moving_img_scalar_np,
        f"{output_dir}/plots/deformable_pointset_alignment_before.png",
    )

    SCRIPT_DIR = Path(__file__).resolve().parent
    parameter_map_paths = [
        f"{SCRIPT_DIR}/ParameterMap_rigid_pointset.txt",
        f"{SCRIPT_DIR}/ParameterMap_bspline_pointset_rough.txt",
        f"{SCRIPT_DIR}/ParameterMap_bspline_pointset_fine.txt",
    ]

    logging.info("Start registration")
    log_name = "elastix_log_deformable.log"
    result_image, result_transform_parameters = elastix_pointset_registration(
        fixed_img,
        moving_img,
        parameter_map_paths,
        fixed_pointset,
        moving_pointset,
        str(output_dir),
        log_name=log_name,
    )

    result_img_np = itk.GetArrayFromImage(result_image)
    result_resolution = list(result_image.GetSpacing())[::-1]  # XYZ -> ZYX
    plot_overlay(
        fixed_img_scalar_np,
        result_img_np,
        f"{output_dir}/plots/deformable_pointset_alignment_semantic.pdf",
    )
    plot_overlay(
        fixed_img_scalar_np,
        result_img_np,
        f"{output_dir}/plots/deformable_pointset_alignment_semantic.png",
    )

    logging.info("Apply transform to all channels")
    logging.info(f"Moving image shape {moving_img_np.shape}")
    result_img_np = apply_transform_chanwise(
        result_transform_parameters, moving_img_np.astype(np.float32), moving_resolution
    )
    plot_overlay(
        fixed_img_np,
        result_img_np,
        f"{output_dir}/plots/deformable_pointset_alignment_final.pdf",
    )
    plot_overlay(
        fixed_img_np,
        result_img_np,
        f"{output_dir}/plots/deformable_pointset_alignment_final.png",
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
    plot_overlay(fixed_img_np, grid_img_np, f"{output_dir}/plots/grid_before.png")
    plot_overlay(fixed_img_np, transformed_grid_np, f"{output_dir}/plots/grid_after.png")

    return result_img_np, result_resolution


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
    setup_logging(output_dir, "elastix_deformable_pointset_registration.log")

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

    matched_label_df = pd.read_csv(match_path)
    print(matched_label_df)

    moving_aligned, aligned_resolution = run_pointset_registration(
        fixed_img,
        fixed_resolution,
        moving_img,
        moving_resolution,
        matched_label_df,
        output_dir,
    )

    logging.info("Save registered moving image")
    # Elastix resamples into the fixed image domain, so this output is on the fixed grid
    aligned_attributes = dict(get_attrs(moving_path, moving_key))
    aligned_attributes["resolution"] = aligned_resolution
    write_volume(
        f=moving_path, arr=moving_aligned, key=output_key, attrs=aligned_attributes
    )

    prealignment_transform = read_transform_dict(prealignment_transform)
    matrix = prealignment_transform["fixed_prealignment"]["matrix"]
    output_shape = prealignment_transform["fixed_prealignment"]["output_shape"]
    prealigned_moving_aligned = rotate_img(
        moving_aligned, matrix, output_shape=output_shape
    )
    prealigned_fixed = rotate_img(fixed_img, matrix, output_shape=output_shape)

    # the prealignment transform maps the fixed grid into the isotropic prealignment space
    prealigned_attributes = dict(aligned_attributes)
    prealigned_attributes["resolution"] = prealignment_spacing(fixed_resolution)
    write_volume(
        f=moving_path,
        arr=prealigned_moving_aligned,
        key=prealigned_output_key,
        attrs=prealigned_attributes,
    )
    plot_overlay(
        prealigned_fixed,
        prealigned_moving_aligned,
        f"{output_dir}/plots/deformable_pointset_alignment_prealigned.pdf",
    )
    plot_overlay(
        prealigned_fixed,
        prealigned_moving_aligned,
        f"{output_dir}/plots/deformable_pointset_alignment_prealigned.png",
    )


if __name__ == "__main__":
    main()
