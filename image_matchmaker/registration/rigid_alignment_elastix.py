import itk
import click
import logging
import numpy as np
from pathlib import Path

from image_matchmaker.utils import (
    read_volume,
    write_volume,
    get_attrs,
    plot_overlay,
    itk_scalar_img,
    elastix_registration,
    apply_transform_chanwise,
    setup_logging,
    get_parameter_map_paths,
)

DEFAULT_RIGID_PARAMETER_MAPS = (
    "ParameterMap_segm_rigid_registration_corr.txt",
)


def elastix_segm_rigid_alignment(
    fixed_img_np, fixed_resolution, moving_img_np, moving_resolution, output_dir,
    parameter_map_paths=None,
):
    """
    Run rigid alignment of the ventral and dorsal datasets using elastix.
    """

    logging.info("Do rigid transform of unnormalized images")
    logging.info(f"Fixed resolution {fixed_resolution}")
    logging.info(f"Moving resolution {moving_resolution}")

    fixed_img_semantic_np = (fixed_img_np > 0).astype(np.float32)
    moving_img_semantic_np = (moving_img_np > 0).astype(np.float32)

    fixed_img = itk_scalar_img(fixed_img_semantic_np, fixed_resolution)
    moving_img = itk_scalar_img(moving_img_semantic_np, moving_resolution)

    logging.info("Fixed image")
    logging.info(f"{fixed_img}")
    logging.info("Moving image")
    logging.info(f"{moving_img}")

    parameter_map_paths = get_parameter_map_paths(parameter_map_paths, DEFAULT_RIGID_PARAMETER_MAPS)

    logging.info("Run rigid registration with elastix")
    result_image, result_transform_parameters = elastix_registration(
        fixed_img,
        moving_img,
        parameter_map_paths,
        output_dir,
        log_name="elastix_log_rigid.log",
        set_threads=True,
    )

    logging.info(f"Result image shape {result_image.shape}")
    result_img_np = itk.GetArrayFromImage(result_image)
    result_resolution = list(result_image.GetSpacing())[::-1]  # XYZ -> ZYX
    fixed_img_scalar_np = itk.GetArrayFromImage(fixed_img)
    plot_overlay(
        fixed_img_scalar_np,
        result_img_np,
        f"{output_dir}/plots/overlay_after_rigid_alignment.pdf",
    )
    plot_overlay(
        fixed_img_scalar_np,
        result_img_np,
        f"{output_dir}/plots/overlay_after_rigid_alignment.png",
    )

    logging.info("Apply transform to all channels")
    result_img_np = apply_transform_chanwise(
        result_transform_parameters, moving_img_np, moving_resolution
    )
    logging.info(f"Result image shape {result_img_np.shape}")

    return result_img_np, result_resolution


def run_rigid_alignment(
    fixed_img,
    fixed_resolution,
    moving_img,
    moving_resolution,
    output_dir
):
    """
    Perform rigid alignment of a moving image to a fixed image using Elastix.

    This function reads the fixed and moving images from the specified paths,
    performs a rigid alignment using Elastix, and saves the aligned moving image
    to the output directory. If the MoBIE export flag is set, the aligned image
    is also exported to a MoBIE project.

    Args:
        fixed_img (np.ndarray): Fixed (prealigned) image volume.
        fixed_resolution (sequence of float): Voxel spacing of the fixed image.
        moving_img (np.ndarray): Moving (prealigned) image volume.
        moving_resolution (sequence of float): Voxel spacing of the moving image.
        output_dir (str): Directory where the aligned image and plots are saved.

    Returns:
        np.ndarray: The rigidly aligned moving image.
    """

    Path(f"{output_dir}/plots").mkdir(parents=True, exist_ok=True)

    logging.info("Start rigid alignment")

    fixed_img_np = fixed_img.astype(np.float32)
    moving_dtype = moving_img.dtype
    moving_img_np = moving_img.astype(np.float32)

    logging.info("Compute rigid alignment of moving image...")

    moving_img_np, aligned_resolution = elastix_segm_rigid_alignment(
        fixed_img_np=fixed_img_np,
        fixed_resolution=fixed_resolution,
        moving_img_np=moving_img_np,
        moving_resolution=moving_resolution,
        output_dir=output_dir
    )
    # Preserve the caller's storage dtype. In particular, the CPD tuning
    # workflow embeds landmark IDs in uint32 segmentations before this step.
    moving_img_np = moving_img_np.astype(moving_dtype)

    return moving_img_np, aligned_resolution


@click.command()
@click.option("-fi", "--fixed_path", required=True, help="Fixed prealigned input .n5 file")
@click.option("-fk", "--fixed_key", required=True, help="Fixed input key")
@click.option("-mi", "--moving_path", required=True, help="Moving prealigned input .n5 file")
@click.option("-mk", "--moving_key", required=True, help="Moving input key")
@click.option("-o", "--output_dir", required=True, help="Output directory")
@click.option("-ok", "--output_key", required=True, help="Output key (same in both n5)")
@click.option("-tif", "--save_tif", is_flag=True, help="Whether to save tif or not")
def main(fixed_path, fixed_key, moving_path, moving_key, output_dir, output_key, save_tif=False):
    setup_logging(output_dir, "rigid_alignment.log")

    logging.info("Reading fixed image")
    fixed_img = read_volume(fixed_path, fixed_key)
    fixed_resolution = get_attrs(fixed_path, fixed_key)["resolution"]
    logging.info(f"Fixed image shape: {fixed_img.shape}, dtype {fixed_img.dtype}")

    logging.info("Reading moving image")
    moving_img = read_volume(moving_path, moving_key)
    moving_resolution = get_attrs(moving_path, moving_key)["resolution"]
    logging.info(f"Moving image shape: {moving_img.shape}, dtype {moving_img.dtype}")

    moving_rigid_aligned, aligned_resolution = run_rigid_alignment(
        fixed_img,
        fixed_resolution,
        moving_img,
        moving_resolution,
        output_dir
    )

    logging.info("Save rigid aligned moving image")
    moving_attributes = dict(get_attrs(moving_path, moving_key))
    moving_attributes["resolution"] = aligned_resolution
    write_volume(
        f=moving_path,
        arr=moving_rigid_aligned,
        key=output_key,
        attrs=moving_attributes
    )

    if save_tif:
        import tifffile as tiff
        tiff.imwrite(f"{output_dir}/moving_rigid_aligned.tif", moving_rigid_aligned)


if __name__ == "__main__":
    main()
