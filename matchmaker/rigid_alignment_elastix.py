import os
import sys
import itk
import click
import logging
import numpy as np
from pathlib import Path

from matchmaker.utils import (read_volume, write_volume, get_attrs, plot_overlay, itk_scalar_img,
                                run_registration, itk_to_np_order, apply_transform_chanwise,
                                setup_logging)


def elastix_segm_rigid_alignment(
    fixed_img_np, fixed_resolution, moving_img_np, moving_resolution, output_dir
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

    SCRIPT_DIR = Path(__file__).resolve().parent
    parameter_map_paths = [
        f"{SCRIPT_DIR}/ParameterMap_segm_rigid_registration_corr.txt"
    ]

    logging.info("Run rigid registration with elastix")
    result_image, result_transform_parameters = run_registration(
        fixed_img,
        moving_img,
        parameter_map_paths,
        output_dir,
        log_name="elastix_log_rigid.log",
        set_threads=True,
    )

    logging.info(f"Result image shape {result_image.shape}")
    result_img_np = itk_to_np_order(itk.GetArrayFromImage(result_image))
    plot_overlay(
        itk_to_np_order(itk.GetArrayFromImage(fixed_img)),
        result_img_np,
        f"{output_dir}/plots/overlay_after_rigid_alignment.png",
    )

    logging.info("Apply transform to all channels")
    result_img_np = apply_transform_chanwise(
        result_transform_parameters, moving_img_np, moving_resolution
    )
    logging.info(f"Result image shape {result_img_np.shape}")

    return result_img_np


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
        fixed_path (str): Path to the fixed image .n5 file.
        fixed_key (str): Key to the fixed image data in the .n5 file.
        moving_path (str): Path to the moving image .n5 file.
        moving_key (str): Key to the moving image data in the .n5 file.
        output_dir (str): Directory where the aligned image should be saved.

    Returns:
        np.ndarray: The rigidly aligned moving image.
    """

    if not os.path.exists(f"{output_dir}/plots"):
        os.makedirs(f"{output_dir}/plots")

    logging.info("Start rigid alignment")

    fixed_img_np = fixed_img.astype(np.float32)
    moving_img_np = moving_img.astype(np.float32)

    logging.info("Compute rigid alignment of moving image...")

    moving_img_np = elastix_segm_rigid_alignment(
        fixed_img_np=fixed_img_np,
        fixed_resolution=fixed_resolution,
        moving_img_np=moving_img_np,
        moving_resolution=moving_resolution,
        output_dir=output_dir
    )
    moving_img_np = moving_img_np.astype(np.uint16)

    return moving_img_np


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

    moving_rigid_aligned = run_rigid_alignment(
        fixed_img,
        fixed_resolution,
        moving_img,
        moving_resolution,
        output_dir
    )

    logging.info("Save rigid aligned moving image")
    moving_attributes = dict(get_attrs(moving_path, moving_key))
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

# python rigid_alignment_elastix.py -fi ../examples/data/test/platy1_muscles_stardist_fixed_prealigned.n5 -fk seg -mi ../examples/data/test/platy1_muscles_stardist_moving_prealigned.n5 -mk seg -o ../examples/data/test -ok rigid -trans ../examples/data/test/rigid_transform.json
