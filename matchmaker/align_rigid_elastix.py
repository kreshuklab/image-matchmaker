from pathlib import Path
import numpy as np
import argparse
from matchmaker.n5_utils import read_volume, write_volume, get_attrs
from matchmaker.vis import plot_overlay
import logging
import sys
from matchmaker import elastix_utils
from matchmaker.mobie_export import export_to_mobie
import itk
import click
import os


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

    fixed_img = elastix_utils.itk_scalar_img(fixed_img_semantic_np, fixed_resolution)
    moving_img = elastix_utils.itk_scalar_img(moving_img_semantic_np, moving_resolution)

    logging.info("Fixed image")
    logging.info(f"{fixed_img}")
    logging.info("Moving image")
    logging.info(f"{moving_img}")

    plot_overlay(
        elastix_utils.itk_to_np_order(itk.GetArrayFromImage(fixed_img)),
        elastix_utils.itk_to_np_order(itk.GetArrayFromImage(moving_img)),
        f"{output_dir}/plots/intersample_segm_overlay_before_alignment.png",
    )

    parameter_map_paths = [
        "../ParameterMap_segm_rigid_registration_corr.txt"
    ]
    logging.info("Run rigid registration")
    result_image, result_transform_parameters = elastix_utils.run_registration(
        fixed_img,
        moving_img,
        parameter_map_paths,
        output_dir,
        log_name="elastix_log_rigid.log",
        set_threads=True,
    )

    logging.info(f"Result image shape {result_image.shape}")
    result_img_np = elastix_utils.itk_to_np_order(itk.GetArrayFromImage(result_image))
    plot_overlay(
        elastix_utils.itk_to_np_order(itk.GetArrayFromImage(fixed_img)),
        result_img_np,
        f"{output_dir}/plots/intersample_segm_rigid_alignment_semantic.png",
    )
    # NOTE: difference between result_img_np before and after applying transform?
    logging.info("Apply transform to all channels")
    result_img_np = elastix_utils.apply_transform_chanwise(
        result_transform_parameters, moving_img_np, moving_resolution
    )
    logging.info(f"Result image shape {result_img_np.shape}")

    return result_img_np


def run_rigid_alignment(
    fixed_path,
    fixed_key,
    moving_path,
    moving_key,
    output_dir,
    mobie_export
):
    if not os.path.exists(f"{output_dir}/plots"):
        os.makedirs(f"{output_dir}/plots")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{output_dir}/rigid_alignment.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("Read image file")
    fixed_img_np = read_volume(fixed_path, fixed_key)
    print(fixed_img_np.dtype)

    if fixed_path == moving_path:
        logging.info("Same n5 for fixed and moving image, not doing registration")
        moving_img_np = fixed_img_np

    else:
        moving_img_np = read_volume(moving_path, moving_key)

        fixed_img_np = fixed_img_np.astype(np.float32)
        moving_img_np = moving_img_np.astype(np.float32)

        logging.info("Start registration")

        moving_img_np = elastix_segm_rigid_alignment(
            fixed_img_np=fixed_img_np,
            fixed_resolution=get_attrs(fixed_path, fixed_key)["resolution"],
            moving_img_np=moving_img_np,
            moving_resolution=get_attrs(moving_path, moving_key)["resolution"],
            output_dir=output_dir,
        )
    moving_img_np = moving_img_np.astype(np.uint16)
    logging.info("Write results")
    attributes = dict(get_attrs(moving_path, moving_key))

    file_name = os.path.splitext(os.path.basename(moving_path))[0]
    file_name = file_name.removesuffix("_prealigned")
    print(file_name)

    write_volume(
        f"{output_dir}/{file_name}_rigid_aligned.n5",
        moving_img_np,
        moving_key,
        chunks=(128, 512, 512),
        attrs=attributes,
    )

    # export rigid alignment to mobie
    if mobie_export:
        export_to_mobie(
            input_path=f"{output_dir}/{file_name}_rigid_aligned.n5",
            input_key=moving_key,
            output_dir=output_dir,
            segmentation_name=f"{file_name}_rigid_aligned",
            menu_name="moving"
        )


@click.command()
@click.option("-fi", "--fixed_path", required=True, help="Fixed input .n5 file")
@click.option("-fk", "--fixed_key", required=True, help="Fixed input key")
@click.option("-mi", "--moving_path", required=True, help="Moving input .n5 file")
@click.option("-mk", "--moving_key", required=True, help="Moving input key")
@click.option("-o", "--output_dir", required=True, help="Output directory")
@click.option("-m", "--mobie_export", required=False, is_flag=True, help="MoBIE export")
def main(fixed_path, fixed_key, moving_path, moving_key, output_dir, mobie_export):

    run_rigid_alignment(fixed_path, fixed_key, moving_path, moving_key, output_dir, mobie_export)


if __name__ == "__main__":
    main()

    # python align_rigid_elastix.py ../examples/data/test/platy1_muscles_stardist_fixed_prealigned.n5 seg ../examples/data/test/platy1_muscles_stardist_moving_prealigned.n5 seg ../examples/data/test/platy1_muscles_stardist_moving_rigid_aligned.n5 seg
