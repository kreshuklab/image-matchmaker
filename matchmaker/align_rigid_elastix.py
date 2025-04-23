from pathlib import Path
import numpy as np
import argparse
from platy_reg.n5_utils import read_volume, write_volume, get_attrs
from matchmaker.vis import plot_overlay
import logging
import sys
from matchmaker import elastix_utils
import itk


def elastix_segm_rigid_alignment(
    fixed_img_np, fixed_resolution, moving_img_np, moving_resolution, log_dir
):
    """
    Run rigid alignment of the ventral and dorsal datasets using elastix.
    """

    logging.info("Do rigid transform of unnormalized images")
    logging.info(f"Fixed resolution {fixed_resolution}")
    logging.info(f"Moving resolution {moving_resolution}")
    logging.info("Do rigid transform of unnormalized images")

    fixed_img_semantic_np = (fixed_img_np > 0).astype(np.float32)
    moving_img_semantic_np = (moving_img_np > 0).astype(np.float32)

    fixed_img = elastix_utils.itk_scalar_img(fixed_img_semantic_np, fixed_resolution)
    moving_img = elastix_utils.itk_scalar_img(moving_img_semantic_np, moving_resolution)

    logging.info("Fixed image")
    logging.info(f"{fixed_img}")
    logging.info("Moving image")
    logging.info(f"{moving_img}")

    # plot_overlay(fixed_img_np, moving_img_np, log_dir / "intersample_segm_overlay_before_alignment.png")

    plot_overlay(
        elastix_utils.itk_to_np_order(itk.GetArrayFromImage(fixed_img)),
        elastix_utils.itk_to_np_order(itk.GetArrayFromImage(moving_img)),
        log_dir / "intersample_segm_overlay_before_alignment.png",
    )

    parameter_map_paths = [
        "pipeline_steps/inter_sample_registration/ParameterMap_segm_rigid_registration_corr.txt"
    ]
    logging.info("Run rigid registration")
    result_image, result_transform_parameters = elastix_utils.run_registration(
        fixed_img,
        moving_img,
        parameter_map_paths,
        str(log_dir),
        log_name="elastix_log_rigid.log",
        set_threads=True,
    )
    # serialize_parameter_object(result_transform_parameters, "ParameterMap_rigid_transform", log_dir)

    logging.info(f"Result image shape {result_image.shape}")
    result_img_np = elastix_utils.itk_to_np_order(itk.GetArrayFromImage(result_image))
    plot_overlay(
        elastix_utils.itk_to_np_order(itk.GetArrayFromImage(fixed_img)),
        result_img_np,
        log_dir / "intersample_segm_rigid_alignment_semantic.png",
    )

    logging.info("Apply transform to all channels")
    result_img_np = elastix_utils.apply_transform_chanwise(
        result_transform_parameters, moving_img_np, moving_resolution
    )
    logging.info(f"Result image shape {result_img_np.shape}")

    return result_img_np


def main():
    parser = argparse.ArgumentParser(
        description="""Align rigidly moving segmentation volume to fixed volume."""
    )

    parser.add_argument("fixed_n5", type=str, help="Path of the output n5")
    parser.add_argument("fixed_key", type=str, help="Key of ventral dataset")
    parser.add_argument("moving_n5", type=str, help="Path of the output n5")
    parser.add_argument("moving_key", type=str, help="Key of ventral dataset")
    parser.add_argument("output_n5", type=str, help="Path of the output n5")
    parser.add_argument("output_key", type=str, help="Key to write aligned ventral dataset to in the output n5")
    parser.add_argument("log_dir", type=str, help="Directory to store diagnostic plots etc")
    parser.add_argument("log_path", type=str, help="Path to log file")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("Read image file")
    fixed_img_np = read_volume(args.fixed_n5, args.fixed_key)

    # pad_z = 50
    # fixed_img_np = np.pad(fixed_img_np, pad_width=((pad_z, pad_z), (0, 0), (0, 0)))

    # fixed_mask = gaussian(fixed_img_np, sigma=[6, 12, 12])
    # fixed_mask_bin = fixed_mask > np.quantile(fixed_mask, 0.7)
    # fixed_img_np = fixed_img_np * fixed_mask_bin

    if args.fixed_n5 == args.moving_n5:
        logging.info("Same n5 for fixed and moving image, not doing registration")
        moving_img_np = fixed_img_np

    else:
        moving_img_np = read_volume(args.moving_n5, args.moving_key)
        # moving_mask = gaussian(moving_img_np, sigma=[6, 12, 12])
        # moving_mask_bin = moving_mask > np.quantile(moving_mask, 0.7)
        # moving_img_np = moving_img_np * moving_mask_bin

        fixed_img_np = fixed_img_np.astype(np.float32)
        moving_img_np = moving_img_np.astype(np.float32)

        logging.info("Start registration")

        # Dirty hack for the sample size mismatch
        # For no apparent reason the size of the light samples seems to be ~20% larger than the EM
        moving_resolution = get_attrs(args.moving_n5, args.moving_key)["resolution"]
        moving_resolution = [r * 0.9 for r in moving_resolution]

        moving_img_np = elastix_segm_rigid_alignment(
            fixed_img_np,
            get_attrs(args.fixed_n5, args.fixed_key)["resolution"],
            moving_img_np,
            moving_resolution,
            log_dir,
        )

    logging.info("Write results")
    attributes = dict(get_attrs(args.moving_n5, args.moving_key))
    attributes["resolution"] = get_attrs(args.fixed_n5, args.fixed_key)["resolution"]

    write_volume(
        args.output_n5,
        moving_img_np,
        args.output_key,
        chunks=(128, 512, 512),
        attrs=attributes,
    )


if __name__ == "__main__":
    main()
