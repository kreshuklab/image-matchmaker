import itk
import click
import logging
import numpy as np
from pathlib import Path

from utils import (setup_logging, read_volume, write_volume, get_attrs, rotate_img,
                    read_transform_dict, apply_transform_chanwise, plot_three_slices,
                    plot_overlay,)


@click.command()
@click.option("-fi", "--fixed_path", required=True, help="Fixed input .n5 file")
@click.option("-fk", "--fixed_key", required=True, help="Fixed input key")
@click.option("-mi", "--moving_path", required=True, help="Moving input .n5 file")
@click.option("-mk", "--moving_key", required=True, help="Moving input key")
@click.option("-o", "--output_dir", required=True, help="Output directory")
@click.option("-ok", "--output_key", required=True, help="Output key for the regsitered moving image",)
@click.option("-pm", "--parameter_map_path", required=True, help="Path to the parameter map",)
@click.option("-pt", "--prealignment_transform_path", required=True, help="Prealignment transform path",)
@click.option("-io", "--interpolation_order", default=0, help="Order of interpolation",)
def apply_transform(fixed_path, fixed_key, moving_path, moving_key, output_dir, output_key,
                    parameter_map_path, prealignment_transform_path, interpolation_order):

    log_dir = Path(output_dir)
    log_dir.mkdir(exist_ok=True)

    setup_logging(log_dir, "apply_transform.log")

    logging.info("Read image file")
    fixed_img_np = read_volume(fixed_path, fixed_key).astype(np.float32)
    fixed_resolution = get_attrs(fixed_path, fixed_key)["resolution"]

    moving_img_np = read_volume(moving_path, moving_key).astype(np.float32)
    moving_resolution = get_attrs(moving_path, moving_key)["resolution"]

    if moving_img_np.ndim == 3:
        moving_img_np = moving_img_np[None, ...]
        chunks = (128, 512, 512)
    else:
        chunks = (1, 128, 512, 512)

    logging.info("Start transformation")

    logging.info(f"Read parameter file {parameter_map_path}")
    parameter_object = itk.ParameterObject.New()
    parameter_object.ReadParameterFile(parameter_map_path)
    parameter_object.SetParameter("FinalBSplineInterpolationOrder", str(interpolation_order))
    logging.info(parameter_object)
    result_img_np = apply_transform_chanwise(parameter_object, moving_img_np, moving_resolution)
    logging.info(f"Result image shape {result_img_np.shape}")

    result_img_np = np.squeeze(result_img_np)

    logging.info("Plot transformed image")
    plot_three_slices(result_img_np, save_path=log_dir / "deformable_pointset_transformed_moving.png")
    plot_overlay(
        fixed_img_np,
        result_img_np,
        log_dir / f"deformable_pointset_transformed_overlay.png",
    )

    if prealignment_transform_path is not None:
        logging.info("Rotate image using prealignment transform")
        prealignment_transform = read_transform_dict(prealignment_transform_path)["fixed_prealignment"]
        T_fixed, output_shape = prealignment_transform["matrix"], prealignment_transform["output_shape"]

        fixed_img_np = rotate_img(fixed_img_np, T_fixed, output_shape=output_shape)
        result_img_np = rotate_img(result_img_np, T_fixed, output_shape=output_shape)

        logging.info("Plot transformed images after pre-alignment")
        plot_three_slices(result_img_np, save_path=log_dir / "deformable_pointset_transformed_moving_prealigned.png")
        plot_overlay(
            fixed_img_np,
            result_img_np,
            log_dir / f"deformable_pointset_transformed_prealigned_overlay.png",
        )

    logging.info("Write results")
    attributes = dict(get_attrs(moving_path, moving_key))
    resolution = [float(res) for res in parameter_object.GetParameter(0, "Spacing")]
    attributes["resolution"] = resolution

    write_volume(moving_path, result_img_np, output_key, chunks=chunks, attrs=attributes)


if __name__ == "__main__":
    apply_transform()
