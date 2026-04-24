import itk
import click
import logging
import numpy as np
import tifffile as tiff
from pathlib import Path

from utils import (setup_logging, read_volume, write_volume, get_attrs, rotate_img,
                    read_transform_dict, apply_transform_chanwise, plot_three_slices,
                    plot_overlay,)


def load_data(path, key=None):
    if path.endswith(".n5"):
        assert key
        data = read_volume(path, key)
    elif path.endswith((".tif", ".tiff")):
        data = tiff.imread(path)
        if data.ndim == 2:
            data = data[None, ...]
    else:
        raise NotImplementedError

    return data.astype(np.float32)


def save_data(data, input_path, output_dir, output_key=None, **kwargs):
    if input_path.endswith((".tif", ".tiff")):
        output_path = f"{output_dir}/{Path(input_path).stem}_transformed.tif"
        print("output_path", output_path)
        tiff.imwrite(output_path, data)
    elif input_path.endswith(".n5"):
        assert output_key
        write_volume(input_path, data, output_key, **kwargs)
    else:
        raise NotImplementedError


@click.command()
@click.option("-fi", "--fixed_path", default=None, help="Fixed input .n5 file")
@click.option("-fk", "--fixed_key", default=None, help="Fixed input key")
@click.option("-mi", "--moving_path", required=True, help="Moving input .n5 file")
@click.option("-mk", "--moving_key", required=True, help="Moving input key")
@click.option("-mr", "--moving_resolution", type=float, nargs=3, required=True, help="Moving resolution")
@click.option("-o", "--output_dir", required=True, help="Output directory")
@click.option("-ok", "--output_key", default=None, help="Output key for the regsitered moving image",)
@click.option("-pm", "--parameter_map_path", required=True, help="Path to the parameter map",)
@click.option("-pt", "--prealignment_transform_path", default=None, help="Prealignment transform path",)
@click.option("-io", "--interpolation_order", default=0, help="Order of interpolation",)
def apply_transform(fixed_path, fixed_key, moving_path, moving_key, moving_resolution,
                    output_dir, output_key, parameter_map_path, prealignment_transform_path,
                    interpolation_order):

    log_dir = Path(output_dir)
    log_dir.mkdir(exist_ok=True)

    setup_logging(log_dir, "apply_transform.log")

    if fixed_path:
        logging.info("Read fixed image")
        fixed_img_np = load_data(fixed_path, fixed_key)

    logging.info("Read moving image")
    moving_img_np = load_data(moving_path, moving_key)
    if moving_path.endswith(".n5"):
        if list(moving_resolution) != get_attrs(moving_path, moving_key)["resolution"]:
            raise ValueError("Moving resolution from config is different from n5 file")

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
    if fixed_path:
        logging.info("Plot overlay image")
        plot_overlay(
            fixed_img_np,
            result_img_np,
            log_dir / f"deformable_pointset_transformed_overlay.png",
        )

    if prealignment_transform_path:
        logging.info("Rotate moving image using prealignment transform")
        prealignment_transform = read_transform_dict(prealignment_transform_path)["fixed_prealignment"]
        T_fixed, output_shape = prealignment_transform["matrix"], prealignment_transform["output_shape"]
        result_img_np = rotate_img(result_img_np, T_fixed, output_shape=output_shape)

        logging.info("Plot transformed moving images after pre-alignment")
        plot_three_slices(result_img_np, save_path=log_dir / "deformable_pointset_transformed_moving_prealigned.png")

        if fixed_path:
            logging.info("Rotate fixed image using prealignment transform")
            fixed_img_np = rotate_img(fixed_img_np, T_fixed, output_shape=output_shape)

            logging.info("Plot overlay images after pre-alignment")
            plot_overlay(
                fixed_img_np,
                result_img_np,
                log_dir / f"deformable_pointset_transformed_prealigned_overlay.png",
            )

    logging.info("Write results")
    resolution = [float(res) for res in parameter_object.GetParameter(0, "Spacing")]

    save_attrs = {}
    if moving_path.endswith(".n5"):
        attributes = dict(get_attrs(moving_path, moving_key))
        attributes["resolution"] = resolution
        save_attrs["chunks"] = chunks
        save_attrs["attrs"] = attributes

    save_data(result_img_np, moving_path, output_dir, output_key=output_key, **save_attrs)


if __name__ == "__main__":
    apply_transform()
