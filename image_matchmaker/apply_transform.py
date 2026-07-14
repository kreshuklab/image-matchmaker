import itk
import json
import click
import logging
import numpy as np
import tifffile as tiff
from pathlib import Path

from image_matchmaker.utils import (
    setup_logging,
    read_volume,
    write_volume,
    get_attrs,
    rotate_img,
    read_transform_dict,
    apply_transform_chanwise,
    plot_three_slices,
    plot_overlay,
)


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


def save_data(data, output_path, output_key=None, **kwargs):
    logging.info("Write results")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if output_path.endswith((".tif", ".tiff")):
        tiff.imwrite(output_path, data)
    elif output_path.endswith(".n5"):
        assert output_key
        write_volume(output_path, data, output_key, **kwargs)
    else:
        raise NotImplementedError


def apply_transform(moving_img, moving_resolution, parameter_object, interpolation_order,
                    T_fixed=None, output_shape=None):
    """
    Apply a (registration) transform to a moving image.

    Warps ``moving_img`` channel-wise using the Elastix ``parameter_object``. If a
    pre-alignment transform ``T_fixed`` is given, the warped image is additionally
    rotated into the pre-alignment space.

    Parameters
    ----------
    moving_img : numpy.ndarray
        Image to warp.
    moving_resolution : sequence of float
        Voxel spacing of ``moving_img``.
    parameter_object : itk.ParameterObject
        Elastix transform parameters to apply.
    interpolation_order : int
        Final B-spline interpolation order (use ``0`` for label masks).
    T_fixed : numpy.ndarray, optional
        Pre-alignment transform; if given, ``output_shape`` is required.
    output_shape : sequence of int, optional
        Output shape for the pre-alignment rotation.

    Returns
    -------
    warped : numpy.ndarray
        The warped image.
    warp_prealigned : numpy.ndarray or None
        The warped image in pre-alignment space, or ``None`` if ``T_fixed`` was
        not given.
    """
    if T_fixed is not None:
        assert output_shape is not None

    logging.info(f"Set interpolation order {interpolation_order}")
    parameter_object.SetParameter("FinalBSplineInterpolationOrder", str(interpolation_order))

    warped = apply_transform_chanwise(parameter_object, moving_img, moving_resolution)
    logging.info(f"transformed image shape {warped.shape}")

    warped = np.squeeze(warped)

    if T_fixed is not None:
        logging.info("Rotate moving image using prealignment transform")
        warp_prealigned = rotate_img(warped, T_fixed, output_shape=output_shape)
    else:
        warp_prealigned = None

    return warped, warp_prealigned


@click.command()
@click.option("-mp", "--moving_path", required=True, help="Path to moving input")
@click.option("-mk", "--moving_key", required=True, help="Key of moving input")
@click.option("-mr", "--moving_resolution", required=True, help="Resolution of moving input")
@click.option("-op", "--output_path", required=True, help="Path to save warped image")
@click.option("-ok", "--output_key", required=True, help="Key of moving output")
@click.option("-io", "--interpolation_order", required=True, help="Order of interpolation")
@click.option("-ld", "--log_dir", required=True, help="Log directory")
@click.option("-pm", "--parameter_map_path", required=True, help="Path to the parameter map",)
@click.option("-pt", "--prealignment_transform_path", default=None, help="Prealignment transform path",)
@click.option("-fi", "--fixed_path", default=None, help="Fixed input .n5 file")
@click.option("-fk", "--fixed_key", default=None, help="Fixed input key")
@click.option("-vb", "--verbose", is_flag=True, default=False, help="Show verbose logs")
def apply_transforms(
    moving_path,
    moving_key,
    moving_resolution,
    output_path,
    output_key,
    interpolation_order,
    log_dir,
    parameter_map_path,
    prealignment_transform_path,
    fixed_path,
    fixed_key,
    verbose,
):
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    setup_logging(log_dir, "apply_transform.log")

    logging.info(f"Read parameter file {parameter_map_path}")
    parameter_object = itk.ParameterObject.New()
    parameter_object.ReadParameterFile(parameter_map_path)
    if verbose:
        logging.info(parameter_object)

    if prealignment_transform_path:
        logging.info("Read prealignment transform")
        prealignment_transform = read_transform_dict(prealignment_transform_path)["fixed_prealignment"]
        T_fixed, output_shape = prealignment_transform["matrix"], prealignment_transform["output_shape"]
    else:
        logging.info("Process without prealignment transform")
        T_fixed, output_shape = None, None

    if fixed_path:
        logging.info("Read fixed image")
        fixed_img = load_data(fixed_path, fixed_key)
        if T_fixed is not None:
            logging.info("Rotate fixed image using prealignment transform")
            fixed_prealigned = rotate_img(fixed_img, T_fixed, output_shape=output_shape)

    logging.info(f"Start processing moving image: {moving_path}")
    moving_resolution = json.loads(moving_resolution)

    moving_name = Path(moving_path).stem
    logging.info("Read moving image")
    moving_img = load_data(moving_path, moving_key)
    if moving_path.endswith(".n5"):
        if list(moving_resolution) != get_attrs(moving_path, moving_key)["resolution"]:
            raise ValueError("Moving resolution from config is different from n5 file")

    if moving_img.ndim == 3:
        moving_img = moving_img[None, ...]
        chunks = (128, 512, 512)
    else:
        chunks = (1, 128, 512, 512)

    logging.info("Start transformation")
    warped, warp_prealigned = apply_transform(
        moving_img,
        moving_resolution,
        parameter_object,
        interpolation_order,
        T_fixed=T_fixed,
        output_shape=output_shape,
    )

    resolution = [float(res) for res in parameter_object.GetParameter(0, "Spacing")]

    save_attrs = {}
    if moving_path.endswith(".n5"):
        attributes = dict(get_attrs(moving_path, moving_key))
        attributes["resolution"] = resolution
        save_attrs["chunks"] = chunks
        save_attrs["attrs"] = attributes

    logging.info("Plot warped image")
    plot_three_slices(warped, save_path=log_dir / f"{moving_name}_warped.png")
    if fixed_path:
        logging.info("Plot overlay image")
        plot_overlay(fixed_img, warped, log_dir / f"{moving_name}_warped_overlay.png",)

    if T_fixed is not None:
        logging.info("Plot warped moving image after pre-alignment")
        plot_three_slices(warp_prealigned, save_path=log_dir / f"{moving_name}_warp_prealigned.png")

        if fixed_path:
            logging.info("Plot overlay image after pre-alignment")
            plot_overlay(fixed_prealigned, warp_prealigned, log_dir / f"{moving_name}_warp_prealigned_overlay.png",)

        save_data(warp_prealigned, output_path, output_key=output_key, **save_attrs)

    else:
        save_data(warped, output_path, output_key=output_key, **save_attrs)


if __name__ == "__main__":
    apply_transforms()
