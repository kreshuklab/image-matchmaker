import sys
import click
import logging
import numpy as np
import tifffile as tif
from pathlib import Path

from matchmaker.utils import (read_volume, write_volume, plot_three_slices, convert_to_int)


def preprocess_tif_input(input_path, output_path, output_key, log_dir, x_res, y_res, z_res):
    logging.info(f"Reading input image from {input_path}, resolution {z_res}, {y_res}, {x_res}")

    image = tif.imread(input_path)
    logging.info(f"Image size: {image.shape}, dtype {image.dtype}, ndim {image.ndim}")
    logging.info(f"Value range: min={image.min()}, max={image.max()}")

    assert (image.ndim == 3) or (
        image.ndim == 4
    ), f"Currently pipeline only works with ZYX or CZYX images, input has {image.ndim} dimensions"

    image = convert_to_int(image)    
    logging.info(f"Writing output image to {output_path}")

    attrs = {"resolution": [z_res, y_res, x_res]}

    print(log_dir / f"input_image_{Path(input_path).stem}.png")
    if image.ndim == 4:
        chunks = (1, 128, 512, 512)
        for chan in range(image.shape[0]):
            plot_three_slices(
                image[chan],
                log_dir / f"input_image_{Path(input_path).stem}_{chan}.png",
                cmap="gnuplot2_r",
            )

    else:
        chunks = (128, 512, 512)
        plot_three_slices(
            image,
            log_dir / f"input_image_{Path(input_path).stem}.png",
            cmap="gnuplot2_r",
        )

    write_volume(output_path, image, output_key, chunks=chunks, attrs=attrs)


def preprocess_n5_input(input_path, input_key, output_path, output_key, log_dir, x_res, y_res, z_res):
    logging.info(f"Reading input image from {input_path}, resolution {z_res}, {y_res}, {x_res}")
    print(input_key)
    image = read_volume(input_path, input_key)
    logging.info(f"Image size: {image.shape}, dtype {image.dtype}, ndim {image.ndim}")
    logging.info(f"Value range: min={image.min()}, max={image.max()}")

    assert (image.ndim == 3) or (
        image.ndim == 4
    ), f"Currently pipeline only works with ZYX or CZYX images, input has {image.ndim} dimensions"

    image = convert_to_int(image)

    logging.info(f"Writing output image to {output_path}")

    attrs = {"resolution": [z_res, y_res, x_res]}

    print(log_dir / f"input_image_{Path(input_path).stem}.png")
    if image.ndim == 4:
        chunks = (1, 128, 512, 512)
        for chan in range(image.shape[0]):
            plot_three_slices(
                image[chan],
                log_dir / f"input_image_{Path(input_path).stem}_{chan}.png",
                cmap="gnuplot2_r",
            )

    else:
        chunks = (128, 512, 512)
        plot_three_slices(
            image,
            save_path=log_dir / f"input_image_{Path(input_path).stem}.png",
            cmap="gnuplot2_r",
        )

    write_volume(output_path, image, output_key, chunks=chunks, attrs=attrs)


@click.command()
@click.option("-in", "--input_path", required=True, help="Path of the input image in .tif format")
@click.option("-ink", "--input_key", required=False, default=None, help="Key of the input image in .n5 format")
@click.option("-out", "--output_path", required=True, help="Path of the output .n5 file")
@click.option("-outk", "--output_key", required=True, help="Key in the output .n5 file")
@click.option("-log", "--log_dir", required=True, help="Log directory")
@click.option("--x_res", required=False, type=float, default=1.0, help="The image is interpreted as (C)ZYX")
@click.option("--y_res", required=False, type=float, default=1.0, help="The image is interpreted as (C)ZYX")
@click.option("--z_res", required=False, type=float, default=1.0, help="The image is interpreted as (C)ZYX")
def main(input_path, input_key, output_path, output_key, log_dir, x_res, y_res, z_res):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/raw_to_n5.log", mode="a"),
            logging.StreamHandler(sys.stdout)
        ],
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_dir = Path(log_dir)

    logging.info("Checking input image file format...")
    ext = Path(input_path).suffix.lower()
    logging.info(f"Input image file extension: {ext}")

    assert ext in [".tif", ".tiff", ".n5"], f"Unsupported input image format: {ext}"

    if ext in [".tif", ".tiff"]:
        logging.info(f"Input image is in {ext} format, reading with tifffile")
        preprocess_tif_input(input_path, output_path, output_key, log_dir, x_res, y_res, z_res)
        return

    if ext == ".n5":
        logging.info(f"Input image is in {ext} format, reading with z5py")
        preprocess_n5_input(input_path, input_key, output_path, output_key, log_dir, x_res, y_res, z_res)


if __name__ == "__main__":
    main()
