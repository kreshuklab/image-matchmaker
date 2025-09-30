import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import click
import sys
import tifffile as tif

from matchmaker.data import create_point_cloud
from matchmaker.mobie_export import export_to_mobie, update_default_view
from matchmaker.transform_utils import get_transformation_matrix, rotate_img
from matchmaker.n5_utils import write_volume
from matchmaker.vis import plot_three_slices, plot_overlay
from pathlib import Path


@click.command()
@click.option("-in", "--input_path", required=True, help="Path of the input image in .tif forma")
@click.option("-out", "--output_path", required=True, help="Path of the output .n5 file")
@click.option("-outk", "--output_key", required=True, help="Key in the output .n5 file")
@click.option("-log", "--log_dir", required=True, help="Log directory")
@click.option("--x_res", required=False, default=1, help="The image is interpreted as (C)ZYX")
@click.option("--y_res", required=False, default=1, help="The image is interpreted as (C)ZYX")
@click.option("--z_res", required=False, default=1, help="The image is interpreted as (C)ZYX")
def main(input_path, output_path, output_key, log_dir, x_res, y_res, z_res):
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

    logging.info(f"Reading input image from {input_path}, resolution {z_res}, {y_res}, {x_res}")

    image = tif.imread(input_path)
    logging.info(f"Image size: {image.shape}, dtype {image.dtype}, ndim {image.ndim}")
    logging.info(f"Value range: min={image.min()}, max={image.max()}")

    assert (image.ndim == 3) or (image.ndim == 4), "Currently pipeline only works with ZYX or CZYX images, input has {image.ndim} dimensions"



    logging.info(f"Writing output image to  {input_path}")

    attrs = {"resolution": [z_res, y_res, x_res]}

    print(log_dir / f"input_image_{Path(input_path).stem}.png")
    if image.ndim == 4:
        chunks = (1, 128, 512, 512)
        for chan in range(image.shape[0]):
            plot_three_slices(image[chan], log_dir / f"input_image_{Path(input_path).stem}_{chan}.png")

    else:
        chunks = (128, 512, 512)
        plot_three_slices(image, log_dir / f"input_image_{Path(input_path).stem}.png")


    write_volume(output_path, image, output_key, chunks=chunks, attrs=attrs)



if __name__ == "__main__":
    main()
