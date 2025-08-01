import os
import z5py
import numpy as np

from matchmaker.prealignment import prealignment_per_image
from matchmaker.mobie_export import export_to_mobie
from matchmaker.transform_utils import rotate_img
from matchmaker.vis import plot_three_slices, plot_overlay


def create_mobie_project(
        fixed_input_path,
        fixed_input_key,
        moving_input_path,
        moving_input_key,
        output_dir,
    ):

    # export fixed image to MoBIE
    fixed_file_name = os.path.splitext(os.path.basename(fixed_input_path))[0]
    export_to_mobie(
        fixed_input_path,
        fixed_input_key,
        output_dir,
        segmentation_name=f"{fixed_file_name}_original",
        menu_name="fixed",
    )

    # export moving image to MoBIE
    fixed_file_name = os.path.splitext(os.path.basename(moving_input_path))[0]
    export_to_mobie(
        moving_input_path,
        moving_input_key,
        output_dir,
        segmentation_name=f"{fixed_file_name}_original",
        menu_name="moving",
    )


def run_prealignment(
        fixed_input_path,
        fixed_input_key,
        moving_input_path,
        moving_input_key,
        output_dir,
        mobie_export=True,
        ):

    fixed_prealigned = prealignment_per_image(
        fixed_input_path,
        fixed_input_key,
        output_dir,
        mobie_export,
        image_type="fixed"
    )

    moving_prealigned = prealignment_per_image(
        moving_input_path,
        moving_input_key,
        output_dir,
        mobie_export,
        image_type="moving"
    )

    plot_overlay(
        fixed_prealigned,
        moving_prealigned,
        save_path=f"{output_dir}/plots/overlay_prealignment.png",
    )

    return fixed_prealigned, moving_prealigned


def main():
    fixed_input_path = "../examples/data/platy1_muscles_stardist_fixed.n5"
    fixed_input_key = "seg"
    moving_input_path = "../examples/data/platy1_muscles_stardist_moving.n5"
    moving_input_key = "seg"

    output_dir = "../examples/data/test"
    if not os.path.exists(f"{output_dir}/plots"):
        os.makedirs(f"{output_dir}/plots")

    create_mobie_project(
        fixed_input_path,
        fixed_input_key,
        moving_input_path,
        moving_input_key,
        output_dir,
    )

    fixed_prealigned, moving_prealigned = run_prealignment(
        fixed_input_path,
        fixed_input_key,
        moving_input_path,
        moving_input_key,
        output_dir,
        mobie_export=True,
    )


if __name__ == "__main__":
    main()
