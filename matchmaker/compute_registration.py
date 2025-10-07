import os
import sys
import click
import logging
from matchmaker.mobie_export import create_mobie_project
from matchmaker.prealignment import run_prealignment
from matchmaker.align_rigid_elastix import run_rigid_alignment


@click.command()
@click.option("-fi", "--fixed_path", required=True, help="Fixed input .n5 file")
@click.option("-fk", "--fixed_key", required=True, help="Fixed input key")
@click.option("-mi", "--moving_path", required=True, help="Moving input .n5 file")
@click.option("-mk", "--moving_key", required=True, help="Moving input key")
@click.option("-o", "--output_dir", required=True, help="Output directory")
@click.option("-m", "--mobie_export", required=False, is_flag=True, help="MoBIE export")  # TODO: unplug MoBIE export
def main(fixed_path, fixed_key, moving_path, moving_key, output_dir, mobie_export):
    """
    Main function to perform registration of moving image to fixed image.

    This function orchestrates the sequence of steps required to register a moving image to a fixed image.
    It handles the creation of necessary directories, configures logging, and invokes prealignment and 
    rigid alignment functions. Optionally, it can create a MoBIE project for visualization.

    Args:
        fixed_path (str): Path to the fixed input .n5 file.
        fixed_key (str): Key to the fixed image data in the .n5 file.
        moving_path (str): Path to the moving input .n5 file.
        moving_key (str): Key to the moving image data in the .n5 file.
        output_dir (str): Directory where the results should be saved.
        mobie_export (bool): Flag indicating whether to export results to a MoBIE project.
    """

    if not os.path.exists(f"{output_dir}/plots"):
        os.makedirs(f"{output_dir}/plots")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{output_dir}/registration.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    dataset_name = "platy1_muscles_stardist"
    if mobie_export:
        create_mobie_project(
            fixed_path,
            fixed_key,
            moving_path,
            moving_key,
            output_dir,
            dataset_name
        )

    run_prealignment(
        fixed_path,
        fixed_key,
        moving_path,
        moving_key,
        output_dir,
        mobie_export,
        dataset_name,
    )

    fixed_prealigned_path = f"{output_dir}/{os.path.splitext(os.path.basename(fixed_path))[0]}_prealigned.n5"
    moving_prealigned_path = f"{output_dir}/{os.path.splitext(os.path.basename(moving_path))[0]}_prealigned.n5"

    run_rigid_alignment(
        fixed_prealigned_path,
        fixed_key,
        moving_prealigned_path,
        moving_key,
        output_dir,
        mobie_export,
        dataset_name,
    )

    # fixed_rigid_aligned_path = f"{output_dir}/{os.path.splitext(os.path.basename(fixed_path))[0]}_rigid_aligned.n5"
    # moving_rigid_aligned_path = f"{output_dir}/{os.path.splitext(os.path.basename(moving_path))[0]}_rigid_aligned.n5"

    # run_cpd(...)


if __name__ == "__main__":
    main()


# python compute_registration.py -fi ../examples/data/platy1_muscles_stardist_fixed.n5 -fk seg 
# -mi ../examples/data/platy1_muscles_stardist_moving.n5 -mk seg -o ../examples/data/test -m