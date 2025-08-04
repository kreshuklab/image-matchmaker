import os
import numpy as np
import click
from matchmaker.prealignment import run_prealignment
from matchmaker.mobie_export import create_mobie_project, export_to_mobie


@click.command()
@click.option("-fi", "--fixed_path", required=True, help="Fixed input .n5 file")
@click.option("-fk", "--fixed_key", required=True, help="Fixed input key")
@click.option("-mi", "--moving_path", required=True, help="Moving input .n5 file")
@click.option("-mk", "--moving_key", required=True, help="Moving input key")
@click.option("-o", "--output_dir", required=True, help="Output directory")
@click.option("-m", "--mobie_export", required=False, is_flag=True, help="MoBIE export")
def main(fixed_path, fixed_key, moving_path, moving_key, output_dir, mobie_export):
    # fixed_input_path = "../examples/data/platy1_muscles_stardist_fixed.n5"
    # fixed_input_key = "seg"
    # moving_input_path = "../examples/data/platy1_muscles_stardist_moving.n5"
    # moving_input_key = "seg"

    # output_dir = "../examples/data/test"
    if not os.path.exists(f"{output_dir}/plots"):
        os.makedirs(f"{output_dir}/plots")

    if mobie_export:
        create_mobie_project(
            fixed_path,
            fixed_key,
            moving_path,
            moving_key,
            output_dir,
        )

    fixed_prealigned, moving_prealigned = run_prealignment(
        fixed_path,
        fixed_key,
        moving_path,
        moving_key,
        output_dir,
        mobie_export,
    )


if __name__ == "__main__":
    main()
