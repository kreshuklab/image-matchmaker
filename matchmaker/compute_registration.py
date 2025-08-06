import os
import click
from matchmaker.mobie_export import create_mobie_project
from matchmaker.prealignment import run_prealignment
from matchmaker.align_rigid_elastix import run_rigid_alignment


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

    run_prealignment(
        fixed_path,
        fixed_key,
        moving_path,
        moving_key,
        output_dir,
        mobie_export,
    )

    fixed_prealigned_path = f"{output_dir}/{os.path.splitext(os.path.basename(fixed_path))[0]}_prealigned.n5"
    moving_prealigned_path = f"{output_dir}/{os.path.splitext(os.path.basename(moving_path))[0]}_prealigned.n5"

    run_rigid_alignment(
        fixed_prealigned_path,
        fixed_key,
        moving_prealigned_path,
        moving_key,
        output_dir,
        mobie_export
    )

    # fixed_rigid_aligned_path = f"{output_dir}/{os.path.splitext(os.path.basename(fixed_path))[0]}_rigid_aligned.n5"
    # moving_rigid_aligned_path = f"{output_dir}/{os.path.splitext(os.path.basename(moving_path))[0]}_rigid_aligned.n5"

    # run_cpd(...)


if __name__ == "__main__":
    main()


# python compute_registration.py -fi ../examples/data/platy1_muscles_stardist_fixed.n5 -fk seg 
# -mi ../examples/data/platy1_muscles_stardist_moving.n5 -mk seg -o ../examples/data/test -m