import os
import logging
import click
import mobie
from matchmaker.n5_utils import get_attrs


def export_to_mobie(input_path, input_key, output_dir, dataset_name, segmentation_name, menu_name):
    """
    Export segmentation from n5 file to MoBIE project.

    Args:
        input_path (str): Path to the n5 file containing the segmentation.
        input_key (str): Key to the segmentation data in the n5 file.
        output_dir (str): Directory where the MoBIE project should be saved.
        dataset_name (str): Name of the MoBIE dataset.
        segmentation_name (str): Name of the segmentation in the MoBIE project.
        menu_name (str): Name of the menu in the MoBIE project.
    """
    if not os.path.exists(f"{output_dir}/mobie_project"):
        os.makedirs(f"{output_dir}/mobie_project")

    # Set parameters for MOBIE
    mobie_folder = f"{output_dir}/mobie_project"
    resolution = get_attrs(input_path, input_key)["resolution"]
    chunks = (64, 64, 64)
    scale_factors = 4 * [[2, 2, 2]]

    mobie.add_segmentation(
        input_path=input_path,
        input_key=input_key,
        root=mobie_folder,
        dataset_name=dataset_name,
        segmentation_name=segmentation_name,
        resolution=resolution,
        scale_factors=scale_factors,
        chunks=chunks,
        menu_name=menu_name,
        file_format="ome.zarr",
        is_default_dataset=True
    )
    logging.info(f"Added segmentation: {segmentation_name}")


@click.command()
@click.option("-i", "--input_path", required=True, help="Input .n5 file")
@click.option("-k", "--input_key", required=True, help="Input key")
@click.option("-t", "--input_type", required=True, help="Fixed or moving image?")
@click.option("-d", "--dataset_name", required=True, help="Name of the MoBIE dataset")
@click.option("-o", "--output_dir", required=True, help="Output directory")
def main(input_path, input_key, input_type, dataset_name, output_dir):
    # TODO: add logging

    file_name = os.path.splitext(os.path.basename(input_path))[0]
    # NOTE: maybe add specific segmentation name?
    export_to_mobie(
        input_path,
        input_key,
        output_dir,
        dataset_name=dataset_name,
        segmentation_name=f"{file_name}",
        menu_name=input_type,
    )
    print(f"MoBIE project created/updated at {output_dir}/mobie_project")


if __name__ == "__main__":
    main()
