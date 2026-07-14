import os
import logging
import click
from shutil import rmtree
import pandas as pd
import numpy as np
from elf.io import open_file
import mobie

from mobie import add_segmentation
from mobie.import_data import import_segmentation
from mobie.metadata import read_dataset_metadata
from mobie.utils import get_data_key
from mobie.tables import compute_default_table

from image_matchmaker.utils import read_volume, write_volume, get_attrs, setup_logging


def instance_to_semantic(input_path, input_key, output_key):
    """
    Convert instance segmentation (.n5) to semantic segmentation (.n5).
    """

    instance_seg = read_volume(input_path, input_key)
    semantic_seg = (instance_seg > 0).astype(np.uint8)

    attrs = get_attrs(input_path, input_key)

    write_volume(input_path, semantic_seg, output_key, chunks=(128, 512, 512), attrs=attrs)


def add_to_mobie(input_path, input_key, mobie_folder, dataset_name, segmentation_name, menu_name):
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

    # Set parameters for MOBIE
    resolution = get_attrs(input_path, input_key)["resolution"]
    chunks = (64, 64, 64)
    scale_factors = 4 * [[2, 2, 2]]

    add_segmentation(
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


def check_consistency(table_path, seg_path, key):
    tab = pd.read_csv(table_path, sep="\t")
    tab_max_id = tab["label_id"].max().item()

    with open_file(seg_path, "r") as f:
        seg_max_id = f[key].attrs["maxId"]

    assert tab_max_id == seg_max_id, f"{tab_max_id}, {seg_max_id}"
    print("Done!")


def update_in_mobie(input_path, input_key, mobie_folder, dataset_name, segmentation_name, menu_name):

    ds_folder = os.path.join(mobie_folder, dataset_name)
    metadata = read_dataset_metadata(ds_folder)
    seg_path = metadata["sources"][segmentation_name]["segmentation"]["imageData"]["ome.zarr"][
        "relativePath"
    ]
    seg_path = os.path.join(ds_folder, seg_path)

    # Set parameters for MOBIE
    resolution = get_attrs(input_path, input_key)["resolution"]
    chunks = (64, 64, 64)
    scale_factors = 4 * [[2, 2, 2]]

    rmtree(seg_path)

    max_jobs = 8
    tmp_folder = f"tmp_{dataset_name}_{segmentation_name}"

    import_segmentation(
        input_path,
        input_key,
        seg_path,
        resolution,
        scale_factors,
        chunks,
        tmp_folder=tmp_folder,
        file_format="ome.zarr",
        max_jobs=max_jobs,
        target="local",
    )

    table_folder = os.path.join(ds_folder, "tables", segmentation_name)
    table_path = os.path.join(table_folder, "default.tsv")
    os.makedirs(table_folder, exist_ok=True)
    key = get_data_key(file_format="ome.zarr", scale=0, path=seg_path)
    compute_default_table(
        seg_path,
        key,
        table_path,
        resolution,
        tmp_folder=tmp_folder,
        target="local",
        max_jobs=max_jobs,
    )

    check_consistency(table_path, seg_path, key)


def export_to_mobie(input_path, input_key, mobie_folder, dataset_name, segmentation_name, menu_name):
    """
    Export a segmentation to a MoBIE project.

    Adds the segmentation to the dataset, creating the dataset if it does not
    exist yet, or updating it (delete and re-upload) if a segmentation with the
    same name is already present.

    Parameters
    ----------
    input_path : str
        Path to the input ``.n5`` container holding the segmentation.
    input_key : str
        Dataset key to read.
    mobie_folder : str
        Root folder of the MoBIE project.
    dataset_name : str
        Name of the MoBIE dataset.
    segmentation_name : str
        Name of the segmentation source in MoBIE.
    menu_name : str
        MoBIE menu group the source is placed under.
    """
    metadata = mobie.metadata.read_dataset_metadata(os.path.join(mobie_folder, dataset_name))
    if not metadata:
        add_to_mobie(input_path, input_key, mobie_folder, dataset_name, segmentation_name, menu_name)
        return

    if segmentation_name in metadata["sources"]:
        logging.info(f"Segmentation already exists. Updating (delete and re-upload) segmentation: {segmentation_name}")
        update_in_mobie(input_path, input_key, mobie_folder, dataset_name, segmentation_name, menu_name)
    else:
        add_to_mobie(input_path, input_key, mobie_folder, dataset_name, segmentation_name, menu_name)


@click.command()
@click.option("-i", "--input_path", required=True, help="Input .n5 file")
@click.option("-k", "--input_key", required=True, help="Input key")
@click.option("-t", "--input_type", required=True, help="Fixed or moving image?")
@click.option("-s", "--semantic_seg", is_flag=True, default=False, help="semantic or instance?")
@click.option("-d", "--dataset_name", required=True, help="Name of the MoBIE dataset")
@click.option("-o", "--output_dir", required=True, help="Output directory")
def main(input_path, input_key, input_type, semantic_seg, dataset_name, output_dir):

    setup_logging(output_dir, "mobie_export.log")

    logging.info(f"MoBIE upload semantic segmentation: {semantic_seg}")
    if semantic_seg:
        output_key = f"{input_key}_binary"
        instance_to_semantic(input_path, input_key, output_key)
        input_key = output_key

    file_name = os.path.splitext(os.path.basename(input_path))[0]
    logging.info(f"Start uploading {file_name}/{input_key} to MoBIE ...")

    mobie_folder = f"{output_dir}/mobie_project"
    if not os.path.exists(mobie_folder):
        os.makedirs(mobie_folder)

    export_to_mobie(
        input_path,
        input_key,
        mobie_folder,
        dataset_name=dataset_name,
        segmentation_name=f"{file_name}_{input_key}",
        menu_name=input_type,
    )
    logging.info(f"MoBIE project created/updated at {output_dir}/mobie_project")


if __name__ == "__main__":
    main()
