import os
import json
import logging
import mobie
from matchmaker.n5_utils import get_attrs


def update_default_view(dataset_json_path, new_segmentation_name):
    """
    Update the name and sources for the segmentation in the 'default' view
    in a MoBIE dataset.json file.

    Args:
        dataset_json_path (str): Path to the dataset.json file.
        new_segmentation_name (str): New segmentation name to set in the default view.
    """
    if not os.path.exists(dataset_json_path):
        raise FileNotFoundError(f"Could not find: {dataset_json_path}")

    with open(dataset_json_path, "r") as f:
        data = json.load(f)

    views = data.get("views", {})
    default_view = views.get("default", {})

    source_displays = default_view.get("sourceDisplays", [])
    for display in source_displays:
        if "segmentationDisplay" in display:
            display["segmentationDisplay"]["name"] = new_segmentation_name
            display["segmentationDisplay"]["sources"] = [new_segmentation_name]

    # Save the updated dataset.json
    with open(dataset_json_path, "w") as f:
        json.dump(data, f, indent=2)

    logging.info(f"Updated default view to use segmentation: '{new_segmentation_name}'")


def create_mobie_project(
    fixed_input_path,
    fixed_key,
    moving_input_path,
    moving_key,
    output_dir,
    dataset_name
):
    """
    Create initial MoBIE project with the fixed and moving images.

    Args:
        fixed_input_path (str): Path to the fixed image n5 file.
        fixed_key (str): Key to the fixed image data in the n5 file.
        moving_input_path (str): Path to the moving image n5 file.
        moving_key (str): Key to the moving image data in the n5 file.
        output_dir (str): Directory where the MoBIE project should be saved.
    """

    # export fixed image to MoBIE
    fixed_file_name = os.path.splitext(os.path.basename(fixed_input_path))[0]
    export_to_mobie(
        fixed_input_path,
        fixed_key,
        output_dir,
        dataset_name,
        segmentation_name=f"{fixed_file_name}_original",
        menu_name="fixed",
    )

    # export moving image to MoBIE
    moving_file_name = os.path.splitext(os.path.basename(moving_input_path))[0]
    export_to_mobie(
        moving_input_path,
        moving_key,
        output_dir,
        dataset_name,
        segmentation_name=f"{moving_file_name}_original",
        menu_name="moving",
    )
    logging.info("Created initial MoBIE project with fixed and moving images.")


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


def main():
    input_path = "../examples/CLI_test/platy1_muscles_stardist_fixed_prealigned.n5"
    input_key = "seg"
    output_dir = "../examples/data/test"

    file_name = os.path.splitext(os.path.basename(input_path))[0]

    export_to_mobie(
        input_path,
        input_key,
        output_dir,
        dataset_name="platy1_muscles_stardist",
        segmentation_name=f"{file_name}_prealigned",
        menu_name="fixed",
    )
    print(f"MoBIE project created at {output_dir}/mobie_project")


if __name__ == "__main__":
    main()
