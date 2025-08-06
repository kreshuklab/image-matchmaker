# export results as mobie project
# menu 1: dataset
# menu 2: fixed_segmentation; then original, pre-aligned, ...
# menu 3: moving_segmentation; then original, pre-aligned, ...
import os
import json
import mobie
# from mobie import add_segmentation


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

    print(f"Updated default view to use segmentation: '{new_segmentation_name}'")


def create_mobie_project(
    fixed_input_path,
    fixed_key,
    moving_input_path,
    moving_key,
    output_dir,
):

    # export fixed image to MoBIE
    fixed_file_name = os.path.splitext(os.path.basename(fixed_input_path))[0]
    export_to_mobie(
        fixed_input_path,
        fixed_key,
        output_dir,
        segmentation_name=f"{fixed_file_name}_original",
        menu_name="fixed",
    )

    # export moving image to MoBIE
    moving_file_name = os.path.splitext(os.path.basename(moving_input_path))[0]
    export_to_mobie(
        moving_input_path,
        moving_key,
        output_dir,
        segmentation_name=f"{moving_file_name}_original",
        menu_name="moving",
    )


def export_to_mobie(input_path, input_key, output_dir, segmentation_name, menu_name):
    if not os.path.exists(f"{output_dir}/mobie_project"):
        os.makedirs(f"{output_dir}/mobie_project")

    # Set parameters for MOBIE
    mobie_folder = f"{output_dir}/mobie_project"
    dataset_name = "platy1_muscles_stardist"  # TODO: specify as input
    # resolution = get_attrs(input_path, input_key)["resolution"]
    chunks = (64, 64, 64)
    scale_factors = 4 * [[2, 2, 2]]

    mobie.add_segmentation(
        input_path=input_path,
        input_key=input_key,
        root=mobie_folder,
        dataset_name=dataset_name,
        segmentation_name=segmentation_name,
        resolution=[1, 1, 1],
        scale_factors=scale_factors,
        chunks=chunks,
        menu_name=menu_name,
        file_format="ome.zarr",
        is_default_dataset=True
    )


def main():
    input_path = "/Users/marei/git-repositories/matchmaker/examples/CLI_test/platy1_muscles_stardist_fixed_prealigned.n5"
    input_key = "seg"
    output_dir = "/Users/marei/git-repositories/matchmaker/examples/data/test"

    file_name = os.path.splitext(os.path.basename(input_path))[0]

    export_to_mobie(input_path, input_key, output_dir, segmentation_name=f"{file_name}_prealigned", menu_name="fixed")
    print(f"MoBIE project created at {output_dir}/mobie_project")


if __name__ == "__main__":
    main()
