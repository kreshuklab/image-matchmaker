# export results as mobie project
# menu 1: dataset
# menu 2: fixed_segmentation; then original, pre-aligned, ...
# menu 3: moving_segmentation; then original, pre-aligned, ...
import os
import mobie
# from mobie import add_segmentation


# def add_segmentation_to_mobie(input_path, input_key, seg_name):

#     resolution = get_attrs(input_path, input_key)["resolution"]
#     unit = "micrometer"
#     chunks = (64, 64, 64)
#     scale_factors = 4 * [[2, 2, 2]]

#     chunks = (128, 128, 128)
#     max_jobs = 8

#     add_segmentation(
#         input_path=input_path,
#         input_key=input_key,
#         root=MOBIE_FOLDER,
#         dataset_name=DS_NAME,
#         segmentation_name=seg_name,
#         resolution=resolution,
#         scale_factors=scale_factors,
#         chunks=chunks,
#         file_format="ome.zarr",
#         max_jobs=max_jobs,
#     )

# def process_segmentation(input_path, input_key, seg_name, scale):
#     metadata = read_dataset_metadata(os.path.join(MOBIE_FOLDER, DS_NAME))
#     if seg_name in metadata["sources"]:
#         update_segmentation(input_path, input_key, seg_name, scale)
#     else:
#         add_segmentation_to_mobie(input_path, input_key, seg_name, scale)


def create_mobie_project(input_path, input_key, output_dir):
    if not os.path.exists(f"{output_dir}/mobie_project"):
        os.makedirs(f"{output_dir}/mobie_project")

    # Set parameters for MOBIE
    mobie_folder = f"{output_dir}/mobie_project"
    dataset_name = "platy1_muscles_stardist"
    breakpoint()
    # resolution = get_attrs(input_path, input_key)["resolution"]
    chunks = (64, 64, 64)
    scale_factors = 4 * [[2, 2, 2]]

    mobie.add_segmentation(
        input_path=input_path,
        input_key=input_key,
        root=mobie_folder,
        dataset_name=dataset_name,
        segmentation_name="original",
        resolution=[1, 1, 1],
        scale_factors=scale_factors,
        chunks=chunks,
        menu_name="fixed",
        file_format="ome.zarr",
        is_default_dataset=True
    )


def main():
    input_path = "/Users/marei/git-repositories/matchmaker/examples/data/platy1_muscles_stardist_fixed.n5"
    input_key = "seg"
    output_dir = "/Users/marei/git-repositories/matchmaker/examples/data/test"

    create_mobie_project(input_path, input_key, output_dir)
    print(f"MoBIE project created at {output_dir}/mobie_project")


if __name__ == "__main__":
    main()
