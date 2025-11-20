import pandas as pd
from pathlib import Path

root_dir = f"{Path(workflow.basedir).resolve().parent}/"
print(f"working directory: {root_dir}")
workdir: root_dir
configfile: "examples/register_config_test.yaml"

print(config["fixed_image"])
print(config["moving_image"])

fixed_name = config["fixed_name"] if "fixed_name" in config else "fixed_image"
fixed_n5_path = f"{config['log_dir']}/{fixed_name}.n5"
moving_name = config["moving_name"] if "moving_name" in config else "moving_image"
moving_n5_path = f"{config['log_dir']}/{moving_name}.n5"
log_dir = config["log_dir"]
final_transform = config["final_transform_path"]

print(fixed_name, fixed_n5_path, moving_name, moving_n5_path)

# define global variables for n5 keys
raw_n5_key = config["input_key"] if "input_key" in config else "input"
prealignment_n5_key = "svd_prealignment"
rigid_alignment_n5_key = "rigid_alignment"

# define global MoBIE variables
fixed_type = "fixed"
moving_type = "moving"
dataset_name = config["mobie_dataset_name"]


rule all:
    input:
        preprocessing_check = f"{log_dir}/preprocessing.done",
        input_to_n5_fixed = fixed_n5_path,
        input_to_n5_moving = moving_n5_path,
        svd_prealignment = f"{fixed_n5_path}/{prealignment_n5_key}",
        svd_transform = f"{log_dir}/{prealignment_n5_key}/{prealignment_n5_key}_transform.json",
        rigid_alignment = f"{moving_n5_path}/{rigid_alignment_n5_key}",
        rigid_transform = f"{log_dir}/{rigid_alignment_n5_key}/TransformParameters.0.txt",
        mobie_fixed_input_check = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{fixed_name}_{raw_n5_key}.done",
        mobie_moving_input_check = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{moving_name}_{raw_n5_key}.done",
        mobie_fixed_prealign_check = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{fixed_name}_{prealignment_n5_key}.done",
        mobie_moving_prealign_check = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{moving_name}_{prealignment_n5_key}.done",
        mobie_moving_rigid_align_check = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{moving_name}_{rigid_alignment_n5_key}.done"


"""
Convert whatever is the input image format (supporting only .tif at the moment) to the internal pipeline's format
"""
rule input_to_n5:
    input:
        fixed_image_path = config["fixed_image"]["path"],
        moving_image_path = config["moving_image"]["path"]
    output:
        directory(f"{fixed_n5_path}/{raw_n5_key}/"),
        directory(f"{moving_n5_path}/{raw_n5_key}/"),
        fixed_image_n5 = directory(fixed_n5_path),
        moving_image_n5 = directory(moving_n5_path),
        preprocessing_check = f"{log_dir}/preprocessing.done"
    params:
        output_key = raw_n5_key
    log: f"{log_dir}/matchmaker.log"
    conda: "matchmaker_env"
    shell:
        f"python matchmaker/raw_to_n5.py --input_path {{input.fixed_image_path}} --input_key {{params.output_key}} --output_path {{output.fixed_image_n5}} --output_key {{params.output_key}} --log_dir {log_dir} --x_res {config['fixed_image']['x_res']} --y_res {config['fixed_image']['y_res']} --z_res {config['fixed_image']['z_res']};"
        f"python matchmaker/raw_to_n5.py --input_path {{input.moving_image_path}} --input_key {{params.output_key}} --output_path {{output.fixed_image_n5}} --output_key {{params.output_key}} --log_dir {log_dir} --x_res {config['moving_image']['x_res']} --y_res {config['moving_image']['y_res']} --z_res {config['moving_image']['z_res']};"
        f"touch {{output.preprocessing_check}};"


"""
Run pre-alignment with SVD
"""
rule SVD_prealignment:
    input:
        fixed_input_ds = f"{fixed_n5_path}/{raw_n5_key}",
        moving_input_ds = f"{moving_n5_path}/{raw_n5_key}",
        fixed_image_n5 = fixed_n5_path,
        moving_image_n5 = moving_n5_path,
        preprocessing_check = f"{log_dir}/preprocessing.done"
    output:
        directory(f"{fixed_n5_path}/{prealignment_n5_key}"),
        directory(f"{moving_n5_path}/{prealignment_n5_key}"),
        output_transform = f"{log_dir}/{prealignment_n5_key}/{prealignment_n5_key}_transform.json"
    params:
        input_key = raw_n5_key,
        output_key = prealignment_n5_key
    log: f"{log_dir}/matchmaker.log"
    conda: "matchmaker_env"
    shell:
        f"python matchmaker/prealignment.py --fixed_path {{input.fixed_image_n5}} --fixed_key {{params.input_key}} --moving_path {{input.moving_image_n5}} --moving_key {{params.input_key}} --output_dir {log_dir}/{{params.output_key}}  --output_key {{params.output_key}} --output_transform_path {{output.output_transform}};"


"""
Run rigid alignment with elastix
"""
rule rigid_alignment:
    input:
        fixed_input_ds = f"{fixed_n5_path}/{prealignment_n5_key}",
        moving_input_ds = f"{moving_n5_path}/{prealignment_n5_key}",
        fixed_image_n5 = fixed_n5_path,
        moving_image_n5 = moving_n5_path
    output:
        directory(f"{moving_n5_path}/{rigid_alignment_n5_key}"),
        output_transform = f"{log_dir}/{rigid_alignment_n5_key}/TransformParameters.0.txt"
    params:
        input_key = prealignment_n5_key,
        output_key = rigid_alignment_n5_key
    log: f"{log_dir}/matchmaker.log"
    conda: "matchmaker_env"
    shell:
        f"python matchmaker/rigid_alignment_elastix.py --fixed_path {{input.fixed_image_n5}} --fixed_key {{params.input_key}} --moving_path {{input.moving_image_n5}} --moving_key {{params.input_key}} --output_dir {log_dir}/{{params.output_key}} --output_key {{params.output_key}};"


"""
Create mobie project with raw data
"""
rule create_mobie_project:
    input:
        fixed_input_ds = f"{fixed_n5_path}/{raw_n5_key}",
        moving_input_ds = f"{moving_n5_path}/{raw_n5_key}",
        fixed_image_n5 = fixed_n5_path,
        moving_image_n5 = moving_n5_path,
        preprocessing_check = f"{log_dir}/preprocessing.done"
    output:
        fixed_check = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{fixed_name}_{raw_n5_key}.done",
        moving_check = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{moving_name}_{raw_n5_key}.done"
    params:
        input_key = raw_n5_key
    log: f"{log_dir}/matchmaker.log"
    conda: "matchmaker_env"
    shell:
        f"python matchmaker/mobie_export.py --input_path {{input.fixed_image_n5}} --input_key {{params.input_key}} --input_type {fixed_type} --dataset_name {dataset_name} --output_dir {log_dir};"
        f"touch {{output.fixed_check}};"
        f"rm -rf ./tmp_{dataset_name}_{fixed_name}_{{params.input_key}};"
        f"python matchmaker/mobie_export.py --input_path {{input.moving_image_n5}} --input_key {{params.input_key}} --input_type {moving_type} --dataset_name {dataset_name} --output_dir {log_dir};"
        f"touch {{output.moving_check}};"
        f"rm -rf ./tmp_{dataset_name}_{moving_name}_{{params.input_key}};"


"""
Add prealignment to mobie project
"""
rule add_prealignment_to_mobie:
    input:
        fixed_input_ds = f"{fixed_n5_path}/{prealignment_n5_key}",
        moving_input_ds = f"{moving_n5_path}/{prealignment_n5_key}",
        fixed_image_n5 = fixed_n5_path,
        moving_image_n5 = moving_n5_path,
        fixed_uploaded = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{fixed_name}_{raw_n5_key}.done",
        moving_uploaded = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{moving_name}_{raw_n5_key}.done"
    output:
        fixed_check = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{fixed_name}_{prealignment_n5_key}.done",
        moving_check = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{moving_name}_{prealignment_n5_key}.done"
    params:
        input_key = prealignment_n5_key
    log: f"{log_dir}/matchmaker.log"
    conda: "matchmaker_env"
    shell:
        f"python matchmaker/mobie_export.py --input_path {{input.fixed_image_n5}} --input_key {{params.input_key}} --input_type {fixed_type} --dataset_name {dataset_name} --output_dir {log_dir};"
        f"touch {{output.fixed_check}};"
        f"rm -rf ./tmp_{dataset_name}_{fixed_name}_{{params.input_key}};"
        f"python matchmaker/mobie_export.py --input_path {{input.moving_image_n5}} --input_key {{params.input_key}} --input_type {moving_type} --dataset_name {dataset_name} --output_dir {log_dir};"
        f"touch {{output.moving_check}};"
        f"rm -rf ./tmp_{dataset_name}_{moving_name}_{{params.input_key}};"


"""
Add rigid alignment to mobie project
"""
rule add_rigid_alignment_to_mobie:
    input:
        moving_input_ds = f"{moving_n5_path}/{rigid_alignment_n5_key}",
        moving_image_n5 = moving_n5_path,
        fixed_uploaded = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{fixed_name}_{prealignment_n5_key}.done",
        moving_uploaded = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{moving_name}_{prealignment_n5_key}.done"
    output:
        moving_check = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{moving_name}_{rigid_alignment_n5_key}.done"
    params:
        input_key = rigid_alignment_n5_key
    log: f"{log_dir}/matchmaker.log"
    conda: "matchmaker_env"
    shell:
        f"python matchmaker/mobie_export.py --input_path {{input.moving_image_n5}} --input_key {{params.input_key}} --input_type {moving_type} --dataset_name {dataset_name} --output_dir {log_dir};"
        f"touch {{output.moving_check}};"
        f"rm -rf ./tmp_{dataset_name}_{moving_name}_{{params.input_key}};"



"""
Combine transforms
"""
# rule SVD_prealignment_mobie:
        # f"python matchmaker/mobie_add_segmentation.py fixed_prealigned ...;"
        # f"python matchmaker/mobie_add_segmentation.py moving_prealigned ...;"



"""
Combine transforms to create one final transform
"""
# rule combine_transforms:
# input:
#     svd_trans = f"{log_dir}/svd_prealignment_transform.yaml",