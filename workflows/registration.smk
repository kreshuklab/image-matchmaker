import pandas as pd
from pathlib import Path

root_dir = f"{Path(workflow.basedir).resolve().parent}/"
print(f"working directory: {root_dir}")
workdir: root_dir
configfile: "examples/register_config_test_rigid.yaml"

print(config["fixed_image"])
print(config["moving_image"])

fixed_name = config["fixed_image"]["name"] if "name" in config["fixed_image"] else "fixed_image"
fixed_input_key = config["fixed_image"]["input_key"] if "input_key" in config["fixed_image"] else None
fixed_n5_path = f"{config['log_dir']}/{fixed_name}.n5"
moving_name = config["moving_image"]["name"] if "name" in config["moving_image"] else "moving_image"
moving_input_key = config["moving_image"]["input_key"] if "input_key" in config["moving_image"] else None
moving_n5_path = f"{config['log_dir']}/{moving_name}.n5"
log_dir = config["log_dir"]
final_transform = config["final_transform_path"]
fixed_spacing = [config["fixed_image"]["z_res"], config["fixed_image"]["y_res"], config["fixed_image"]["x_res"]]
moving_spacing = [config["moving_image"]["z_res"], config["moving_image"]["y_res"], config["moving_image"]["x_res"]]

print(fixed_name, fixed_n5_path, moving_name, moving_n5_path)

# define global variables for n5 keys
raw_n5_key = "input"
prealignment_n5_key = "svd_prealignment"
rigid_alignment_n5_key = "rigid_alignment"
pointset_alignment_n5_key = "pointset_alignment"
pointset_alignment_input_space_n5_key = "pointset_alignment_input_space"

# define global MoBIE variables
fixed_type = "fixed"
moving_type = "moving"
dataset_name = config["mobie_dataset_name"]
semantic_seg = config["semantic_seg"] if "semantic_seg" in config else False
print(f'Upload to MoBIE: {config["mobie_export"]}, upload semantic segmentation: {semantic_seg}')

# Coherent Point Drift parameters
w = config["coherent_point_drift"]["w"]
beta = config["coherent_point_drift"]["beta"]
lmd = config["coherent_point_drift"]["lmd"]
maxiter = config["coherent_point_drift"]["maxiter"]

# ILP matching parameters
min_neighbours = config["ILP"]["min_neighbours"]
max_dist = config["ILP"]["max_dist"]

if config["mobie_export"]:
    mobie_outputs = [
        f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{fixed_name}_{raw_n5_key}.done",
        f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{moving_name}_{raw_n5_key}.done",
        f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{fixed_name}_{prealignment_n5_key}.done",
        f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{moving_name}_{prealignment_n5_key}.done",
        f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{moving_name}_{rigid_alignment_n5_key}.done",
        f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{moving_name}_{pointset_alignment_input_space_n5_key}.done"
    ]
else:
    mobie_outputs = []



rule all:
    input:
        input_to_n5_fixed = fixed_n5_path,
        input_to_n5_moving = moving_n5_path,
        svd_prealignment = f"{fixed_n5_path}/{prealignment_n5_key}",
        svd_transform = f"{log_dir}/{prealignment_n5_key}/{prealignment_n5_key}_transform.json",
        rigid_alignment = f"{moving_n5_path}/{rigid_alignment_n5_key}",
        rigid_transform = f"{log_dir}/{rigid_alignment_n5_key}/TransformParameters.0.txt",
        fixed_pcd = f"{log_dir}/cpd_nonrigid_registration/fixed_pcd.pcd",
        registered_pcd = f"{log_dir}/cpd_nonrigid_registration/registered_pcd.pcd",
        match_path = f"{log_dir}/match_pointclouds/matched_labels.csv",
        pointset_alignment = f"{moving_n5_path}/{pointset_alignment_input_space_n5_key}",
        mobie_outputs = mobie_outputs


"""
Convert whatever is the input image format (supporting only .tif and .n5 at the moment) to the internal pipeline's format
"""
rule input_to_n5:
    input:
        fixed_image_path = config["fixed_image"]["path"],
        moving_image_path = config["moving_image"]["path"]
    output:
        directory(f"{fixed_n5_path}/input/"),
        directory(f"{moving_n5_path}/input/"),
        fixed_image_n5 = directory(fixed_n5_path),
        moving_image_n5 = directory(moving_n5_path),
    params:
        output_key = raw_n5_key
    log: f"{log_dir}/matchmaker.log"
    conda: "matchmaker_env"
    shell:
        f"rm -r {{output.fixed_image_n5}};"
        f"rm -r {{output.moving_image_n5}};"
        f"python matchmaker/raw_to_n5.py --input_path {{input.fixed_image_path}} --input_key {fixed_input_key} --output_path {{output.fixed_image_n5}} --output_key {{params.output_key}} --log_dir {log_dir} --x_res {config['fixed_image']['x_res']} --y_res {config['fixed_image']['y_res']} --z_res {config['fixed_image']['z_res']};"
        f"python matchmaker/raw_to_n5.py --input_path {{input.moving_image_path}} --input_key {moving_input_key} --output_path {{output.moving_image_n5}} --output_key {{params.output_key}} --log_dir {log_dir} --x_res {config['moving_image']['x_res']} --y_res {config['moving_image']['y_res']} --z_res {config['moving_image']['z_res']};"


"""
Run pre-alignment with SVD
"""
rule SVD_prealignment:
    input:
        fixed_input_ds = f"{fixed_n5_path}/{raw_n5_key}",
        moving_input_ds = f"{moving_n5_path}/{raw_n5_key}",
        fixed_image_n5 = fixed_n5_path,
        moving_image_n5 = moving_n5_path,
    output:
        directory(f"{fixed_n5_path}/{prealignment_n5_key}"),
        directory(f"{moving_n5_path}/{prealignment_n5_key}"),
        output_transform = f"{log_dir}/{prealignment_n5_key}/{prealignment_n5_key}_transform.json"
    params:
        input_key = raw_n5_key,
        output_key = prealignment_n5_key,
        axis_orientation = config["prealignment"]["axis_orientation"]
    log: f"{log_dir}/matchmaker.log"
    conda: "matchmaker_env"
    shell:
        f"python matchmaker/prealignment.py --fixed_path {{input.fixed_image_n5}} --fixed_key {{params.input_key}} --fixed_spacing {{fixed_spacing}} --moving_path {{input.moving_image_n5}} --moving_key {{params.input_key}} --moving_spacing {{moving_spacing}} --output_dir {log_dir}/{{params.output_key}}  --output_key {{params.output_key}} --output_transform_path {{output.output_transform}} --axis_orientation {{params.axis_orientation}};"


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
    output:
        fixed_check = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{fixed_name}_{raw_n5_key}.done",
        moving_check = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{moving_name}_{raw_n5_key}.done"
    params:
        input_key = raw_n5_key
    log: f"{log_dir}/matchmaker.log"
    conda: "matchmaker_env"
    shell:
        f"python matchmaker/mobie_export.py --input_path {{input.fixed_image_n5}} --input_key {{params.input_key}} --input_type {fixed_type} {'--semantic_seg' if semantic_seg else ''} --dataset_name {dataset_name} --output_dir {log_dir};"
        f"touch {{output.fixed_check}};"
        f"rm -rf ./tmp_{dataset_name}_{fixed_name}_{{params.input_key}};"
        f"rm -rf ./tmp_{dataset_name}_{fixed_name}_{{params.input_key}}_binary;"
        f"python matchmaker/mobie_export.py --input_path {{input.moving_image_n5}} --input_key {{params.input_key}} --input_type {moving_type} {'--semantic_seg' if semantic_seg else ''} --dataset_name {dataset_name} --output_dir {log_dir};"
        f"touch {{output.moving_check}};"
        f"rm -rf ./tmp_{dataset_name}_{moving_name}_{{params.input_key}};"
        f"rm -rf ./tmp_{dataset_name}_{moving_name}_{{params.input_key}}_binary;"


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
        f"python matchmaker/mobie_export.py --input_path {{input.fixed_image_n5}} --input_key {{params.input_key}} --input_type {fixed_type} {'--semantic_seg' if semantic_seg else ''} --dataset_name {dataset_name} --output_dir {log_dir};"
        f"touch {{output.fixed_check}};"
        f"rm -rf ./tmp_{dataset_name}_{fixed_name}_{{params.input_key}};"
        f"rm -rf ./tmp_{dataset_name}_{fixed_name}_{{params.input_key}}_binary;"
        f"python matchmaker/mobie_export.py --input_path {{input.moving_image_n5}} --input_key {{params.input_key}} --input_type {moving_type} {'--semantic_seg' if semantic_seg else ''} --dataset_name {dataset_name} --output_dir {log_dir};"
        f"touch {{output.moving_check}};"
        f"rm -rf ./tmp_{dataset_name}_{moving_name}_{{params.input_key}};"
        f"rm -rf ./tmp_{dataset_name}_{moving_name}_{{params.input_key}}_binary;"


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
        f"python matchmaker/mobie_export.py --input_path {{input.moving_image_n5}} --input_key {{params.input_key}} --input_type {moving_type} {'--semantic_seg' if semantic_seg else ''} --dataset_name {dataset_name} --output_dir {log_dir};"
        f"touch {{output.moving_check}};"
        f"rm -rf ./tmp_{dataset_name}_{moving_name}_{{params.input_key}};"
        f"rm -rf ./tmp_{dataset_name}_{moving_name}_{{params.input_key}}_binary;"


rule cpd_nonrigid_registration:
    input:
        fixed_input_ds = f"{fixed_n5_path}/{prealignment_n5_key}",
        moving_input_ds = f"{moving_n5_path}/{rigid_alignment_n5_key}",
        fixed_image_n5 = fixed_n5_path,
        moving_image_n5 = moving_n5_path,
    output:
        fixed_pcd = f"{log_dir}/cpd_nonrigid_registration/fixed_pcd.pcd",
        registered_pcd = f"{log_dir}/cpd_nonrigid_registration/registered_pcd.pcd"
    params:
        log_dir = f"{log_dir}/cpd_nonrigid_registration",
        fixed_key = prealignment_n5_key,
        moving_key = rigid_alignment_n5_key
    log: f"{log_dir}/matchmaker.log"
    conda: "matchmaker_env"
    shell:
        f"python matchmaker/cpd_nonrigid_registration.py --moving_path {{input.moving_image_n5}} --moving_key {{params.moving_key}} --fixed_path {{input.fixed_image_n5}} --fixed_key {{params.fixed_key}} -o {{params.log_dir}} --w {w} --beta {beta} --lmd {lmd} --maxiter {maxiter};"


rule ilp_matching:
    input:
        fixed_pcd = f"{log_dir}/cpd_nonrigid_registration/fixed_pcd.pcd",
        moving_pcd = f"{log_dir}/cpd_nonrigid_registration/registered_pcd.pcd"
    output:
        match_path = f"{log_dir}/match_pointclouds/matched_labels.csv"
    params:
        log_dir = f"{log_dir}/match_pointclouds"
    log: f"{log_dir}/matchmaker.log"
    conda: "matchmaker_env"
    shell:
        f"python matchmaker/match_pointclouds.py --fixed_pcd {{input.fixed_pcd}} --moving_pcd {{input.moving_pcd}} -o {{params.log_dir}} --min_neighbours {min_neighbours} --max_dist {max_dist};"


rule elastix_deformable_pointset:
    input:
        match_path = f"{log_dir}/match_pointclouds/matched_labels.csv",
        fixed_input_ds = f"{fixed_n5_path}/{raw_n5_key}",
        moving_input_ds = f"{moving_n5_path}/{raw_n5_key}",
        fixed_image_n5 = fixed_n5_path,
        moving_image_n5 = moving_n5_path,
        prealignment_transform = f"{log_dir}/{prealignment_n5_key}/{prealignment_n5_key}_transform.json"
    output:
        directory(f"{moving_n5_path}/{pointset_alignment_n5_key}"),
        directory(f"{moving_n5_path}/{pointset_alignment_input_space_n5_key}"),
        f"{log_dir}/elastix_deformable_pointset_registration/TransformParameters.0.txt",
        f"{log_dir}/elastix_deformable_pointset_registration/TransformParameters.1.txt",
        f"{log_dir}/elastix_deformable_pointset_registration/TransformParameters.2.txt"
    params:
        input_key = raw_n5_key,
        output_key = pointset_alignment_input_space_n5_key,
        prealigned_output_key = pointset_alignment_n5_key,
        log_dir = f"{log_dir}/elastix_deformable_pointset_registration"
    log: f"{log_dir}/matchmaker.log"
    conda: "matchmaker_env"
    shell:
        f"python matchmaker/elastix_deformable_pointset_registration.py --fixed_path {{input.fixed_image_n5}} --fixed_key {{params.input_key}} --moving_path {{input.moving_image_n5}} --moving_key {{params.input_key}} --output_dir {{params.log_dir}}  --output_key {{params.output_key}} --match_path {{input.match_path}} --prealigned_output_key {{params.prealigned_output_key}} --prealignment_transform {{input.prealignment_transform}};"


rule add_elastix_deformable_pointset_to_mobie:
    input:
        moving_input_ds = f"{moving_n5_path}/{pointset_alignment_input_space_n5_key}",
        moving_image_n5 = moving_n5_path,
        moving_uploaded = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{moving_name}_{rigid_alignment_n5_key}.done"
    output:
        moving_check = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{moving_name}_{pointset_alignment_input_space_n5_key}.done"
    params:
        input_key = pointset_alignment_input_space_n5_key
    log: f"{log_dir}/matchmaker.log"
    conda: "matchmaker_env"
    shell:
        f"python matchmaker/mobie_export.py --input_path {{input.moving_image_n5}} --input_key {{params.input_key}} --input_type {moving_type} {'--semantic_seg' if semantic_seg else ''} --dataset_name {dataset_name} --output_dir {log_dir};"
        f"touch {{output.moving_check}};"
        f"rm -rf ./tmp_{dataset_name}_{moving_name}_{{params.input_key}};"
        f"rm -rf ./tmp_{dataset_name}_{moving_name}_{{params.input_key}}_binary;"
