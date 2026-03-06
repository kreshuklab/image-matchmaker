import pandas as pd
from pathlib import Path

root_dir = f"{Path(workflow.basedir).resolve().parent}/"
print(f"working directory: {root_dir}")
workdir: root_dir
configfile: "examples/register_config_test_rigid.yaml"

print(config["fixed_image"])
print(config["moving_image"])

fixed_name = "fixed_image"
fixed_n5_path = f"{config['log_dir']}/{fixed_name}.n5"
moving_name = "moving_image"
moving_n5_path = f"{config['log_dir']}/{moving_name}.n5"
log_dir = config["log_dir"]
final_transform = config["final_transform_path"]

# define global variables for n5 keys
raw_n5_key = "input"
prealignment_n5_key = "svd_prealignment"
rigid_alignment_n5_key = "rigid_alignment"

# define global MoBIE variables
fixed_type = "fixed"
moving_type = "moving"
dataset_name = config["mobie_dataset_name"]

# Coherent Point Drift parameters
w = config["coherent_point_drift"]["w"]
beta = config["coherent_point_drift"]["beta"]
lmd = config["coherent_point_drift"]["lmd"]
maxiter = config["coherent_point_drift"]["maxiter"]


rule all:
    input:
        input_to_n5_fixed = fixed_n5_path,
        input_to_n5_moving = moving_n5_path,
        svd_prealignment = f"{fixed_n5_path}/{prealignment_n5_key}",
        svd_transform = f"{log_dir}/{prealignment_n5_key}/{prealignment_n5_key}_transform.json",
        rigid_alignment = f"{moving_n5_path}/{rigid_alignment_n5_key}",
        rigid_transform = f"{log_dir}/{rigid_alignment_n5_key}/TransformParameters.0.txt",
        fixed_pcd = f"{log_dir}/cpd_nonrigid_registration/fixed_pcd.pcd",
        registered_pcd = f"{log_dir}/cpd_nonrigid_registration/registered_pcd.pcd"
        # match_path = f"{log_dir}/match_pointclouds/matched_pairs.csv"
        # mobie_fixed_input_check = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{fixed_name}_{raw_n5_key}.done",
        # mobie_moving_input_check = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{moving_name}_{raw_n5_key}.done",
        # mobie_fixed_prealign_check = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{fixed_name}_{prealignment_n5_key}.done",
        # mobie_moving_prealign_check = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{moving_name}_{prealignment_n5_key}.done",
        # mobie_moving_rigid_align_check = f"{log_dir}/mobie_project/{dataset_name}/images/ome-zarr/{moving_name}_{rigid_alignment_n5_key}.done"


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

    params:
        output_key = raw_n5_key
    log: f"{log_dir}/matchmaker.log"
    conda: "matchmaker_env"
    shell:
        f"rm -r {{output.fixed_image_n5}};"
        f"rm -r {{output.moving_image_n5}};"
        f"python matchmaker/raw_to_n5.py --input_path {{input.fixed_image_path}} --output_path {{output.fixed_image_n5}} --output_key {{params.output_key}} --log_dir {log_dir};"
        f"python matchmaker/raw_to_n5.py --input_path {{input.moving_image_path}} --output_path {{output.moving_image_n5}} --output_key {{params.output_key}} --log_dir {log_dir};"


"""
Run pre-alignment with SVD
"""
rule SVD_prealignment:
    input:
        fixed_input_ds = f"{fixed_n5_path}/{raw_n5_key}",
        moving_input_ds = f"{moving_n5_path}/{raw_n5_key}",
        fixed_image_n5 = fixed_n5_path,
        moving_image_n5 = moving_n5_path
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
        moving_image_n5 = moving_n5_path
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

# """
# Create point clouds
# """

# rule create_point_clouds:
#     input:
#         f"{log_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/template_aligned_rigid/segmentation",
#         gene_assignment = f"{log_dir}/{{sample_name}}/rigid_registration/gene_assignment_rigid.csv"
#     output:
#         f"{log_dir}/{{sample_name}}/rigid_registration/point_cloud_rigid.pcd"
#     params:
#         log_dir = f"{log_dir}/{{sample_name}}/create_pcd",
#         n5_path = f"{log_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5"
#     log: f"{log_dir}/{{sample_name}}/logs/csv_to_pcd.log"
#     conda: "open3d_env"
#     resources:
#         time="00:15:00",
#     shell:
#         f"python pipeline_steps/inter_sample_registration/csv_to_pcd.py {{params.n5_path}} template_aligned_rigid/segmentation {{input.gene_assignment}} {{output}} {{params.log_dir}} {{log}};"


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
    conda: "cvxpy_cpd_env"
    resources:
        time="01:00:00",
        cpus_per_task=16,
        mem_mb="128GB"
    shell:
        f"python matchmaker/cpd_nonrigid_registration.py --moving_path {{input.moving_image_n5}} --moving_key {{params.moving_key}} --fixed_path {{input.fixed_image_n5}} --fixed_key {{params.fixed_key}} -o {{params.log_dir}} --w {w} --beta {beta} --lmd {lmd} --maxiter {maxiter};"




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
