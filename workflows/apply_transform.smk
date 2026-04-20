from pathlib import Path

root_dir = f"{Path(workflow.basedir).resolve().parent}/"
print(f"working directory: {root_dir}")
configfile: "examples/register_config_test_rigid.yaml"

fixed_name = config["fixed_name"] if "fixed_name" in config else "fixed_image"
moving_name = config["moving_name"] if "moving_name" in config else "moving_image"

fixed_n5_path = f"{config['log_dir']}/{fixed_name}.n5"
moving_n5_path = f"{config['log_dir']}/{moving_name}.n5"
log_dir = config["log_dir"]
parameter_map_path = f"{log_dir}/elastix_deformable_pointset_registration/TransformParameters.2.txt"

# define global variables for n5 keys
raw_n5_key = "input"
prealignment_n5_key = "svd_prealignment"
pointset_alignment_transform_n5_key = "pointset_alignment_transform"


rule all:
    input:
        apply_transform = f"{moving_n5_path}/{pointset_alignment_transform_n5_key}",


rule apply_transform:
    input:
        fixed_image_n5 = fixed_n5_path,
        moving_image_n5 = moving_n5_path,
        parameter_map_path = parameter_map_path,
        prealignment_transform_path = f"{log_dir}/{prealignment_n5_key}/{prealignment_n5_key}_transform.json"
    output:
        directory(f"{moving_n5_path}/{pointset_alignment_transform_n5_key}"),
    params:
        input_key = raw_n5_key,
        output_key = pointset_alignment_transform_n5_key,
        log_dir = f"{log_dir}/apply_transform"
    log: f"{log_dir}/matchmaker.log"
    conda: "matchmaker_env"
    shell:
        f"python matchmaker/apply_transform.py --fixed_path {{input.fixed_image_n5}} --fixed_key {{params.input_key}} --moving_path {{input.moving_image_n5}} --moving_key {{params.input_key}} --output_dir {{params.log_dir}}  --output_key {{params.output_key}} --parameter_map_path {{input.parameter_map_path}} --prealignment_transform_path {{input.prealignment_transform_path}};"
