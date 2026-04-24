from pathlib import Path

configfile: "examples/register_config_test_rigid_apply_transform.yaml"

fixed_path = config["fixed_image"]["path"]
fixed_key = config["fixed_image"]["input_key"]

moving_path = config["moving_image"]["path"]
moving_key = config["moving_image"]["input_key"]
output_key = config["moving_image"]["output_key"]
log_dir = config["log_dir"]
output_dir = f"{log_dir}/apply_transform"
parameter_map_path = config["parameter_map_path"]
prealignment_transform_path = config["prealignment_transform_path"]
moving_resolution = [config["moving_image"]["z_res"], config["moving_image"]["y_res"], config["moving_image"]["x_res"]]

if moving_path.endswith((".tif", ".tiff")):
    output_path = f"{output_dir}/{Path(moving_path).stem}_transformed.tif"
elif moving_path.endswith(".n5"):
    output_path = f"{moving_path}/{output_key}"
else:
    raise NotImplementedError


def get_all_opts(d):
    opts = []
    for k, v in d.items():
        if v:
            opts.extend([f"--{k}", str(v)])
    return opts


rule all:
    input:
        apply_transform = output_path,


rule apply_transform:
    input:
        moving_path = moving_path,
        parameter_map_path = parameter_map_path,
    output:
        directory(output_path) if moving_path.endswith(".n5") else output_path,
    params:
        opts = lambda w: get_all_opts({
            "fixed_path": fixed_path,
            "fixed_key": fixed_key,
            "output_key": output_key,
            "prealignment_transform_path": prealignment_transform_path,
        }),

        moving_key = moving_key,
        output_dir = output_dir,
        moving_resolution = moving_resolution,
    log: f"{log_dir}/matchmaker.log"
    conda: "matchmaker_env"
    shell:
        """
        python matchmaker/apply_transform.py \
            {params.opts} \
            --moving_path {input.moving_path} \
            --moving_key {params.moving_key} \
            --moving_resolution {params.moving_resolution} \
            --output_dir {params.output_dir} \
            --parameter_map_path {input.parameter_map_path}
        """
