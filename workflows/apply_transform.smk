import json

configfile: "examples/register_config_test_rigid_apply_transform.yaml"

moving_images = config["moving_images"]
moving_paths = [item["input_path"] for item in moving_images]
moving_keys = [item["input_key"] for item in moving_images]
moving_resolutions = [[item["z_res"], item["y_res"], item["x_res"]] for item in moving_images]
moving_resolutions = [json.dumps(r, separators=(',', ':')) for r in moving_resolutions]
output_paths = [item["output_path"] for item in moving_images]
output_keys = [item["output_key"] for item in moving_images]
interpolation_orders = [item["interpolation_order"] for item in moving_images]

log_dir = config["log_dir"]
parameter_map_path = config["parameter_map_path"]
prealignment_transform_path = config["prealignment_transform_path"]

fixed_path = config["fixed_image"]["input_path"]
fixed_key = config["fixed_image"]["input_key"]


def expand_flag(flag, values):
    return " ".join(f"{flag} {v}" for v in values)


def get_all_opts(d):
    opts = []
    for k, v in d.items():
        if v:
            opts.extend([f"--{k}", str(v)])
    return opts


rule all:
    input:
        output_paths,


rule apply_transform:
    input:
        parameter_map_path = parameter_map_path,
    output:
        [directory(path) if path.endswith(".n5") else path for path in output_paths],
    params:
        opts = lambda w: get_all_opts({
            "prealignment_transform_path": prealignment_transform_path,
            "fixed_path": fixed_path,
            "fixed_key": fixed_key,
        }),

        moving_paths = expand_flag("--moving_paths", moving_paths),
        moving_keys = expand_flag("--moving_keys", moving_keys),
        moving_resolutions = expand_flag("--moving_resolutions", moving_resolutions),
        output_paths = expand_flag("--output_paths", output_paths),
        output_keys = expand_flag("--output_keys", output_keys),
        interpolation_orders = expand_flag("--interpolation_orders", interpolation_orders),
        log_dir = log_dir,

    log: f"{log_dir}/matchmaker.log"
    conda: "matchmaker_env"
    shell:
        """
        python matchmaker/apply_transform.py \
            {params.opts} \
            {params.moving_paths} \
            {params.moving_keys} \
            {params.moving_resolutions} \
            {params.output_paths} \
            {params.output_keys} \
            {params.interpolation_orders} \
            --log_dir {params.log_dir} \
            --parameter_map_path {input.parameter_map_path}
        """
