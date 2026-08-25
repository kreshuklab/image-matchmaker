import json
import re

moving_images = config["moving_images"]
moving_paths = [item["input_path"] for item in moving_images]
moving_keys = [item["input_key"] for item in moving_images]
output_paths = [item["output_path"] for item in moving_images]
output_keys = [item["output_key"] for item in moving_images]
interpolation_orders = [item["interpolation_order"] for item in moving_images]

input_resolutions, output_resolutions = [], []
for item in moving_images:
    in_r = item["input_resolution"]
    ou_r = item.get("output_resolution")
    input_resolutions.append([in_r["z_res"], in_r["y_res"], in_r["x_res"]])
    if ou_r is None or ou_r.get("z_res") is None or ou_r.get("y_res") is None or ou_r.get("x_res") is None:
        output_resolutions.append(None)
    else:
        output_resolutions.append([ou_r["z_res"], ou_r["y_res"], ou_r["x_res"]])

input_resolutions = [json.dumps(r, separators=(',', ':')) for r in input_resolutions]
output_resolutions = [
    json.dumps(r, separators=(',', ':')) if r is not None else None
    for r in output_resolutions
]

log_dir = config["log_dir"]
parameter_map_path = config["parameter_map_path"]
prealignment_transform_path = config["prealignment_transform_path"]

fixed_image = config.get("fixed_image", {})
fixed_path = fixed_image.get("input_path")
fixed_key = fixed_image.get("input_key")

TARGET_OUTPUTS = [f"{p}/{k}" if p.endswith(".n5") else p for p, k in zip(output_paths, output_keys)]
FILE_OUTPUTS = [p for p in TARGET_OUTPUTS if not ".n5/" in p]
N5_OUTPUTS = [p for p in TARGET_OUTPUTS if ".n5/" in p]


def get_all_opts(d):
    opts = []
    for k, v in d.items():
        if v:
            opts.extend([f"--{k}", str(v)])
    return opts


def get_output_resolution_opt(index):
    output_resolution = output_resolutions[index]

    if output_resolution is None:
        return ""

    return f"--output_resolution '{output_resolution}'"


rule all:
    input:
        TARGET_OUTPUTS,


rule apply_transform_file:
    input:
        parameter_map_path = parameter_map_path,
        moving_img = lambda w: moving_paths[TARGET_OUTPUTS.index(w.out_file)]
    output:
        out_file = "{out_file}"
    wildcard_constraints:
        out_file = "^(" + "|".join(re.escape(p) for p in FILE_OUTPUTS) + ")$" if FILE_OUTPUTS else "$^"
    params:
        opts = lambda w: get_all_opts({
            "prealignment_transform_path": prealignment_transform_path,
            "fixed_path": fixed_path,
            "fixed_key": fixed_key,
        }),
        moving_path = lambda w: moving_paths[TARGET_OUTPUTS.index(w.out_file)],
        moving_key = lambda w: moving_keys[TARGET_OUTPUTS.index(w.out_file)],
        input_resolution = lambda w: input_resolutions[TARGET_OUTPUTS.index(w.out_file)],
        output_path = lambda w: output_paths[TARGET_OUTPUTS.index(w.out_file)],
        output_key = lambda w: output_keys[TARGET_OUTPUTS.index(w.out_file)],
        output_resolution_opt = lambda w: get_output_resolution_opt(TARGET_OUTPUTS.index(w.out_file)),
        interpolation_order = lambda w: interpolation_orders[TARGET_OUTPUTS.index(w.out_file)],
    shell:
        """
        python -m image_matchmaker.apply_transform \
            {params.opts} \
            --moving_path {params.moving_path} \
            --moving_key {params.moving_key} \
            --input_resolution '{params.input_resolution}' \
            --output_path {params.output_path} \
            --output_key {params.output_key} \
            {params.output_resolution_opt} \
            --interpolation_order {params.interpolation_order} \
            --log_dir {log_dir} \
            --parameter_map_path {input.parameter_map_path}
        """


rule apply_transform_n5:
    input:
        parameter_map_path = parameter_map_path,
        moving_img = lambda w: moving_paths[TARGET_OUTPUTS.index(w.out_dir)]
    output:
        out_dir = directory("{out_dir}")
    wildcard_constraints:
        out_dir = "^(" + "|".join(re.escape(p) for p in N5_OUTPUTS) + ")$" if N5_OUTPUTS else "$^"
    params:
        opts = lambda w: get_all_opts({
            "prealignment_transform_path": prealignment_transform_path,
            "fixed_path": fixed_path,
            "fixed_key": fixed_key,
        }),
        moving_path = lambda w: moving_paths[TARGET_OUTPUTS.index(w.out_dir)],
        moving_key = lambda w: moving_keys[TARGET_OUTPUTS.index(w.out_dir)],
        input_resolution = lambda w: input_resolutions[TARGET_OUTPUTS.index(w.out_dir)],
        output_path = lambda w: output_paths[TARGET_OUTPUTS.index(w.out_dir)],
        output_key = lambda w: output_keys[TARGET_OUTPUTS.index(w.out_dir)],
        output_resolution_opt = lambda w: get_output_resolution_opt(TARGET_OUTPUTS.index(w.out_dir)),
        interpolation_order = lambda w: interpolation_orders[TARGET_OUTPUTS.index(w.out_dir)],
    shell:
        """
        python -m image_matchmaker.apply_transform \
            {params.opts} \
            --moving_path {params.moving_path} \
            --moving_key {params.moving_key} \
            --input_resolution '{params.input_resolution}' \
            --output_path {params.output_path} \
            --output_key {params.output_key} \
            {params.output_resolution_opt} \
            --interpolation_order {params.interpolation_order} \
            --log_dir {log_dir} \
            --parameter_map_path {input.parameter_map_path}
        """
