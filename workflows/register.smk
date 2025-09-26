import pandas as pd


workdir: "/g/kreshuk/buglakova/projects/matchmaker/"
configfile: "examples/register_config_test.yaml"


print(config["fixed_image"])
print(config["moving_image"])

fixed_n5_path = f"{config['log_dir']}/fixed_image.n5"
moving_n5_path = f"{config['log_dir']}/moving_image.n5"
log_dir = config["log_dir"]


rule all:
    input:
        input_to_n5_fixed = fixed_n5_path,
        input_to_n5_moving = moving_n5_path,

"""
Convert whatever is the input image format (supporting only .tif at the moment) to the internal pipeline's format
"""
rule input_to_n5:
    input:
        fixed_image_path = config["fixed_image"]["path"],
        moving_image_path = config["moving_image"]["path"]
    output:
        f"{fixed_n5_path}/input/attributes.json",
        f"{moving_n5_path}/input/attributes.json",
        fixed_image_n5 = directory(fixed_n5_path),
        moving_image_n5 = directory(moving_n5_path),
    params:
        output_n5_key = "input"
    log: f"{log_dir}/matchmaker.log"
    conda: "matchmaker_env"
    shell:
        f"rm -r {{output.fixed_image_n5}};"
        f"rm -r {{output.moving_image_n5}};"
        f"python matchmaker/raw_to_n5.py --input_path {{input.fixed_image_path}} --output_path {{output.fixed_image_n5}} --output_key {{params.output_n5_key}} --log_dir {log_dir};"
        f"python matchmaker/raw_to_n5.py --input_path {{input.moving_image_path}} --output_path {{output.moving_image_n5}} --output_key {{params.output_n5_key}} --log_dir {log_dir};"


