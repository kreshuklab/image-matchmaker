from pathlib import Path

root_dir = f"{Path(workflow.basedir).resolve().parent}/"
workdir: root_dir

fixed_input_key   = config["fixed_image"]["input_key"]
fixed_aligned_key = config["fixed_image"]["aligned_key"]

moving_input_key   = config["moving_image"]["input_key"]
moving_aligned_key = config["moving_image"]["aligned_key"]

fixed_lm_csv  = config["landmarks"]["fixed"]
moving_lm_csv = config["landmarks"]["moving"]

log_dir          = config["log_dir"]
axis_orientation = config["prealignment"]["axis_orientation"]

fixed_name    = config.get("fixed_name", "fixed_image")
moving_name   = config.get("moving_name", "moving_image")
fixed_n5_path = f"{log_dir}/{fixed_name}.n5"
moving_n5_path = f"{log_dir}/{moving_name}.n5"

raw_n5_key        = "input"
lm_input_key      = "input_with_lm"
landmark_ids_json = f"{log_dir}/prepare_landmarks/landmark_label_ids.json"

fixed_spacing  = [config["fixed_image"]["z_res"], config["fixed_image"]["y_res"], config["fixed_image"]["x_res"]]
moving_spacing = [config["moving_image"]["z_res"], config["moving_image"]["y_res"], config["moving_image"]["x_res"]]


rule all:
    input:
        best_params   = f"{log_dir}/best_cpd_params.yaml",
        study_results = f"{log_dir}/study_results.csv",


rule input_to_n5:
    input:
        fixed_image_path  = config["fixed_image"]["path"],
        moving_image_path = config["moving_image"]["path"],
    output:
        directory(f"{fixed_n5_path}/{raw_n5_key}/"),
        directory(f"{moving_n5_path}/{raw_n5_key}/"),
        fixed_image_n5  = directory(fixed_n5_path),
        moving_image_n5 = directory(moving_n5_path),
    params:
        output_key = raw_n5_key,
    log: f"{log_dir}/matchmaker.log"
    shell:
        f"rm -rf {{output.fixed_image_n5}};"
        f"rm -rf {{output.moving_image_n5}};"
        f"python matchmaker/raw_to_n5.py --input_path {{input.fixed_image_path}} --input_key {fixed_input_key} --output_path {{output.fixed_image_n5}} --output_key {{params.output_key}} --log_dir {log_dir} --x_res {config['fixed_image']['x_res']} --y_res {config['fixed_image']['y_res']} --z_res {config['fixed_image']['z_res']};"
        f"python matchmaker/raw_to_n5.py --input_path {{input.moving_image_path}} --input_key {moving_input_key} --output_path {{output.moving_image_n5}} --output_key {{params.output_key}} --log_dir {log_dir} --x_res {config['moving_image']['x_res']} --y_res {config['moving_image']['y_res']} --z_res {config['moving_image']['z_res']};"


rule add_landmarks:
    """Embed landmarks into both raw segmentations."""
    input:
        fixed_n5        = fixed_n5_path,
        moving_n5       = moving_n5_path,
        fixed_input_ds  = f"{fixed_n5_path}/{raw_n5_key}",
        moving_input_ds = f"{moving_n5_path}/{raw_n5_key}",
        fixed_lm_csv    = fixed_lm_csv,
        moving_lm_csv   = moving_lm_csv,
    output:
        directory(f"{fixed_n5_path}/{lm_input_key}"),
        directory(f"{moving_n5_path}/{lm_input_key}"),
        landmark_ids = landmark_ids_json,
    params:
        fixed_input_key  = raw_n5_key,
        moving_input_key = raw_n5_key,
        lm_input_key     = lm_input_key,
        log_dir          = f"{log_dir}/prepare_landmarks",
    log: f"{log_dir}/prepare_landmarks/add_landmarks.log"
    shell:
        "python matchmaker/cpd_parameter_tuning/add_landmarks.py "
        "--fixed_path {input.fixed_n5} "
        "--fixed_key {params.fixed_input_key} "
        "--fixed_output_key {params.lm_input_key} "
        "--fixed_landmarks_csv {input.fixed_lm_csv} "
        "--moving_path {input.moving_n5} "
        "--moving_key {params.moving_input_key} "
        "--moving_output_key {params.lm_input_key} "
        "--moving_landmarks_csv {input.moving_lm_csv} "
        "--log_dir {params.log_dir}"


rule prealignment_with_lm:
    """Run SVD prealignment on the landmark-embedded segmentations."""
    input:
        fixed_ds  = f"{fixed_n5_path}/{lm_input_key}",
        moving_ds = f"{moving_n5_path}/{lm_input_key}",
    output:
        directory(f"{fixed_n5_path}/{fixed_aligned_key}"),
        directory(f"{moving_n5_path}/{fixed_aligned_key}"),
        transform = f"{log_dir}/prealignment_with_lm/prealignment_transform.json",
    params:
        fixed_path       = fixed_n5_path,
        moving_path      = moving_n5_path,
        input_key        = lm_input_key,
        output_key       = fixed_aligned_key,
        fixed_spacing    = f"{config['fixed_image']['z_res']} {config['fixed_image']['y_res']} {config['fixed_image']['x_res']}",
        moving_spacing   = f"{config['moving_image']['z_res']} {config['moving_image']['y_res']} {config['moving_image']['x_res']}",
        axis_orientation = axis_orientation,
        output_dir       = f"{log_dir}/prealignment_with_lm",
    log: f"{log_dir}/prealignment_with_lm/prealignment.log"
    shell:
        "python matchmaker/prealignment.py "
        "--fixed_path {params.fixed_path} "
        "--fixed_key {params.input_key} "
        "--fixed_spacing {params.fixed_spacing} "
        "--moving_path {params.moving_path} "
        "--moving_key {params.input_key} "
        "--moving_spacing {params.moving_spacing} "
        "--output_dir {params.output_dir} "
        "--output_key {params.output_key} "
        "--output_transform_path {output.transform} "
        "--axis_orientation {params.axis_orientation}"


rule rigid_alignment_with_lm:
    """Run elastix rigid alignment on the prealigned landmark-embedded segmentations."""
    input:
        fixed_ds  = f"{fixed_n5_path}/{fixed_aligned_key}",
        moving_ds = f"{moving_n5_path}/{fixed_aligned_key}",
    output:
        directory(f"{moving_n5_path}/{moving_aligned_key}"),
    params:
        fixed_path  = fixed_n5_path,
        moving_path = moving_n5_path,
        input_key   = fixed_aligned_key,
        output_key  = moving_aligned_key,
        output_dir  = f"{log_dir}/rigid_alignment_with_lm",
    log: f"{log_dir}/rigid_alignment_with_lm/rigid_alignment.log"
    shell:
        "python matchmaker/rigid_alignment_elastix.py "
        "--fixed_path {params.fixed_path} "
        "--fixed_key {params.input_key} "
        "--moving_path {params.moving_path} "
        "--moving_key {params.input_key} "
        "--output_dir {params.output_dir} "
        "--output_key {params.output_key}"


rule optimize_cpd:
    """Run Optuna grid search over CPD parameters, evaluated by mean LRE.
    Beta ranges for dataset-specific mode are computed inside cpd_optimization.py
    from the extracted point clouds.
    """
    input:
        fixed_ds          = f"{fixed_n5_path}/{fixed_aligned_key}",
        moving_ds         = f"{moving_n5_path}/{moving_aligned_key}",
        config            = workflow.configfiles[0],
        landmark_ids_json = landmark_ids_json,
    output:
        best_params   = f"{log_dir}/best_cpd_params.yaml",
        study_results = f"{log_dir}/study_results.csv",
    params:
        fixed_path  = fixed_n5_path,
        moving_path = moving_n5_path,
    log: f"{log_dir}/cpd_optimization.log"
    shell:
        "python matchmaker/cpd_parameter_tuning/cpd_optimization.py "
        "--config {input.config} "
        "--landmark_ids_json {input.landmark_ids_json} "
        "--fixed_path {params.fixed_path} "
        "--moving_path {params.moving_path}"
