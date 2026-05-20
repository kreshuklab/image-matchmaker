"""Snakemake workflow for CPD parameter optimization via Optuna.

Separate from the main registration pipeline. Performs:
  1. Embed anatomical landmarks into the input segmentations as single-voxel labels.
  2. Apply the stored SVD prealignment transform to the fixed (Igor) image with landmarks.
  3. Apply the stored SVD + rigid transforms to the moving (Seymour) image with landmarks.
  4. Run an Optuna grid search over CPD parameters, evaluating each combination by the
     mean Landmark Registration Error (LRE) between corresponding landmarks after CPD.

Output: best_cpd_params.yaml (drop-in replacement for the coherent_point_drift section
of the main registration config).

Usage:
    snakemake -s workflows/cpd_optimization.smk \
              --configfile data/brain_matching/cpd_optimization_config.yaml \
              --cores 1
"""

from pathlib import Path

root_dir = f"{Path(workflow.basedir).resolve().parent}/"
workdir: root_dir
configfile: "data/brain_matching/cpd_optimization_config.yaml"

fixed_n5        = config["fixed_image"]["path"]
fixed_input_key = config["fixed_image"]["input_key"]
fixed_aligned_key = config["fixed_image"]["aligned_key"]

moving_n5        = config["moving_image"]["path"]
moving_input_key = config["moving_image"]["input_key"]
moving_aligned_key = config["moving_image"]["aligned_key"]

prealignment_transform = config["prealignment_transform"]
rigid_transform        = config["rigid_transform"]

fixed_lm_csv  = config["landmarks"]["fixed"]
moving_lm_csv = config["landmarks"]["moving"]

log_dir = config["log_dir"]


rule all:
    input:
        best_params = f"{log_dir}/best_cpd_params.yaml",
        study_results = f"{log_dir}/study_results.csv",


rule prepare_fixed_with_lm:
    """Add Igor landmarks to the input segmentation, apply SVD prealignment transform."""
    input:
        seg_n5        = fixed_n5,
        lm_csv        = fixed_lm_csv,
        transform_json = prealignment_transform,
    output:
        directory(f"{fixed_n5}/{fixed_aligned_key}"),
    params:
        input_key             = fixed_input_key,
        output_key            = fixed_aligned_key,
        transform_sample_key  = "fixed_prealignment",
        log_dir               = f"{log_dir}/prepare_fixed",
    log: f"{log_dir}/prepare_fixed/add_landmarks.log"
    conda: "matchmaker_env"
    shell:
        "python matchmaker/add_landmarks.py "
        "--input_path {input.seg_n5} "
        "--input_key {params.input_key} "
        "--output_path {input.seg_n5} "
        "--output_key {params.output_key} "
        "--landmarks_csv {input.lm_csv} "
        "--transform_json {input.transform_json} "
        "--transform_sample_key {params.transform_sample_key} "
        "--log_dir {params.log_dir}"


rule prepare_moving_with_lm:
    """Add Seymour landmarks to the input segmentation, apply SVD + rigid transforms."""
    input:
        seg_n5         = moving_n5,
        lm_csv         = moving_lm_csv,
        transform_json = prealignment_transform,
        rigid_transform = rigid_transform,
    output:
        directory(f"{moving_n5}/{moving_aligned_key}"),
    params:
        input_key            = moving_input_key,
        output_key           = moving_aligned_key,
        transform_sample_key = "moving_prealignment",
        log_dir              = f"{log_dir}/prepare_moving",
    log: f"{log_dir}/prepare_moving/add_landmarks.log"
    conda: "matchmaker_env"
    shell:
        "python matchmaker/add_landmarks.py "
        "--input_path {input.seg_n5} "
        "--input_key {params.input_key} "
        "--output_path {input.seg_n5} "
        "--output_key {params.output_key} "
        "--landmarks_csv {input.lm_csv} "
        "--transform_json {input.transform_json} "
        "--transform_sample_key {params.transform_sample_key} "
        "--rigid_transform_path {input.rigid_transform} "
        "--log_dir {params.log_dir}"


rule optimize_cpd:
    """Run Optuna grid search over CPD parameters, evaluated by mean LRE."""
    input:
        fixed_ds         = f"{fixed_n5}/{fixed_aligned_key}",
        moving_ds        = f"{moving_n5}/{moving_aligned_key}",
        config           = "data/brain_matching/cpd_optimization_config.yaml",
        landmark_ids_json = f"{log_dir}/prepare_fixed/landmark_label_ids.json",
    output:
        best_params   = f"{log_dir}/best_cpd_params.yaml",
        study_results = f"{log_dir}/study_results.csv",
    log: f"{log_dir}/cpd_optimization.log"
    conda: "matchmaker_env"
    shell:
        "python matchmaker/cpd_optimization.py "
        "--config {input.config} "
        "--landmark_ids_json {input.landmark_ids_json}"
