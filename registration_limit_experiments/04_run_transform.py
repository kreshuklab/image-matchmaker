import shutil
from pathlib import Path

import yaml

from image_matchmaker.workflows import run_snakemake
from image_matchmaker.utils import load_config


def set_setting_paths(config, tag):
    """Replace the "deformed_data" placeholder: inputs in data/deformed_data_<tag>/, outputs in results/<tag>/."""
    def to_results(path):
        return path.replace("results/deformed_data", f"results/{tag}")

    config["log_dir"] = to_results(config["log_dir"])
    config["parameter_map_path"] = to_results(config["parameter_map_path"])
    config["prealignment_transform_path"] = to_results(config["prealignment_transform_path"])
    for moving_image in config["moving_images"]:
        moving_image["input_path"] = moving_image["input_path"].replace(
            "data/deformed_data", f"data/deformed_data_{tag}")
        moving_image["output_path"] = to_results(moving_image["output_path"])
    return config


if __name__ == "__main__":
    config_path = "register_config_test_elastic_apply_transform.yaml"
    results_folder = Path("results")
    snakemake_folder = Path(".snakemake")

    for result_dir in sorted(results_folder.iterdir()):
        transform_path = result_dir / "05_elastix_deformable_pointset_registration" / "TransformParameters.2.txt"
        if not transform_path.exists():
            continue

        if snakemake_folder.exists():
            shutil.rmtree(snakemake_folder)

        try:
            config = set_setting_paths(load_config(config_path), result_dir.name)

            log_dir = Path(config["log_dir"])
            log_dir.mkdir(parents=True, exist_ok=True)

            tmp_config_path = log_dir / "apply_transform.yaml"
            with open(tmp_config_path, "w") as f:
                yaml.dump(config, f)

            # Run apply_transform snakemake workflow
            run_snakemake(
                workflow="apply",
                configfile=tmp_config_path,
                cores=8,
            )

        except Exception as e:
            print(f"Error encountered: {e}. Skipping...")
            continue
