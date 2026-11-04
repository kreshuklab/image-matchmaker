import shutil
from pathlib import Path

import yaml

from image_matchmaker.workflows import run_snakemake
from image_matchmaker.utils import load_config


if __name__ == "__main__":
    config_path = "register_config_test_elastic.yaml"
    src_folder = Path("data")
    snakemake_folder = Path(".snakemake")
    select_folders = None   # specify a keyword to select folders

    for folder in sorted(src_folder.glob("deformed_data_*")):
        if not folder.is_dir():
            continue
        if select_folders is not None:
            if not any(select_folder in folder.name for select_folder in select_folders):
                continue

        if snakemake_folder.exists():
            shutil.rmtree(snakemake_folder)

        tag = folder.name.removeprefix("deformed_data_")

        try:
            config = load_config(config_path)

            log_dir = Path("results") / tag
            log_dir.mkdir(parents=True, exist_ok=True)
            config["log_dir"] = str(log_dir)

            # The fixed image is shared by all settings, only the moving image changes.
            moving_path = config["moving_image"]["path"]
            config["moving_image"]["path"] = moving_path.replace("deformed_data", folder.name)

            tmp_config_path = log_dir / "registration.yaml"
            with open(tmp_config_path, "w") as f:
                yaml.dump(config, f)

            # Run registration snakemake workflow
            run_snakemake(
                workflow="register",
                configfile=tmp_config_path,
                cores=8,
            )
        except Exception as e:
            print(f"Error encountered: {e}. Skipping...")
            continue
