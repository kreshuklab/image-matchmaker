import json
import click
import logging
import numpy as np
import pandas as pd
import yaml
from functools import reduce
from pathlib import Path

import optuna
from optuna.samplers import GridSampler

from matchmaker.utils import (
    read_volume, extract_centroids, run_cpd, create_pcd, setup_logging,
)

_ranges_file = Path(__file__).parent / "default_cpd_ranges.yaml"
with open(_ranges_file) as _f:
    DEFAULT_SEARCH_SPACE = yaml.safe_load(_f)


def compute_lre(fixed_pcd, registered_pcd, id_map):
    """Compute mean and per-landmark LRE (µm) between fixed and CPD-registered point clouds.

    Args:
        fixed_pcd: Open3D point cloud for the fixed image, with landmark labels embedded
        registered_pcd: Open3D point cloud after CPD, with moving landmark labels at new positions
        id_map: {landmark_name: label_id}

    Returns:
        (mean_lre, per_landmark_dict)
    """
    def pcd_to_label_pos(pcd):
        labels = pcd.point.label.numpy()[:, 0].astype(int)
        positions = pcd.point.positions.numpy()
        return dict(zip(labels, positions))

    fixed_pos = pcd_to_label_pos(fixed_pcd)
    reg_pos = pcd_to_label_pos(registered_pcd)

    distances, per_lm = [], {}
    for name, lbl in id_map.items():
        if lbl in fixed_pos and lbl in reg_pos:
            d = float(np.linalg.norm(fixed_pos[lbl] - reg_pos[lbl]))
            distances.append(d)
            per_lm[name] = d
        else:
            logging.warning(f"Landmark {name!r} (id={lbl}) missing from point clouds")

    mean_lre = float(np.mean(distances)) if distances else float("inf")
    return mean_lre, per_lm


def suggest_beta_ranges(coords):
    """Compute 4 beta values covering extent/[40, 20, 10, 5] from point cloud coords (µm)."""
    if len(coords) == 0:
        logging.warning("Empty point cloud, falling back to default beta ranges")
        return DEFAULT_SEARCH_SPACE["beta"]
    extents = coords.max(axis=0) - coords.min(axis=0)
    mean_extent = float(np.mean(extents))
    logging.info(f"Point cloud extent per axis (µm): {extents}, mean: {mean_extent:.1f}")
    betas = sorted({
        mean_extent / 40,
        mean_extent / 20,
        mean_extent / 10,
        mean_extent / 5,
    })
    return np.round(betas, 2).tolist()


def run_optimization(fixed_pcd, moving_pcd, fixed_labels, moving_labels,
                     id_map, search_space, n_trials, study_name, output_dir):
    """Run Optuna grid search and save results."""

    logging.info(f"Fixed point cloud: {len(fixed_labels)} points")
    logging.info(f"Moving point cloud: {len(moving_labels)} points")
    min_lm_id = min(id_map.values())
    n_landmark_fixed = sum(1 for lbl in fixed_labels if lbl >= min_lm_id)
    n_landmark_moving = sum(1 for lbl in moving_labels if lbl >= min_lm_id)
    logging.info(f"Landmarks in fixed pcd: {n_landmark_fixed}, in moving pcd: {n_landmark_moving}")

    trial_results = []

    def objective(trial):
        w = trial.suggest_categorical("w", search_space["w"])
        beta = trial.suggest_categorical("beta", search_space["beta"])
        lmd = trial.suggest_categorical("lmd", search_space["lmd"])
        maxiter = trial.suggest_categorical("maxiter", search_space["maxiter"])

        logging.info(f"Trial {trial.number}: w={w}, beta={beta}, lmd={lmd}, maxiter={maxiter}")
        registered_pcd = run_cpd(fixed_pcd, moving_pcd, w, beta, lmd, maxiter)
        mean_lre, per_lm = compute_lre(fixed_pcd, registered_pcd, id_map)
        logging.info(f"  → mean LRE = {mean_lre:.2f} µm")

        result = {
            "trial": trial.number,
            "w": w,
            "beta": beta,
            "lmd": lmd,
            "maxiter": maxiter,
            "mean_lre_um": mean_lre,
            "per_landmark_um": per_lm,
        }
        trial_results.append(result)
        with open(output_dir / f"trial_{trial.number:04d}.json", "w") as f:
            json.dump(result, f, indent=2)

        return mean_lre

    sampler = GridSampler(search_space)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    logging.info(
        f"Best trial {best.number}: w={best.params['w']}, beta={best.params['beta']}, "
        f"lmd={best.params['lmd']}, maxiter={best.params['maxiter']} "
        f"→ mean LRE = {best.value:.2f} µm"
    )

    best_params = {
        "coherent_point_drift": {
            "w": best.params["w"],
            "beta": best.params["beta"],
            "lmd": best.params["lmd"],
            "maxiter": best.params["maxiter"],
        },
        "optimization": {
            "mean_lre_um": best.value,
            "n_trials_run": len(study.trials),
        },
    }
    best_params_path = output_dir / "best_cpd_params.yaml"
    with open(best_params_path, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)
    logging.info(f"Best parameters written to {best_params_path}")

    summary = pd.DataFrame([
        {
            "trial": r["trial"],
            "w": r["w"],
            "beta": r["beta"],
            "lmd": r["lmd"],
            "maxiter": r["maxiter"],
            "mean_lre_um": r["mean_lre_um"],
        }
        for r in trial_results
    ])
    summary_path = output_dir / "study_results.csv"
    summary.sort_values("mean_lre_um").to_csv(summary_path, index=False)
    logging.info(f"Study summary written to {summary_path}")

    return best_params_path


@click.command()
@click.option("--config", required=True, help="Path to cpd_optimization_config.yaml")
@click.option("--landmark_ids_json", default=None, help="Path to landmark_label_ids.json (default: <log_dir>/landmark_label_ids.json)")
def main(config, landmark_ids_json):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["log_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_dir), "cpd_optimization.log")

    fixed_path = cfg["fixed_image"]["path"]
    fixed_key = cfg["fixed_image"]["aligned_key"]
    fixed_resolution = [
        cfg["fixed_image"]["z_res"],
        cfg["fixed_image"]["y_res"],
        cfg["fixed_image"]["x_res"],
    ]

    moving_path = cfg["moving_image"]["path"]
    moving_key = cfg["moving_image"]["aligned_key"]
    moving_resolution = [
        cfg["moving_image"]["z_res"],
        cfg["moving_image"]["y_res"],
        cfg["moving_image"]["x_res"],
    ]

    landmark_ids_path = Path(landmark_ids_json) if landmark_ids_json else output_dir / "landmark_label_ids.json"
    with open(landmark_ids_path) as f:
        id_map = {name: int(lbl) for name, lbl in json.load(f).items()}
    logging.info(f"Loaded {len(id_map)} landmark label IDs from {landmark_ids_path}")

    logging.info("Extracting point clouds (done once, reused across all trials)")
    fixed_img = read_volume(fixed_path, fixed_key)
    moving_img = read_volume(moving_path, moving_key)
    fixed_labels, fixed_coords = extract_centroids(fixed_img, fixed_resolution)
    moving_labels, moving_coords = extract_centroids(moving_img, moving_resolution)
    fixed_pcd = create_pcd(fixed_coords, fixed_labels)
    moving_pcd = create_pcd(moving_coords, moving_labels)

    search_space_cfg = cfg.get("optuna", {}).get("search_space", "default")
    if isinstance(search_space_cfg, dict):
        search_space = {k: list(v) for k, v in search_space_cfg.items()}
        logging.info("Using manually specified search space from config")
    elif search_space_cfg == "dataset-specific":
        search_space = dict(DEFAULT_SEARCH_SPACE)
        search_space["beta"] = suggest_beta_ranges(fixed_coords)
        logging.info(f"Dataset-specific beta ranges: {search_space['beta']}")
    else:
        search_space = DEFAULT_SEARCH_SPACE
        logging.info("Using default search space from default_cpd_ranges.yaml")
    n_combinations = reduce(lambda a, b: a * b, (len(v) for v in search_space.values()), 1)
    logging.info(f"Grid search space: {search_space}")
    logging.info(f"Total combinations: {n_combinations}")

    study_name = cfg.get("optuna", {}).get("study_name", "cpd_optimization")

    run_optimization(
        fixed_pcd=fixed_pcd,
        moving_pcd=moving_pcd,
        fixed_labels=fixed_labels,
        moving_labels=moving_labels,
        id_map=id_map,
        search_space=search_space,
        n_trials=n_combinations,
        study_name=study_name,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
