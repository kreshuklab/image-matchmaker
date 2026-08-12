import json
import os
import shutil
import tempfile
import threading
import click
import logging
import numpy as np
import open3d as o3d
import pandas as pd
import yaml
from functools import reduce
from pathlib import Path

import optuna
from optuna.samplers import GridSampler

from image_matchmaker.utils import (
    read_volume, get_attrs, extract_centroids, cpd_from_pcds, create_pcd, setup_logging,
    plot_displacement_field_panels,
)
from image_matchmaker.cpd_parameter_tuning.suggest_cpd_ranges import suggest_beta_ranges

import matplotlib
matplotlib.use("agg")  # non-interactive backend required for worker threads

_ranges_file = Path(__file__).parent / "default_cpd_ranges.yaml"
with open(_ranges_file) as _f:
    DEFAULT_SEARCH_SPACE = yaml.safe_load(_f)


def pcd_to_label_pos(pcd):
    """{label_id: (x, y, z)} for every point of a labelled point cloud."""
    # reshape(-1): the label attribute is (N, 1) in memory but (N,) when read back from .pcd
    labels = np.asarray(pcd.point.label.numpy()).reshape(-1).astype(int)
    positions = pcd.point.positions.numpy()
    return dict(zip(labels, positions))


def compute_lre(fixed_pcd, registered_pcd, id_map):
    """Compute mean and per-landmark LRE (µm) between fixed and CPD-registered point clouds.

    Args:
        fixed_pcd: Open3D point cloud for the fixed image, with landmark labels embedded
        registered_pcd: Open3D point cloud after CPD, with moving landmark labels at new positions
        id_map: {landmark_name: label_id}

    Returns:
        (mean_lre, per_landmark_dict)
    """
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


def run_optimization(fixed_pcd, moving_pcd, fixed_labels, moving_labels,
                     id_map, search_space, n_trials, study_name, output_dir, n_jobs=1):
    """Run Optuna grid search and save results."""

    logging.info(f"Fixed point cloud: {len(fixed_labels)} points")
    logging.info(f"Moving point cloud: {len(moving_labels)} points")
    min_lm_id = min(id_map.values())
    n_landmark_fixed = sum(1 for lbl in fixed_labels if lbl >= min_lm_id)
    n_landmark_moving = sum(1 for lbl in moving_labels if lbl >= min_lm_id)
    logging.info(f"Landmarks in fixed pcd: {n_landmark_fixed}, in moving pcd: {n_landmark_moving}")

    trial_results = []
    results_lock = threading.Lock()
    best_so_far = {"lre": float("inf"), "pcd": None}

    def objective(trial):
        w = trial.suggest_categorical("w", search_space["w"])
        beta = trial.suggest_categorical("beta", search_space["beta"])
        lmd = trial.suggest_categorical("lmd", search_space["lmd"])
        maxiter = trial.suggest_categorical("maxiter", search_space["maxiter"])

        logging.info(f"Trial {trial.number}: w={w}, beta={beta}, lmd={lmd}, maxiter={maxiter}")
        registered_pcd = cpd_from_pcds(fixed_pcd, moving_pcd, w, beta, lmd, maxiter)
        mean_lre, per_lm = compute_lre(fixed_pcd, registered_pcd, id_map)
        logging.info(f"  → mean LRE = {mean_lre:.2f} µm")

        plot_title = f"Trial {trial.number:04d} | w={w}, β={beta}, λ={lmd}, maxiter={maxiter} | LRE={mean_lre:.2f} µm"
        plot_displacement_field_panels(
            moving_pcd,
            registered_pcd,
            title=plot_title,
            save_path=plots_dir / f"trial_{trial.number:04d}.pdf",
        )

        result = {
            "trial": trial.number,
            "w": w,
            "beta": beta,
            "lmd": lmd,
            "maxiter": maxiter,
            "mean_lre_um": mean_lre,
            "per_landmark_um": per_lm,
        }
        with results_lock:
            trial_results.append(result)
            if mean_lre < best_so_far["lre"]:
                best_so_far.update(lre=mean_lre, pcd=registered_pcd)
        with open(output_dir / f"trial_{trial.number:04d}.json", "w") as f:
            json.dump(result, f, indent=2)

        return mean_lre

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    final_db_path = output_dir / "optuna_study.db"
    local_base = os.environ.get("LOCAL_TMPDIR") or tempfile.gettempdir()
    local_db_dir = Path(tempfile.mkdtemp(prefix="optuna_", dir=local_base))
    local_db_path = local_db_dir / "optuna_study.db"
    if final_db_path.exists():
        shutil.copy2(final_db_path, local_db_path)
        logging.info(f"Seeded local study DB from existing {final_db_path}")
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{local_db_path}",
        engine_kwargs={"connect_args": {"timeout": 100}},
    )

    sampler = GridSampler(search_space)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )
    logging.info(f"Optuna study stored on local disk at {local_db_path}")
    logging.info(f"Running {n_trials} trials with n_jobs={n_jobs}")
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    finally:
        try:
            shutil.copy2(local_db_path, final_db_path)
            logging.info(f"Copied study DB to {final_db_path}")
        except OSError as e:
            logging.warning(f"Could not copy study DB to {final_db_path}: {e}")

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

    if best_so_far["pcd"] is None:
        raise RuntimeError(
            f"No trial ran in this process, so there is no point cloud to save. The study "
            f"'{study_name}' in {final_db_path} is already complete. Delete that database as "
            "well to rerun the grid search from scratch."
        )
    if best_so_far["lre"] > best.value:
        logging.warning(
            f"Best trial of the study has LRE {best.value:.4g} µm but ran in an earlier process; "
            f"the best cloud available here is from a trial with LRE {best_so_far['lre']:.4g} µm. "
            f"registered_pcd.pcd is that cloud, not the one described by {best_params_path.name}, "
            f"so the best_cpd overlay does not match those parameters. "
            f"Delete {final_db_path.name} and rerun to make the two agree."
        )
    registered_pcd_path = output_dir / "registered_pcd.pcd"
    o3d.t.io.write_point_cloud(str(registered_pcd_path), best_so_far["pcd"], write_ascii=True)
    logging.info(f"Registered point cloud of the best trial written to {registered_pcd_path}")

    return best_params_path


@click.command()
@click.option("--config", required=True, help="Path to cpd_optimization_config.yaml")
@click.option(
    "--landmark_ids_json",
    default=None,
    help="Path to landmark_label_ids.json (default: <log_dir>/01_prepare_landmarks/landmark_label_ids.json)",
)
@click.option(
    "--fixed_path", default=None, help="Override fixed image n5 path from config"
)
@click.option(
    "--moving_path", default=None, help="Override moving image n5 path from config"
)
def main(config, landmark_ids_json, fixed_path, moving_path):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    log_dir = Path(cfg["log_dir"])
    output_dir = log_dir / "04_cpd_optimization"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_dir), "cpd_optimization.log")

    fixed_path = fixed_path or cfg["fixed_image"]["path"]
    fixed_key = cfg["fixed_image"]["aligned_key"]

    moving_path = moving_path or cfg["moving_image"]["path"]
    moving_key = cfg["moving_image"]["aligned_key"]

    landmark_ids_path = (
        Path(landmark_ids_json) if landmark_ids_json
        else log_dir / "01_prepare_landmarks" / "landmark_label_ids.json"
    )
    with open(landmark_ids_path) as f:
        id_map = {name: int(lbl) for name, lbl in json.load(f).items()}
    logging.info(f"Loaded {len(id_map)} landmark label IDs from {landmark_ids_path}")

    logging.info("Extracting point clouds (done once, reused across all trials)")

    logging.info("Reading fixed image")
    fixed_img = read_volume(fixed_path, fixed_key)
    fixed_resolution = get_attrs(fixed_path, fixed_key)["resolution"]
    logging.info(f"Fixed image shape: {fixed_img.shape}, dtype {fixed_img.dtype}")

    logging.info("Reading moving image")
    moving_img = read_volume(moving_path, moving_key)
    moving_resolution = get_attrs(moving_path, moving_key)["resolution"]
    logging.info(f"Moving image shape: {moving_img.shape}, dtype {moving_img.dtype}")

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
        search_space["beta"] = suggest_beta_ranges(fixed_coords, fallback_betas=DEFAULT_SEARCH_SPACE["beta"])
        logging.info(f"Dataset-specific beta ranges: {search_space['beta']}")
    else:
        search_space = DEFAULT_SEARCH_SPACE
        logging.info("Using default search space from default_cpd_ranges.yaml")
    n_combinations = reduce(lambda a, b: a * b, (len(v) for v in search_space.values()), 1)
    logging.info(f"Grid search space: {search_space}")
    logging.info(f"Total combinations: {n_combinations}")

    study_name = cfg.get("optuna", {}).get("study_name", "cpd_optimization")
    n_jobs = cfg.get("optuna", {}).get("n_jobs", 1)

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
        n_jobs=n_jobs,
    )


if __name__ == "__main__":
    main()
