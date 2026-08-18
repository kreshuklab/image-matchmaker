import json
import logging
from pathlib import Path

import click
import matplotlib
import numpy as np
import open3d as o3d
import yaml

from image_matchmaker.utils import (
    read_volume, get_attrs, extract_centroids, create_pcd, setup_logging, plot_landmark_overlay,
    PLOT_FORMAT,
)
from image_matchmaker.cpd_parameter_tuning.cpd_optimization import compute_lre, pcd_to_label_pos

matplotlib.use("agg")  # non-interactive backend for headless runs


def volume_pcd(path, key):
    """Point cloud of instance centroids (in µm) for one n5 dataset."""
    img = read_volume(path, key)
    resolution = get_attrs(path, key)["resolution"]
    logging.info(f"Read {path}/{key}: shape {img.shape}, resolution {resolution}")
    labels, coords = extract_centroids(img, resolution)
    del img
    return create_pcd(coords, labels)


@click.command()
@click.option("--config", required=True, help="Path to cpd_optimization_config.yaml")
@click.option(
    "--landmark_ids_json",
    default=None,
    help="Path to landmark_label_ids.json (default: <log_dir>/01_prepare_landmarks/landmark_label_ids.json)",
)
@click.option(
    "--fixed_path",
    required=True,
    help="Fixed image n5 container written by the workflow (not the raw input volume)",
)
@click.option(
    "--moving_path",
    required=True,
    help="Moving image n5 container written by the workflow (not the raw input volume)",
)
@click.option(
    "--lm_input_key",
    default="input_with_lm",
    show_default=True,
    help="Key of the unaligned segmentations with landmarks embedded",
)
@click.option(
    "--output_dir",
    default=None,
    help="Where to write the overlays (default: <log_dir>/05_landmark_overlays)",
)
def main(config, landmark_ids_json, fixed_path, moving_path, lm_input_key, output_dir):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    log_dir = Path(cfg["log_dir"])
    cpd_dir = log_dir / "04_cpd_optimization"
    output_dir = Path(output_dir) if output_dir else log_dir / "05_landmark_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_dir), "plot_overlays.log")

    fixed_aligned_key = cfg["fixed_image"]["aligned_key"]
    moving_aligned_key = cfg["moving_image"]["aligned_key"]

    landmark_ids_path = (
        Path(landmark_ids_json) if landmark_ids_json
        else log_dir / "01_prepare_landmarks" / "landmark_label_ids.json"
    )
    with open(landmark_ids_path) as f:
        id_map = {name: int(lbl) for name, lbl in json.load(f).items()}
    logging.info(f"Loaded {len(id_map)} landmark label IDs from {landmark_ids_path}")

    cache = {}

    def stage_pcd(path, key):
        if (path, key) not in cache:
            cache[(path, key)] = volume_pcd(path, key)
        return cache[(path, key)]

    def draw(name, label, fixed_pcd, moving_pcd):
        mean_lre, _ = compute_lre(fixed_pcd, moving_pcd, id_map)
        # PLOT_FORMAT is None when vis.py infers the format from the path, so name a suffix here
        save_path = output_dir / f"landmark_overlay_{name}.{PLOT_FORMAT or 'pdf'}"
        plot_landmark_overlay(
            pcd_to_label_pos(fixed_pcd),
            pcd_to_label_pos(moving_pcd),
            id_map,
            save_path=save_path,
            title=f"{label} | mean LRE {mean_lre:.2f} µm",
            fixed_bg=fixed_pcd.point.positions.numpy(),
            moving_bg=moving_pcd.point.positions.numpy(),
        )
        logging.info(f"{name}: mean LRE {mean_lre:.2f} µm, written to {save_path}")
        return mean_lre

    stages = [
        ("input", "input", (fixed_path, lm_input_key), (moving_path, lm_input_key)),
        ("prealignment", "after prealignment",
         (fixed_path, fixed_aligned_key), (moving_path, fixed_aligned_key)),
        ("rigid_alignment", "after rigid alignment",
         (fixed_path, fixed_aligned_key), (moving_path, moving_aligned_key)),
    ]

    lres = {}
    for name, label, fixed_spec, moving_spec in stages:
        lres[name] = draw(name, label, stage_pcd(*fixed_spec), stage_pcd(*moving_spec))

    with open(cpd_dir / "best_cpd_params.yaml") as f:
        params = yaml.safe_load(f)["coherent_point_drift"]
    registered_pcd = o3d.t.io.read_point_cloud(str(cpd_dir / "registered_pcd.pcd"))
    lres["best_cpd"] = draw(
        "best_cpd",
        f"after CPD | w={params['w']}, β={params['beta']}, λ={params['lmd']}, "
        f"maxiter={params['maxiter']}",
        stage_pcd(fixed_path, fixed_aligned_key),
        registered_pcd,
    )

    logging.info(
        "Mean LRE per stage (µm): "
        + ", ".join(f"{name} {lre:.2f}" for name, lre in lres.items())
    )


if __name__ == "__main__":
    main()
