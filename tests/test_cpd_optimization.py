"""End-to-end test of the CPD parameter search on a synthetic point cloud.

No image data or Snakemake run involved: the search operates on point clouds, so a
jittered grid of "cells" plus five landmarks is enough to check that the grid search
ranks its trials, writes every artefact the workflow depends on, and that the point
cloud it refits for the winning parameters really is that trial's result.
"""
import numpy as np
import open3d as o3d
import pandas as pd
import pytest
import yaml

from image_matchmaker.utils import create_pcd
from image_matchmaker.cpd_parameter_tuning.cpd_optimization import (
    compute_lre,
    run_optimization,
)

N_SIDE = 5
SPACING = 15.0
EXTENT = (N_SIDE - 1) * SPACING
LANDMARK_ID = 1000  # landmark labels start well above the cell labels, as add_landmarks does

# Two beta values and two lmd values, so the four trials rank differently.
SEARCH_SPACE = {"w": [1.0e-4], "beta": [20.0, 80.0], "lmd": [0.1, 1.0], "maxiter": [50]}


def make_point_clouds():
    """A fixed cloud and a smoothly deformed copy of it, both with the same labels.

    The deformation is a shift plus a gentle bend along z - small relative to the point
    spacing, so the correspondence stays unambiguous and CPD can recover it.
    """
    rng = np.random.default_rng(0)
    grid = np.stack(
        np.meshgrid(*(np.arange(N_SIDE) * SPACING,) * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    cells = grid + rng.uniform(-3, 3, grid.shape)
    landmarks = EXTENT * np.array(
        [[0.2, 0.2, 0.2], [0.8, 0.2, 0.5], [0.2, 0.8, 0.5], [0.5, 0.5, 0.8], [0.8, 0.8, 0.2]]
    )

    fixed = np.vstack([cells, landmarks])
    labels = np.concatenate(
        [np.arange(1, len(cells) + 1), np.arange(LANDMARK_ID, LANDMARK_ID + len(landmarks))]
    )

    moving = fixed.copy()
    moving[:, 0] += 6.0 * np.sin(fixed[:, 2] / (EXTENT / 3)) + 3.0
    moving[:, 1] += 2.25

    id_map = {f"lm{i}": int(LANDMARK_ID + i) for i in range(len(landmarks))}
    return create_pcd(fixed, labels), create_pcd(moving, labels), labels, id_map


def test_cpd_optimization(tmp_path, monkeypatch):
    # keeps the Optuna study DB (and the temp dir run_optimization makes for it) out of /tmp
    monkeypatch.setenv("LOCAL_TMPDIR", str(tmp_path))
    output_dir = tmp_path / "04_cpd_optimization"
    output_dir.mkdir()

    fixed_pcd, moving_pcd, labels, id_map = make_point_clouds()
    lre_before, _ = compute_lre(fixed_pcd, moving_pcd, id_map)

    best_params_path = run_optimization(
        fixed_pcd=fixed_pcd,
        moving_pcd=moving_pcd,
        fixed_labels=labels,
        moving_labels=labels,
        id_map=id_map,
        search_space=SEARCH_SPACE,
        n_trials=4,
        study_name="test_cpd_optimization",
        output_dir=output_dir,
        n_jobs=1,
    )

    # every artefact the workflow (and plot_overlays) expects
    assert best_params_path == output_dir / "best_cpd_params.yaml"
    for name in ("best_cpd_params.yaml", "study_results.csv", "registered_pcd.pcd",
                 "optuna_study.db"):
        assert (output_dir / name).exists(), f"{name} missing"
    assert len(list(output_dir.glob("trial_*.json"))) == 4
    assert len(list((output_dir / "plots").glob("trial_*.*"))) == 4  # suffix follows PLOT_FORMAT

    best = yaml.safe_load(best_params_path.read_text())
    cpd_params = best["coherent_point_drift"]
    assert set(cpd_params) == {"w", "beta", "lmd", "maxiter"}
    for key, value in cpd_params.items():
        assert value in SEARCH_SPACE[key]

    # the search ran the full grid and ranked it, best first
    summary = pd.read_csv(output_dir / "study_results.csv")
    assert len(summary) == 4
    assert summary["mean_lre_um"].is_monotonic_increasing
    assert summary["mean_lre_um"].notna().all()

    lre_best = best["optimization"]["mean_lre_um"]
    assert lre_best == pytest.approx(summary["mean_lre_um"].min())
    for key, value in cpd_params.items():
        assert summary.iloc[0][key] == value

    # the winning parameters beat the unregistered baseline by a wide margin (~16x here)
    assert lre_best < lre_before / 5, f"LRE {lre_before:.2f} -> {lre_best:.2f} µm is no better"

    # the refit point cloud reproduces the winning trial, and keeps its labels through
    # the .pcd round trip - plot_overlays reads it back to draw the CPD stage
    registered_pcd = o3d.t.io.read_point_cloud(str(output_dir / "registered_pcd.pcd"))
    assert len(registered_pcd.point.positions) == len(moving_pcd.point.positions)
    lre_refit, per_landmark = compute_lre(fixed_pcd, registered_pcd, id_map)
    assert len(per_landmark) == len(id_map)
    assert abs(lre_refit - lre_best) < 1e-3, "refit does not match the reported best trial"
