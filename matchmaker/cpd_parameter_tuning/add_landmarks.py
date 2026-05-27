"""Add anatomical landmark voxels to nuclei segmentations, then run prealignment and rigid alignment.

Landmarks are embedded as small sphere labels with high IDs into the raw (unaligned)
segmentation volumes first, before any registration. They are then carried through SVD
prealignment and elastix rigid alignment together with the nuclei, ending up in the same
aligned space ready for CPD hyperparameter tuning.
"""

import json
import click
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from matchmaker.utils import read_volume, write_volume, get_attrs, setup_logging
from matchmaker.prealignment import run_prealignment
from matchmaker.rigid_alignment_elastix import run_rigid_alignment


def landmark_label_ids(sorted_names, dtype):
    """Return dict {name: label_id} placing IDs at the top of the dtype range.

    IDs are assigned from (dtype_max - n_landmarks + 1) to dtype_max in sorted-name order,
    so the mapping is fully determined by the landmark names and dtype — independent of
    which segmentation (fixed or moving) is being processed, ensuring consistency.
    """
    max_val = int(np.iinfo(dtype).max)
    offset = max_val - len(sorted_names) + 1
    return {name: offset + i for i, name in enumerate(sorted_names)}


def add_landmarks_to_seg(seg, landmarks_df, resolution, id_map, radius=3):
    """Embed landmark spheres into a segmentation at positions given by landmarks_df.

    Each landmark is written as a filled sphere of the given radius (in voxels).
    This ensures the landmark label survives affine transforms and nearest-neighbor
    interpolation, where a single voxel would often be missed.

    Args:
        seg: integer ZYX numpy array
        landmarks_df: DataFrame with columns [name, x, y, z] in physical µm
        resolution: [z_res, y_res, x_res] in µm
        id_map: {landmark_name: label_id}
        radius: sphere radius in voxels

    Returns modified segmentation (copy).
    """
    min_lm_id = min(id_map.values())
    if int(seg.max()) >= min_lm_id:
        logging.warning(
            f"Existing label max ({int(seg.max())}) >= landmark ID offset ({min_lm_id}). "
            "Landmark labels may overwrite nucleus labels."
        )
    seg_out = seg.copy()
    shape = np.array(seg.shape)
    offsets = [
        (dz, dy, dx)
        for dz in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dz**2 + dy**2 + dx**2 <= radius**2
    ]
    for row in landmarks_df.itertuples(index=False):
        if row.name not in id_map:
            logging.warning(f"Landmark {row.name!r} not in id_map, skipping")
            continue
        lbl = id_map[row.name]
        z_c = row.z / resolution[0]
        y_c = row.y / resolution[1]
        x_c = row.x / resolution[2]
        for dz, dy, dx in offsets:
            z_v = int(np.clip(round(z_c + dz), 0, shape[0] - 1))
            y_v = int(np.clip(round(y_c + dy), 0, shape[1] - 1))
            x_v = int(np.clip(round(x_c + dx), 0, shape[2] - 1))
            seg_out[z_v, y_v, x_v] = lbl
        logging.info(f"  {row.name}: label={lbl}, center=({z_c:.1f},{y_c:.1f},{x_c:.1f}), r={radius}")
    return seg_out


@click.command()
@click.option("--fixed_path", required=True, help="Path to the fixed raw .n5 segmentation")
@click.option("--fixed_key", required=True, help="Dataset key of the raw (unaligned) fixed segmentation")
@click.option("--fixed_output_key", required=True, help="Dataset key to write the prealigned fixed segmentation with landmarks")
@click.option("--fixed_landmarks_csv", required=True, help="CSV with columns [name, x, y, z] in µm for the fixed image")
@click.option("--moving_path", required=True, help="Path to the moving raw .n5 segmentation")
@click.option("--moving_key", required=True, help="Dataset key of the raw (unaligned) moving segmentation")
@click.option("--moving_output_key", required=True, help="Dataset key to write the aligned moving segmentation with landmarks")
@click.option("--moving_landmarks_csv", required=True, help="CSV with columns [name, x, y, z] in µm for the moving image")
@click.option("--axis_orientation", required=True,
              type=click.Choice(["auto", "IDENTITY", "X", "Y", "Z"]),
              help="Axis orientation correction to apply to the moving image after SVD prealignment")
@click.option("--log_dir", required=True, help="Output directory for logs, plots, and the landmark label ID mapping")
def main(fixed_path, fixed_key, fixed_output_key, fixed_landmarks_csv,
         moving_path, moving_key, moving_output_key, moving_landmarks_csv,
         axis_orientation, log_dir):

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, "add_landmarks.log")

    # Read raw segmentations
    logging.info(f"Reading fixed segmentation from {fixed_path}/{fixed_key}")
    fixed_seg = read_volume(fixed_path, fixed_key)
    fixed_attrs = dict(get_attrs(fixed_path, fixed_key))
    fixed_resolution = fixed_attrs["resolution"]
    logging.info(f"Fixed shape: {fixed_seg.shape}, dtype: {fixed_seg.dtype}, resolution: {fixed_resolution}")

    logging.info(f"Reading moving segmentation from {moving_path}/{moving_key}")
    moving_seg = read_volume(moving_path, moving_key)
    moving_attrs = dict(get_attrs(moving_path, moving_key))
    moving_resolution = moving_attrs["resolution"]
    logging.info(f"Moving shape: {moving_seg.shape}, dtype: {moving_seg.dtype}, resolution: {moving_resolution}")

    # Only use landmarks that have a correspondence in both files
    fixed_lm_df = pd.read_csv(fixed_landmarks_csv)
    moving_lm_df = pd.read_csv(moving_landmarks_csv)
    shared_names = sorted(set(fixed_lm_df["name"].tolist()) & set(moving_lm_df["name"].tolist()))
    fixed_only = set(fixed_lm_df["name"]) - set(shared_names)
    moving_only = set(moving_lm_df["name"]) - set(shared_names)
    if fixed_only:
        logging.warning(f"Landmarks only in fixed CSV, skipping: {sorted(fixed_only)}")
    if moving_only:
        logging.warning(f"Landmarks only in moving CSV, skipping: {sorted(moving_only)}")
    fixed_lm_df = fixed_lm_df[fixed_lm_df["name"].isin(shared_names)]
    moving_lm_df = moving_lm_df[moving_lm_df["name"].isin(shared_names)]
    id_map = landmark_label_ids(shared_names, fixed_seg.dtype)
    min_lm_id = min(id_map.values())
    logging.info(f"{len(shared_names)} corresponding landmarks, ids {min_lm_id}–{max(id_map.values())}")

    # Embed landmarks into both raw segmentations before any alignment
    logging.info(f"Embedding {len(fixed_lm_df)} landmarks into fixed segmentation")
    fixed_seg_lm = add_landmarks_to_seg(fixed_seg, fixed_lm_df, fixed_resolution, id_map)

    logging.info(f"Embedding {len(moving_lm_df)} landmarks into moving segmentation")
    moving_seg_lm = add_landmarks_to_seg(moving_seg, moving_lm_df, moving_resolution, id_map)

    # SVD prealignment — landmarks are carried through the transform together with the nuclei
    fixed_spacing = np.asarray(fixed_resolution, dtype=np.float32)
    moving_spacing = np.asarray(moving_resolution, dtype=np.float32)
    new_spacing = np.full_like(fixed_spacing, fixed_spacing.min())

    logging.info("Running SVD prealignment")
    fixed_prealigned, moving_prealigned, _ = run_prealignment(
        fixed_seg_lm, moving_seg_lm,
        fixed_spacing, moving_spacing, new_spacing,
        log_dir, axis_orientation,
    )

    # Rigid alignment of the moving image
    logging.info("Running elastix rigid alignment of moving image")
    prealigned_resolution = new_spacing.tolist()
    moving_rigid_aligned = run_rigid_alignment(
        fixed_prealigned, prealigned_resolution,
        moving_prealigned, prealigned_resolution,
        log_dir,
    ).astype(moving_seg.dtype)

    # Write outputs
    logging.info(f"Writing fixed prealigned with landmarks to {fixed_path}/{fixed_output_key}")
    fixed_out_attrs = fixed_attrs.copy()
    fixed_out_attrs["resolution"] = prealigned_resolution
    write_volume(fixed_path, fixed_prealigned, fixed_output_key, attrs=fixed_out_attrs)

    logging.info(f"Writing moving aligned with landmarks to {moving_path}/{moving_output_key}")
    moving_out_attrs = moving_attrs.copy()
    moving_out_attrs["resolution"] = prealigned_resolution
    write_volume(moving_path, moving_rigid_aligned, moving_output_key, attrs=moving_out_attrs)

    label_id_path = Path(log_dir) / "landmark_label_ids.json"
    with open(label_id_path, "w") as f:
        json.dump(id_map, f, indent=2)
    logging.info(f"Label ID mapping written to {label_id_path}")


if __name__ == "__main__":
    main()
