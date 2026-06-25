import json
import click
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from matchmaker.utils import read_volume, write_volume, get_attrs, setup_logging, plot_landmark_qc, PINK, CYAN


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
@click.option("--fixed_key", required=True, help="Dataset key of the raw fixed segmentation")
@click.option("--fixed_output_key", required=True, help="Dataset key to write the fixed segmentation with landmarks")
@click.option("--fixed_landmarks_csv", required=True, help="CSV with columns [name, x, y, z] in µm for the fixed image")
@click.option("--moving_path", required=True, help="Path to the moving raw .n5 segmentation")
@click.option("--moving_key", required=True, help="Dataset key of the raw moving segmentation")
@click.option("--moving_output_key", required=True, help="Dataset key to write the moving segmentation with landmarks")
@click.option(
    "--moving_landmarks_csv",
    required=True,
    help="CSV with columns [name, x, y, z] in µm for the moving image")
@click.option("--log_dir", required=True, help="Output directory for logs and landmark_label_ids.json")
def main(fixed_path, fixed_key, fixed_output_key, fixed_landmarks_csv,
         moving_path, moving_key, moving_output_key, moving_landmarks_csv,
         log_dir):

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, "add_landmarks.log")

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

    logging.info(f"Embedding {len(fixed_lm_df)} landmarks into fixed segmentation")
    fixed_seg_lm = add_landmarks_to_seg(fixed_seg, fixed_lm_df, fixed_resolution, id_map)

    logging.info(f"Embedding {len(moving_lm_df)} landmarks into moving segmentation")
    moving_seg_lm = add_landmarks_to_seg(moving_seg, moving_lm_df, moving_resolution, id_map)

    logging.info(f"Writing fixed with landmarks to {fixed_path}/{fixed_output_key}")
    write_volume(fixed_path, fixed_seg_lm, fixed_output_key, attrs=fixed_attrs)

    logging.info(f"Writing moving with landmarks to {moving_path}/{moving_output_key}")
    write_volume(moving_path, moving_seg_lm, moving_output_key, attrs=moving_attrs)

    label_id_path = Path(log_dir) / "landmark_label_ids.json"
    with open(label_id_path, "w") as f:
        json.dump(id_map, f, indent=2)
    logging.info(f"Label ID mapping written to {label_id_path}")

    plots_dir = Path(log_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_landmark_qc(fixed_seg_lm, id_map,
                     save_path=plots_dir / "fixed_landmarks_qc.png",
                     cell_cmap=PINK, landmark_color="red")
    plot_landmark_qc(moving_seg_lm, id_map,
                     save_path=plots_dir / "moving_landmarks_qc.png",
                     cell_cmap=CYAN, landmark_color="blue")
    logging.info(f"Landmark QC plots written to {plots_dir}")


if __name__ == "__main__":
    main()
