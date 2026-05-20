"""Add anatomical landmark voxels to a nuclei segmentation and apply registration transforms.

Landmarks are embedded as single-voxel labels with IDs starting at LANDMARK_ID_OFFSET,
so they are carried through SVD and rigid alignment transforms identically to nuclei.
The label ID for a given landmark name is determined by its position in the sorted list of
landmark names, ensuring consistent IDs between fixed and moving images.
"""

import json
import click
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from matchmaker.utils import (
    read_volume, write_volume, get_attrs, read_transform_dict, rotate_img,
    apply_transform_chanwise, setup_logging,
)

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
@click.option("--input_path", required=True, help="Path to input .n5 file")
@click.option("--input_key", required=True, help="Dataset key to read from input .n5")
@click.option("--output_path", required=True, help="Path to output .n5 file (may equal input_path)")
@click.option("--output_key", required=True, help="Dataset key to write in output .n5")
@click.option("--landmarks_csv", required=True, help="CSV with columns [name, x, y, z] in µm")
@click.option("--transform_json", required=True, help="SVD prealignment transform JSON")
@click.option("--transform_sample_key", required=True,
              type=click.Choice(["fixed_prealignment", "moving_prealignment"]),
              help="Which transform entry to apply from the JSON")
@click.option("--rigid_transform_path", default=None,
              help="Elastix TransformParameters.txt for the moving image (optional)")
@click.option("--log_dir", required=True)
def main(input_path, input_key, output_path, output_key, landmarks_csv,
         transform_json, transform_sample_key, rigid_transform_path, log_dir):

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, "add_landmarks.log")

    logging.info(f"Reading segmentation from {input_path}/{input_key}")
    seg = read_volume(input_path, input_key)
    attrs = dict(get_attrs(input_path, input_key))
    resolution = attrs["resolution"]
    logging.info(f"Shape: {seg.shape}, dtype: {seg.dtype}, resolution: {resolution}")

    lm_df = pd.read_csv(landmarks_csv)
    sorted_names = sorted(lm_df["name"].tolist())
    id_map = landmark_label_ids(sorted_names, seg.dtype)
    min_lm_id = min(id_map.values())
    logging.info(f"Adding {len(lm_df)} landmarks (ids {min_lm_id}–{max(id_map.values())})")

    seg_with_lm = add_landmarks_to_seg(seg, lm_df, resolution, id_map)

    transform_dict = read_transform_dict(transform_json)
    T = transform_dict[transform_sample_key]["matrix"]
    output_shape = tuple(transform_dict[transform_sample_key]["output_shape"])
    logging.info(f"Applying SVD transform ({transform_sample_key}) → shape {output_shape}")
    seg_prealigned = rotate_img(seg_with_lm, T, output_shape=output_shape)
    logging.info(f"SVD output shape: {seg_prealigned.shape}")

    result = seg_prealigned

    if rigid_transform_path:
        import itk
        logging.info(f"Applying rigid elastix transform from {rigid_transform_path}")
        parameter_object = itk.ParameterObject.New()
        parameter_object.ReadParameterFile(str(rigid_transform_path))
        # Nearest-neighbor interpolation to preserve integer label values
        parameter_object.SetParameter("FinalBSplineInterpolationOrder", "0")
        seg_rigid = apply_transform_chanwise(
            parameter_object, seg_prealigned.astype(np.float32), resolution
        )
        result = np.round(np.squeeze(seg_rigid)).astype(seg.dtype)
        logging.info(f"Rigid output shape: {result.shape}")

    write_volume(output_path, result, output_key, attrs=attrs)
    logging.info(f"Written to {output_path}/{output_key}")

    label_id_path = Path(log_dir) / "landmark_label_ids.json"
    with open(label_id_path, "w") as f:
        json.dump(id_map, f, indent=2)
    logging.info(f"Label ID mapping written to {label_id_path}")


if __name__ == "__main__":
    main()
