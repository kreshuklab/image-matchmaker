"""Register all settings with NucleiSky3D.

Runs in its own environment with nucleisky (Python >= 3.12), so it does not import image_matchmaker.
"""
from pathlib import Path

import tifffile as tiff
from tqdm import tqdm

from nucleisky3d.features import extract_nuclear_features_3d, centroids_from_df_3d
from nucleisky3d.pipeline import NucleiSky3D
from nucleisky3d.export import export_aligned_crop_tiff


FIXED_PATH = Path("data/platy1_muscles_stardist_fixed_rotated.tif")
MOVING_NAME = "platy1_muscles_stardist_elastic"
SPACING = (1., 1., 1.)  # (z, y, x) in µm, the same for fixed and moving


def features(labels):
    """Return the NucleiSky feature table and the centroids in µm."""
    df = extract_nuclear_features_3d(labels, pixel_size_um=SPACING)
    return df, centroids_from_df_3d(df, use_um=True)


if __name__ == "__main__":
    fixed = tiff.imread(FIXED_PATH)
    df_fixed, centroids_fixed = features(fixed)

    for data_dir in tqdm(sorted(Path("data").glob("deformed_data_*"))):
        tag = data_dir.name.removeprefix("deformed_data_")
        out_dir = Path("results") / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            moving = tiff.imread(data_dir / f"{MOVING_NAME}.tif")
            df_moving, centroids_moving = features(moving)

            # NucleiSky aligns the moving ("crop") to the fixed ("full") with a similarity transform.
            res = NucleiSky3D(
                centroids_crop_um=centroids_moving,
                centroids_full_um=centroids_fixed,
                full_shape_px_zyx=fixed.shape,
                crop_shape_px_zyx=moving.shape,
                pixel_size_full_um_zyx=SPACING,
                pixel_size_crop_um_zyx=SPACING,
                matcher="pyramid",
                df_full=df_fixed,
                df_crop=df_moving,
            )
            if res.get("best_R") is None:
                print(f"{tag}: NucleiSky found no match, skipping")
                continue

            # Warp the same images as for IMM: the full moving for all settings,
            # the jittered moving only for ji settings.
            to_warp = {"full": data_dir / f"{MOVING_NAME}_full.tif"}
            if tag.startswith("ji-"):
                to_warp["jitter"] = data_dir / f"{MOVING_NAME}.tif"

            for variant, path in to_warp.items():
                export_aligned_crop_tiff(
                    img_full=fixed,
                    img_crop=tiff.imread(path),
                    output_path=out_dir / f"nucleisky_{variant}.tif",
                    pixel_size_full_um=SPACING,
                    pixel_size_crop_um=SPACING,
                    res=res,
                    export_region="full",
                    order=0,
                )

        except Exception as e:
            print(f"{tag}: error {e}, skipping")
            continue
