from pathlib import Path

import numpy as np
from tqdm import tqdm

from image_matchmaker.utils import load_data, read_volume
from lre import FIXED_PATH, LANDMARKS_PATH, TABLE_HEADER, evaluate


RESULTS = Path("results")
SAVE_PATH = RESULTS / "lre_imm.tsv"
# Variant -> key of the warped moving image in results/<tag>/moving_image.n5.
# "full": moving with all instances and without jitter, warped in step 04.
# "jitter": the registered (jittered) moving, warped by the pipeline. Only for ji settings.
VARIANTS = {"full": "full_pointset_alignment_transform", "jitter": "pointset_alignment"}


if __name__ == "__main__":
    fixed = load_data(FIXED_PATH).astype(np.int32)
    landmarks = np.loadtxt(LANDMARKS_PATH, dtype=int)
    assert np.isin(landmarks, np.unique(fixed)).all(), "All landmarks must exist in the fixed image"
    print(f"Landmarks: {len(landmarks)}")

    rows = []
    for result_dir in tqdm(sorted(RESULTS.iterdir())):
        moving_path = result_dir / "moving_image.n5"
        if not moving_path.exists():
            continue

        for variant, key in VARIANTS.items():
            if variant == "jitter" and not result_dir.name.startswith("ji-"):
                continue

            warped = read_volume(str(moving_path), key)
            if warped is None:
                print(f"{result_dir.name}: key {key} not found, skipping")
                continue

            rows.append(evaluate(fixed, warped.astype(np.int32), landmarks, result_dir.name, variant,
                                 result_dir / f"lre_imm_{variant}.txt"))

    with open(SAVE_PATH, "w") as f:
        f.write(TABLE_HEADER + "\n".join(rows) + "\n")
    print(f"Saved to {SAVE_PATH}")
