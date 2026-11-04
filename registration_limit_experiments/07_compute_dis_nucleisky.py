from pathlib import Path

import numpy as np
from tqdm import tqdm

from image_matchmaker.utils import load_data
from lre import FIXED_PATH, LANDMARKS_PATH, TABLE_HEADER, evaluate


RESULTS = Path("results")
SAVE_PATH = RESULTS / "lre_nucleisky.tsv"
# Same variants as for IMM (see 05_compute_dis_imm.py), warped by 06_run_nucleisky.py.
VARIANTS = ("full", "jitter")


if __name__ == "__main__":
    fixed = load_data(FIXED_PATH).astype(np.int32)
    landmarks = np.loadtxt(LANDMARKS_PATH, dtype=int)
    assert np.isin(landmarks, np.unique(fixed)).all(), "All landmarks must exist in the fixed image"
    print(f"Landmarks: {len(landmarks)}")

    rows = []
    for result_dir in tqdm(sorted(RESULTS.iterdir())):
        for variant in VARIANTS:
            warped_path = result_dir / f"nucleisky_{variant}.tif"
            if not warped_path.exists():
                continue

            warped = load_data(str(warped_path)).astype(np.int32)
            rows.append(evaluate(fixed, warped, landmarks, result_dir.name, variant,
                                 result_dir / f"lre_nucleisky_{variant}.txt"))

    with open(SAVE_PATH, "w") as f:
        f.write(TABLE_HEADER + "\n".join(rows) + "\n")
    print(f"Saved to {SAVE_PATH}")
