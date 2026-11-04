import numpy as np
import tifffile as tiff


def extract_ids(mask, fraction=0.05, seed=42, output_txt="metric_ids.txt"):
    """
    Randomly select a fraction of unique segmentation IDs, excluding 0,
    and save the selected IDs in ascending order.

    Args:
        mask: 3D NumPy segmentation mask.
        fraction: Fraction of foreground IDs to select [0, 1].
        seed: Random seed for reproducibility.
        output_txt: Path to the output text file.
    """
    # Get unique IDs and exclude background
    unique_ids = np.unique(mask)
    foreground_ids = unique_ids[unique_ids != 0]

    # Validate fraction
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1.")

    # Randomly select IDs reproducibly
    rng = np.random.default_rng(seed)
    n_select = int(round(len(foreground_ids) * fraction))
    selected_ids = rng.choice(foreground_ids, size=n_select, replace=False)

    # Sort IDs in ascending order
    selected_ids = np.sort(selected_ids)

    # Save IDs to a text file, one ID per line
    np.savetxt(output_txt, selected_ids, fmt="%d")

    print(f"Total foreground IDs: {len(foreground_ids)}")
    print(f"Selected IDs: {len(selected_ids)}")
    print(f"Saved to: {output_txt}")

    return selected_ids


if __name__ == "__main__":
    src_path = "data/platy1_muscles_stardist_fixed_rotated.tif"

    mask = tiff.imread(src_path)

    selected_ids = extract_ids(
        mask,
        fraction=0.05,
        seed=42,
        output_txt="metric_ids.txt"
    )

    # print(selected_ids)
