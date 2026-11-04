import numpy as np
from scipy.ndimage import center_of_mass
from scipy.spatial import cKDTree

from image_matchmaker.data_processing.deform_test_data import deform_test_data
from image_matchmaker.utils import load_config, load_data


def mean_instance_distance(seg):
    """Return the mean distance from each instance centroid to its nearest neighbour centroid."""
    ids = np.unique(seg)
    ids = ids[ids > 0]
    centroids = np.array(center_of_mass(seg > 0, seg, ids))
    dists, _ = cKDTree(centroids).query(centroids, k=2)
    return dists[:, 1].mean()


CFG_PATH = "register_config_test_elastic.yaml"
STATS_PATH = "data/removal_stats.tsv"
REMOVING_FRACTIONS = [0.1, 0.3, 0.5, 0.7, 0.9]
SPHERE_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]
SPHERE_CENTERS = {1: [159, 203, 228], 2: [139, 118, 90]}  # head, tail (zyx in rotated moving volume)
DISPLACEMENTS = [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5]

# (folder tag, deform_test_data kwargs). Data is written to data/deformed_data_<tag>/.
cases = (
    [(f"ri-{f}", dict(remove_mode="random_instances", remove_fraction=f)) for f in REMOVING_FRACTIONS]
    + [(f"sphere-center{i}-{f}", dict(remove_mode="sphere", remove_fraction=f, remove_center=c))
       for i, c in SPHERE_CENTERS.items() for f in SPHERE_FRACTIONS]
)


if __name__ == "__main__":
    source = load_data(load_config(CFG_PATH)["fixed_image"]["source_path"])
    mean_dist = mean_instance_distance(source)
    print(f"Mean instance distance: {mean_dist:.2f} voxels")

    for f in DISPLACEMENTS:
        d = round(f * mean_dist)
        print(f"ji-{f}: max displacement {d} voxels")
        cases.append(
            (
                f"ji-{f}",
                dict(
                    move_mode="jitter_instances",
                    move_max_displacement=d,
                    remove_fraction=0,
                ),
            )
        )

    rows = []
    for tag, kwargs in cases:
        config = load_config(CFG_PATH)
        # The fixed image is the same for all settings, so it is written to data/ each time.
        moving = config["moving_image"]
        moving["path"] = moving["path"].replace("deformed_data", f"deformed_data_{tag}")
        stats = deform_test_data(config=config, enable_elastic=True, save_full_moving=True, **kwargs)

        before, left = stats["num_before"], stats["num_left"]
        rows.append(f"{tag}\t{before}\t{left}\t{before - left}\t{100 * (before - left) / before:.2f}")

    with open(STATS_PATH, "w") as f:
        f.write("setting\tnum_before\tnum_left\tnum_removed\tremoved_percent\n")
        f.write("\n".join(rows) + "\n")
    print(f"Removal stats saved to {STATS_PATH}")
