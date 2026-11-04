"""Landmark registration error (LRE), shared by 05_compute_dis_imm.py and 07_compute_dis_nucleisky.py."""
import numpy as np

from image_matchmaker.utils import compute_centroids


FIXED_PATH = "data/platy1_muscles_stardist_fixed_rotated.tif"
LANDMARKS_PATH = "metric_ids.txt"
TABLE_HEADER = "setting\tvariant\tnum_landmarks\tnum_missing\tnum_cut\tMean\tP95\tP80\n"


def border_ids(seg):
    """Return the IDs that touch the outer faces of a 3D volume."""
    faces = [seg[0], seg[-1], seg[:, 0], seg[:, -1], seg[:, :, 0], seg[:, :, -1]]
    return np.unique(np.concatenate([f.ravel() for f in faces]))


def landmark_errors(fixed, warped, landmarks):
    """Compute the centroid distance between fixed and warped moving for each landmark.

    Landmarks are skipped if they are missing in the warped image (removed, cut by the
    moving bbox in step 01, or mapped outside the fixed volume), or if they touch the
    border of the warped image, because then they may be cut and their centroid is wrong.

    Returns:
        ids, distances (in voxels), number of missing landmarks, number of cut landmarks.
    """
    present = landmarks[np.isin(landmarks, np.unique(warped))]
    cut = np.intersect1d(present, border_ids(warped))
    ids = np.setdiff1d(present, cut)

    centroids_fixed, ids_fixed = compute_centroids(fixed, ids=ids)
    centroids_warped, ids_warped = compute_centroids(warped, ids=ids)
    assert np.array_equal(ids_fixed, ids) and np.array_equal(ids_warped, ids)

    distances = np.linalg.norm(centroids_fixed - centroids_warped, axis=1)
    return ids, distances, len(landmarks) - len(present), len(cut)


def summarize(distances):
    """Return the mean, and the means of all distances up to the 95th and 80th percentile."""
    if len(distances) == 0:
        return np.nan, np.nan, np.nan
    return tuple(distances[distances <= np.percentile(distances, p)].mean() for p in (100, 95, 80))


def evaluate(fixed, warped, landmarks, setting, variant, per_landmark_path):
    """Compute the LRE of one warped image, save the per-landmark distances and return a table row."""
    assert warped.shape == fixed.shape, f"{setting}: {warped.shape} != {fixed.shape}"
    ids, distances, n_missing, n_cut = landmark_errors(fixed, warped, landmarks)
    np.savetxt(per_landmark_path, np.column_stack([ids, distances]),
               fmt=["%d", "%.4f"], header="id\tdistance", delimiter="\t")

    mean, p95, p80 = summarize(distances)
    return (f"{setting}\t{variant}\t{len(ids)}\t{n_missing}\t{n_cut}"
            f"\t{mean:.3f}\t{p95:.3f}\t{p80:.3f}")
