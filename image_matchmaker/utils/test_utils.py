import yaml
import numpy as np
import urllib.request
from pathlib import Path


def load_config(config_path):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to a ``.yaml`` config file. Must exist.

    Returns
    -------
    dict
        The parsed configuration.
    """
    assert config_path.endswith("yaml")
    assert Path(config_path).exists()

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f'Unable to load config file due to: {e}')
        raise e

    return config


def download_file(path, url):
    """
    Download a file from ``url`` to ``path`` if it does not already exist.

    On failure, prints instructions for downloading the file manually rather
    than raising.

    Parameters
    ----------
    path : str
        Local destination path.
    url : str
        URL to download from.
    """
    if Path(path).exists():
        print(f"✅ File already exists at {path}")
        return

    print(f"Downloading file from {url} ...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"✅ Download complete: {path}")
    except Exception as e:
        print(f"❌ Failed to download file: {e}")
        print(f"Please manually download the file from:\n{url}")
        print(f"and save it to:\n{path}")


def compute_centroids(mask, exclude_id=None):
    """
    Compute centroids for all instance IDs within a segmentation mask.

    Each non-excluded instance ID is treated as a separate label, and the
    centroid is computed as the mean coordinate of all voxels/pixels
    belonging to that label.

    Args:
        mask (np.ndarray): Segmentation mask of arbitrary dimension.
        exclude_id (int | None): Optional label ID to ignore (e.g. background).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - centroids (N x D array): Centroid coordinates for each label.
            - valid_ids (N array): Corresponding label IDs for each centroid.
    """

    ids = np.unique(mask)
    if exclude_id is not None:
        assert isinstance(exclude_id, int), '`exclude_id` should be an integer ID'
        ids = ids[ids != exclude_id]

    if len(ids) == 0:
        return np.zeros((0, mask.ndim)), np.zeros((0,), dtype=int)

    if exclude_id is not None:
        coords = np.argwhere(mask != exclude_id)
        labels = mask[mask != exclude_id]
    else:
        coords = np.argwhere(np.ones_like(mask, dtype=bool))
        labels = mask.ravel()

    labels = labels.astype(int)

    max_id = int(labels.max())
    sums = np.zeros((max_id + 1, mask.ndim), dtype=np.float64)
    np.add.at(sums, labels, coords)

    counts = np.bincount(labels)
    valid = counts > 0
    centroids = sums[valid] / counts[valid, None]
    valid_ids = np.arange(len(counts))[valid]

    # Only keep IDs that actually exist in `ids`
    mask_valid = np.isin(valid_ids, ids)
    return centroids[mask_valid], valid_ids[mask_valid]


def compute_centroid_distances(mask1, mask2, exclude_id=None, matching=None,
                               match1_idx=None, match2_idx=None,):
    """
    Compute centroid distances between two segmentation masks.

    If `matching` is None, distances are computed only for shared instance IDs.
    Otherwise, distances are computed using the provided matched index pairs
    (`match1_idx`, `match2_idx`).

    Args:
        mask1 (np.ndarray): First segmentation mask.
        mask2 (np.ndarray): Second segmentation mask.
        exclude_id (int | None): Optional label to ignore (e.g., background).
        matching (bool | None): Whether matched index pairs are provided.
        match1_idx (np.ndarray | None): Indices of matched instances in mask1.
        match2_idx (np.ndarray | None): Indices of matched instances in mask2.

    Returns:
        dict: Contains mean, median, and maximum centroid distance values.
    """
    exclude_id = exclude_id if matching is None else None

    mask1_centroids, mask1_ids = compute_centroids(mask1, exclude_id=exclude_id)
    mask2_centroids, mask2_ids = compute_centroids(mask2, exclude_id=exclude_id)

    if matching is None:
        shared_ids = np.intersect1d(mask1_ids, mask2_ids)
        if len(shared_ids) > 0:
            mask1_idx = np.isin(mask1_ids, shared_ids)
            mask2_idx = np.isin(mask2_ids, shared_ids)
            mask1_c = mask1_centroids[mask1_idx]
            mask2_c = mask2_centroids[mask2_idx]
            centroid_distances = np.linalg.norm(mask1_c - mask2_c, axis=1)
        else:
            raise RuntimeError("No shared id found.")

    else:
        assert match1_idx is not None and match2_idx is not None

        centroid_distances = []
        for idx1, idx2 in zip(match1_idx, match2_idx):
            c1 = mask1_centroids[idx1]
            c2 = mask2_centroids[idx2]
            centroid_distances.append(np.linalg.norm(c1 - c2))

        centroid_distances = np.array(centroid_distances)

    return centroid_distances


def check_no_new_ids(vol1, vol2):
    """
    Check whether vol2 contains any instance IDs that do not exist in vol1.
    """
    ids1 = np.unique(vol1)
    ids2 = np.unique(vol2)

    new_ids = np.setdiff1d(ids2, ids1)

    return len(new_ids) == 0, new_ids
