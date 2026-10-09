import yaml
import numpy as np
import urllib.request
from pathlib import Path


def load_config(config_path):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to a ``.yaml`` config file. Must exist.

    Returns
    -------
    dict
        The parsed configuration.
    """
    assert str(config_path).endswith("yaml")
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


def compute_centroids(mask, exclude_id=None, ids=None):
    """
    Compute centroids for selected instance IDs within a segmentation mask.

    Args:
        mask (np.ndarray): Segmentation mask.
        exclude_id (int | None): Optional label ID to ignore.
        ids (list[int] | None): Optional instance IDs to evaluate.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - centroids (N x D array): Centroid coordinates.
            - valid_ids (N array): Corresponding instance IDs.
    """
    if exclude_id is not None:
        assert isinstance(exclude_id, int), '`exclude_id` should be an integer ID'

    if ids is not None:
        assert all(isinstance(i, (int, np.integer)) for i in ids), \
            '`ids` should be a list of integers'
        ids = np.asarray(ids, dtype=int)
        ids = np.unique(ids)
    else:
        ids = np.unique(mask)

    if exclude_id is not None:
        ids = ids[ids != exclude_id]

    if len(ids) == 0:
        return np.zeros((0, mask.ndim)), np.zeros((0,), dtype=int)

    valid = np.isin(mask, ids)
    coords = np.argwhere(valid)
    labels = mask[valid]

    unique_ids, inverse = np.unique(labels, return_inverse=True)

    sums = np.zeros((len(unique_ids), mask.ndim), dtype=np.float64)
    np.add.at(sums, inverse, coords)

    counts = np.bincount(inverse)
    centroids = sums / counts[:, None]

    return centroids, unique_ids


def compute_centroid_distances(mask1, mask2, ids=None, exclude_id=None, matching=False,
        match1_idx=None, match2_idx=None,):
    """
    Compute centroid distances between two segmentation masks.

    If matching=False, `ids` specifies the instance IDs to evaluate, and the
    same IDs must exist in both masks.

    If matching=True, `match1_idx` and `match2_idx` specify corresponding
    centroid indices between mask1 and mask2. In this case, `ids` is ignored.

    Args:
        mask1 (np.ndarray): First segmentation mask.
        mask2 (np.ndarray): Second segmentation mask.
        ids (list[int] | None): Instance IDs to evaluate when matching=False.
        exclude_id (int | None): Optional label ID to ignore.
        matching (bool): Whether explicit centroid matching is provided.
        match1_idx (np.ndarray | None): Matched centroid indices in mask1.
        match2_idx (np.ndarray | None): Matched centroid indices in mask2.

    Returns:
        np.ndarray: Centroid distance for each evaluated instance.
    """
    assert isinstance(matching, bool), '`matching` should be a boolean'

    if matching:
        assert match1_idx is not None and match2_idx is not None, \
            '`match1_idx` and `match2_idx` are required when matching=True'
        assert len(match1_idx) == len(match2_idx), \
            '`match1_idx` and `match2_idx` should have the same length'

        centroids1, ids1 = compute_centroids(mask1, exclude_id=exclude_id)
        centroids2, ids2 = compute_centroids(mask2, exclude_id=exclude_id)

        match1_idx = np.asarray(match1_idx, dtype=int)
        match2_idx = np.asarray(match2_idx, dtype=int)

        if (
            np.any(match1_idx < 0) or np.any(match1_idx >= len(centroids1)) or
            np.any(match2_idx < 0) or np.any(match2_idx >= len(centroids2))
        ):
            raise IndexError("Matched centroid index is out of range.")

        centroids1 = centroids1[match1_idx]
        centroids2 = centroids2[match2_idx]

    else:
        centroids1, ids1 = compute_centroids(mask1, exclude_id=exclude_id, ids=ids)
        centroids2, ids2 = compute_centroids(mask2, exclude_id=exclude_id, ids=ids)

        if not np.array_equal(ids1, ids2):
            raise ValueError("Instance IDs do not match between mask1 and mask2.")

    return np.linalg.norm(centroids1 - centroids2, axis=1)


def check_no_new_ids(vol1, vol2):
    """
    Check whether vol2 contains any instance IDs that do not exist in vol1.
    """
    ids1 = np.unique(vol1)
    ids2 = np.unique(vol2)

    new_ids = np.setdiff1d(ids2, ids1)

    return len(new_ids) == 0, new_ids


def assert_arrays_equal(arr1, arr2):
    assert arr1.dtype == arr2.dtype
    assert np.array_equal(arr1, arr2)
