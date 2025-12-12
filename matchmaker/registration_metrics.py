import json
import numpy as np

from matchmaker.utils import (pad_to_same_shape, read_volume, to_dense_mask)


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


def compute_centroid_distances(mask1, mask2, match1_idx, match2_idx,
                                exclude_id=None, matching=None):
    """
    Compute centroid distances between two segmentation masks.

    Works for dense integer-labeled masks. If `matching` is None, distances are
    computed only for shared instance IDs. Otherwise, distances are computed
    using the provided matched index pairs (`match1_idx`, `match2_idx`).

    Args:
        mask1 (np.ndarray): First segmentation mask.
        mask2 (np.ndarray): Second segmentation mask.
        match1_idx (np.ndarray | None): Indices of matched instances in mask1.
        match2_idx (np.ndarray | None): Indices of matched instances in mask2.
        exclude_id (int | None): Optional label to ignore (e.g., background).
        matching (bool | None): Whether matched index pairs are provided.

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

        if len(centroid_distances) == 0:
            return {
                "mean_centroid_distance": "NaN",
                "median_centroid_distance": "NaN",
                "max_centroid_distance": "NaN",
            }

        centroid_distances = np.array(centroid_distances)

    return {
        "mean_centroid_distance": np.mean(centroid_distances),
        "median_centroid_distance": np.median(centroid_distances),
        "max_centroid_distance": np.max(centroid_distances),
    }


def greedy_match(inter):
    """
    Greedy one-to-one matching based on intersection values.

    Repeatedly selects the (gt, pred) pair with the highest overlap and
    marks both as matched.

    Args:
        inter (np.ndarray): (G x P) matrix, inter[g,p] = overlap size.

    Returns:
        (np.ndarray, np.ndarray): Matched GT indices, matched Pred indices.
    """

    G, P = inter.shape
    gt_idx, pred_idx = np.nonzero(inter)
    vals = inter[gt_idx, pred_idx]
    order = np.argsort(-vals)
    gt_idx, pred_idx = gt_idx[order], pred_idx[order]

    used_gt = np.zeros(G, dtype=bool)
    used_pred = np.zeros(P, dtype=bool)
    matched_gt_ids, matched_pred_ids = [], []

    for g, p in zip(gt_idx, pred_idx):
        if not used_gt[g] and not used_pred[p]:
            matched_gt_ids.append(g)
            matched_pred_ids.append(p)
            used_gt[g] = True
            used_pred[p] = True

    return np.array(matched_gt_ids), np.array(matched_pred_ids)


def compute_segmentation_stats(gt, pred, matching=None, eps=1e-8):
    """
    Compute raw intersection statistics between GT and Pred masks.

    Builds the intersection matrix, instance areas, and unions.
    For matching=None, a square matrix based on max label is used.
    For greedy/hungarian, a (G x P) matrix is built.

    Args:
        gt (np.ndarray): Ground-truth instance mask.
        pred (np.ndarray): Predicted instance mask.
        matching (str | None): None, 'greedy', or 'hungarian'.
        eps (float): Small constant to avoid division by zero.

    Returns:
        inter, gt_sum, pred_sum, union:
            - inter (array): Intersection counts.
            - gt_sum (array): GT instance areas.
            - pred_sum (array): Predicted instance areas.
            - union (array): Union = gt_sum + pred_sum - inter.
    """

    gt = gt.astype(np.int64)
    pred = pred.astype(np.int64)

    if matching is None:
        max_label = int(max(gt.max(), pred.max()))

        hist = np.bincount(
            gt.ravel() * (max_label + 1) + pred.ravel(),
            minlength=(max_label + 1) ** 2
        ).reshape(max_label + 1, max_label + 1)

        inter = np.diag(hist)
        gt_sum = hist.sum(axis=1)
        pred_sum = hist.sum(axis=0)
        union = gt_sum + pred_sum - inter

    else:
        G = int(gt.max()) + 1
        P = int(pred.max()) + 1

        hist = np.bincount(
            gt.ravel() * P + pred.ravel(),
            minlength=G * P
        ).reshape(G, P)

        inter = hist
        gt_sum = hist.sum(axis=1)
        pred_sum = hist.sum(axis=0)
        union = gt_sum[:, None] + pred_sum[None, :] - inter

    return inter, gt_sum, pred_sum, union


def rm_idx(arr, exclude_id):
    """
    Remove an element from a 1D NumPy array by index.
    """
    if exclude_id is None:
        return arr
    return np.delete(arr, exclude_id)


def compute_segmentation_metrics(gt, pred, exclude_id=None, matching=None, min_iou=1e-3, eps=1e-8):
    """
    Compute segmentation metrics (IoU, Dice, Precisions, Recalls, F1s) between two segmentation masks.

    Works for N-dimensional integer-labeled segmentation masks. Supports both
    corresponding or non-corresponding instance IDs.

    Args:
        gt (np.ndarray): Ground truth segmentation mask.
        pred (np.ndarray): Predicted segmentation mask.
        exclude_id (int | None): Optional label ID to ignore (e.g. background).
        matching (str):
            None -> corresponding IDs (direct ID match)
            'greedy' -> greedy matching
            'hungarian' -> hungarian matching
        min_iou (float): IoU cutoff for valid matches (non-corresponding case).
        eps (float): Small constant to avoid division by zero.

    Returns:
        dict: Dictionary containing averaged metrics and label statistics.
    """

    assert gt.shape == pred.shape, "gt and pred must have the same shape"
    assert matching in {None, "greedy", "hungarian"}

    gt = gt.astype(np.int64)
    pred = pred.astype(np.int64)

    valid_gt = rm_idx(np.unique(gt), exclude_id)
    valid_pred = rm_idx(np.unique(pred), exclude_id)

    if len(valid_gt) == 0 or len(valid_pred) == 0:
        return {
            "matched_instances": 0,
            "unmatched_gt_ids_instances": len(valid_gt),
            "unmatched_pred_ids_instances": len(valid_pred),
            "mean_IoU": 0.0,
            "mean_Dice": 0.0,
            "mean_Precision": 0.0,
            "mean_Recall": 0.0,
            "mean_F1_score": 0.0,
        }, (None, None)

    inter, gt_sum, pred_sum, union = compute_segmentation_stats(gt, pred, matching=matching, eps=eps)

    if matching is None:
        inter, union = rm_idx(inter, exclude_id), rm_idx(union, exclude_id)
        gt_sum, pred_sum = rm_idx(gt_sum, exclude_id), rm_idx(pred_sum, exclude_id)
        row_ind = col_ind = matched_gt_ids = matched_pred_ids = np.intersect1d(valid_gt, valid_pred)
        IoUs = inter / (union + eps)
        Dices = 2 * inter / (gt_sum + pred_sum + eps)
        Precisions = inter / (pred_sum + eps)
        Recalls = inter / (gt_sum + eps)
        F1s = 2 * Precisions * Recalls / (Precisions + Recalls + eps)

    else:
        if matching == "hungarian":
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-inter)
        elif matching == "greedy":
            row_ind, col_ind = greedy_match(inter)
        else:
            raise ValueError(f"Unknown matching method: {matching}")

        row_ind, col_ind = rm_idx(row_ind, exclude_id), rm_idx(col_ind, exclude_id)

        TP = inter[row_ind, col_ind]
        FN = gt_sum[row_ind] - TP
        FP = pred_sum[col_ind] - TP
        IoUs = TP / (TP + FP + FN + eps)

        valid_match = IoUs >= min_iou
        row_ind, col_ind = row_ind[valid_match], col_ind[valid_match]
        matched_gt_ids, matched_pred_ids = set(row_ind), set(col_ind)

        TP, FN, FP = TP[valid_match], FN[valid_match], FP[valid_match]
        IoUs = IoUs[valid_match]

        Dices = 2 * TP / (2 * TP + FP + FN + eps)
        Precisions = TP / (TP + FP + eps)
        Recalls = TP / (TP + FN + eps)
        F1s = 2 * Precisions * Recalls / (Precisions + Recalls + eps)

        n_zero = len(valid_gt) - len(matched_gt_ids) + len(valid_pred) - len(matched_pred_ids)
        if n_zero > 0:
            zeros = np.zeros(n_zero, dtype=np.float64)
            IoUs = np.concatenate([IoUs, zeros])
            Dices = np.concatenate([Dices, zeros])
            Precisions = np.concatenate([Precisions, zeros])
            Recalls = np.concatenate([Recalls, zeros])
            F1s = np.concatenate([F1s, zeros])

    return {
        "matched_instances": len(matched_gt_ids),
        "unmatched_gt_ids_instances": len(valid_gt) - len(matched_gt_ids),
        "unmatched_pred_ids_instances": len(valid_pred) - len(matched_pred_ids),
        "mean_IoU": float(IoUs.mean()),
        "mean_Dice": float(Dices.mean()),
        "mean_Precision": float(Precisions.mean()),
        "mean_Recall": float(Recalls.mean()),
        "mean_F1_score": float(F1s.mean()),
    }, (row_ind, col_ind)


def evaluate_registration(fixed, moving, save_path="metrics.json", exclude_id=None, matching=None):
    assert isinstance(fixed, np.ndarray) and isinstance(moving, np.ndarray)
    if exclude_id is not None:
        assert exclude_id <= max(fixed.max(), moving.max())

    fixed, fixed_mapping = to_dense_mask(fixed, background=exclude_id)

    mapping = fixed_mapping if matching is None else None
    moving, moving_mapping = to_dense_mask(moving, background=exclude_id, mapping=mapping)

    if fixed_mapping[exclude_id] != exclude_id or moving_mapping[exclude_id] != exclude_id:
        assert fixed_mapping[exclude_id] == moving_mapping[exclude_id]
        exclude_id = fixed_mapping[exclude_id]

    metrics = {"Matching method": matching}

    fixed, moving = pad_to_same_shape(fixed, moving)

    seg_metrics, matched_idx = compute_segmentation_metrics(fixed, moving, exclude_id=exclude_id,
                                                            matching=matching)
    metrics.update(seg_metrics)

    centroid_metrics = compute_centroid_distances(fixed, moving, *matched_idx, exclude_id=exclude_id,
                                                    matching=matching)
    metrics.update(centroid_metrics)

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved at path {save_path}.")

    print("Evaluation results:")
    max_key_len = max(len(k) for k in metrics)
    for k, v in metrics.items():
        print(f"  {k:<{max_key_len}} : {v}")

    return metrics


if __name__ == "__main__":
    fixed_path = "data/test_registration/fixed_image.n5"
    moving_path = "data/test_registration/moving_image.n5"
    save_path = "data/test_registration/metrics.json"
    matching = None

    fixed_key = "svd_prealignment"
    moving_key = "rigid_alignment"

    fixed_img = read_volume(fixed_path, fixed_key)
    moving_img = read_volume(moving_path, moving_key)

    evaluate_registration(fixed_img, moving_img, save_path=save_path, exclude_id=0, matching=matching)
