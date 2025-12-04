import json
import numpy as np

from matchmaker.utils import (pad_to_same_shape, read_volume)


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


def compute_centroid_distances(mask1, mask2, exclude_id=None):
    """
    Compute centroid distances between two segmentation masks.

    Centroids are computed per instance ID in each mask. Only IDs shared
    between the two masks are compared.

    Args:
        mask1 (np.ndarray): First segmentation mask.
        mask2 (np.ndarray): Second segmentation mask.
        exclude_id (int | None): Optional label ID to ignore (e.g. background).

    Returns:
        dict: Contains mean, median, and maximum centroid distance values.
    """

    mask1_centroids, mask1_ids = compute_centroids(mask1, exclude_id=exclude_id)
    mask2_centroids, mask2_ids = compute_centroids(mask2, exclude_id=exclude_id)

    shared_ids = np.intersect1d(mask1_ids, mask2_ids)
    if len(shared_ids) > 0:
        mask1_idx = np.isin(mask1_ids, shared_ids)
        mask2_idx = np.isin(mask2_ids, shared_ids)
        mask1_c = mask1_centroids[mask1_idx]
        mask2_c = mask2_centroids[mask2_idx]
        centroid_distances = np.linalg.norm(mask1_c - mask2_c, axis=1)
    else:
        raise RuntimeError("No shared id found.")

    return {
        "mean_centroid_distance": np.mean(centroid_distances),
        "median_centroid_distance": np.median(centroid_distances),
        "max_centroid_dist": np.max(centroid_distances),
    }


def compute_segmentation_metrics(gt, pred, exclude_id=None, average="macro", eps=1e-8):
    """
    Compute segmentation metrics (IoU, Dice, Precision, Recall, F1) between two segmentation masks.

    Works for N-dimensional integer-labeled segmentation masks. Supports both
    'macro' (mean across instances) and 'micro' (aggregate) averaging.

    Args:
        gt (np.ndarray): Ground truth segmentation mask.
        pred (np.ndarray): Predicted segmentation mask.
        exclude_id (int | None): Optional label ID to ignore (e.g. background).
        average (str): 'macro' (mean across instances) or 'micro' (aggregate).
        eps (float): Small constant to avoid division by zero.

    Returns:
        dict: Dictionary containing averaged metrics and label statistics.
    """

    gt = gt.astype(np.int64)
    pred = pred.astype(np.int64)
    assert gt.shape == pred.shape, "gt and pred must have the same shape"
    assert average in {"macro", "micro"}, "average must be 'macro' or 'micro'"

    max_label = int(max(gt.max(), pred.max()))
    hist = np.bincount(
        gt.ravel() * (max_label + 1) + pred.ravel(),
        minlength=(max_label + 1) ** 2
    ).reshape(max_label + 1, max_label + 1)

    gt_sum = hist.sum(axis=1)
    pred_sum = hist.sum(axis=0)
    inter = np.diag(hist)
    union = gt_sum + pred_sum - inter

    labels_gt = np.where(gt_sum > 0)[0]
    labels_pred = np.where(pred_sum > 0)[0]

    if exclude_id is not None:
        labels_gt = labels_gt[labels_gt != exclude_id]
        labels_pred = labels_pred[labels_pred != exclude_id]

    common = np.intersect1d(labels_gt, labels_pred)
    missing_in_gt = np.setdiff1d(labels_pred, labels_gt)
    missing_in_pred = np.setdiff1d(labels_gt, labels_pred)

    if len(common) == 0:
        return {
            "shared_instances": 0, "missing_gt_instances": len(missing_in_gt),
            "missing_pred_instances": len(missing_in_pred), "mean_iou": 0.0,
            "mean_dice": 0.0, "mean_precision": 0.0, "mean_recall": 0.0, "mean_f1_score": 0.0,
        }

    if average == "macro":
        iou_per_class = inter / (union + eps)
        dice_per_class = 2 * inter / (gt_sum + pred_sum + eps)
        p_per_class = inter / (pred_sum + eps)
        r_per_class = inter / (gt_sum + eps)
        f1_per_class = 2 * p_per_class * r_per_class / (p_per_class + r_per_class + eps)

        iou = np.mean(iou_per_class)
        dice = np.mean(dice_per_class)
        precision = np.mean(p_per_class)
        recall = np.mean(r_per_class)
        f1 = np.mean(f1_per_class)
    else:  # micro
        tp_total = inter.sum()
        union_total = union.sum()
        gt_sum_total = gt_sum.sum()
        pred_sum_total = pred_sum.sum()

        iou = tp_total / (union_total + eps)
        dice = 2 * tp_total / (gt_sum_total + pred_sum_total + eps)
        precision = tp_total / (pred_sum_total + eps)
        recall = tp_total / (gt_sum_total + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "shared_instances": len(common),
        "missing_gt_instances": len(missing_in_gt),
        "missing_pred_instances": len(missing_in_pred),
        "mean_iou": float(iou),
        "mean_dice": float(dice),
        "mean_precision": float(precision),
        "mean_recall": float(recall),
        "mean_f1_score": float(f1),
    }


def evaluate_registration(fixed, moving, save_path="metrics.json", exclude_id=None):
    fixed, moving = pad_to_same_shape(fixed, moving)

    centroid_metrics = compute_centroid_distances(fixed, moving, exclude_id=exclude_id)

    metrics = compute_segmentation_metrics(fixed, moving, exclude_id=exclude_id)

    metrics.update(centroid_metrics)

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f'Metrics saved at path {save_path}.')

    print("Evaluation results:")
    max_key_len = max(len(k) for k in metrics)
    for k, v in metrics.items():
        print(f"  {k:<{max_key_len}} : {v}")
    return metrics


if __name__ == "__main__":
    fixed_path = "data/test_registration/fixed_image.n5"
    moving_path = "data/test_registration/moving_image.n5"
    save_path="data/test_registration/metrics.json"

    fixed_key = "svd_prealignment"
    moving_key = "rigid_alignment"

    fixed_img = read_volume(fixed_path, fixed_key)
    moving_img = read_volume(moving_path, moving_key)

    evaluate_registration(fixed_img, moving_img, save_path=save_path, exclude_id=0)
