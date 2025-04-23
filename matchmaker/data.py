from skimage.measure import regionprops_table
import pandas as pd
import numpy as np


def create_point_cloud(segm):
    props = regionprops_table(segm, properties=("label", "centroid"))
    props = pd.DataFrame(props)
    segm_labels = props["label"].to_numpy()
    centroid_columns = [col for col in props.columns if col.startswith("centroid")]
    pos = props[centroid_columns].to_numpy().astype(np.float32)
    # TODO: center, maybe w/o labels
    return pos, segm_labels


def read_pcd(pcd_path: str):
    pcd_df = pd.read_csv(pcd_path)
    pcd_df.set_index("order_idx", drop=False)
    return pcd_df


def write_pcd(pcd_df, pcd_path):
    pcd_df.to_csv(pcd_path, index=False)


def pcd_np_to_df(pos: np.array, segm_labels=None, reg_gt=None, matching=None):
    # print("pos len", len(pos))
    pcd_df = pd.DataFrame()
    N = pos.shape[0]
    # print(N)
    pcd_df["order_idx"] = range(N)
    for col in range(pos.shape[1]):
        pcd_df[f"coord_{col}"] = pos[:, col]

    if segm_labels is not None:
        assert len(segm_labels) == N, print(
            f"Labels of shape {segm_labels.shape} don't correspond to coordinates of shape {pos.shape}"
        )
        pcd_df["segm_labels"] = segm_labels

    if reg_gt is not None:
        assert len(reg_gt) == N, print(
            f"Labels of shape {reg_gt.shape} don't correspond to coordinates of shape {pos.shape}"
        )
        pcd_df["reg_gt"] = reg_gt

    pcd_df.set_index("order_idx", drop=False)
    return pcd_df


def pcd_df_to_np(pcd_df: pd.DataFrame):
    coord_columns = [col for col in pcd_df.columns if col.startswith("coord")]
    pos = pcd_df[coord_columns].to_numpy().astype(np.float32)
    coord_idx = pcd_df["order_idx"].to_numpy()

    return pos, coord_idx


def matching_to_index_pairs(labels_1: np.array, labels_2: np.array, ignore_index=-1):
    pairs = []

    for idx_1, l1 in enumerate(labels_1):
        if l1 != ignore_index:
            for idx_2 in np.argwhere(labels_2 == l1):
                pairs.append((idx_1, idx_2.item()))

    return pairs


def index_pairs_to_matching():
    pass


def write_index_pairs(pairs, pairs_path):
    with open(pairs_path, "w+") as f:
        for pair in pairs:
            f.write(str(pair[0]) + "," + str(pair[1]) + "\n")


def read_index_pairs(pairs_path):
    with open(pairs_path, "r") as f:
        pairs = [(int(loc.split(",")[0]), int(loc.split(",")[1])) for loc in f.readlines()]
    return pairs
