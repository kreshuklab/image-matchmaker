import numpy as np
import cvxpy as cp

from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

def sparse_ilp_matching(pos_1, pos_2, max_dist=35, min_neighbours=10):
    logging.info("Calculate cost matrix")
    kd_tree1 = cKDTree(pos_1)
    kd_tree2 = cKDTree(pos_2)
    sdm = kd_tree1.sparse_distance_matrix(kd_tree2, max_dist)
    logging.info("Get a list of nonzero elements")
    nonzero_idx = sdm.nonzero()
    nonzero_mask = sdm > 0
    logging.info(nonzero_mask)
    logging.info(nonzero_mask.sum(axis=1))
    unconnected_pcd1 = np.argwhere(nonzero_mask.sum(axis=1) < min_neighbours)[:, 0]
    unconnected_pcd2 = np.argwhere((nonzero_mask > 0).sum(axis=0) < min_neighbours)[:, 1]
    logging.info(f"Number of nonzero elements: {len(nonzero_idx[0])}")
    logging.info(f"Unconnected points in pcd2 after adding knn {unconnected_pcd1}, {len(nonzero_mask.sum(axis=0)[0, unconnected_pcd2])} out of {sdm.sum(axis=0).shape}")
    logging.info(f"Unconnected points in pcd1 after adding knn {unconnected_pcd2}, {len(nonzero_mask.sum(axis=1)[unconnected_pcd1, 0])} out of {sdm.sum(axis=1).shape}")
    logging.info(f"Cost matrix shape: {sdm.shape}")
    logging.info("Add closest neighbours for unconnected points in pcd2")
    for idx in unconnected_pcd2:
        p_2 = pos_2[idx]
        knn_distances, knn_i = kd_tree1.query(p_2, k=min_neighbours, workers=8)
        for d, knn_idx in zip(knn_distances, knn_i):
            # logging.info(idx, knn_idx, d)
            sdm[knn_idx, idx] = d
    logging.info("Add closest neighbours for unconnected points in pcd1")
    for idx in unconnected_pcd1:
        p_1 = pos_1[idx]
        knn_distances, knn_i = kd_tree2.query(p_1, k=min_neighbours, workers=8)
        for d, knn_idx in zip(knn_distances, knn_i):
            # logging.info(idx, knn_idx, d)
            sdm[idx, knn_idx] = d
    nonzero_mask = sdm > 0
    unconnected_pcd1 = np.argwhere(nonzero_mask.sum(axis=1) < min_neighbours)[:, 0]
    unconnected_pcd2 = np.argwhere((nonzero_mask > 0).sum(axis=0) < min_neighbours)[:, 1]
    logging.info(f"Unconnected points in pcd2 after adding knn {unconnected_pcd1}, {len(nonzero_mask.sum(axis=0)[0, unconnected_pcd2])} out of {sdm.sum(axis=0).shape}")
    logging.info(f"Unconnected points in pcd1 after adding knn {unconnected_pcd2}, {len(nonzero_mask.sum(axis=1)[unconnected_pcd1, 0])} out of {sdm.sum(axis=1).shape}")
    logging.info("Get a list of nonzero elements")
    nonzero_idx = sdm.nonzero()
    logging.info(f"Number of nonzero elements: {len(nonzero_idx[0])}")
    logging.info("Match")
    potential_pairs = [(p1.item(), p2.item()) for p1, p2 in zip(nonzero_idx[0], nonzero_idx[1])]
    nonzero_costs = sdm[nonzero_idx]
    logging.info(f"nonzero costs shape {nonzero_costs.shape}")
    X = cp.Variable(shape=nonzero_costs.shape, name="X", boolean=True)
    objective = cp.Minimize(cp.sum(cp.multiply(nonzero_costs, X)))
    logging.info(objective)

    logging.info("Create constraints")
    constraints = []
    for point in range(sdm.shape[1]):
        point_idx = np.argwhere(nonzero_idx[1] == point).flatten()
        constraints += [cp.sum(X[:, point_idx]) == 1]
    for point in range(sdm.shape[0]):
        point_idx = np.argwhere(nonzero_idx[0] == point).flatten()
        constraints += [cp.sum(X[:, point_idx]) >= 1]
    
    # X = cp.Variable(shape=cost_matrix.shape, name="X", boolean=True)
    # ones_1 = np.ones((cost_matrix.shape[0], 1))
    # ones_2 = np.ones((cost_matrix.shape[1], 1))
    # objective = cp.Minimize(cp.sum(cp.multiply(cost_matrix, X)))
    # constraints = []
    # constraints += [X.T @ ones_1 == ones_2]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True, canon_backend=cp.SCIPY_CANON_BACKEND)

    solution = np.array(X.value[0]).astype(np.int8)
    matched_pairs = [(p1.item(), p2.item()) for p1, p2 in zip(nonzero_idx[0][solution==1], nonzero_idx[1][solution==1])]
    # logging.info("matched pairs", matched_pairs)
    # X_sol = (X.value==1)

    # matched_pairs = np.where(X_sol)
    # matched_pairs = [(p1.item(), p2.item()) for p1, p2 in zip(matched_pairs[0], matched_pairs[1])]
    return matched_pairs


def write_index_pairs(pairs, pairs_path):
    with open(pairs_path,"w+") as f:
        for pair in pairs:
            f.write(str(pair[0]) + "," + str(pair[1]) + '\n')


def read_index_pairs(pairs_path):
    with open(pairs_path,"r") as f:
        pairs = [(int(l.split(",")[0]), int(l.split(",")[1])) for l in f.readlines()]
    return pairs