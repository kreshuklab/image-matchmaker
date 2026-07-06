import numpy as np
import cvxpy as cp

from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging


def create_cost_matrix(pos_1, pos_2, max_dist=35, min_neighbours=10):
    logging.info("Calculate cost matrix")
    kd_tree1 = cKDTree(pos_1)
    kd_tree2 = cKDTree(pos_2)
    sdm = kd_tree1.sparse_distance_matrix(kd_tree2, max_dist)
    logging.info("Get a list of nonzero elements")
    nonzero_idx = sdm.nonzero()
    nonzero_idx = sdm > 0
    unconnected_pcd1 = np.argwhere((nonzero_idx > 0).sum(axis=1) < min_neighbours)[:, 0]
    unconnected_pcd2 = np.argwhere((nonzero_idx > 0).sum(axis=0) < min_neighbours)[:, 1]
    logging.info(
        f"N points in pcd2  with less than {min_neighbours} neighbours: {len(nonzero_idx.sum(axis=0)[0, unconnected_pcd2])} out of {len(pos_2)}"
    )
    logging.info(
        f"N points in pcd1  with less than {min_neighbours} neighbours: {len(nonzero_idx.sum(axis=1)[unconnected_pcd1, 0])} out of {len(pos_1)}"
    )
    logging.info(f"Cost matrix shape: {sdm.shape}")
    logging.info(
        f"Add closest neighbours for points with less than {min_neighbours} neighbours in pcd2"
    )
    for idx in unconnected_pcd2:
        p_2 = pos_2[idx]
        knn_distances, knn_i = kd_tree1.query(p_2, k=min_neighbours, workers=8)
        for d, knn_idx in zip(knn_distances, knn_i):
            sdm[knn_idx, idx] = d
    logging.info(
        f"Add closest neighbours for points with less than {min_neighbours} neighbours in pcd1"
    )
    for idx in unconnected_pcd1:
        p_1 = pos_1[idx]
        knn_distances, knn_i = kd_tree2.query(p_1, k=min_neighbours, workers=8)
        for d, knn_idx in zip(knn_distances, knn_i):
            sdm[idx, knn_idx] = d
    nonzero_idx = sdm > 0
    unconnected_pcd1 = np.argwhere((nonzero_idx > 0).sum(axis=1) < min_neighbours)[:, 0]
    unconnected_pcd2 = np.argwhere((nonzero_idx > 0).sum(axis=0) < min_neighbours)[:, 1]
    logging.info(
        f"N points in pcd2  with less than {min_neighbours} neighbours after adding knn: {len(nonzero_idx.sum(axis=0)[0, unconnected_pcd2])} out of {len(pos_2)}"
    )
    logging.info(
        f"N points in pcd1  with less than {min_neighbours} neighbours after adding knn: {len(nonzero_idx.sum(axis=1)[unconnected_pcd1, 0])} out of {len(pos_1)}"
    )

    return sdm


def problem_setup(sdm):
    nonzero_idx = sdm.nonzero()

    nonzero_costs = sdm[nonzero_idx]
    logging.info(f"nonzero costs shape {nonzero_costs.shape}")
    X = cp.Variable(shape=nonzero_costs.shape, name="X", boolean=True)
    objective = cp.Minimize(cp.sum(cp.multiply(nonzero_costs, X)))

    logging.info("Create constraints")
    constraints = []
    for point in range(sdm.shape[1]):
        point_idx = np.argwhere(nonzero_idx[1] == point).flatten()
        constraints += [cp.sum(X[:, point_idx]) == 1]
    
    #########################################
    # Comment this out to get a one-to-one matching
    #########################################
    for point in range(sdm.shape[0]):
        point_idx = np.argwhere(nonzero_idx[0] == point).flatten()
        constraints += [cp.sum(X[:, point_idx]) >= 1]
    #########################################
    prob = cp.Problem(objective, constraints)
    return X, prob


def sparse_ilp_matching(pos_1, pos_2, max_dist=35, min_neighbours=10):
    """
    Match two point sets via sparse integer linear programming.

    Builds a sparse cost matrix from each point's neighbours and solves an ILP
    for the optimal one-to-one assignment.

    Parameters
    ----------
    pos_1 : numpy.ndarray
        ``(N, 3)`` coordinates of the first point set.
    pos_2 : numpy.ndarray
        ``(M, 3)`` coordinates of the second point set.
    max_dist : float, optional
        Maximum distance between candidate neighbours (default ``35``).
    min_neighbours : int, optional
        Minimum number of neighbours considered per point (default ``10``).

    Returns
    -------
    list of tuple of int
        Matched index pairs ``(i, j)`` into ``pos_1`` and ``pos_2``.
    """
    sdm = create_cost_matrix(
        pos_1, pos_2, max_dist=max_dist, min_neighbours=min_neighbours
    )
    nonzero_idx = sdm.nonzero()
    X, prob = problem_setup(sdm)

    prob.solve(verbose=True, canon_backend=cp.SCIPY_CANON_BACKEND)

    solution = np.array(X.value[0]).astype(np.int8)
    matched_pairs = [
        (p1.item(), p2.item())
        for p1, p2 in zip(nonzero_idx[0][solution == 1], nonzero_idx[1][solution == 1])
    ]

    return matched_pairs


def write_index_pairs(pairs, pairs_path):
    with open(pairs_path, "w+") as f:
        for pair in pairs:
            f.write(str(pair[0]) + "," + str(pair[1]) + "\n")


def read_index_pairs(pairs_path):
    with open(pairs_path, "r") as f:
        pairs = [(int(l.split(",")[0]), int(l.split(",")[1])) for l in f.readlines()]
    return pairs
