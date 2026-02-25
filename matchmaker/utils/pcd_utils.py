import numpy as np
from pathlib import Path
import open3d as o3d
import copy
import time
from probreg import cpd, callbacks
import logging
import argparse
import sys
import matplotlib.pyplot as plt
from probreg.transformation import Transformation
from skimage.measure import regionprops_table
import seaborn as sns

import pandas as pd


use_cuda = False
if use_cuda:
    import cupy as cp
    to_cpu = cp.asnumpy
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    asnumpy = cp.asnumpy
else:
    cp = np
    to_cpu = lambda x: x
    
    def asnumpy(x):
        return x
    



def extract_centroids(segm, resolution):
    coords_df = pd.DataFrame(regionprops_table(segm, properties=("label", 'centroid')))
    center_coords = np.array([coords_df["centroid-2"] * resolution[2], coords_df["centroid-1"] * resolution[1], coords_df["centroid-0"] * resolution[0]]).T.astype(np.float64)
    labels = np.array(coords_df['label'])
    return labels, center_coords


def create_pcd(center_coords, labels):
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3d.core.Tensor(center_coords)
    pcd.point.label = o3d.core.Tensor(labels[:, None])
    return pcd




def run_cpd(fixed_pcd, moving_pcd, w, beta, lmd, maxiter):
    # source_pt = asnumpy(moving_pcd.point.positions.numpy())
    # target_pt = asnumpy(fixed_pcd.point.positions.numpy())

    source_pt = cp.asarray(moving_pcd.point.positions.numpy(), dtype=cp.float32)
    target_pt = cp.asarray(fixed_pcd.point.positions.numpy(), dtype=cp.float32)
    
    start = time.time()
    
    tf_param, _, _ = cpd.registration_cpd(source_pt, target_pt, use_cuda=use_cuda, maxiter=maxiter, tf_type_name="nonrigid", w=w, beta=float(beta), lmd=lmd)
    elapsed = time.time() - start
    logging.info(f"time: {elapsed}")

    # print("result: ", to_cpu(tf_param.w), to_cpu(tf_param.g))

    result = to_cpu(tf_param.transform(source_pt))
    registered_pcd = copy.deepcopy(moving_pcd)
    registered_pcd.point.positions = o3d.core.Tensor(result)


    return registered_pcd
