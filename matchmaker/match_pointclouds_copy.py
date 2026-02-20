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
import seaborn as sns
from platy_reg.preprocessing import percentile_norm

from platy_reg.pcd_utils import evaluate_registration, plot_point_clouds, visualize_reg_results, map_features
from platy_reg.ilp import sparse_ilp_matching, plot_matching_qc,  write_index_pairs, read_index_pairs
import pandas as pd


use_cuda = True
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
    

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def create_pcd(coords_df: pd.DataFrame, resolution: List):
    # Resolution: ZYX, physical dimensions of pixels
    # coords_df: centroid-0, centroid-1, centroid-2 is ZYX in pixels
    # Open3D expects XYZ in physical coordinates
    
    # Scale coordinates and create points
    center_coords = np.array([coords_df["centroid-2_rigid"] * resolution[2], coords_df["centroid-1_rigid"] * resolution[1], coords_df["centroid-0_rigid"] * resolution[0]]).T.astype(np.float64)
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3d.core.Tensor(center_coords)
    
    # Assign gene signal
    # Type and reshape, because o3d.core.Tensor requires numpy array as input and storing to .pcd specifically wants attributes of shape [N, :]
    coords_df = coords_df.fillna(0)
    pcd.point.gene1_spots = o3d.core.Tensor(np.array(coords_df['counts_gene1'])[:, None])
    pcd.point.gene2_spots = o3d.core.Tensor(np.array(coords_df['counts_gene2'])[:, None])
    pcd.point.gene1_mean = o3d.core.Tensor(np.array(coords_df['intensity_mean_gene1'])[:, None])
    pcd.point.gene2_mean = o3d.core.Tensor(np.array(coords_df['intensity_mean_gene2'])[:, None])
    pcd.point.gene1_sum = o3d.core.Tensor(np.array(coords_df['intensity_sum_gene1'])[:, None])
    pcd.point.gene2_sum = o3d.core.Tensor(np.array(coords_df['intensity_sum_gene2'])[:, None])
    pcd.point.gene1_median = o3d.core.Tensor(np.array(coords_df['intensity_median_gene1'])[:, None])
    pcd.point.gene2_median = o3d.core.Tensor(np.array(coords_df['intensity_median_gene2'])[:, None])
    pcd.point.label = o3d.core.Tensor(np.array(coords_df['label'])[:, None])
    
    # Assign colors: red for gene1 and yellow for gene2
    # red_cmap = sns.light_palette("crimson", as_cmap=True)
    # yellow_cmap = sns.light_palette("gold", as_cmap=True)
    
    # coords_df = coords_df.fillna(0)
    # pcd.point.gene1_counts_red = o3d.core.Tensor([red_cmap(val) for val in percentile_norm(coords_df['counts_gene1'], pmin=0, pmax=98)])
    # pcd.point.gene1_intensity_red = o3d.core.Tensor([red_cmap(val) for val in percentile_norm(coords_df['intensity_mean_gene1'], pmin=0, pmax=98)])
    # pcd.point.gene2_counts_yellow = o3d.core.Tensor([yellow_cmap(val) for val in percentile_norm(coords_df['counts_gene2'], pmin=0, pmax=98)])
    # pcd.point.gene2_intensity_yellow = o3d.core.Tensor([yellow_cmap(val) for val in percentile_norm(coords_df['intensity_mean_gene2'], pmin=0, pmax=98)])
    
    # Assign colors: get a random color for both genes and overall for all points to plot overlays nicely
    rand_color_palette = sns.color_palette("husl", 30) # arbitrary number 30, because that's a maximum of how many samples I'll have
    rand_col_uniform = rand_color_palette[np.random.randint(len(rand_color_palette))]
    pcd.paint_uniform_color(rand_col_uniform)
    
    # rand_col_gene1 = rand_color_palette[np.random.randint(len(rand_color_palette))]
    # rand_cmap = sns.light_palette(rand_col_gene1, as_cmap=True)
    # pcd.point.gene1_counts_rand = o3d.core.Tensor([rand_cmap(val) for val in percentile_norm(coords_df['counts_gene1'], pmin=0, pmax=98)])
    # pcd.point.gene1_intensity_rand = o3d.core.Tensor([rand_cmap(val) for val in percentile_norm(coords_df['intensity_mean_gene1'], pmin=0, pmax=98)])
    
    # rand_col_gene2 = rand_color_palette[np.random.randint(len(rand_color_palette))]
    # rand_cmap = sns.light_palette(rand_col_gene2, as_cmap=True)
    # pcd.point.gene2_counts_rand = o3d.core.Tensor([rand_cmap(val) for val in percentile_norm(coords_df['counts_gene2'], pmin=0, pmax=98)])
    # pcd.point.gene2_intensity_rand = o3d.core.Tensor([rand_cmap(val) for val in percentile_norm(coords_df['intensity_mean_gene2'], pmin=0, pmax=98)])
    
    
    # Assign colors: always same color if this is a fixed image and same color if it's moving - to do later
    
    return pcd






class Plot3DCallback(object):
    """Display the 3D registration result of each iteration by making max projection along z axis (if axes were in XYZ order).

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        save (bool, optional): If this flag is True,
            each iteration image is saved in a sequential number.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray, source_color="turquoise", target_color="violet", save_path: bool|str = False, keep_window: bool = True):
        self._source = source
        self._target = target
        self._result = copy.deepcopy(self._source)
        self.save_path = save_path
        
        if self.save_path:
            self.save_path = Path(self.save_path)
            self.save_path.mkdir(exist_ok=True)
            (self.save_path / "xy").mkdir(exist_ok=True)
            (self.save_path / "xz").mkdir(exist_ok=True)
            (self.save_path / "yz").mkdir(exist_ok=True)
            
        self._cnt = 0
        
        self.source_color = source_color
        self.target_color = target_color
        
        plt.axis("equal")
        source = asnumpy(self._source)
        target = asnumpy(self._target)
        result = asnumpy(self._result)
        plt.plot(source[:, 0], source[:, 1], "ro", label="source", markersize=0.2)
        plt.plot(target[:, 0], target[:, 1], "g^", label="target", markersize=0.2)
        plt.plot(result[:, 0], result[:, 1], "bo", label="result", markersize=0.2)
        plt.legend()
        plt.draw()

    def __call__(self, transformation: Transformation) -> None:
        # print(transformation.w, transformation.g)
        # print("result: ", transformation.b, transformation.t)
        # print("result: ", transformation.rot, transformation.t)
        logging.info(f"Iteration {self._cnt}")
        self._result = transformation.transform(self._source)
        plt.figure(figsize=(10, 10))
        plt.cla()
        plt.axis("equal")
        source = asnumpy(self._source)
        target = asnumpy(self._target)
        result = asnumpy(self._result)
        # plt.plot(source[:, 0], source[:, 1], "ro", label="source", markersize=0.2)
        plt.title(f"Iteration {self._cnt}")
        plt.scatter(target[:, 0], target[:, 1], s=0.5, c=self.target_color, alpha=0.5, label="target")
        plt.scatter(result[:, 0], result[:, 1], s=0.5, c=self.source_color, alpha=0.5, label="result")
        plt.legend()
        if self.save_path:
            plt.savefig(self.save_path / "xy" / f"image_{self._cnt}.png", dpi=300)
        plt.close()
            
        plt.figure(figsize=(10, 10))
        plt.cla()
        plt.axis("equal")
        source = asnumpy(self._source)
        target = asnumpy(self._target)
        result = asnumpy(self._result)
        # plt.plot(source[:, 0], source[:, 1], "ro", label="source", markersize=0.2)
        plt.title(f"Iteration {self._cnt}")
        plt.scatter(target[:, 0], target[:, 2], s=0.5, c=self.target_color, alpha=0.5, label="target")
        plt.scatter(result[:, 0], result[:, 2], s=0.5, c=self.source_color, alpha=0.5, label="result")
        plt.legend()
        if self.save_path:
            plt.savefig(self.save_path / "xz" / f"image_{self._cnt}.png", dpi=300)
        plt.close()
        
        plt.figure(figsize=(10, 10))
        plt.cla()
        plt.axis("equal")
        source = asnumpy(self._source)
        target = asnumpy(self._target)
        result = asnumpy(self._result)
        # plt.plot(source[:, 0], source[:, 1], "ro", label="source", markersize=0.2)
        plt.title(f"Iteration {self._cnt}")
        plt.scatter(target[:, 1], target[:, 2], s=0.5, c=self.target_color, alpha=0.5, label="target")
        plt.scatter(result[:, 1], result[:, 2], s=0.5, c=self.source_color, alpha=0.5, label="result")
        plt.legend()
        if self.save_path:
            plt.savefig(self.save_path / "yz" / f"image_{self._cnt}.png", dpi=300)    
        plt.close()
        
        self._cnt += 1



def run_cpd(fixed_pcd, moving_pcd, log_dir, w, beta, lmd, maxiter, fixed_neuropil, moving_neuropil):
    source_pt = cp.asarray(np.vstack([moving_pcd.point.positions.numpy(), moving_neuropil]), dtype=cp.float32)
    target_pt = cp.asarray(np.vstack([fixed_pcd.point.positions.numpy(), fixed_neuropil]), dtype=cp.float32)

    # Plot point positions
    points = moving_pcd.point.positions.numpy()
    reg_points = fixed_pcd.point.positions.numpy()
    plt.figure(figsize=(15, 15))
    z_mean = np.mean(points[:, 2])
    points_slice = points[abs(points[:, 2] - z_mean) < 10, :]
    reg_points_slice = reg_points[abs(reg_points[:, 2] - z_mean) < 10, :]
    plt.scatter(points_slice[:, 0], points_slice[:, 1], c="green", label="fixed")
    plt.scatter(reg_points_slice[:, 0], reg_points_slice[:, 1], c="red", label="moving")

    points = moving_neuropil
    reg_points = fixed_neuropil
    points_slice = points[abs(points[:, 2] - z_mean) < 10, :]
    reg_points_slice = reg_points[abs(reg_points[:, 2] - z_mean) < 10, :]
    plt.scatter(points_slice[:, 0], points_slice[:, 1], c="blue", label="fixed")
    plt.scatter(reg_points_slice[:, 0], reg_points_slice[:, 1], c="yellow", label="moving")

    plt.legend()
    plt.savefig(log_dir / f"point_slice_before.png", dpi=300)
    
    turquoise_cmap = sns.blend_palette(["paleturquoise", "darkslategray"], as_cmap=True)
    violet_cmap = sns.blend_palette(["pink", "purple"], as_cmap=True)
    
    source_color = [turquoise_cmap(val) for val in percentile_norm(moving_pcd.point.gene1_sum.numpy(), pmin=0, pmax=99)]
    source_color += ["blue"] * len(moving_neuropil)
    target_color = [violet_cmap(val) for val in percentile_norm(fixed_pcd.point.gene1_mean.numpy(), pmin=0, pmax=99)]
    target_color += ["crimson"] * len(fixed_neuropil)
    
    (log_dir / "nonrigid_reg_iterations").mkdir(exist_ok=True)
    cbs = [Plot3DCallback(source_pt, target_pt, source_color, target_color, save_path=log_dir / "nonrigid_reg_iterations", keep_window=False)]
    # cbs = [Plot3DCallback(source_pt, target_pt, source_color, target_color, save_path=None, keep_window=False)]
    start = time.time()
    tf_param, _, _ = cpd.registration_cpd(source_pt, target_pt, use_cuda=use_cuda, maxiter=args.maxiter, tf_type_name="nonrigid", callbacks=cbs, w=args.w, beta=args.beta, lmd=args.lmd)
    elapsed = time.time() - start
    logging.info(f"time: {elapsed}")

    # print("result: ", to_cpu(tf_param.w), to_cpu(tf_param.g))

    result = to_cpu(tf_param.transform(source_pt))
    registered_pcd = copy.deepcopy(moving_pcd)
    registered_pcd.point.positions = o3d.core.Tensor(result[:-len(moving_neuropil)])

    # Plot point positions
    points = fixed_pcd.point.positions.numpy()
    reg_points = registered_pcd.point.positions.numpy()
    plt.figure(figsize=(15, 15))
    z_mean = np.mean(points[:, 2])
    points_slice = points[abs(points[:, 2] - z_mean) < 10, :]
    reg_points_slice = reg_points[abs(reg_points[:, 2] - z_mean) < 10, :]
    plt.scatter(points_slice[:, 0], points_slice[:, 1], c="green", label="fixed")
    plt.scatter(reg_points_slice[:, 0], reg_points_slice[:, 1], c="red", label="moving")
    plt.legend()
    plt.savefig(log_dir / f"point_slice.png", dpi=300)


    # Plot point positions
    points = fixed_pcd.point.positions.numpy()
    reg_points = registered_pcd.point.positions.numpy()
    plt.figure(figsize=(15, 15))
    z_mean = 120
    points_slice = points[abs(points[:, 2] - z_mean) < 10, :]
    reg_points_slice = reg_points[abs(reg_points[:, 2] - z_mean) < 10, :]
    plt.scatter(points_slice[:, 0], points_slice[:, 1], c="green", label="fixed")
    plt.scatter(reg_points_slice[:, 0], reg_points_slice[:, 1], c="red", label="moving")
    plt.legend()
    plt.savefig(log_dir / f"point_slice_120.png", dpi=300)

    return registered_pcd


def match_points(fixed_pcd, registered_pcd, log_dir):

    print("Number of points in fixed pcd:", len(fixed_pcd.point.positions))
    print("Number of points in moving pcd:", len(registered_pcd.point.positions))

    if len(fixed_pcd.point.positions) <= len(registered_pcd.point.positions):
        pos_1 = fixed_pcd.point.positions.numpy()
        pos_2 = registered_pcd.point.positions.numpy()
        swap_order = False
    else:
        pos_1 = registered_pcd.point.positions.numpy()
        pos_2 = fixed_pcd.point.positions.numpy()
        swap_order = True

    matched_pairs = sparse_ilp_matching(pos_1, pos_2, max_dist=8, min_neighbours=30)

    if swap_order:
        matched_pairs = [(p2, p1) for p1, p2 in matched_pairs]

    # matched_pairs = read_index_pairs(str(log_dir / "matched_pairs.txt"))

    write_index_pairs(matched_pairs, str(log_dir / "matched_pairs.txt"))

    z_mean = np.mean(pos_1[:, 2])
    z_slice = (z_mean - 5, z_mean + 5)
    
    if swap_order:
        plot_matching_qc(pos_2, pos_1, log_dir / "point_matching.png", pairs=matched_pairs, z_slice=z_slice)

    else:
        plot_matching_qc(pos_1, pos_2, log_dir / "point_matching.png", pairs=matched_pairs, z_slice=z_slice)

    return matched_pairs



def assign_features(fixed_pcd: o3d.t.geometry.PointCloud, registered_pcd: o3d.t.geometry.PointCloud, matched_pairs):
    mapped_features = {}
    for feat_name, val in registered_pcd.point.items():
        if feat_name not in ["colors", "positions"]:
            registered_feat = registered_pcd.point[feat_name].numpy()
            print(registered_feat.shape)
            fixed_feat = np.zeros(len(fixed_pcd.point.positions))
            for fixed_idx, registered_idx in matched_pairs:
                fixed_feat[fixed_idx] = registered_feat[registered_idx]
            mapped_features[feat_name] = fixed_feat
    mapped_features = pd.DataFrame(mapped_features)
    return mapped_features


def create_final_pos_pcd(fixed_pcd: o3d.t.geometry.PointCloud, registered_pcd: o3d.t.geometry.PointCloud, matched_pairs):
    final_pos_pcd = copy.deepcopy(registered_pcd)
    em_points = fixed_pcd.point.positions.numpy()
    final_pos = np.zeros_like(registered_pcd.point.positions.numpy())
    for fixed_p, moving_p in matched_pairs:
        final_pos[moving_p] = em_points[fixed_p]
    final_pos_pcd.point.positions = o3d.core.Tensor(final_pos)
    return final_pos_pcd



if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="""Nonrigid registration of point clouds using CPD algorithm.
        """
        )

    parser.add_argument("fixed_pcd", type=str, help="Path of the input n5")
    parser.add_argument("moving_pcd", type=str, help="Key of the corresponding segmentation dataset")
    parser.add_argument("output_pcd", type=str, help="Path to output point cloud file. .pcd format allows to store attributes")
    parser.add_argument("em_neuropil", type=str, help="Path of the input n5")
    parser.add_argument("light_neuropil", type=str, help="Path of the input n5")
    parser.add_argument("log_dir", type=str, help="Directory to store diagnostic plots etc")
    parser.add_argument("log_path", type=str, help="Path to log file")
    parser.add_argument("--w", type=float, default=0.0, help="Weight of uniform distribution in the model, 0 <= w <= 1. The closer to 1, the more noise is expected.")
    parser.add_argument("--beta", type=float, default=2.0, help="Width of the gaussian filter used for regularization of the displacement field. The bigger, the less regularization")
    parser.add_argument("--lmd", type=float, default=2.0, help="Weight of the regularization term in the objective. The bigger, the stronger is the regularization.")
    parser.add_argument("--maxiter", type=int, default=100, help="Number of iterations of CPD. Usually 100-150 should be enough.")
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[logging.FileHandler(args.log_path, mode="w"),
                            logging.StreamHandler(sys.stdout)],
                datefmt='%Y-%m-%d %H:%M:%S')        
    
    logging.info(f"Register fixed pcd {args.fixed_pcd} and moving pcd {args.moving_pcd}")
    logging.info(f"Registration parameters w = {args.w}, lambda = {args.lmd}, beta = {args.beta}")
    
    fixed_pcd = o3d.t.io.read_point_cloud(args.fixed_pcd)
    moving_pcd = o3d.t.io.read_point_cloud(args.moving_pcd)

    # registered_pcd = o3d.t.io.read_point_cloud(str(log_dir / "nonrigid_reg_em_backup.pcd"))
    fixed_neuropil = np.array(pd.read_csv(args.em_neuropil))
    moving_neuropil = np.array(pd.read_csv(args.light_neuropil))
    registered_pcd = run_cpd(fixed_pcd, moving_pcd, log_dir, args.w, args.beta, args.lmd, args.maxiter, fixed_neuropil, moving_neuropil)
    o3d.t.io.write_point_cloud(args.output_pcd, registered_pcd, write_ascii=True)

    matched_pairs = match_points(fixed_pcd, registered_pcd, log_dir)

    mapped_features = assign_features(fixed_pcd, registered_pcd, matched_pairs)
    print(mapped_features)
    mapped_features.to_csv(log_dir / "mapped_features.csv", index=False)

    final_pos_pcd = create_final_pos_pcd(fixed_pcd, registered_pcd, matched_pairs)
    o3d.t.io.write_point_cloud(str(log_dir / "em_matched_points.pcd"), final_pos_pcd, write_ascii=True)

    # # Plot point clouds in 3D - should be a separate script, really
    # # o3d.visualization.draw_geometries([source, pc, target])
    
    # params = {"w": args.w, "lmd": args.lmd, "beta": args.beta, "maxiter": args.maxiter}
    # before_registration_path = log_dir / "plots_before"
    # before_registration_path.mkdir(exist_ok=True)
    # visualize_reg_results(fixed_pcd, moving_pcd, before_registration_path, features=["gene1_mean", "gene1_spots", "gene2_mean", "gene2_spots"], params=params, prefix="nonrigid_reg_before")
    
    # after_registration_path = log_dir / "plots_after"
    # after_registration_path.mkdir(exist_ok=True)
    # visualize_reg_results(fixed_pcd, registered_pcd, after_registration_path, features=["gene1_mean", "gene1_spots", "gene2_mean", "gene2_spots"], params=params, prefix="nonrigid_reg_after")
    
    # points = moving_pcd.point.positions.numpy()
    # reg_points = registered_pcd.point.positions.numpy()
    # plt.figure(figsize=(15, 15))
    # z_mean = np.mean(points[:, 2])
    # for idx in range(len(points)):
    #     if abs(points[idx, 2] - z_mean) < 10:
    #         plt.plot([points[idx, 0], reg_points[idx, 0]], [points[idx, 1], reg_points[idx, 1]])
    # plt.savefig(log_dir / f"point_paths.png", dpi=300)
    