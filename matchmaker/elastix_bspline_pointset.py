from pathlib import Path
import numpy as np
import argparse

from platy_reg.n5_utils import read_volume, write_volume, get_attrs
from platy_reg.vis import plot_overlay


import logging
import sys

from platy_reg.elastix_utils import *

import itk



def elastix_deformable_pointset_alignment(fixed_img_np, fixed_resolution, moving_img_np, moving_resolution, fixed_pointset, moving_pointset, log_dir):
    """
    Run deformable alignment of the ventral and dorsal datasets using elastix.
    """

    logging.info(f"Do deformable alignment")
    
    fixed_img_semantic_np = (fixed_img_np > 0).astype(np.float32)
    moving_img_semantic_np = (moving_img_np > 0).astype(np.float32)
    
    fixed_img = itk_scalar_img(fixed_img_semantic_np, fixed_resolution)
    moving_img = itk_scalar_img(moving_img_semantic_np, moving_resolution)
    
    plot_overlay(itk_to_np_order(itk.GetArrayFromImage(fixed_img)), itk_to_np_order(itk.GetArrayFromImage(moving_img)), log_dir / f"deformable_pointset_alignment_before.png")

    parameter_map_paths = ["pipeline_steps/platybrowser_export/ParameterMap_rigid_pointset.txt", "pipeline_steps/platybrowser_export/ParameterMap_bspline_pointset_rough.txt", "pipeline_steps/platybrowser_export/ParameterMap_bspline_pointset_fine.txt"]
    # parameter_map_paths = ["pipeline_steps/platybrowser_export/ParameterMap_rigid_pointset.txt"]
    
    print("Start registration")
    log_name = f"elastix_log_deformable.log"
    result_image, result_transform_parameters = run_pointset_registration(fixed_img, moving_img, parameter_map_paths, fixed_pointset, moving_pointset, str(log_dir), log_name=log_name)
    
    result_img_np = itk_to_np_order(itk.GetArrayFromImage(result_image))
    plot_overlay(itk_to_np_order(itk.GetArrayFromImage(fixed_img)), result_img_np, log_dir / f"deformable_pointset_alignment_semantic.png")
    
    logging.info(f"Apply transform to all channels")
    result_img_np = apply_transform_chanwise(result_transform_parameters, moving_img_np, moving_resolution)
    plot_overlay(fixed_img_np, result_img_np, log_dir / f"deformable_pointset_alignment_final.png")
    logging.info(f"Result image shape {result_img_np.shape}")
    
    # Transform a grid
    grid_img_np = np.zeros_like(moving_img_np)
    grid_img_np[::10] = 1
    grid_img_np[:, ::10, :] = 1
    grid_img_np[:, :, ::10] = 1
    transformed_grid_np = apply_transform_chanwise(result_transform_parameters, grid_img_np, moving_resolution)
    plot_overlay(fixed_img_np, grid_img_np, log_dir / f"grid_before.png")
    plot_overlay(fixed_img_np, transformed_grid_np, log_dir / f"grid_after.png")
    
    
    return fixed_img_np, result_img_np



def main():
        parser = argparse.ArgumentParser(
        description="""Deformable alignment of LM sample nuclei segmentation to EM nuclei segmentation using registered points as landmarks.
        """
        )
        parser.add_argument("em_n5", type=str, help="Path of the input n5 for EM segmentation")
        parser.add_argument("em_key", type=str, help="Key of EM segmentation")
        parser.add_argument("input_n5", type=str, help="Path of the input n5 for LM segmentation")
        parser.add_argument("input_key", type=str, help="Key of LM segmentation")
        parser.add_argument("output_n5", type=str, help="Path of the output n5")
        parser.add_argument("output_key", type=str, help="Key to write aligned LM segmentation dataset to in the output n5")
        parser.add_argument("fixed_pointset", type=str, help="Path of the output n5")
        parser.add_argument("moving_pointset", type=str, help="Path of the output n5")
        parser.add_argument("log_dir", type=str, help="Directory to store diagnostic plots etc")
        parser.add_argument("log_path", type=str, help="Path to log file")
        args = parser.parse_args()
        
        log_dir = Path(args.log_dir)
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(args.log_path, mode="w"),
                                logging.StreamHandler(sys.stdout)],
                    datefmt='%Y-%m-%d %H:%M:%S')        
        
        logging.info("Read image file")
        em_img_np = read_volume(args.em_n5, args.em_key)
        moving_img_np = read_volume(args.input_n5, args.input_key)
        
        # Debug by transforming a grid
        # moving_img_np = np.zeros_like(moving_img_np)
        # moving_img_np[::10] = 1

        
        em_img_np, moving_img_aligned = elastix_deformable_pointset_alignment(em_img_np,
                                                                                  get_attrs(args.em_n5, args.em_key)["resolution"],
                                                                                  moving_img_np,
                                                                                  get_attrs(args.input_n5, args.input_key)["resolution"],
                                                                                  args.fixed_pointset,
                                                                                  args.moving_pointset,
                                                                                  log_dir)
        logging.info("Plot overlay")
        # plot_overlay(ventral_elastix_aligned[args.dapi_chan, ...], dorsal_elastix_aligned[args.dapi_chan, ...], log_dir / "ventral_dorsal_registered_deformable_overlay.png")
        
        logging.info("Write results")

        attributes = get_attrs(args.input_n5, args.input_key)
        write_volume(args.output_n5, moving_img_aligned, args.output_key, chunks=(128, 512, 512), attrs=attributes)        
        
        
if __name__=="__main__":
    main()
    
    
