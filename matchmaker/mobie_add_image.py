from pathlib import Path
import numpy as np
import argparse
import os

from platy_reg.n5_utils import read_volume, write_volume, get_attrs
from platy_reg.vis import plot_three_slices

from scipy.ndimage import affine_transform


import logging
import sys

import mobie



def main():
        parser = argparse.ArgumentParser(
        description="""Add image source to the platy browser.
        """
        )
        parser.add_argument("input_n5", type=str, help="Path of the input n5")
        parser.add_argument("input_key", type=str, help="Key of the volume. Has to be one channel volume, because mobie doesn't handle multichannel")
        parser.add_argument("mobie_project_folder", type=str, help="Path to the mobie project")
        parser.add_argument("dataset_name", type=str, help="Dataset in mobie")
        parser.add_argument("menu_name", type=str, help="Name of the menu in mobie interface (name of the drop-down menu on the left)")
        parser.add_argument("source_name", type=str, help="Name of the volume in mobie")
        parser.add_argument("log_dir", type=str, help="Directory for images, additional files, etc")
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
        
        # Check that the dataset is correct and get the resolution
        # img = read_volume(args.input_n5, args.input_key)
        resolution = get_attrs(args.input_n5, args.input_key)["resolution"]
        # plot_three_slices(img , save_path=log_dir / "n5_volume.png")
        
        # Set parameters for MOBIE
        mobie_project_folder = args.mobie_project_folder
        target ="local"
        max_jobs = 8
        dataset_name = args.dataset_name
        unit = "micrometer"
        chunks = (64, 64, 64)
        scale_factors = 4 * [[2, 2, 2]]
        
        # Delete source if already exists
        dataset_folder = os.path.join(mobie_project_folder, dataset_name)
        dataset_metadata = mobie.metadata.read_dataset_metadata(dataset_folder)
        sources = dataset_metadata["sources"]

        if args.source_name in sources:
            print("Source already exists, remove before adding again")
            mobie.remove_source(dataset_folder, args.source_name, remove_data=True)
             
        # Run adding the image
        mobie.add_image(
            input_path=args.input_n5, 
            input_key=args.input_key,  
            root=mobie_project_folder,
            dataset_name=dataset_name,
            image_name=args.source_name,
            menu_name=args.menu_name,
            resolution=resolution,
            chunks=chunks,
            scale_factors=scale_factors,
            is_default_dataset=False,
            target=target,
            max_jobs=max_jobs,
            unit=unit
        )
                
        
        
if __name__=="__main__":
    main()
    
    