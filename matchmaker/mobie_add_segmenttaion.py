import os
import sys
import mobie
import logging
import argparse
from pathlib import Path

from matchmaker.utils import (read_volume, get_attrs, plot_three_slices)


def main():
        parser = argparse.ArgumentParser(
        description="""Add segmentation source to the platy browser.
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
        em_img_np = read_volume(args.input_n5, args.input_key)
        resolution = get_attrs(args.input_n5, args.input_key)["resolution"]
        plot_three_slices(em_img_np , save_path=log_dir / "n5_volume.png")
        
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

        # update the source metadata
        source_metadata = sources.pop(args.source_name, None)

        if not (source_metadata is None):
            print("Source already exists, remove before adding")
            mobie.remove_source(dataset_folder, args.source_name, remove_data=True)
             
        # Run adding the image
        
        mobie.add_segmentation(
        input_path=args.input_n5,
        input_key=args.input_key,
        root=mobie_project_folder,
        dataset_name=dataset_name,
        segmentation_name=args.source_name,
        menu_name=args.menu_name,
        resolution=resolution,
        chunks=chunks,
        scale_factors=scale_factors,
        add_default_table=True  # add the default table with the properties mobie needs to interact with table and segmentation
        )
                    
        
        
if __name__=="__main__":
    main()
    