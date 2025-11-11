import os
import sys
import logging


def setup_logging(log_dir, log_filename, mode="w"):
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode=mode),
            logging.StreamHandler(sys.stdout),
        ],
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info(f"Logging initialized. Writing to {log_path}")
