import logging
from pathlib import Path

import click
import numpy as np
import yaml
from scipy.spatial import cKDTree

from matchmaker.utils import read_volume, get_attrs, extract_centroids

_DEFAULT_RANGES_FILE = Path(__file__).parent / "default_cpd_ranges.yaml"


def compute_point_spacing(coords: np.ndarray) -> float:
    """Mean nearest-neighbor distance as a proxy for point spacing (µm)."""
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)  # k=2: first hit is self (dist=0)
    return float(np.mean(distances[:, 1]))


def suggest_beta_ranges(coords: np.ndarray, fallback_betas: list | None = None) -> list:
    """4 beta values as max(n*h, f*D) with h=point spacing, D=mean extent.

    Values: max(2h,0.025D), max(4h,0.05D), max(8h,0.1D), max(16h,0.2D).
    The h-based terms set a lower bound relative to point density; the
    D-based terms prevent pathologically small betas when landmarks are
    very densely packed relative to the object scale.

    Args:
        coords: (N, 3) array of point positions in µm.
        fallback_betas: returned as-is when coords has fewer than 2 points.

    Returns:
        List of 4 beta values rounded to 2 decimal places.
    """
    if len(coords) < 2:
        logging.warning("Fewer than 2 points; returning fallback beta ranges")
        return fallback_betas if fallback_betas is not None else []

    h = compute_point_spacing(coords)
    D = float(np.mean(coords.max(axis=0) - coords.min(axis=0)))
    logging.info(f"Mean NN spacing h={h:.2f} µm, mean extent D={D:.1f} µm")

    betas = [
        max(2  * h, 0.025 * D),
        max(4  * h, 0.05  * D),
        max(8  * h, 0.1   * D),
        max(16 * h, 0.2   * D),
    ]
    betas = np.round(betas, 2).tolist()
    logging.info(f"Beta ranges: {betas}")
    return betas


@click.command()
@click.option("--path", required=True, help="Path to .n5 segmentation file")
@click.option("--key", required=True, help="Dataset key to read from .n5")
@click.option("--log_dir", required=True, type=click.Path(), help="Directory to write dataset_cpd_ranges.yaml")
@click.option("--x_res", type=float, default=None, help="X resolution in µm (overrides .n5 attrs)")
@click.option("--y_res", type=float, default=None, help="Y resolution in µm (overrides .n5 attrs)")
@click.option("--z_res", type=float, default=None, help="Z resolution in µm (overrides .n5 attrs)")
def main(path, key, log_dir, x_res, y_res, z_res):
    seg = read_volume(path, key)

    if x_res is None or y_res is None or z_res is None:
        attrs = dict(get_attrs(path, key))
        resolution = attrs["resolution"]
    else:
        resolution = [z_res, y_res, x_res]

    _, positions = extract_centroids(seg, resolution)
    n_points = len(positions)

    extents = positions.max(axis=0) - positions.min(axis=0)
    betas = suggest_beta_ranges(positions)

    with open(_DEFAULT_RANGES_FILE) as f:
        search_space = yaml.safe_load(f)
    search_space["beta"] = betas

    output_path = Path(log_dir) / "dataset_cpd_ranges.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(search_space, f, default_flow_style=False)

    click.echo(f"# Point cloud: {n_points} points, extent (x,y,z) µm: "
               f"[{extents[0]:.1f}, {extents[1]:.1f}, {extents[2]:.1f}]")
    click.echo(f"Dataset-specific beta values: {betas}")
    click.echo(f"# Dataset-specific CPD ranges written to {output_path}")


if __name__ == "__main__":
    main()
