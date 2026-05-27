"""Suggest dataset-specific CPD parameter search ranges for use in cpd_optimization_config.yaml.

Reads a segmentation and computes basic point-cloud statistics (spatial extent, point density)
to derive a beta search range proportional to the data scale.  Prints a YAML block that can
be pasted into the optuna.search_space section of the optimization config.

Usage:
    python matchmaker/cpd_parameter_tuning/suggest_cpd_ranges.py \
        --path data/brain_matching/igor_fixed_image.n5 \
        --key svd_prealignment \
        --x_res 0.4 --y_res 0.4 --z_res 0.4
"""

import click
import numpy as np
import yaml

from matchmaker.utils import read_volume, get_attrs, extract_centroids


def suggest_ranges(positions: np.ndarray) -> dict:
    """Derive CPD search space from point-cloud spatial statistics.

    beta should scale with dataset extent: a reasonable range covers
    one decade centred on ~extent/15 (empirically, beta ≈ extent/20 to
    extent/5 works well for biological instance segmentations).

    Args:
        positions: (N, 3) array of point positions in µm

    Returns:
        dict with keys w, beta, lmd, maxiter — same structure as
        cpd_optimization_config.yaml optuna.search_space
    """
    extents = positions.max(axis=0) - positions.min(axis=0)
    mean_extent = float(np.mean(extents))

    # Three beta values spanning one decade around mean_extent / 15,
    # rounded to nearest 5 for readability
    beta_centre = mean_extent / 15.0
    betas = sorted({
        max(5.0, round(beta_centre / 2.0 / 5) * 5),
        max(5.0, round(beta_centre / 5) * 5),
        max(5.0, round(beta_centre * 2.0 / 5) * 5),
    })

    return {
        "w": [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
        "beta": [float(b) for b in betas],
        "lmd": [0.01, 0.1, 1.0],
        "maxiter": [100, 150],
    }


@click.command()
@click.option("--path", required=True, help="Path to .n5 segmentation file")
@click.option("--key", required=True, help="Dataset key to read from .n5")
@click.option("--x_res", type=float, default=None, help="X resolution in µm (overrides .n5 attrs)")
@click.option("--y_res", type=float, default=None, help="Y resolution in µm (overrides .n5 attrs)")
@click.option("--z_res", type=float, default=None, help="Z resolution in µm (overrides .n5 attrs)")
def main(path, key, x_res, y_res, z_res):
    seg = read_volume(path, key)

    if x_res is None or y_res is None or z_res is None:
        attrs = dict(get_attrs(path, key))
        resolution = attrs["resolution"]
    else:
        resolution = [z_res, y_res, x_res]

    _, positions = extract_centroids(seg, resolution)
    n_points = len(positions)

    extents = positions.max(axis=0) - positions.min(axis=0)
    search_space = suggest_ranges(positions)

    click.echo(f"# Point cloud: {n_points} points, extent (x,y,z) µm: "
               f"[{extents[0]:.1f}, {extents[1]:.1f}, {extents[2]:.1f}]")
    click.echo("#")
    click.echo("# Paste the block below into optuna.search_space in your cpd_optimization_config.yaml:")
    click.echo("#")
    block = yaml.dump({"search_space": search_space}, default_flow_style=False)
    for line in block.splitlines():
        click.echo(line)


if __name__ == "__main__":
    main()
