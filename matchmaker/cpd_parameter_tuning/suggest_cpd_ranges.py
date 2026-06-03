from pathlib import Path

import click
import numpy as np
import yaml

from matchmaker.utils import read_volume, get_attrs, extract_centroids

_DEFAULT_RANGES_FILE = Path(__file__).parent / "default_cpd_ranges.yaml"


def suggest_beta_ranges(positions: np.ndarray) -> dict:
    """Derive CPD search space from point-cloud spatial statistics.

    beta should scale with dataset extent: the range covers
    extent/40 to extent/5 across 4 values (extent/40, extent/20,
    extent/10, extent/5).

    Args:
        positions: (N, 3) array of point positions in µm

    Returns:
        betas: sorted list of beta values to try in CPD optimization
    """
    extents = positions.max(axis=0) - positions.min(axis=0)
    mean_extent = float(np.mean(extents))

    betas = sorted({
        mean_extent / 40,
        mean_extent / 20,
        mean_extent / 10,
        mean_extent / 5,
    })

    return np.round(betas, 2).tolist()


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
