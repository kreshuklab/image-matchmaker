import os
import sys
import click
import logging
import numpy as np
import matplotlib.pyplot as plt

from matchmaker.data import create_point_cloud
from matchmaker.utils import (get_transformation_matrix, rotate_img, read_volume, get_attrs, write_volume,
                                write_transform_dict, plot_three_slices, plot_overlay)


def get_SVD_transform(img, save_path=None):
    """Convert image to point cloud by thresholding, then run SVD on resulting point cloud.

    Args:
        img: _description_
        plot_path: _description_. Defaults to None.

    Returns:
        Variance matrix and principal axes matrix.
    """

    pos, _ = create_point_cloud(img)
    gc = pos.mean(axis=0)
    gc = np.array(img.shape) // 2
    pos_c = pos - gc
    logging.info(f"Point cloud shape {pos.shape}")
    logging.info(f"Point cloud center {gc}")

    logging.info("Run SVD ...")
    U, S, Vt = np.linalg.svd(pos_c, full_matrices=False)
    logging.info("U")
    logging.info(str(U))
    logging.info("S")
    logging.info(str(S))
    logging.info("Vt")
    logging.info(str(Vt))

    logging.info("Rotate point cloud")
    vr = pos_c @ Vt.T

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Vertices")
    plt.scatter(pos_c[:, 0], pos_c[:, 1], alpha=0.2)
    plt.subplot(1, 2, 2)
    plt.title("Rotated Vertices")
    plt.scatter(vr[:, 0], vr[:, 1], alpha=0.1)
    if save_path:
        plt.savefig(save_path, dpi=300)

    return gc, Vt


def orient_axis(img, axis, save_path=None):
    """
    Plot the sum intensity along the given axis and return True if the maximum is closer to the
    upper boundary than the lower boundary, False otherwise.

    Args:
        img: 3D image
        axis: Axis to sum along
        save_path: Path to save the plot to. If None, show the plot instead.

    Returns:
        True if the maximum is closer to the upper boundary than the lower boundary, False otherwise.
    """
    assert img.ndim == 3, f"Input image should have 3 dimensions, has {img.ndim}"
    if axis == 0:
        int_profile = np.sum(img, axis=(1, 2))  # profile along z
    if axis == 1:
        int_profile = np.sum(img, axis=(0, 2))  # profile along y
    if axis == 2:
        int_profile = np.sum(img, axis=(0, 1))  # profile along x

    plt.figure()
    plt.plot(int_profile)
    plt.xlabel(f"Axis {axis} Coordinate")
    plt.ylabel(f"Sum intensity along axis = {axis}")
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    max_pos = int_profile.argmax()
    logging.info(f"Max position is {max_pos}, dimension shape is {img.shape[axis]}")
    if max_pos < img.shape[axis] // 2:
        return False
    else:
        return True


def prealign_sample(img):
    """
    Pre-align a sample segmentation with its principal components.

    Args:
        img: Segmentation volume
        file_name: Name of the sample
        save_path: Folder to save results

    Returns:
        Pre-aligned segmentation volume.
    """
    gc, Vt = get_SVD_transform(img)
    T, new_shape = get_transformation_matrix(img, gc, Vt)
    img_rotated = rotate_img(img, T, output_shape=new_shape)

    if np.linalg.det(Vt.T) < 0:
        logging.warning("V includes a reflection (mirroring)")
        R_3x3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        logging.info("Mirror back...")
        img_center = 0.5 * (np.array(img_rotated.shape)-1)
        offset = img_center - R_3x3 @ img_center
        R = np.eye(4)
        R[:3, :3] = R_3x3
        R[:3, 3] = offset
        img_rotated = rotate_img(img_rotated, R, output_shape=img_rotated.shape)

        # update transformation matrix
        T = T @ R

    return img_rotated, T


def run_prealignment(
    fixed_img,
    moving_img,
    output_dir
):
    """
    Run prealignment of a fixed and moving 3D image volume.

    This function reads two volumetric datasets (a fixed and a moving image),
    performs prealignment to roughly register them into a common space, and
    saves diagnostic plots, transformation matrices, and prealigned volumes.
    It also checks axis orientation consistency between the two images and
    applies corrective rotations if necessary. Optionally, the results can be
    exported into a MoBIE project for interactive visualization.

    Steps performed:
        1. Load fixed and moving volumes.
        2. Plot reference slices and overlays before alignment.
        3. Apply prealignment to both volumes.
        4. Check and correct axis orientations if required.
        5. Save prealigned volumes and transformation matrices.
        6. Generate plots before and after prealignment.
        7. Optionally export results to a MoBIE project.

    Parameters
    ----------
    fixed_path : str
        Path to the fixed volume file (e.g. N5, OME-Zarr).
    fixed_key : str
        Dataset key inside the fixed volume file.
    moving_path : str
        Path to the moving volume file (e.g. N5, OME-Zarr).
    moving_key : str
        Dataset key inside the moving volume file.
    output_dir : str
        Directory where outputs (plots, volumes, transformations) will be saved.

    Outputs
    -------
    - Plots of slices and overlays before and after prealignment, saved in
      ``{output_dir}/plots/``.
    - Prealigned fixed and moving volumes saved as N5 containers in
      ``{output_dir}/``.
    - Transformation matrix for the moving image saved as a text file.
    - (Optional) Exported MoBIE project with updated views.

    Notes
    -----
    - The function assumes the input volumes are large 3D datasets.
    - Axis orientation is checked via intensity profile analysis; axes may be
      flipped by 180° if misaligned.
    - The MoBIE export modifies the `dataset.json` to set the prealigned fixed
      volume as the default view.
    """
    if not os.path.exists(f"{output_dir}/plots"):
        os.makedirs(f"{output_dir}/plots")

    logging.info("Start prealignment")
    logging.info("Start prealignment of fixed image ...")

    plot_three_slices(
        fixed_img,
        save_path=f"{output_dir}/plots/fixed_input.png"
    )

    fixed_prealigned, T_fixed = prealign_sample(fixed_img)

    logging.info("Start prealignment of moving image ...")

    plot_three_slices(
        moving_img,
        save_path=f"{output_dir}/plots/moving_input.png"
    )

    plot_overlay(
        fixed_img,
        moving_img,
        save_path=f"{output_dir}/plots/overlay_input.png",
    )

    moving_prealigned, T_moving = prealign_sample(moving_img)

    # check orientation (if moving fits to fixed)
    logging.info("Check axis orientation ...")
    R_3x3 = np.eye(3)
    change_orientation = False
    for axis in range(3):
        rotate_axis_fixed = orient_axis(
            fixed_prealigned,
            axis=axis,
            save_path=f"{output_dir}/plots/fixed_prealigned_intensity_profile_{axis}.png",
        )
        rotate_axis_moving = orient_axis(
            moving_prealigned,
            axis=axis,
            save_path=f"{output_dir}/plots/moving_prealigned_intensity_profile_{axis}.png",
        )

        if rotate_axis_fixed or rotate_axis_moving:
            print(f"Rotate axis {axis} 180 degrees to align...")
            change_orientation = True
            R_3x3[axis, axis] = -1
        else:
            print(f"Correct orientation in axis {axis}.")

    if not change_orientation:
        logging.info("Correct orientation.")
    else:
        logging.info("Rotate moving image to align orientation...")
        logging.info(str(R_3x3))
        img_center = 0.5 * (np.array(moving_prealigned.shape)-1)
        offset = img_center - R_3x3 @ img_center
        R = np.eye(4)
        R[:3, :3] = R_3x3
        R[:3, 3] = offset
        moving_prealigned = rotate_img(moving_prealigned, R, output_shape=moving_prealigned.shape)

        # update transformation matrix
        T_moving = T_moving @ R

    logging.info("Prealignment done.")

    prealignment_transform = {
        "fixed_prealignment": {
            "matrix": T_fixed,
            "output_shape": fixed_prealigned.shape,
        },
        "moving_prealignment": {
            "matrix": T_moving,
            "output_shape": moving_prealigned.shape,
        },
    }

    plot_three_slices(
        fixed_prealigned,
        save_path=f"{output_dir}/plots/fixed_prealigned.png"
    )

    plot_three_slices(
        moving_prealigned,
        save_path=f"{output_dir}/plots/moving_prealigned.png"
    )

    plot_overlay(
        fixed_prealigned,
        moving_prealigned,
        save_path=f"{output_dir}/plots/overlay_after_prealignment.png",
    )

    return fixed_prealigned, moving_prealigned, prealignment_transform


@click.command()
@click.option("-fi", "--fixed_path", required=True, help="Fixed input .n5 file")
@click.option("-fk", "--fixed_key", required=True, help="Fixed input key")
@click.option("-mi", "--moving_path", required=True, help="Moving input .n5 file")
@click.option("-mk", "--moving_key", required=True, help="Moving input key")
@click.option("-o", "--output_dir", required=True, help="Output directory")
@click.option("-ok", "--output_key", required=True, help="Output key (same in both n5)")
@click.option("-trans", "--output_transform_path", required=True, help="Path to write the final transform")
def main(fixed_path, fixed_key, moving_path, moving_key, output_dir, output_key, output_transform_path):
    """
    Perform prealignment of moving image to fixed image.

    This function orchestrates the sequence of steps required to prealign a moving image to a fixed image.
    It handles the creation of necessary directories, configures logging, and invokes prealignment functions.
    Optionally, it can create a MoBIE project for visualization.

    Args:
        fixed_path (str): Path to the fixed input .n5 file.
        fixed_key (str): Key to the fixed image data in the .n5 file.
        moving_path (str): Path to the moving input .n5 file.
        moving_key (str): Key to the moving image data in the .n5 file.
        output_dir (str): Directory where the results should be saved.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{output_dir}/prealignment.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("Reading fixed image")
    fixed_img = read_volume(fixed_path, fixed_key)
    logging.info(f"Fixed image shape: {fixed_img.shape}, dtype {fixed_img.dtype}")

    logging.info("Reading moving image")
    moving_img = read_volume(moving_path, moving_key)
    logging.info(f"Moving image shape: {moving_img.shape}, dtype {moving_img.dtype}")

    fixed_prealigned, moving_prealigned, prealignment_transform = run_prealignment(
        fixed_img,
        moving_img,
        output_dir
    )

    logging.info("Save prealigned fixed image")
    fixed_attributes = dict(get_attrs(fixed_path, fixed_key))
    write_volume(
        f=fixed_path,
        arr=fixed_prealigned,
        key=output_key,
        attrs=fixed_attributes
    )

    logging.info("Save prealigned moving image")
    moving_attributes = dict(get_attrs(moving_path, moving_key))
    write_volume(
        f=moving_path,
        arr=moving_prealigned,
        key=output_key,
        attrs=moving_attributes
    )

    logging.info("Save prealignment tranform")

    print(prealignment_transform)

    write_transform_dict(prealignment_transform, output_transform_path)


if __name__ == "__main__":
    main()
