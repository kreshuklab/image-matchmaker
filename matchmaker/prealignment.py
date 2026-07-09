import click
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from matchmaker.data import create_point_cloud
from matchmaker.utils import (
    get_transformation_matrix,
    rotate_img,
    read_volume,
    get_attrs,
    write_volume,
    write_transform_dict,
    plot_three_slices,
    plot_overlay,
    setup_logging,
    get_axis_orient_matrix,
    transform_axes_vis,
)

from matchmaker.utils.vis import CYAN_HEX, PINK, CYAN, PINK_HEX, LABEL, _savefig


def get_SVD_transform(img, spacing, save_path=None):
    """Convert image to point cloud by thresholding, then run SVD on resulting point cloud.

    Args:
        img: _description_
        plot_path: _description_. Defaults to None.

    Returns:
        Variance matrix and principal axes matrix.
    """

    pos, _ = create_point_cloud(img)

    pos *= spacing

    gc = pos.mean(axis=0)
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

    gc /= spacing

    if save_path:
        logging.info("Rotate point cloud")
        vr = pos_c @ Vt.T

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Vertices")
        plt.scatter(pos_c[:, 0], pos_c[:, 1], alpha=0.2)
        plt.subplot(1, 2, 2)
        plt.title("Rotated Vertices")
        plt.scatter(vr[:, 0], vr[:, 1], alpha=0.1)
        _savefig(save_path)

    return gc, Vt


def orient_axis(fixed_prealigned, moving_prealigned, output_dir):
    int_prof_z = np.sum(moving_prealigned > 0, axis=(1, 2)) / np.sum(
        moving_prealigned > 0
    )
    asymm_coeff_z = (
        np.corrcoef(int_prof_z, int_prof_z)[0, 1]
        - np.corrcoef(int_prof_z, int_prof_z[::-1])[0, 1]
    )
    int_prof_y = np.sum(moving_prealigned > 0, axis=(0, 2)) / np.sum(
        moving_prealigned > 0
    )
    asymm_coeff_y = (
        np.corrcoef(int_prof_y, int_prof_y)[0, 1]
        - np.corrcoef(int_prof_y, int_prof_y[::-1])[0, 1]
    )
    int_prof_x = np.sum(moving_prealigned > 0, axis=(1, 0)) / np.sum(
        moving_prealigned > 0
    )
    asymm_coeff_x = (
        np.corrcoef(int_prof_x, int_prof_x)[0, 1]
        - np.corrcoef(int_prof_x, int_prof_x[::-1])[0, 1]
    )

    logging.info("Asymmetry coefficients in moving image:")
    logging.info(f"Z: {asymm_coeff_z}")
    logging.info(f"Y: {asymm_coeff_y}")
    logging.info(f"X: {asymm_coeff_x}")

    int_prof_z_fixed = np.sum(fixed_prealigned > 0, axis=(1, 2)) / np.sum(
        fixed_prealigned > 0
    )
    int_prof_y_fixed = np.sum(fixed_prealigned > 0, axis=(0, 2)) / np.sum(
        fixed_prealigned > 0
    )
    int_prof_x_fixed = np.sum(fixed_prealigned > 0, axis=(1, 0)) / np.sum(
        fixed_prealigned > 0
    )

    plt.figure()
    plt.plot(int_prof_z_fixed, label="fixed", color=PINK_HEX)
    plt.plot(int_prof_z, label="moving", color=CYAN_HEX)
    plt.xlabel("Axis Z Coordinate")
    plt.ylabel("Sum intensity along axis = Z")
    plt.legend()
    _savefig(f"{output_dir}/plots/axis_int_profile_Z.png")

    plt.figure()
    plt.plot(int_prof_y_fixed, label="fixed", color=PINK_HEX)
    plt.plot(int_prof_y, label="moving", color=CYAN_HEX)
    plt.xlabel("Axis Y Coordinate")
    plt.ylabel("Sum intensity along axis = Y")
    plt.legend()
    _savefig(f"{output_dir}/plots/axis_int_profile_Y.png")

    plt.figure()
    plt.plot(int_prof_x_fixed, label="fixed", color=PINK_HEX)
    plt.plot(int_prof_x, label="moving", color=CYAN_HEX)
    plt.xlabel("Axis X Coordinate")
    plt.ylabel("Sum intensity along axis = X")
    plt.legend()
    _savefig(f"{output_dir}/plots/axis_int_profile_X.png")

    if (
        np.corrcoef(int_prof_z, int_prof_z_fixed)[0, 1]
        > np.corrcoef(int_prof_z[::-1], int_prof_z_fixed)[0, 1]
    ):
        z_correct = True
    else:
        z_correct = False

    if (
        np.corrcoef(int_prof_y, int_prof_y_fixed)[0, 1]
        > np.corrcoef(int_prof_y[::-1], int_prof_y_fixed)[0, 1]
    ):
        y_correct = True
    else:
        y_correct = False

    if (
        np.corrcoef(int_prof_x, int_prof_x_fixed)[0, 1]
        > np.corrcoef(int_prof_x[::-1], int_prof_x_fixed)[0, 1]
    ):
        x_correct = True
    else:
        x_correct = False

    logging.info(
        f"Orientations are correct: Z - {z_correct}, Y - {y_correct}, X - {x_correct}"
    )

    if (asymm_coeff_x < asymm_coeff_y) and (asymm_coeff_x < asymm_coeff_z):
        logging.info("Using axes Y and Z for determining orientation")
        if z_correct and y_correct:
            logging.info("Orientation along both axes is correct")
            R = np.eye(4, 4)
        elif z_correct and not y_correct:
            logging.info("Rotate 180 degrees around Z axis")
            R = get_axis_orient_matrix(moving_prealigned, "xyz")
        else:
            logging.info("Rotate 180 degrees around X axis")
            R = get_axis_orient_matrix(moving_prealigned, "zyx")

    elif (asymm_coeff_y < asymm_coeff_x) and (asymm_coeff_y < asymm_coeff_z):
        logging.info("Using axes X and Z for determining orientation")
        if z_correct and x_correct:
            logging.info("Orientation along both axes is correct")
            R = np.eye(4, 4)
        elif z_correct and not x_correct:
            logging.info("Rotate 180 degrees around Z axis")
            R = get_axis_orient_matrix(moving_prealigned, "xyz")
        else:
            logging.info("Rotate 180 degrees around Y axis")
            R = get_axis_orient_matrix(moving_prealigned, "yzx")

    else:
        logging.info("Using axes X and Y for determining orientation")
        if x_correct and y_correct:
            logging.info("Orientation along both axes is correct")
            R = np.eye(4, 4)
        elif x_correct and not y_correct:
            logging.info("Rotate 180 degrees around X axis")
            R = get_axis_orient_matrix(moving_prealigned, "zyx")
        else:
            logging.info("Rotate 180 degrees around Y axis")
            R = get_axis_orient_matrix(moving_prealigned, "yzx")

    return R


def generate_rotation_overlays(fixed_prealigned, moving_prealigned, output_dir):
    plot_dir = Path(f"{output_dir}/manual_prealignment_options/")
    plot_dir.mkdir(exist_ok=True)

    plot_overlay(
        fixed_prealigned,
        moving_prealigned,
        save_path=f"{output_dir}/manual_prealignment_options/IDENTITY.png"
    )

    R = get_axis_orient_matrix(moving_prealigned, "zyx")
    moving_rotated = rotate_img(moving_prealigned, R, output_shape=moving_prealigned.shape)
    plot_overlay(
        fixed_prealigned,
        moving_rotated,
        save_path=f"{output_dir}/manual_prealignment_options/X.png"
    )

    R = get_axis_orient_matrix(moving_prealigned, "yzx")
    moving_rotated = rotate_img(moving_prealigned, R, output_shape=moving_prealigned.shape)
    plot_overlay(
        fixed_prealigned,
        moving_rotated,
        save_path=f"{output_dir}/manual_prealignment_options/Y.png"
    )

    R = get_axis_orient_matrix(moving_prealigned, "xyz")
    moving_rotated = rotate_img(moving_prealigned, R, output_shape=moving_prealigned.shape)
    plot_overlay(
        fixed_prealigned,
        moving_rotated,
        save_path=f"{output_dir}/manual_prealignment_options/Z.png"
    )


def prealign_samples(fixed_img, moving_img, fixed_spacing, moving_spacing, new_spacing):
    """
    Pre-align two segmentation volumes using PCA.

    Aligns the centroids and principal axes of both volumes into a common
    output space, correcting reflections if the PCA rotation includes a mirror.

    Parameters
    ----------
    fixed_img : numpy.ndarray
        Fixed segmentation volume.
    moving_img : numpy.ndarray
        Moving segmentation volume.
    fixed_spacing, moving_spacing : sequence of float
        Voxel spacing of the fixed and moving volumes.
    new_spacing : sequence of float
        Voxel spacing of the shared output space.

    Returns
    -------
    dict
        Mapping with keys ``"fixed"`` and ``"moving"``, each a list
        ``[prealigned_volume, transform, centroid, Vt, output_shape]``.
    """
    gc_fixed, Vt_fixed = get_SVD_transform(fixed_img, fixed_spacing)
    gc_moving, Vt_moving = get_SVD_transform(moving_img, moving_spacing)

    T_fixed, fixed_shape = get_transformation_matrix(
        fixed_img,
        gc_fixed,
        Vt_fixed,
        fixed_spacing,
        img_ref=moving_img,
        Vt_ref=Vt_moving,
        spacing_ref=moving_spacing,
        spacing_out=new_spacing,
    )
    T_moving, moving_shape = get_transformation_matrix(
        moving_img,
        gc_moving,
        Vt_moving,
        moving_spacing,
        img_ref=fixed_img,
        Vt_ref=Vt_fixed,
        spacing_ref=fixed_spacing,
        spacing_out=new_spacing,
    )
    assert np.array_equal(fixed_shape, moving_shape)

    fixed_rot = rotate_img(fixed_img, T_fixed, output_shape=fixed_shape)
    moving_rot = rotate_img(moving_img, T_moving, output_shape=fixed_shape)

    def mirror_img(img, T):
        logging.warning("V includes a reflection (mirroring)")
        R_3x3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        logging.info("Mirror back...")
        img_center = 0.5 * (np.array(img.shape)-1)
        offset = img_center - R_3x3 @ img_center
        R = np.eye(4)
        R[:3, :3] = R_3x3
        R[:3, 3] = offset
        img_rotated = rotate_img(img, R, output_shape=img.shape)

        # update transformation matrix
        T = T @ R
        return img_rotated, T

    if np.linalg.det(Vt_fixed.T) < 0:
        fixed_rot, T_fixed = mirror_img(fixed_rot, T_fixed)

    if np.linalg.det(Vt_moving.T) < 0:
        moving_rot, T_moving = mirror_img(moving_rot, T_moving)

    return {"fixed": [fixed_rot, T_fixed, gc_fixed, Vt_fixed, fixed_shape],
            "moving": [moving_rot, T_moving, gc_moving, Vt_moving, moving_shape],}


def run_prealignment(
    fixed_img,
    moving_img,
    fixed_spacing,
    moving_spacing,
    new_spacing,
    output_dir,
    axis_orientation
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
    fixed_img : numpy.ndarray
        Fixed segmentation volume.
    moving_img : numpy.ndarray
        Moving segmentation volume.
    fixed_spacing, moving_spacing : sequence of float
        Voxel spacing of the fixed and moving volumes.
    new_spacing : sequence of float
        Voxel spacing of the shared output space.
    output_dir : str
        Directory where outputs (plots, volumes, transformations) will be saved.
    axis_orientation : str
        How to orient the principal axes: ``auto``, ``IDENTITY``, ``X``, ``Y``,
        or ``Z``.

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
    Path(f"{output_dir}/plots").mkdir(parents=True, exist_ok=True)

    logging.info("Start prealignment")

    logging.info("Start prealignment of fixed and moving images ...")

    prealigned_results = prealign_samples(fixed_img, moving_img, fixed_spacing, moving_spacing, new_spacing)

    fixed_prealigned, T_fixed, gc_fixed, Vt_fixed, fixed_shape = prealigned_results["fixed"]
    moving_prealigned, T_moving, gc_moving, Vt_moving, _ = prealigned_results["moving"]

    plot_three_slices(
        fixed_img,
        save_path=f"{output_dir}/plots/fixed_input.pdf",
        gc=gc_fixed,
        Vt=Vt_fixed,
        cmap=LABEL
    )

    plot_three_slices(
        moving_img,
        save_path=f"{output_dir}/plots/moving_input.pdf",
        gc=gc_moving,
        Vt=Vt_moving,
        cmap=LABEL
    )

    plot_three_slices(
        fixed_img,
        save_path=f"{output_dir}/plots/fixed_input_semantic.pdf",
        gc=gc_fixed,
        Vt=Vt_fixed,
        cmap=PINK,
    )

    plot_three_slices(
        moving_img,
        save_path=f"{output_dir}/plots/moving_input_semantic.pdf",
        gc=gc_moving,
        Vt=Vt_moving,
        cmap=CYAN
    )

    plot_overlay(
        fixed_img,
        moving_img,
        save_path=f"{output_dir}/plots/overlay_input.pdf",
        gc1=gc_fixed,
        Vt1=Vt_fixed,
        gc2=gc_moving,
        Vt2=Vt_moving,
    )

    plot_overlay(
        fixed_prealigned,
        moving_prealigned,
        save_path=f"{output_dir}/plots/overlay_after_prealignment_before_axis_orient.png",
        gc1=(np.linalg.inv(T_fixed) @ np.append(gc_fixed, 1))[:3],
        Vt1=transform_axes_vis(Vt_fixed, T_fixed),
        gc2=(np.linalg.inv(T_moving) @ np.append(gc_moving, 1))[:3],
        Vt2=transform_axes_vis(Vt_moving, T_moving),
    )
    # check orientation (if moving fits to fixed)

    if axis_orientation == "auto":
        logging.info("Try to determine axis orientation based on intensity profile of the samples")
        R = orient_axis(fixed_prealigned, moving_prealigned, output_dir)
        # In case the automatic estimation is incorrect, generate possible rotations
        generate_rotation_overlays(fixed_prealigned, moving_prealigned, output_dir)

        moving_prealigned = rotate_img(moving_prealigned, R, output_shape=moving_prealigned.shape)
        logging.info("Axis orientation matrix")
        logging.info(str(R))
        # update transformation matrix
        T_moving = T_moving @ R

    elif axis_orientation == "IDENTITY":
        pass

    elif axis_orientation == "X":
        logging.info("Rotate 180 degrees around X to orient axes")
        R = get_axis_orient_matrix(moving_prealigned, "zyx")
        logging.info("Axis orientation matrix")
        logging.info(str(R))
        moving_prealigned = rotate_img(moving_prealigned, R, output_shape=moving_prealigned.shape)
        T_moving = T_moving @ R

    elif axis_orientation == "Y":
        logging.info("Rotate 180 degrees around Y to orient axes")
        R = get_axis_orient_matrix(moving_prealigned, "yzx")
        logging.info("Axis orientation matrix")
        logging.info(str(R))
        moving_prealigned = rotate_img(moving_prealigned, R, output_shape=moving_prealigned.shape)
        T_moving = T_moving @ R

    elif axis_orientation == "Z":
        logging.info("Rotate 180 degrees around Z to orient axes")
        R = get_axis_orient_matrix(moving_prealigned, "xyz")
        logging.info("Axis orientation matrix")
        logging.info(str(R))
        moving_prealigned = rotate_img(moving_prealigned, R, output_shape=moving_prealigned.shape)
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
        save_path=f"{output_dir}/plots/fixed_prealigned.pdf",
        gc=(np.linalg.inv(T_fixed) @ np.append(gc_fixed, 1))[:3],
        Vt=transform_axes_vis(Vt_fixed, T_fixed),
        cmap=PINK,
    )

    plot_three_slices(
        moving_prealigned,
        save_path=f"{output_dir}/plots/moving_prealigned.pdf",
        gc=(np.linalg.inv(T_moving) @ np.append(gc_moving, 1))[:3],
        Vt=transform_axes_vis(Vt_moving, T_moving),
        cmap=CYAN,
    )

    plot_overlay(
        fixed_prealigned,
        moving_prealigned,
        save_path=f"{output_dir}/plots/overlay_after_prealignment.png",
        gc1=(np.linalg.inv(T_fixed) @ np.append(gc_fixed, 1))[:3],
        Vt1=transform_axes_vis(Vt_fixed, T_fixed),
        gc2=(np.linalg.inv(T_moving) @ np.append(gc_moving, 1))[:3],
        Vt2=transform_axes_vis(Vt_moving, T_moving),
    )

    return fixed_prealigned, moving_prealigned, prealignment_transform


@click.command()
@click.option("-fi", "--fixed_path", required=True, help="Fixed input .n5 file")
@click.option("-fk", "--fixed_key", required=True, help="Fixed input key")
@click.option("-fs", "--fixed_spacing", nargs=3, required=True, help="Fixed input spacing")
@click.option("-mi", "--moving_path", required=True, help="Moving input .n5 file")
@click.option("-mk", "--moving_key", required=True, help="Moving input key")
@click.option("-ms", "--moving_spacing", nargs=3, required=True, help="Moving input spacing")
@click.option("-o", "--output_dir", required=True, help="Output directory")
@click.option("-ok", "--output_key", required=True, help="Output key (same in both n5)")
@click.option("-trans", "--output_transform_path", required=True, help="Path to write the final transform")
@click.option("-axis_orientation", "--axis_orientation", required=True,
              help="How to find the correct orientation along the principal axes")
@click.option("-tif", "--save_tif", is_flag=True, help="Whether to save tif or not")
def main(
    fixed_path,
    fixed_key,
    fixed_spacing,
    moving_path,
    moving_key,
    moving_spacing,
    output_dir,
    output_key,
    output_transform_path,
    axis_orientation,
    save_tif=False,
):
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
        output_key (str): Key to the output image data in the .n5 file.
    Returns:
        None
    """
    setup_logging(output_dir, "prealignment.log")

    fixed_spacing = np.asarray(fixed_spacing, dtype=np.float32)
    moving_spacing = np.asarray(moving_spacing, dtype=np.float32)
    new_spacing = np.full_like(fixed_spacing, fixed_spacing.min())

    logging.info("Reading fixed image")
    fixed_img = read_volume(fixed_path, fixed_key)
    logging.info(f"Fixed image shape: {fixed_img.shape}, dtype {fixed_img.dtype}, resolution {fixed_spacing}")

    logging.info("Reading moving image")
    moving_img = read_volume(moving_path, moving_key)
    logging.info(f"Moving image shape: {moving_img.shape}, dtype {moving_img.dtype}, resolution {moving_spacing}")

    fixed_prealigned, moving_prealigned, prealignment_transform = run_prealignment(
        fixed_img,
        moving_img,
        fixed_spacing,
        moving_spacing,
        new_spacing,
        output_dir,
        axis_orientation
    )

    logging.info("Save prealigned fixed image")
    fixed_attributes = dict(get_attrs(fixed_path, fixed_key))
    if not np.array_equal(fixed_spacing, new_spacing):
        assert new_spacing is not None
        fixed_attributes["resolution"] = new_spacing.tolist()
    write_volume(
        f=fixed_path,
        arr=fixed_prealigned,
        key=output_key,
        attrs=fixed_attributes
    )

    logging.info("Save prealigned moving image")
    moving_attributes = dict(get_attrs(moving_path, moving_key))
    if not np.array_equal(moving_spacing, new_spacing):
        moving_attributes["resolution"] = new_spacing.tolist()
    write_volume(
        f=moving_path,
        arr=moving_prealigned,
        key=output_key,
        attrs=moving_attributes
    )

    if save_tif:
        import tifffile as tiff
        tiff.imwrite(f"{output_dir}/fixed_prealigned.tif", fixed_prealigned)
        tiff.imwrite(f"{output_dir}/moving_prealigned.tif", moving_prealigned)

    logging.info("Save prealignment tranform")

    print(prealignment_transform)

    write_transform_dict(prealignment_transform, output_transform_path)


if __name__ == "__main__":
    main()
