import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import click
import sys

from matchmaker.data import create_point_cloud
from matchmaker.mobie_export import export_to_mobie, update_default_view
from matchmaker.transform_utils import get_transformation_matrix, rotate_img
from matchmaker.n5_utils import read_volume, get_attrs, write_volume
from matchmaker.vis import plot_three_slices, plot_overlay


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

    # TODO: update axis labels
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("original vertices")
    plt.scatter(pos_c[:, 0], pos_c[:, 1], alpha=0.2)
    plt.subplot(1, 2, 2)
    plt.title("rotated vertices")
    plt.scatter(vr[:, 0], vr[:, 1], alpha=0.1)
    if save_path:
        plt.savefig(save_path, dpi=300)

    return gc, Vt


def orient_axis(img, axis, save_path=None):
    # logging.info(f"Orient along axis {axis}")
    assert img.ndim == 3, f"Input image should have 3 dimensions, has {img.ndim}"
    if axis == 0:
        int_profile = np.sum(img, axis=(1, 2))  # profile along z
    if axis == 1:
        int_profile = np.sum(img, axis=(0, 2))  # profile along y
    if axis == 2:
        int_profile = np.sum(img, axis=(0, 1))  # profile along x

    plt.figure()
    plt.plot(int_profile)
    plt.xlabel("Y Coordinate")
    plt.ylabel(f"Sum intensity along axis = {axis}")
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    max_pos = int_profile.argmax()
    logging.info(f"Max position is {max_pos}, dimension shape is {img.shape[axis]}")
    if max_pos < img.shape[axis] // 2:
        logging.info("Correct orientation")
        return False
    else:
        logging.info("Rotate 180 degree to align")
        return True


def prealign_sample(img, file_name, save_path):
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
    T, new_shape = get_transformation_matrix(img, gc, Vt, save_path=f"{save_path}/{file_name}_T_prealignment.txt")
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
        np.savetxt(f"{save_path}/{file_name}_T_prealignment.txt", T)

    return img_rotated, T


def run_prealignment(
    fixed_path,
    fixed_key,
    moving_path,
    moving_key,
    output_dir,
    mobie_export=True,
):
    if not os.path.exists(f"{output_dir}/plots"):
        os.makedirs(f"{output_dir}/plots")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{output_dir}/prealignment.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Start prealignment of fixed image ...")
    fixed_img = read_volume(fixed_path, fixed_key)
    fixed_file_name = os.path.splitext(os.path.basename(fixed_path))[0]

    plot_three_slices(
        fixed_img,
        save_path=f"{output_dir}/plots/{fixed_file_name}_fixed.png"
    )

    fixed_prealigned, _ = prealign_sample(
        fixed_img,
        fixed_file_name,
        output_dir,
    )

    logging.info("Start prealignment of moving image ...")
    moving_img = read_volume(moving_path, moving_key)
    moving_file_name = os.path.splitext(os.path.basename(moving_path))[0]

    plot_three_slices(
        moving_img,
        save_path=f"{output_dir}/plots/{moving_file_name}_moving.png"
    )

    moving_prealigned, T_moving = prealign_sample(
        moving_img,
        moving_file_name,
        output_dir,
    )

    # check orientation (if moving fits to fixed)
    R_3x3 = np.eye(3)
    change_orientation = False
    for axis in range(3):
        rotate_axis_fixed = orient_axis(
            fixed_prealigned,
            axis=axis,
            save_path=f"{output_dir}/plots/fixed_intensity_profile_{axis}.png",
        )
        rotate_axis_moving = orient_axis(
            moving_prealigned,
            axis=axis,
            save_path=f"{output_dir}/plots/moving_intensity_profile_{axis}.png",
        )

        if rotate_axis_fixed or rotate_axis_moving:
            print(f"Rotate axis {axis} 180 degrees to align...")
            change_orientation = True
            R_3x3[axis, axis] = -1
        else:
            print(f"Correct orientation in axis {axis}")

    if not change_orientation:
        logging.info("Correct orientation")
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
        np.savetxt(f"{output_dir}/moving_T_prealignment.txt", T_moving)

    logging.info("Prealignment done.")

    logging.info("Save prealigned fixed image")
    attributes = dict(get_attrs(fixed_path, fixed_key))
    write_volume(
        f=f"{output_dir}/{fixed_file_name}_prealigned.n5",
        arr=fixed_prealigned,
        key=fixed_key,
        attrs=attributes
    )

    logging.info("Save prealigned moving image")
    attributes = dict(get_attrs(moving_path, moving_key))
    write_volume(
        f=f"{output_dir}/{moving_file_name}_prealigned.n5",
        arr=moving_prealigned,
        key=moving_key,
        attrs=attributes
    )

    plot_three_slices(
        fixed_prealigned,
        save_path=f"{output_dir}/plots/{fixed_file_name}_prealigned.png"
    )

    plot_three_slices(
        moving_prealigned,
        save_path=f"{output_dir}/plots/{moving_file_name}_prealigned.png"
    )

    plot_overlay(
        fixed_prealigned,
        moving_prealigned,
        save_path=f"{output_dir}/plots/overlay_prealignment.png",
    )

    if mobie_export:
        logging.info("Export prealigned images to MoBIE")
        export_to_mobie(
            input_path=f"{output_dir}/{fixed_file_name}_prealigned.n5",
            input_key=fixed_key,
            output_dir=output_dir,
            segmentation_name=f"{fixed_file_name}_prealigned",
            menu_name="fixed"
        )
        update_default_view(
            dataset_json_path=f"{output_dir}/mobie_project/platy1_muscles_stardist/dataset.json",
            # TODO: make name independent
            new_segmentation_name=f"{fixed_file_name}_prealigned"
        )

        export_to_mobie(
            input_path=f"{output_dir}/{moving_file_name}_prealigned.n5",
            input_key=moving_key,
            output_dir=output_dir,
            segmentation_name=f"{moving_file_name}_prealigned",
            menu_name="moving"
        )
        logging.info("MoBIE export done.")


@click.command()
@click.option("-fi", "--fixed_path", required=True, help="Fixed input .n5 file")
@click.option("-fk", "--fixed_key", required=True, help="Fixed input key")
@click.option("-mi", "--moving_path", required=True, help="Moving input .n5 file")
@click.option("-mk", "--moving_key", required=True, help="Moving input key")
@click.option("-o", "--output_dir", required=True, help="Output directory")
@click.option("-m", "--mobie_export", required=False, is_flag=True, help="MoBIE export")
def main(fixed_path, fixed_key, moving_path, moving_key, output_dir, mobie_export):

    run_prealignment(fixed_path, fixed_key, moving_path, moving_key, output_dir, mobie_export)


if __name__ == "__main__":
    main()
