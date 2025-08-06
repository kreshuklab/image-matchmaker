import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import click
import sys
import napari

from matchmaker.data import create_point_cloud
from matchmaker.mobie_export import export_to_mobie
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
    logging.info(f"Orient along axis {axis}")
    assert img.ndim == 3, f"Input image should have 3 dimensions, has {img.ndim}"
    if axis == 0:
        int_profile = np.sum(img, axis=(1, 2))  # profile along z
        R_3x3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    if axis == 1:
        int_profile = np.sum(img, axis=(0, 2))  # profile along y
        R_3x3 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    if axis == 2:
        int_profile = np.sum(img, axis=(0, 1))  # profile along x
        R_3x3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])

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
    if max_pos < img.shape[1] // 2:
        logging.info("Correct orientation")
        return False
    else:
        logging.info("Rotate 180 degree to align")
        return True


# def orient_sample(img, axis, save_path=None):
#     R_3x3 = orient_axis(img, axis=axis)  #, save_path=save_path)
#     if R_3x3 is not None:
#         img_center = 0.5 * (np.array(img_rotated.shape)-1)
#         offset = img_center - R_3x3 @ img_center
#         R = np.eye(4)
#         R[:3, :3] = R_3x3
#         R[:3, 3] = offset
#         img_rotated = rotate_img(img_rotated, R, output_shape=img_rotated.shape)

#         # update transformation matrix
#         T = T @ R
#         np.savetxt(f"{save_path}/{file_name}_T_prealignment.txt", T)
    
#         return img_rotated


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
        print("WARNING: V includes a reflection (mirroring)")
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

    # check orientation in every three dimensions/ if heads up
    # for axis in [0, 1, 2]:
    #     R_3x3 = orient_axis(img_rotated, axis=axis, save_path=f"{save_path}/plots/{file_name}_intensity_profile_{axis}.png")
        # if R_3x3 is not None:
        #     logging.info(f"Rotate axis {axis} 180 degrees to align...")
        #     img_center = 0.5 * (np.array(img_rotated.shape)-1)
        #     offset = img_center - R_3x3 @ img_center
        #     R = np.eye(4)
        #     R[:3, :3] = R_3x3
        #     R[:3, 3] = offset
        #     img_rotated = rotate_img(img_rotated, R, output_shape=img_rotated.shape)

        #     # update transformation matrix
        #     T = T @ R
        #     np.savetxt(f"{save_path}/{file_name}_T_prealignment.txt", T)

    plot_three_slices(
        img_rotated,
        save_path=f"{save_path}/plots/{file_name}_prealigned.png"
    )

    return img_rotated


def prealignment_per_image(input_path, input_key, output_dir, mobie_export, image_type):
    logging.info("Read image file")
    img = read_volume(input_path, input_key)
    file_name = os.path.splitext(os.path.basename(input_path))[0]

    # plot three slices of input image
    plot_three_slices(img, save_path=f"{output_dir}/plots/{image_type}.png")

    logging.info("Prealign image")
    img_prealigned = prealign_sample(img, file_name, save_path=output_dir)

    logging.info("Save prealigned image")
    attributes = dict(get_attrs(input_path, input_key))
    write_volume(
        f=f"{output_dir}/{file_name}_prealigned.n5",
        arr=img_prealigned,
        key=input_key,
        attrs=attributes
    )

    # export pre-aligned image to MoBIE
    if mobie_export:
        logging.info("Export prealigned image to MoBIE")
        export_to_mobie(
            input_path=f"{output_dir}/{file_name}_prealigned.n5",
            input_key=input_key,
            output_dir=output_dir,
            segmentation_name=f"{file_name}_prealigned",
            menu_name=image_type
        )

    return img_prealigned


def run_prealignment(
    fixed_input_path,
    fixed_input_key,
    moving_input_path,
    moving_input_key,
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
    fixed_prealigned = prealignment_per_image(
        fixed_input_path,
        fixed_input_key,
        output_dir,
        mobie_export,
        image_type="fixed"
    )

    logging.info("Start prealignment of moving image ...")
    moving_prealigned = prealignment_per_image(
        moving_input_path,
        moving_input_key,
        output_dir,
        mobie_export,
        image_type="moving",
    )
    # check orientation
    for axis in range(3):
        rotate_axis_fixed = orient_axis(moving_prealigned, axis=axis, save_path=f"{output_dir}/plots/moving_intensity_profile_{axis}.png")
        rotate_axis_moving = orient_axis(fixed_prealigned, axis=axis, save_path=f"{output_dir}/plots/fixed_intensity_profile_{axis}.png")
        
        if not rotate_axis_fixed and not rotate_axis_moving:
            print(f"Correct orientation in axis {axis}")
        else:
            print(f"Rotate axis {axis} 180 degrees to align...")
    R_3x3 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    logging.info(f"Rotate axis {axis} 180 degrees to align...")
    img_center = 0.5 * (np.array(moving_prealigned.shape)-1)
    offset = img_center - R_3x3 @ img_center
    R = np.eye(4)
    R[:3, :3] = R_3x3
    R[:3, 3] = offset
    moving_prealigned = rotate_img(moving_prealigned, R, output_shape=moving_prealigned.shape)

    # update transformation matrix
    # T = T @ R
    # np.savetxt(f"{output_dir}/moving_T_prealignment.txt", T)
    
    logging.info("Prealignment done.")

    v = napari.Viewer()
    v.add_labels(fixed_prealigned)
    v.add_labels(moving_prealigned)
    napari.run()
    breakpoint()

    plot_overlay(
        fixed_prealigned,
        moving_prealigned,
        save_path=f"{output_dir}/plots/overlay_prealignment.png",
    )


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
