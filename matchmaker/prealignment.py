import numpy as np
import matplotlib.pyplot as plt
import logging
import z5py
import os
from matchmaker.data import create_point_cloud
from matchmaker.transform_utils import get_rotated_shape, get_transformation_matrix, rotate_img
from matchmaker.vis import plot_three_slices
import click


def get_SVD_transform(img, save_path=None, percentile_trsh=90):
    """Convert image to point cloud by thresholding, then run SVD on resulting point cloud.

    Args:
        img: _description_
        plot_path: _description_. Defaults to None.
        percentile_trsh: _description_. Defaults to 90.

    Returns:
        Variance matrix and principal axes matrix.
    """

    pos, _ = create_point_cloud(img)
    gc = pos.mean(axis=0)
    gc = np.array(img.shape) // 2
    pos_c = pos - gc
    logging.info(f"Point cloud shape {pos.shape}")
    logging.info(f"Point cloud center {gc}")

    logging.info("Run SVD")
    U, S, Vt = np.linalg.svd(pos_c, full_matrices=False)
    logging.info("U")
    logging.info(U)
    logging.info("S")
    logging.info(str(S))
    logging.info("Vt")
    logging.info(str(Vt))

    logging.info("Rotate point cloud")
    vr = pos_c @ Vt.T

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


def orient_head(img, save_path=None):
    """Euristic to orient all samples "head up": calculate sum intensity profile along the Y axis,
    if max is closer to 0 then do nothing, else rotate 180 degrees.

    Args:
        img: DAPI volume

    Returns:
        True if rotation is needed.
    """
    assert img.ndim == 3, f"Input image should have 3 dimensions, has {img.ndim}"
    int_profile = np.sum(img, axis=(0, 2))

    plt.figure()
    plt.plot(int_profile)
    plt.xlabel("Coordinate")
    plt.ylabel("Sum intensity along Y axis")
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    max_pos = int_profile.argmax()
    logging.info(f"Max position is {max_pos}, dimension shape is {img.shape[1]}")
    if max_pos < img.shape[1] // 2:
        logging.info("Correct head orientation")
        return False
    else:
        logging.info("Rotate 180 degree to align head position")
        return True


def prealign_sample(img, file_name, save_path):

    # align segmentation with PCs
    gc, Vt = get_SVD_transform(img)
    T, new_shape = get_transformation_matrix(img, gc, Vt, save_path=f"{save_path}/{file_name}_T_prealignment.txt")
    img_rotated = rotate_img(img, T, output_shape=new_shape)

    # Check if head is oriented correctly, else rotate 180 degrees
    rotate_head = orient_head(
        img_rotated, f"{save_path}/plots/{file_name}_intensity_profile.png"
    )
    if rotate_head:
        print("Rotate head 180 degrees ...")
        # img_rotated = np.rot90(img_rotated, k=2)  # NOTE: use rotation matrix instead
        R_3x3 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        img_center = 0.5 * (np.array(img_rotated.shape)-1)
        offset = img_center - R_3x3 @ img_center
        R = np.eye(4)
        R[:3, :3] = R_3x3
        R[:3, 3] = offset
        img_rotated = rotate_img(img_rotated, R, output_shape=img_rotated.shape)

        # update transformation matrix
        T = T @ R
        np.savetxt(f"{save_path}/{file_name}_T_prealignment.txt", T)

    plot_three_slices(
        img_rotated,
        save_path=f"{save_path}/plots/{file_name}_prealigned.png"
    )

    return img_rotated


@click.command()
@click.option("-i", "--input-path", required=True, help="Input file")
@click.option("-k", "--input-key", required=True, help="Input key")
@click.option("-o", "--output-dir", required=True, help="Output directory")
def main(input_path, input_key, output_dir):
    with z5py.File(input_path, "r") as f:
        img = f[input_key][:]
    file_name = os.path.splitext(os.path.basename(input_path))[0]

    if not os.path.exists(f"{output_dir}/plots"):
        os.makedirs(f"{output_dir}/plots")

    img_prealigned = prealign_sample(img, file_name, save_path=output_dir)

    with z5py.File(f"{output_dir}/{file_name}_prealigned.n5", "w") as f:
        f.create_dataset(input_key, data=img_prealigned, compression="gzip")

    # create MoBIE project
    # create_mobie_project()


if __name__ == "__main__":
    main()
