import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from matchmaker.preprocessing import percentile_norm


def plot_three_slices(
    img,
    save_path=None,
    x_pos=None,
    y_pos=None,
    z_pos=None,
    cmap="Greys_r",
    max_pos=False,
    alpha=False,
):
    """
    Plot slices of a 3D image along each axis.

    Args:
        img: _description_
        save_path: _description_
        x_pos: _description_. Defaults to None.
        y_pos: _description_. Defaults to None.
        z_pos: _description_. Defaults to None.
        cmap: _description_. Defaults to "Greys".
        max_pos: _description_. Defaults to False.
    """
    assert img.ndim == 3
    if x_pos is None:
        x_pos = int(img.shape[2] // 2)
    if y_pos is None:
        y_pos = int(img.shape[1] // 2)
    if z_pos is None:
        z_pos = int(img.shape[0] // 2)

    if max_pos:
        z_pos, y_pos, x_pos = np.unravel_index(np.argmax(img), img.shape)

    if alpha:
        alpha = (img > 0).astype(np.float32)
    else:
        alpha = np.ones_like(img)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title(f"z slice at {z_pos}")
    plt.imshow(img[z_pos, :, :], cmap=cmap, alpha=alpha[z_pos, :, :])
    plt.subplot(1, 3, 2)
    plt.title(f"y slice at {y_pos}")
    plt.imshow(img[:, y_pos, :], cmap=cmap, alpha=alpha[:, y_pos, :])
    plt.subplot(1, 3, 3)
    plt.title(f"x slice at {x_pos}")
    plt.imshow(img[:, :, x_pos], cmap=cmap, alpha=alpha[:, :, x_pos])
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_overlay(img1, img2, save_path=None, x_pos=None, y_pos=None, z_pos=None):
    """
    Plot slices of two 3D images along each axis.

    Args:
        img1: _description_target_shape = (25, 22, 29)
        img2: _description_
        save_path: _description_. Defaults to None.
        x_pos: _description_. Defaults to None.
        y_pos: _description_. Defaults to None.
        z_pos: _description_. Defaults to None.
    """
    assert img1.ndim == 3
    assert img2.ndim == 3

    if x_pos is None:
        x_pos = min(int(img1.shape[2] // 2), int(img2.shape[2] // 2))
    if y_pos is None:
        y_pos = min(int(img1.shape[1] // 2), int(img2.shape[1] // 2))
    if z_pos is None:
        z_pos = min(int(img1.shape[0] // 2), int(img2.shape[0] // 2))

    plt.figure(figsize=(30, 10), dpi=300)
    plt.subplot(1, 3, 1)
    plt.title(f"z slice at {z_pos}")
    img1_alpha = (percentile_norm(img1, 0, 100) > 0) * 0.5
    img2_alpha = (percentile_norm(img2, 0, 100) > 0) * 0.5
    plt.imshow(img1[z_pos, :, :], cmap="Reds", alpha=img1_alpha[z_pos, :, :])
    plt.imshow(img2[z_pos, :, :], cmap="Blues", alpha=img2_alpha[z_pos, :, :])

    plt.subplot(1, 3, 2)
    plt.title(f"y slice at {y_pos}")
    plt.imshow(img1[:, y_pos, :], cmap="Reds", alpha=img1_alpha[:, y_pos, :])
    plt.imshow(img2[:, y_pos, :], cmap="Blues", alpha=img2_alpha[:, y_pos, :])

    plt.subplot(1, 3, 3)
    plt.title(f"x slice at {x_pos}")
    plt.imshow(img1[:, :, x_pos], cmap="Reds", alpha=img1_alpha[:, :, x_pos])
    plt.imshow(img2[:, :, x_pos], cmap="Blues", alpha=img2_alpha[:, :, x_pos])

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pcd(pcd: o3d.geometry.PointCloud, save_path=None):
    coords = pcd.point.positions.numpy()
    color = pcd.point.colors.numpy()
    logging.info("Point coordinates of shape {coords.shape} and dtype {coords.dtype}")
    # plt.scatter(coords[:, 0], coords[:, 1], s=0.2, c=color)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    
    red_cmap = sns.light_palette("crimson", as_cmap=True)
    yellow_cmap = sns.light_palette("gold", as_cmap=True)
    gene1_counts_red = [red_cmap(val) for val in percentile_norm(pcd.point.gene1_spots.numpy(), pmin=0, pmax=99)]
    gene1_intensity_red = [red_cmap(val) for val in percentile_norm(pcd.point.gene1_mean.numpy(), pmin=0, pmax=98)]
    gene2_counts_yellow = [yellow_cmap(val) for val in percentile_norm(pcd.point.gene2_spots.numpy(), pmin=0, pmax=99)]
    
    # print(percentile_norm(pcd.point.gene2_spots.numpy(), pmin=0, pmax=99))
    gene2_intensity_yellow = [yellow_cmap(val) for val in percentile_norm(pcd.point.gene2_mean.numpy(), pmin=0, pmax=98)]
        
    axs[0, 0].scatter(coords[:, 0], coords[:, 1], s=1, c=gene1_intensity_red, label="Gene 1 intensity", alpha=0.5)
    axs[0, 1].scatter(coords[:, 0], coords[:, 1], s=1, c=gene1_counts_red, label="Gene 1 counts", alpha=0.5)
    axs[1, 0].scatter(coords[:, 0], coords[:, 1], s=1, c=gene2_intensity_yellow, label="Gene 2 intensity", alpha=0.5)
    axs[1, 1].scatter(coords[:, 0], coords[:, 1], s=1, c=gene2_counts_yellow, label="Gene 2 counts", alpha=0.5)
    
    for ax in axs.flat:
        ax.legend()
        ax.set_facecolor("black")
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
    plt.close()