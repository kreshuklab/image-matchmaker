import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import logging

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



def overlay_pcds(fixed_pcd: o3d.t.geometry.PointCloud, moving_pcd: o3d.t.geometry.PointCloud, fixed_col="cornflowerblue", moving_col="orangered", projection="xy", save_path=None, title=""):
    """"
    Overlay two point clouds. Parameters of plotting should be ok to visualize two full platy volumes
    """
    assert len(projection) == 2, f"Projection should be xy, yz or something like that of length 2, not {projection}"
    axis_order = {"x": 0, "y": 1, "z": 2}
    roi_x = np.s_[:, axis_order[projection[0]]]
    roi_y = np.s_[:, axis_order[projection[1]]]
    
    plt.figure(figsize=(10, 10))
    plt.cla()
    plt.axis("equal")
    fixed_np = fixed_pcd.point.positions.numpy()
    moving_np = moving_pcd.point.positions.numpy()

    plt.title(title)
    plt.scatter(fixed_np[roi_x], fixed_np[roi_y], s=0.6, c=fixed_col, alpha=0.5, label="Fixed point cloud")
    plt.scatter(moving_np[roi_x], moving_np[roi_y], s=0.6, c=moving_col, alpha=0.5, label="Moving point cloud")
    plt.legend()
    plt.xlabel(projection[0])
    plt.ylabel(projection[1])
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
    plt.close()

def visualize_displacement_field(moving_pcd: o3d.t.geometry.PointCloud, registered_pcd: o3d.t.geometry.PointCloud, save_path=None, projection="xy"):
    assert len(projection) == 2, f"Projection should be xy, yz or something like that of length 2, not {projection}"
    axis_order = {"x": 0, "y": 1, "z": 2}
    roi_x = np.s_[:, axis_order[projection[0]]]
    roi_y = np.s_[:, axis_order[projection[1]]]
    
    moving_np = moving_pcd.point.positions.numpy()
    registered_np = registered_pcd.point.positions.numpy()


    for idx in range(len(moving_np)):
        # if abs(pcd_in[idx, 2] - 112) < 4:
        #     ax.plot([pcd_in[idx, 0] /res[2], pcd_out[idx, 0]/res[2]], [pcd_in[idx, 1]/ res[1], pcd_out[idx, 1]/ res[1]])

        plt.plot([moving_np[roi_x][idx], registered_np[roi_x][idx]], [moving_np[roi_y][idx], registered_np[roi_y][idx]])
    
    plt.axis("equal")
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
    plt.close()