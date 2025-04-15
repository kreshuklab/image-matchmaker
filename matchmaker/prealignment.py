from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from platy_reg.vis import plot_overlay, plot_three_slices
import logging
from platy_reg.preprocessing import convert_to_point_cloud
from scipy.ndimage import rotate
from skimage.filters import gaussian


def get_SVD_transform(img, plot_path=None, percentile_trsh=90):
    """Convert image to point cloud by thresholding, then run SVD on resulting point cloud.

    Args:
        img: _description_
        plot_path: _description_. Defaults to None.
        percentile_trsh: _description_. Defaults to 90.

    Returns:
        Variance matrix and principal axes matrix.
    """
    trsh = np.percentile(img, percentile_trsh)
    logging.info(f"Threshold used for converting to point cloud: {trsh}")
    X = convert_to_point_cloud(img, trsh)
    gc = X.mean(axis=0)
    gc = np.array(img.shape) // 2
    X_c = X - gc
    logging.info(f"Point cloud shape {X.shape}")
    logging.info(f"Point cloud center {gc}")
    
    if X.shape[1] == 3:
        random_subset = np.random.choice(X.shape[0], 10000, replace=False)
        X_subset = X_c[random_subset, 1:]
    else:
        X_subset = X_c[::100]
    logging.info("Subset point cloud")
    logging.info(f"Subset point cloud shape {X_subset.shape}")
    
    logging.info("Run SVD")
    U, S, Vt = np.linalg.svd(X_subset)
    # logging.info("U")
    # logging.info(U)
    logging.info("S")
    logging.info(str(S))
    logging.info("Vt")
    logging.info(str(Vt))
    
    logging.info("Rotate point cloud")
    vr = X_subset @ Vt.T
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.title('original vertices')
    plt.scatter(X_subset[:, 0], X_subset[:, 1], alpha=0.2)
    plt.subplot(1,2,2)
    plt.title('rotated vertices')
    plt.scatter(vr[:, 0], vr[:, 1], alpha=0.1)
    if plot_path:
            plt.savefig(plot_path, dpi=300)
    
    return gc, Vt
    
    
def orient_head(img, plot_path=None):
    """Euristic to orient all samples "head up": calculate sum intensity profile along the Y axis, if max is closer to 0 then do nothing, else rotate 180 degrees.

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
    plt.savefig(plot_path, dpi=300)
    
    max_pos = int_profile.argmax()
    logging.info(f"Max position is {max_pos}, dimension shape is {img.shape[1]}")
    if max_pos < img.shape[1] // 2:
        logging.info("Correct head orientation")
        return False
    else:
        logging.info("Rotate 180 degree to align head position")
        return True
    

def orient_sample(img, dapi_chan, plot_path, dorsal=False):
    """Preliminary orientation of the samples with body axis along Y, head closer to 0.

    Args:
        img: input volume
        dapi_chan: channel to use for registration
        dorsal: if True, rotate around Y to align with ventral in Z direction

    Returns:
        oriented image
    """
    
    # Rotate around Y if the volume is dorsal
    plot_path = Path(plot_path)
    
    if dorsal:
        logging.info(f"Rotated dorsal sample to align Z")
        img = img[:, ::-1, :, ::-1]
    
    
    # Rotate in XY plane, because samples can be oriented randomly, not only along X or Y
    max_proj = np.max(img, axis=1)[dapi_chan, ...]
    plt.figure()
    plt.imshow(max_proj, cmap="Reds")
    plt.savefig(plot_path / "max_proj_input.png")
    
    logging.info(f"Smooth image with sigma={2}")
    img_smoothed = gaussian(img[dapi_chan, ::10, ::10, ::10], sigma=3)
    gc, Vt = get_SVD_transform(img_smoothed, plot_path / "max_proj_point_cloud_random_angle.png", percentile_trsh=90)
    rot_angle = 90 - np.degrees(np.arctan2(Vt[0, 1], Vt[0, 0]))
    logging.info(f"Rotation angle to correct for random angle is {rot_angle}")
    img = rotate(img, rot_angle, axes=[-2, -1], order=3, mode="constant")
    logging.info(f"Rotated the input volume around Z by {rot_angle} degrees")
    
    plt.figure()
    plt.imshow(np.max(img, axis=1)[dapi_chan, ...], alpha=0.5, cmap="Blues")
    plt.savefig(plot_path / "max_proj_rotated_random_angle.png", dpi=300)
    
    
    # Check again if the sample is along X or along Y
    # Make max projection and determine the direction of principal axes
    max_proj = np.max(img, axis=1)[dapi_chan, ...]
    img_smoothed = gaussian(img[dapi_chan, ::10, ::10, ::10], sigma=2)
    gc, Vt = get_SVD_transform(img_smoothed, plot_path / "max_proj_point_cloud.png")
    
    rot_angle = np.arccos(Vt[0, 0])
    logging.info(f"Rotation angle is {rot_angle}")
    if np.abs(rot_angle - np.pi / 2) < np.pi / 4:
        logging.info("Rotate 90 degrees")
        plt.figure()
        plt.imshow(max_proj, cmap="Reds")
        img_rot = np.rot90(img, axes=(2, 3))
        
    elif np.abs(rot_angle) < np.pi / 4: 
        logging.info("Rotation is already correct")
        img_rot = img
    else:
        logging.info(f"90 degree rotation can't be determined with first principal axis angle of {rot_angle * 180/ np.pi}")
        img_rot = img
        
    # Check if head is oriented correctly, else rotate 180 degrees
    
    if orient_head(img_rot[dapi_chan, ...], plot_path / "sum_intensity_profile.png"):
        img_reg = np.rot90(img_rot, k=2, axes=(2, 3))
    else:
        img_reg = img_rot
    
    logging.info(f"Final image shape is {img_reg.shape}")
    plt.figure()
    plt.imshow(np.max(img_reg, axis=1)[dapi_chan, ...], alpha=0.5, cmap="Blues")
    plt.savefig(plot_path / "max_proj_rotated_final.png", dpi=300)
    
    return img_reg