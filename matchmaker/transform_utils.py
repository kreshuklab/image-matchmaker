import numpy as np
from scipy.ndimage import affine_transform
import transforms3d as tf3d
from elf.wrapper.resized_volume import ResizedVolume
import json


def write_transform_dict(transform_dict, json_path):
    for key, val in transform_dict.items():
        val["matrix"] = val["matrix"].tolist()
    with open(json_path, "w") as f:
	    json.dump(transform_dict, f, indent=2)


def read_transform_dict(json_path):
    with open(json_path, "r") as f:
        transform_dict = json.load(f)
         
    for key, val in transform_dict.items():
        val["matrix"] = np.array(val["matrix"])

    return transform_dict


def downscale_seg(seg, factor):
    new_shape = np.array(seg.shape) // factor
    downsampled_seg = ResizedVolume(seg, shape=new_shape)[:]
    print("Downsampled shape", downsampled_seg.shape)

    return downsampled_seg


def pad_img(img):
    shape = img.shape
    diagonal = int(np.ceil(np.linalg.norm(shape)))
    print("Diagonal", diagonal)

    # Calculate how much padding is needed on each axis
    pad_widths = []
    for dim in shape:
        total_pad = diagonal - dim
        before = total_pad // 2
        after = total_pad - before
        pad_widths.append((before, after))

    # Apply zero padding
    padded = np.pad(img, pad_width=pad_widths, mode='constant', constant_values=0)
    print("New shape after padding:", padded.shape)

    return padded


def crop_to_bbox(img):
    """Crops a 3D volume to the minimal bounding box around all nonzero voxels."""
    # Find where the volume is nonzero (i.e., contains instances)
    nonzero = np.argwhere(img)

    # Compute bounding box from min to max index along each axis
    z_min, y_min, x_min = nonzero.min(axis=0)
    z_max, y_max, x_max = nonzero.max(axis=0) + 1  # +1 to include the max index

    # Crop the volume
    cropped = img[z_min:z_max, y_min:y_max, x_min:x_max]
    return cropped


def get_rotated_shape(img, rotation_matrix):
    # Step 1: Define the 8 corners of the original volume
    dz, dy, dx = img.shape

    corners = np.array([
        [0, 0, 0],
        [0, 0, dx],
        [0, dy, 0],
        [0, dy, dx],
        [dz, 0, 0],
        [dz, 0, dx],
        [dz, dy, 0],
        [dz, dy, dx]
    ])

    # Step 2: Apply the rotation matrix
    rotated_corners = corners @ rotation_matrix[0:3, 0:3]  # Apply rotation

    # Step 3: Find min and max of the rotated corners
    min_coords = rotated_corners.min(axis=0)
    max_coords = rotated_corners.max(axis=0)

    # Step 4: Calculate the new shape
    new_shape = (np.ceil(max_coords - min_coords).astype(int))

    return new_shape


def get_translation_matrix(translation):
    M = np.identity(4)
    M[0:3, 3] = translation
    return M


def get_rotation(angles):
    R = tf3d.euler.euler2mat(*angles, axes='szyx')
    return R


def get_rotation_matrix(R):
    M = np.identity(4)
    M[0:3, 0:3] = R
    return M


def get_transformation_matrix(img, gc, Vt):
    # 1. center image on origin
    center_to_origin = get_translation_matrix(gc)
    # 2. rotate image
    rot = get_rotation_matrix(Vt.T)
    # 3. get new shape
    new_shape = get_rotated_shape(img, rot)
    # 4. center image on new shape
    new_shape_center = np.array(new_shape) // 2
    center_to_new_shape = get_translation_matrix(-new_shape_center)
    # 5. combine all transforms: get transformation matrix
    T = center_to_origin @ rot @ center_to_new_shape


    return T, new_shape


def rotate_img(img, rotation_matrix, output_shape=None, offset=None):
    """
    Rotate an image using a given rotation matrix.

    Parameters
    ----------
    img : array
        The 3D image to be rotated.
    rotation_matrix : array
        A 3x3 or 4x4 rotation matrix.
    output_shape : tuple, optional
        The desired output shape of the rotated image. If not given, the output shape
        will be determined from the rotation matrix.
    offset : tuple, optional
        The offset to apply to the rotated image to ensure it fits within the new
        bounding box. If not given, the offset will be determined from the rotation
        matrix.

    Returns
    -------
    rotated_img : array
        The rotated image with the desired output shape (if given).
    """
    rotated_img = affine_transform(
        img,
        matrix=rotation_matrix,
        output_shape=output_shape,  # new shape after rotation
        offset=offset,  # offset to ensure the image fits within the new bounding box
        order=0,  # interpolation (use 0 for discrete/label data)
        mode='constant',  # fill mode
        cval=0.0  # fill value (if constant mode)
    )

    print(f"Shape after rotation: {rotated_img.shape}")
    return rotated_img
