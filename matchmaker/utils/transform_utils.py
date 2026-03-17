import json
import numpy as np
import transforms3d as tf3d
from scipy.ndimage import affine_transform
from elf.wrapper.resized_volume import ResizedVolume
import logging


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


def pad_to_same_shape(arr1, arr2):
    """Pad two arrays with zeros to the same shape at the end."""
    D1, H1, W1 = arr1.shape
    D2, H2, W2 = arr2.shape
    max_D, max_H, max_W = max(D1, D2), max(H1, H2), max(W1, W2)

    def pad_end(arr, target_shape, value=0):
        pad_D = target_shape[0] - arr.shape[0]
        pad_H = target_shape[1] - arr.shape[1]
        pad_W = target_shape[2] - arr.shape[2]
        return np.pad(arr, ((0, pad_D), (0, pad_H), (0, pad_W)), mode="constant", constant_values=value)

    return pad_end(arr1, (max_D, max_H, max_W)), pad_end(arr2, (max_D, max_H, max_W))


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

    return new_shape, min_coords, max_coords


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


def get_transformation_matrix(img, gc, Vt, img_ref=None, Vt_ref=None):
    # 1. center image on origin
    center_to_origin = get_translation_matrix(gc)
    # 2. rotate image
    rot = get_rotation_matrix(Vt.T)
    # 3. get new shape
    new_shape, min_coords, max_coords = get_rotated_shape(img, rot)
    if img_ref is not None:
        assert Vt_ref is not None
        rot_ref = get_rotation_matrix(Vt_ref.T)
        _, min_coords_ref, max_coords_ref = get_rotated_shape(img_ref, rot_ref)
        union_min = np.minimum(min_coords, min_coords_ref)
        union_max = np.maximum(max_coords, max_coords_ref)
        new_shape = np.ceil(union_max - union_min).astype(int)
    # 4. center image on new shape
    new_shape_center = np.array(new_shape) // 2
    center_to_new_shape = get_translation_matrix(-new_shape_center)
    # 5. combine all transforms: get transformation matrix
    T = center_to_origin @ rot @ center_to_new_shape

    return T, new_shape


def get_axis_orient_matrix(img, axis_order):
    # 1. center image on origin
    center_to_origin = get_translation_matrix(np.array(img.shape) // 2)
    # 2. rotate image
    R = tf3d.euler.euler2mat(np.pi, 0, 0, f"s{axis_order}")
    rot = get_rotation_matrix(R)
    center_to_new_shape = get_translation_matrix(-np.array(img.shape) // 2)
    # 5. combine all transforms: get transformation matrix
    T = center_to_origin @ rot @ center_to_new_shape
    return T


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


def to_dense_mask(mask, background=0, mapping=None):
    """
    Convert a sparse instance segmentation mask into a dense continuous mask.

    Returns:
        dense_mask : np.ndarray (same shape as input)
        mapping    : dict {old_id: new_id}
    """

    mask = mask.astype(np.int64)
    unique_ids = np.unique(mask)

    if mapping is None:
        new_mapping = {background: 0}
        next_id = 1
    else:
        new_mapping = dict(mapping)
        new_mapping[background] = 0
        next_id = max(new_mapping.values()) + 1

    known_ids = np.fromiter(new_mapping.keys(), dtype=np.int64)
    missing_ids = np.setdiff1d(unique_ids, known_ids, assume_unique=False)
    missing_ids = missing_ids[missing_ids != background]

    if missing_ids.size > 0:
        new_ids = np.arange(next_id, next_id + len(missing_ids), dtype=np.int64)
        add_mapping = dict(zip(missing_ids, new_ids))
        new_mapping.update(add_mapping)

    all_old_ids = np.fromiter(new_mapping.keys(), dtype=np.int64)
    all_new_ids = np.fromiter(new_mapping.values(), dtype=np.int64)

    lut_size = all_old_ids.max() + 1
    lut = np.zeros(lut_size, dtype=np.int64)
    lut[all_old_ids] = all_new_ids

    dense_mask = lut[mask]

    return dense_mask, new_mapping


def grid_sample3d(volume, grid, align_corners=False, mode="trilinear"):
    """
    Sample a 3D volume using a normalized sampling grid.

    Args:
        volume (np.ndarray): Input volume of shape (D, H, W).
        grid (np.ndarray): Normalized grid of shape (D, H, W, 3) in [-1, 1].
        align_corners (bool): Whether to align grid corners.
        mode (str): Interpolation mode, "nearest" or "trilinear".

    Returns:
        np.ndarray: Sampled volume of shape (D, H, W).
    """
    assert volume.ndim == 3
    assert mode in ["trilinear", "nearest"]

    D, H, W = volume.shape
    spatial_shape = np.array([W, H, D], dtype=np.float32)

    if align_corners:
        coords = (grid + 1.0) * 0.5 * (spatial_shape - 1.0)
    else:
        coords = ((grid + 1.0) * spatial_shape - 1.0) / 2.0

    if mode == "nearest":
        idx = np.round(coords).astype(np.int32)

        valid = (
            (idx[..., 0] >= 0) & (idx[..., 0] < W) &
            (idx[..., 1] >= 0) & (idx[..., 1] < H) &
            (idx[..., 2] >= 0) & (idx[..., 2] < D)
        )

        out = np.zeros(coords.shape[:-1], dtype=volume.dtype)
        out[valid] = volume[idx[..., 2][valid], idx[..., 1][valid], idx[..., 0][valid]]

    elif mode == "trilinear":
        c0 = np.floor(coords).astype(np.int32)   # (x0, y0, z0)
        c1 = c0 + 1                              # (x1, y1, z1)

        d = coords - c0
        xd, yd, zd = d[..., 0], d[..., 1], d[..., 2]

        wa = (1 - xd) * (1 - yd) * (1 - zd)
        wb = (1 - xd) * (1 - yd) * zd
        wc = (1 - xd) * yd * (1 - zd)
        wd = (1 - xd) * yd * zd
        we = xd * (1 - yd) * (1 - zd)
        wf = xd * (1 - yd) * zd
        wg = xd * yd * (1 - zd)
        wh = xd * yd * zd

        def safe_get(idx):
            ix, iy, iz = idx[..., 0], idx[..., 1], idx[..., 2]
            valid = ((ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & (iz >= 0) & (iz < D))
            out = np.zeros(ix.shape, dtype=volume.dtype)
            out[valid] = volume[iz[valid], iy[valid], ix[valid]]
            return out

        v000 = safe_get(c0)
        v001 = safe_get(c0 + [0, 0, 1])
        v010 = safe_get(c0 + [0, 1, 0])
        v011 = safe_get(c0 + [0, 1, 1])
        v100 = safe_get(c0 + [1, 0, 0])
        v101 = safe_get(c0 + [1, 0, 1])
        v110 = safe_get(c0 + [1, 1, 0])
        v111 = safe_get(c0 + [1, 1, 1])

        out = wa*v000 + wb*v001 + wc*v010 + wd*v011 + we*v100 + wf*v101 + wg*v110 + wh*v111

    return out


def convert_to_int(arr):
    """
    Convert a numpy array to its corresponding int type based on range of values.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    dtype = arr.dtype
    if dtype in [np.int8, np.int16, np.uint8, np.uint16]:
        return arr

    min_val, max_val = np.min(arr), np.max(arr)
    if min_val >= 0:
        if max_val <= 255:
            print(f"{dtype} numpy array is converted to np.uint8")
            return arr.astype(np.uint8)
        elif max_val <= 65535:
            print(f"{dtype} numpy array is converted to np.uint16")
            return arr.astype(np.uint16)
    else:
        if min_val >= -128 and max_val <= 127:
            print(f"{dtype} numpy array is converted to np.int8")
            return arr.astype(np.int8)
        elif min_val >= -32768 and max_val <= 32767:
            print(f"{dtype} numpy array is converted to np.int16")
            return arr.astype(np.int16)

    raise ValueError(f"Array values out of range for int8/int16/uint8/uint16: min={min_val}, max={max_val}")
