import numpy as np
import transforms3d as tf3d
from scipy.ndimage import affine_transform
from elf.wrapper.resized_volume import ResizedVolume


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


# def get_rotation_matrix(angles, save_path=None):
#     rotation_matrix = tf3d.euler.euler2mat(*angles, axes='szyx')
#     print("Rotation matrix:\n", rotation_matrix)
#     if save_path is not None:
#         np.savetxt(save_path, rotation_matrix)
#     return rotation_matrix


def rotate_img(img, rotation_matrix, output_shape=None, offset=None):
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


# NOTE rotate with padding
def rotate_with_padding(img, rot_matrix):
    padded = pad_img(img)
    rotated = rotate_img(padded, rot_matrix)
    cropped = crop_to_bbox(rotated)
    return cropped


# NOTE rotate with new shape
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


# def get_affine_transform(translation, rotation, rotation_order="szyx"):
#     M = get_translation_matrix(translation)
#     M[0:3, 0:3] = tf3d.euler.euler2mat(*rotation, rotation_order)
#     return M


def get_translation_matrix(translation):
    M = np.identity(4)
    M[0:3, 3] = translation
    return M


def get_rotation_matrix(R):
    M = np.identity(4)
    M[0:3, 0:3] = R
    return M


### now again: what do I need?

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
    R = center_to_origin @ rot @ center_to_new_shape

    return R, new_shape
