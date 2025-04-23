
from transforms3d.euler import euler2mat, mat2euler
from scipy.ndimage import affine_transform
import numpy as np

def get_translation_transform(translation):
    M = np.identity(4)
    M[0:3, 3] = translation
    return M


def get_affine_transform(translation, rotation, rotation_order="szyx"):
    M = get_translation_transform(translation)
    M[0:3, 0:3] = euler2mat(*rotation, rotation_order)
    return M


def flip_dorsal(img, n5_fname):
    """
    For dorsal-direction samples flip along z so they are same orientation
    """
    if "dorsal" in n5_fname.lower():
        print("Flip dorsal image")
        img_flipped = np.flip(img, axis=0)
        return img_flipped
    
    else:
        return img

def move_along_z(margin, image_shape):
    """
    After rotation the volume might not fit within the same Z range,
    so move it up by some margin and make output image size bigger
    """
    assert len(image_shape) == 3
    R_t = get_translation_transform([-margin, -margin, -margin])
    new_image_shape = list(image_shape)
    new_image_shape = [sh + 2 * margin for sh in image_shape]
    
    return R_t, new_image_shape


R_t = get_translation_transform(-gc)
R_t_r = get_translation_transform(gc)

R_rot = get_translation_transform([0, 0, 0])
R_rot[0:3, 0:3] = np.linalg.inv(Vt)

R_rot_90_z = get_affine_transform([0, 0, 0], [0, -np.pi / 2, 0], rotation_order="szyx")

R_t_margin, new_image_shape = move_along_z(80, norm_img.shape)

mat =   R_t_r @  R_rot @ R_t @ R_t_margin
print(mat)
print(norm_img.shape)

img_reg = affine_transform(img_isotropic, mat, order=0, output_shape=new_image_shape)