from pathlib import Path
import numpy as np
import logging
import itk


def initial_alignment(ventral_img_np, dorsal_img_np):
    """
    Place samples close enough that elastix registration works. In this case it's just a rotation around Y axis.
    """
    dorsal_img_np_rotated = dorsal_img_np[:, ::-1, :, ::-1]
    return ventral_img_np, dorsal_img_np_rotated


def itk_scalar_img(img: np.array, resolution, ch=0):
    """Convert numpy array to ITK Image object

    Args:
        img: _description_
        resolution: _description_
        ch: _description_. Defaults to 0.

    Returns:
        Scalar ITK Image object for one of the channels.
    """
    logging.info(f"Converting image of type {img.dtype} to ITK object")
    if img.ndim == 4:
        img = np.moveaxis(img[ch, :, :, :], 0, -1)

    else:
        img = np.moveaxis(img[:, :, :], 0, -1)

    itk_img = itk.image_from_array(img)
    # itk_img.SetSpacing(resolution[::-1])
    itk_img.SetSpacing(resolution)
    return itk_img


def itk_to_np_order(img: np.array):
    # In numpy: CZYX
    # In ITK: YXZC
    if img.ndim == 3 or img.ndim == 2:
        return np.moveaxis(img, -1, 0)
    elif img.ndim == 4:
        img = np.swapaxes(img, 0, 3)
        img = np.swapaxes(img, 1, 2)
        return np.flip(img)


def np_to_itk_order(img: np.array):
    # In numpy: CZYX
    # In ITK: XYZC
    if img.ndim == 3 or img.ndim == 2:
        return np.moveaxis(img, 0, -1)
    elif img.ndim == 4:
        img = np.swapaxes(img, 0, 3)
        img = np.swapaxes(img, 1, 2)
        return img


def create_parameter_object(parameter_map_paths):
    parameter_object = itk.ParameterObject.New()
    for parameter_map_path in parameter_map_paths:
        parameter_object.AddParameterFile(parameter_map_path)
    print(parameter_object)

    return parameter_object


def run_registration(
    fixed_img,
    moving_img,
    parameter_map_paths,
    log_dir,
    log_name="elastix.log",
    set_threads=False,
):
    logging.info("Start creating parameter object")
    parameter_object = create_parameter_object(parameter_map_paths)
    # Load Elastix Image Filter Object
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_img, moving_img)
    logging.info(elastix_object)
    logging.info("Created registration object")
    elastix_object.SetParameterObject(parameter_object)
    if set_threads:
        elastix_object.SetNumberOfThreads(32)
    elastix_object.SetLogToConsole(True)
    elastix_object.SetLogToFile(True)
    elastix_object.SetOutputDirectory(log_dir)
    elastix_object.SetLogFileName(log_name)
    logging.info("Set parameter map")

    logging.info("Start registration")
    elastix_object.UpdateLargestPossibleRegion()

    result_image = elastix_object.GetOutput()
    result_transform_parameters = elastix_object.GetTransformParameterObject()

    return result_image, result_transform_parameters


def run_pointset_registration(
    fixed_img,
    moving_img,
    parameter_map_paths,
    fixed_pointset,
    moving_pointset,
    log_dir,
    log_name="elastix.log",
    set_threads=False,
):
    logging.info("Start creating parameter object")
    parameter_object = create_parameter_object(parameter_map_paths)
    # Load Elastix Image Filter Object
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_img, moving_img)
    elastix_object.SetFixedPointSetFileName(fixed_pointset)
    elastix_object.SetMovingPointSetFileName(moving_pointset)
    elastix_object.SetParameterObject(parameter_object)
    if set_threads:
        elastix_object.SetNumberOfThreads(32)
    elastix_object.SetLogToConsole(True)
    elastix_object.SetLogToFile(True)
    elastix_object.SetOutputDirectory(log_dir)
    elastix_object.SetLogFileName(log_name)

    logging.info("Created registration object")
    logging.info(elastix_object)
    logging.info("Set parameter map")

    logging.info("Start registration")
    elastix_object.UpdateLargestPossibleRegion()

    result_image = elastix_object.GetOutput()
    result_transform_parameters = elastix_object.GetTransformParameterObject()

    return result_image, result_transform_parameters


def serialize_parameter_object(parameter_object, prefix, write_dir):
    write_dir = Path(write_dir)
    for index in range(parameter_object.GetNumberOfParameterMaps()):
        parameter_map = parameter_object.GetParameterMap(index)
        parameter_object.WriteParameterFile(
            parameter_map, write_dir / f"{prefix}.{index}.txt"
        )


def deserialize_parameter_object(prefix, cur_dir=Path("./")):
    parameter_files = sorted(list(cur_dir.glob(f"{prefix}*.txt")))
    parameter_files = [str(fname) for fname in parameter_files]
    parameter_object = itk.ParameterObject.New()
    parameter_object.ReadParameterFile(parameter_files)
    return parameter_object


def create_transformix_object(transform_parameter_object):
    ImageType = itk.Image[itk.F, 3]
    transformix_filter = itk.TransformixFilter[ImageType].New()
    transformix_filter.SetTransformParameterObject(transform_parameter_object)
    return transformix_filter


def apply_transform(transformix_filter, moving_img):
    transformix_filter.SetMovingImage(moving_img)
    transformix_filter.Update()
    output_image = transformix_filter.GetOutput()
    output_img_np = itk.GetArrayFromImage(output_image)
    output_img_np = itk_to_np_order(output_img_np)
    return output_img_np


def apply_transform_chanwise(transform_parameter_object, moving_img_np, resolution):
    transformix_filter = create_transformix_object(transform_parameter_object)
    result_img = []

    if moving_img_np.ndim == 4:
        for chan in range(moving_img_np.shape[0]):
            moving_img = itk_scalar_img(moving_img_np, resolution, chan)
            output_img_np = apply_transform(transformix_filter, moving_img)
            result_img.append(output_img_np)
        result_img = np.array(result_img)
    else:
        moving_img = itk_scalar_img(moving_img_np, resolution, 0)
        result_img = apply_transform(transformix_filter, moving_img)

    return result_img
