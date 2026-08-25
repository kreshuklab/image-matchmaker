import itk
import logging
import numpy as np
from pathlib import Path
from contextlib import ExitStack
from importlib.resources import as_file, files


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
        if ch is None:
            raise ValueError("ch must be specified when input image is 4D (C, Z, Y, X).")
        img = img[ch]
    elif img.ndim != 3:
        raise ValueError(
            f"Expected a 3D (Z, Y, X) or 4D (C, Z, Y, X) array, got shape {img.shape}."
        )

    itk_img = itk.image_from_array(img)
    itk_img.SetSpacing(resolution[::-1])  # ZYX -> XYZ
    return itk_img


def create_parameter_object(parameter_map_paths):
    parameter_object = itk.ParameterObject.New()
    for parameter_map_path in parameter_map_paths:
        parameter_object.AddParameterFile(parameter_map_path)
    print(parameter_object)

    return parameter_object


def elastix_registration(
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
    elastix_object.SetLogToConsole(False)
    elastix_object.SetLogToFile(True)
    elastix_object.SetOutputDirectory(log_dir)
    elastix_object.SetLogFileName(log_name)
    logging.info("Set parameter map")

    logging.info("Start registration")
    elastix_object.UpdateLargestPossibleRegion()

    result_image = elastix_object.GetOutput()
    result_transform_parameters = elastix_object.GetTransformParameterObject()

    return result_image, result_transform_parameters


def elastix_pointset_registration(
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
    elastix_object.SetLogToConsole(False)
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
            parameter_map, write_dir / f"{prefix}_{index}.txt"
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


def apply_elastix_transform(transformix_filter, moving_img):
    transformix_filter.SetMovingImage(moving_img)
    transformix_filter.Update()
    output_image = transformix_filter.GetOutput()
    output_img_np = itk.GetArrayFromImage(output_image)  # already in zyx order
    return output_img_np


def apply_transform_chanwise(transform_parameter_object, moving_img_np, resolution):
    transformix_filter = create_transformix_object(transform_parameter_object)

    if moving_img_np.ndim == 4:
        result_img = []
        for chan in range(moving_img_np.shape[0]):
            moving_img = itk_scalar_img(moving_img_np, resolution, chan)
            output_img_np = apply_elastix_transform(transformix_filter, moving_img)
            result_img.append(output_img_np)
        result_img = np.stack(result_img, axis=0)
    elif moving_img_np.ndim == 3:
        moving_img = itk_scalar_img(moving_img_np, resolution)
        result_img = apply_elastix_transform(transformix_filter, moving_img)
    else:
        raise ValueError(
            f"Expected moving image with shape (Z,Y,X) or (C,Z,Y,X), got {moving_img_np.shape}."
        )
    return result_img


def get_parameter_map_paths(parameter_map_paths, default_parameter_maps):
    with ExitStack() as stack:
        if parameter_map_paths is None:
            parameter_map_paths = []
            for default_pm in default_parameter_maps:
                resource = files("image_matchmaker.configs.elastix").joinpath(default_pm)
                parameter_map_path = stack.enter_context(as_file(resource))
                parameter_map_paths.append(str(parameter_map_path))
        else:
            parameter_map_paths = [str(path) for path in parameter_map_paths]
    return parameter_map_paths
