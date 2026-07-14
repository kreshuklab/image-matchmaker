import z5py
import numpy as np
from pathlib import PurePath


def print_key_tree(f: z5py.File):
    print(f"Key structure of z5 file {f.filename}")
    f.visititems(lambda name, obj: print(name))


def read_volume(
    f: z5py.File, key: str, roi: any = np.s_[:]
):
    """
    Read a dataset from an n5/zarr container.

    Parameters
    ----------
    f : z5py.File or str or pathlib.Path
        An open container, or a path to an ``.n5`` file.
    key : str
        Dataset key to read.
    roi : slice, optional
        Region of interest to read (defaults to the whole volume).

    Returns
    -------
    numpy.ndarray or None
        The requested array, or ``None`` if ``key`` does not exist.
    """
    if isinstance(f, (str, PurePath)):
        f = z5py.File(f, "r")

    try:
        ds = f[key]
    except KeyError:
        print(f"No key {key} in file {f.filename}")
        print_key_tree(f)
        return None

    ds.n_threads = 8
    vol = ds[roi]

    return vol


def get_attrs(f: z5py.File, key: str):
    """
    Return the attributes of a dataset in an n5/zarr container.

    Parameters
    ----------
    f : z5py.File or str or pathlib.Path
        An open container, or a path to an ``.n5`` file.
    key : str
        Dataset key whose attributes are returned.

    Returns
    -------
    mapping or None
        The dataset's attributes, or ``None`` if ``key`` does not exist.
    """
    if isinstance(f, (str, PurePath)):
        f = z5py.File(f, "a")

    try:
        ds = f[key]
    except KeyError:
        print(f"No key {key} in file {f.filename}")
        print_key_tree(f)
        return None

    return ds.attrs


def set_attrs(f, key: str, attrs_dict: dict):
    """
    Set (or update) attributes on a dataset in an n5/zarr container.

    Parameters
    ----------
    f : z5py.File or str or pathlib.Path
        An open container, or a path to an ``.n5`` file.
    key : str
        Dataset key whose attributes are set.
    attrs_dict : dict
        Attributes to write; existing keys are overwritten.
    """
    if isinstance(f, (str, PurePath)):
        f = z5py.File(f, "a")

    try:
        ds = f[key]
    except KeyError:
        raise KeyError(f"No key {key} in file {f.filename}")
        print_key_tree(f)
        return None

    # Set or update attributes
    for k, v in attrs_dict.items():
        ds.attrs[k] = v
        print(f"Attributes {k} for {key} written to {f.filename}")


def write_volume(f, arr: np.array, key, chunks=(1, 512, 512), attrs=None):
    """
    Write an array to a dataset in an n5/zarr container.

    Creates the dataset (gzip-compressed) if it does not exist, otherwise
    overwrites it.

    Parameters
    ----------
    f : z5py.File or str or pathlib.Path
        An open container, or a path to an ``.n5`` file.
    arr : numpy.ndarray
        Array to write.
    key : str
        Dataset key to write to.
    chunks : tuple of int, optional
        Chunk shape for the dataset (default ``(1, 512, 512)``).
    attrs : dict, optional
        Attributes to attach to the dataset.
    """
    shape = arr.shape
    compression = "gzip"
    dtype = arr.dtype

    if isinstance(f, (str, PurePath)):
        f = z5py.File(f, "a")

    if key not in f.keys():
        ds = f.create_dataset(
            key, shape=shape, compression=compression, chunks=chunks, dtype=dtype
        )
    else:
        ds = f[key]

    ds.n_threads = 8
    ds[:] = arr

    print(f"Dataset {key} written to {f.filename}")

    if attrs is not None:
        for key in attrs.keys():
            ds.attrs[key] = attrs[key]
