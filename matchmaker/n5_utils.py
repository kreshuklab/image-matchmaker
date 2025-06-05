import z5py
from pathlib import PurePath
import numpy as np


def print_key_tree(f: z5py.File):
    print(f"Key structure of z5 file {f.filename}")
    f.visititems(lambda name, obj: print(name))


def read_volume(
    f: z5py.File, key: str, roi: np.lib.index_tricks.IndexExpression = np.s_[:]
):
    print(type(f))
    if isinstance(f, (str, PurePath)):
        f = z5py.File(f, "r")

    try:
        ds = f[key]
    except KeyError:
        print(f"No key {key} in file {f.filename}")
        print_key_tree(f)
        return None

    ds.n_threads = 8
    print(f"Reading roi {roi} of volume {key} from {f.filename}")
    vol = ds[roi]
    print(f"Read volume with shape {vol.shape}, data type {vol.dtype}")

    return vol


def get_attrs(f: z5py.File, key: str):
    print(type(f))
    if isinstance(f, (str, PurePath)):
        f = z5py.File(f, "a")

    try:
        ds = f[key]
    except KeyError:
        print(f"No key {key} in file {f.filename}")
        print_key_tree(f)
        return None

    return ds.attrs


def write_volume(f, arr: np.array, key, chunks=(1, 512, 512), attrs=None):
    shape = arr.shape
    compression = "gzip"
    dtype = arr.dtype

    print(type(f))
    if isinstance(f, (str, PurePath)):
        f = z5py.File(f, "a")

    if key not in f.keys():
        print(f"Created dataset {key}")
        ds = f.create_dataset(
            key, shape=shape, compression=compression, chunks=chunks, dtype=dtype
        )
    else:
        print(f"Overwriting {key}")
        ds = f[key]

    ds.n_threads = 8
    print(f"Writing array to {key}")
    ds[:] = arr

    if attrs is not None:
        print("Assigning attributes of the dataset")
        for key in attrs.keys():
            ds.attrs[key] = attrs[key]
