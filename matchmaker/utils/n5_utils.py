import z5py
import numpy as np
from pathlib import PurePath


def print_key_tree(f: z5py.File):
    print(f"Key structure of z5 file {f.filename}")
    f.visititems(lambda name, obj: print(name))


def read_volume(
    f: z5py.File, key: str, roi: any = np.s_[:]
):
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
