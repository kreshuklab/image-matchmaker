import numpy as np


def assert_arrays_equal(arr1, arr2):
    assert arr1.dtype == arr2.dtype
    assert np.array_equal(arr1, arr2)
