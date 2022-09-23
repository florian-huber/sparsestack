import numpy as np
from sparsestack.utils import join_arrays


def test_join_arrays():
    arr1r = np.arange(0, 20, 4)
    arr1c = np.arange(0, 5, 1)
    arr2r = np.arange(0, 20, 8)
    arr2c = np.arange(0, 5, 2)

    a, b, c = join_arrays(arr1r, arr1c, arr1c, arr2r, arr2c, 5*arr2c, "test1", join_type="inner")
    assert np.allclose(np.sort(a), [0, 8, 16])
    assert np.allclose(np.sort(b), [0, 2, 4])
    a, b, c = join_arrays(arr1r, arr1c, arr1c, arr2r, arr2c, 5*arr2c, "test1", join_type="left")

    assert np.allclose(np.sort(a), [0, 4, 8, 12, 16])
    assert np.allclose(np.sort(b), [0, 1, 2, 3, 4])
    assert np.allclose([x[0] for x in c], [0, 1, 2, 3, 4])
    assert np.allclose([x[1] for x in c], [0, 0, 10, 0, 20])


def test_join_arrays_int_types():
    arr1r = np.arange(0, 20, 4)
    arr1c = np.arange(0, 5, 1)
    arr2r = np.arange(0, 20, 8).astype(np.int64)
    arr2c = np.arange(0, 5, 2).astype(np.int64)

    a, b, c = join_arrays(arr1r, arr1c, arr1c, arr2r, arr2c, 5*arr2c, "test1", join_type="inner")
    assert np.allclose(np.sort(a), [0, 8, 16])
    assert np.allclose(np.sort(b), [0, 2, 4])
    a, b, c = join_arrays(arr1r, arr1c, arr1c, arr2r, arr2c, 5*arr2c, "test1", join_type="left")

    assert np.allclose(np.sort(a), [0, 4, 8, 12, 16])
    assert np.allclose(np.sort(b), [0, 1, 2, 3, 4])
    assert np.allclose([x[0] for x in c], [0, 1, 2, 3, 4])
    assert np.allclose([x[1] for x in c], [0, 0, 10, 0, 20])