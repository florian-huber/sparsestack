import numpy as np
import pytest
from sparsestack.utils import join_arrays


@pytest.mark.parametrize("row2, col2", [
    [np.arange(0, 20, 8).astype(np.int32), np.arange(0, 5, 2).astype(np.int32)],
    [np.arange(0, 20, 8).astype(np.int64), np.arange(0, 5, 2).astype(np.int64)],
])
def test_join_arrays(row2, col2):
    row1 = np.arange(0, 20, 4)
    col1 = np.arange(0, 5, 1)
    data1 = col1
    data2 = 5 * col2

    a, b, c = join_arrays(row1, col1, data1, row2, col2, data2, "test1", join_type="inner")
    assert np.allclose(np.sort(a), [0, 8, 16])
    assert np.allclose(np.sort(b), [0, 2, 4])
    a, b, c = join_arrays(row1, col1, data1, row2, col2, data2, "test1", join_type="left")

    assert np.allclose(np.sort(a), [0, 4, 8, 12, 16])
    assert np.allclose(np.sort(b), [0, 1, 2, 3, 4])
    assert np.allclose([x[0] for x in c], [0, 1, 2, 3, 4])
    assert np.allclose([x[1] for x in c], [0, 0, 10, 0, 20])

@pytest.mark.parametrize("join_type, expected_data", [
    ["left", np.array([[0, 0], [1, 0], [2, 2], [4, 0], [5, 5]])],
    ["inner", np.array([[2, 2], [5, 5]])],
    ["outer", np.array([[0, 0], [1, 0], [2, 2], [0, 3], [4, 0], [5, 5], [0, 6], [0, 7]])],
])
def test_join_arrays_join_types(join_type, expected_data):
    row1 = np.array([0, 1, 2, 4, 5])
    col1 = np.array([0, 1, 2, 4, 5])
    row2 = np.array([7, 5, 3, 6, 2])
    col2 = np.array([7, 5, 3, 6, 2])
    #row2 = np.array([2, 3, 5, 6, 7])
    #col2 = np.array([2, 3, 5, 6, 7])
    data1 = col1
    data2 = col2

    row, col, data = join_arrays(row1, col1, data1, row2, col2, data2, "test1",
                                 join_type=join_type)
    #assert np.allclose(np.sort(np.array([[x[0], x[1]] for x in data]), axis=0), np.sort(expected_data, axis=0))
    assert np.allclose(np.array([[x[0], x[1]] for x in data]), expected_data)
