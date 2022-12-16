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
    data1 = np.array(col1, dtype=[("layer1", col1.dtype)])
    data2 = np.array(5 * col2, dtype=[("layer1", col2.dtype)])

    a, b, c = join_arrays(row1, col1, data1, row2, col2, data2, "test1", join_type="inner")
    assert np.allclose(np.sort(a), [0, 8, 16])
    assert np.allclose(np.sort(b), [0, 2, 4])
    a, b, c = join_arrays(row1, col1, data1, row2, col2, data2, "test1", join_type="left")

    assert np.allclose(np.sort(a), [0, 4, 8, 12, 16])
    assert np.allclose(np.sort(b), [0, 1, 2, 3, 4])
    assert np.allclose([x[0] for x in c], [0, 1, 2, 3, 4])
    assert np.allclose([x[1] for x in c], [0, 0, 10, 0, 20])


@pytest.mark.parametrize("join_type, expected_data, expected_row", [
    ["left", np.array([[0, 0], [1, 0], [2, 2], [4, 0], [5, 5]]), np.array([0, 1, 2, 4, 5])],
    ["right", np.array([[2, 2],[0, 3], [5, 5], [0, 6], [0, 7],]), np.array([2, 3, 5, 6, 7])],
    ["inner", np.array([[2, 2], [5, 5]]), np.array([2, 5])],
    ["outer", np.array([[0, 0], [1, 0], [2, 2], [0, 3], [4, 0], [5, 5], [0, 6], [0, 7]]),
     np.array([0, 1, 2, 3, 4, 5, 6, 7])],
])
def test_join_arrays_join_types(join_type, expected_data, expected_row):
    row1 = np.array([0, 1, 2, 4, 5])
    col1 = np.array([0, 1, 2, 4, 5])
    row2 = np.array([7, 5, 3, 6, 2])
    col2 = np.array([7, 5, 3, 6, 2])
    data1 = np.array(col1, dtype=[("layer1", col1.dtype)])
    data2 = np.array(col2, dtype=[("layer2", col2.dtype)])

    row, col, data = join_arrays(row1, col1, data1, row2, col2, data2, "test1",
                                 join_type=join_type)
    assert np.allclose(np.array([[x[0], x[1]] for x in data]), expected_data)
    assert np.allclose(row, expected_row)


@pytest.mark.parametrize("join_type", [
    "left", "right", "inner", "outer"
])
def test_join_arrays_larger(join_type):
    """Joining two identical arrays should always give the same result."""
    row = np.array([ 0,  1,  2,  3,  4,  5,  6,  6, 15,  7,  7,  8,  7,  9,  7, 10,  7,
                    11,  8,  8,  9,  8, 10,  8, 11,  9,  9, 10,  9, 11, 10, 10, 11, 11,
                    12, 12, 13, 13, 14, 15, 16, 17, 18, 19])

    col = np.array([ 0,  1,  2,  3,  4,  5,  6, 15,  6,  7,  8,  7,  9,  7, 10,  7, 11, 
                    7,  8,  9,  8, 10,  8, 11,  8,  9, 10,  9, 11,  9, 10, 11, 10, 11,
                    12, 13, 12, 13, 14, 15, 16, 17, 18, 19])
    data1 = np.array(np.arange(0, len(col)), dtype=[("layer1", col.dtype)])
    data2 = np.array(np.arange(0, len(col)), dtype=[("layer2", col.dtype)])

    row_out, col_out, data_out = join_arrays(row, col, data1, row, col, data2, "test1",
                                 join_type=join_type)
    assert np.allclose(sorted(data_out["test1_layer2"]), sorted(np.array([x[0] for x in data2])))
