import numpy as np
import pytest
from scipy.sparse import coo_matrix
from sparsestack.utils import array_to_df


@pytest.fixture
def dense_array_sparse():
    arr = np.arange(0, 120).reshape(12, 10)
    arr[arr % 2 == 1] = 0
    arr[arr % 4 == 0] = 0
    return arr


@pytest.fixture
def structured_numpy_array():
    arr = np.array([[(1, 1.5), (2, 3.3)], [(3, 2.3), (4, 5.5)]],
                   dtype=[("x", int), ("y", float)])
    return arr


def test_array_to_df(dense_array_sparse):
    df = array_to_df(dense_array_sparse)
    assert df.shape == (30, 1)
    assert df.loc[11, 4][0] == 114


def test_array_to_df_structured(structured_numpy_array):
    df = array_to_df(structured_numpy_array)
    assert df.shape == (4, 2)
    assert df.loc[1, 1]["y"] == 5.5
