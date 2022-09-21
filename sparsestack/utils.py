import numba
import numpy as np


@numba.jit(nopython=True)
def get_idx_inner(left_row, left_col, right_row, right_col):
    """Get indexes for entries for a inner join."""
    idx_inner = []
    for i, coo_row_id in enumerate(right_row):
        idx = np.where((left_row == coo_row_id)
                        & (left_col == right_col[i]))[0]
        if len(idx) > 0:
            idx_inner.append(idx[0])
    return idx_inner


if __name__ == '__main__':
    arr1r = np.arange(0, 10000, 6)
    arr1c = np.arange(0, 10000, 6)
    arr2r = np.arange(0, 10000, 8)
    arr2c = np.arange(0, 10000, 8)
    idx = get_idx_inner(arr1r, arr1c, arr2r, arr2c)
    print(idx)
