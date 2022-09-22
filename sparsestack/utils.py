import numba
import numpy as np
import numpy.lib.recfunctions as rfn


def join_arrays(row1, col1, data1,
                row2, col2, data2, name,
                join_type="left"):
    """Joins two (structured) sparse arrays.
    """
    if join_type == "left":
        data_join = rfn.append_fields(data1, name,
                                      np.zeros((len(row1)),
                                      dtype=data2.dtype),
                                      fill_value=0).data
        idx_inner_left, idx_inner_right = get_idx(row1, col1, row2, col2, join_type="inner")
        data_join[name][idx_inner_left] = data2[idx_inner_right]
        return row1, col1, data_join
    if join_type == "right":
        raise NotImplementedError
    if join_type == "inner":
        idx_inner_left, idx_inner_right = get_idx(row1, col1, row2, col2, join_type="inner")
        data_join = rfn.merge_arrays([data1[idx_inner_left],
                                      data2[idx_inner_right]])
        return row1[idx_inner_left], col1[idx_inner_left], data_join
    if join_type == "outer":
        raise NotImplementedError
    raise ValueError("Unknown join_type (must be 'left', 'right', 'inner', 'outer')")


@numba.jit(nopython=True)
def get_idx_inner_brute_force(left_row, left_col, right_row, right_col):
    #Get indexes for entries for a inner join.
    idx_inner_left = []
    idx_inner_right = []
    for i, right_row_id in enumerate(right_row):
        if right_row_id in left_row:
            idx = np.where((left_row == right_row_id)
                            & (left_col == right_col[i]))[0]
            if len(idx) > 0:
                idx_inner_left.append(idx[0])
                idx_inner_right.append(i)
    return idx_inner_left, idx_inner_right


@numba.jit(nopython=True)
def get_idx(left_row, left_col, right_row, right_col,
            join_type="left"):
    list1 = list(zip(left_row, left_col))
    list2 = list(zip(right_row, right_col))
    if join_type == "left":
        uniques = set(list1)
    elif join_type == "right":
        uniques = set(list2)
    elif join_type == "inner":
        uniques = set(list1).intersection(set(list2))
    elif join_type == "outer":
        uniques = set(list1).union(set(list2))
    idx_left = []
    idx_right = []
    for (r, c) in uniques:
        idx_left.append(np.where((left_row == r) & (left_col == c))[0][0])
        idx_right.append(np.where((right_row == r) & (right_col == c))[0][0])
    return idx_left, idx_right


if __name__ == '__main__':
    import time
    arr1r = np.arange(0, 20000, 4)
    arr1c = np.arange(0, 20000, 4)
    arr2r = np.arange(0, 20000, 8)
    arr2c = np.arange(0, 20000, 8)
    
    tstart = time.time()
    for _ in range(100):
        idx1, idx2 = get_idx_inner_brute_force(arr1r, arr1c, arr2r, arr2c)
    print(f"Time: {time.time()-tstart}s")
    
    idx1, idx2 = get_idx_inner_brute_force(arr1r, arr1c, arr2r, arr2c)
    #print(idx1)
    idx1, idx2 = get_idx(arr1r, arr1c, arr2r, arr2c)
    #print(np.sort(idx1))

    tstart = time.time()
    for _ in range(100):
        idx1b, idx2b = get_idx(arr1r, arr1c, arr2r, arr2c)
    print(f"Time: {time.time()-tstart}s")
    #print(idx1, idx2)
    a, b, c = join_arrays(arr1r, arr1c, arr1c, arr2r, arr2c, 1.1*arr2c, "test1", join_type="inner")
    #print(c)
    list1 = list(zip(arr1r, arr1c))
    list2 = list(zip(arr2r, arr2c))
    uniques = set(list1 + list2)
    #uniques = set(zip(arr1r, arr1c) + zip(arr2r, arr2c))
    print(len(uniques), len(list1), len(list2))
