import numba
import numpy as np
import numpy.lib.recfunctions as rfn


def join_arrays(row1, col1, data1,
                row2, col2, data2, name,
                join_type="left"):
    """Joins two (structured) sparse arrays.
    """
    #pylint: disable=too-many-arguments
    #pylint: disable=too-many-locals
    if row1.dtype != row2.dtype:
        row2 = row2.astype(row1.dtype)
    if col1.dtype != col2.dtype:
        col2 = col2.astype(col1.dtype)
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
        data_join = rfn.append_fields(data1[idx_inner_left],
                                      name,
                                      data2[idx_inner_right]).data
        # TODO check if name is handled correctly
        return row1[idx_inner_left], col1[idx_inner_left], data_join
    if join_type == "outer":
        idx_left, idx_left_new, idx_right, idx_right_new, row_new, col_new = get_idx_outer(row1, col1, row2, col2)
        data_join = np.zeros(len(row_new), dtype=data1.dtype)
        data_join[idx_left_new] = data1[idx_left]
        data_join = rfn.append_fields(data_join, name,
                                      np.zeros((len(row_new)), dtype=data2.dtype),
                                      fill_value=0).data
        data_join[name][idx_right_new] = data2[idx_right]
        return np.array(row_new, dtype=row1.dtype), np.array(col_new, dtype=col1.dtype), data_join
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
    #elif join_type == "outer":
    #    uniques = set(list1).union(set(list2))
    else:
        raise ValueError("Unknown join_type")
    uniques = sorted(list(uniques))
    idx_left = []
    idx_right = []
    for (r, c) in uniques:
        i_left = np.where((left_row == r) & (left_col == c))[0]
        if len(i_left) > 0:
            idx_left.append(i_left[0])
        i_right = np.where((right_row == r) & (right_col == c))[0]
        if len(i_right) > 0:
            idx_right.append(i_right[0])
    return idx_left, idx_right


@numba.jit(nopython=True)
def get_idx_outer(left_row, left_col, right_row, right_col):
    #pylint: disable=too-many-locals
    uniques = set(zip(left_row, left_col)).union(set(zip(right_row, right_col)))
    uniques = sorted(list(uniques))

    idx_left = []
    idx_left_new = []
    idx_right = []
    idx_right_new = []
    row_new = []
    col_new = []
    for i, (r, c) in enumerate(uniques):
        row_new.append(r)
        col_new.append(c)
        i_left = np.where((left_row == r) & (left_col == c))[0]
        if len(i_left) > 0:
            idx_left.append(i_left[0])
            idx_left_new.append(i)
        i_right = np.where((right_row == r) & (right_col == c))[0]
        if len(i_right) > 0:
            idx_right.append(i_right[0])
            idx_right_new.append(i)
    return idx_left, idx_left_new, idx_right, idx_right_new, row_new, col_new
