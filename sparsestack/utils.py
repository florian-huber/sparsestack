import numba
import numpy as np


def join_arrays(row1, col1, data1,
                row2, col2, data2, name,
                join_type="left"):
    """Joins two (structured) sparse arrays.
    """
    #pylint: disable=too-many-arguments

    # Harmonize row and col dtype
    if row1.dtype != row2.dtype:
        row2 = row2.astype(row1.dtype)
    if col1.dtype != col2.dtype:
        col2 = col2.astype(col1.dtype)

    r, c, d = _join_arrays(row1, col1, data1,
                           row2, col2, data2, name,
                           join_type=join_type)
    # Sort by rows
    idx = np.lexsort((c, r))
    return r[idx], c[idx], d[idx]


def _join_arrays(row1, col1, data1,
                row2, col2, data2, name,
                join_type="left"):
    """Join array (numpy array, not structured)
    """
    #pylint: disable=too-many-arguments
    #pylint: disable=too-many-locals

    # join types
    if join_type == "left":
        idx_inner_left, idx_inner_right = get_idx(row1, col1, row2, col2, join_type="inner")
        data_join = set_and_fill_new_array(data1, data2, name,
                                           np.arange(0, len(row1)), np.arange(0, len(row1)),
                                           idx_inner_right, idx_inner_left,
                                           len(row1))
        return row1, col1, data_join
    if join_type == "right":
        idx_inner_left, idx_inner_right = get_idx(row1, col1, row2, col2, join_type="inner")
        data_join = set_and_fill_new_array(data1, data2, name,
                                           idx_inner_left, idx_inner_right,
                                           np.arange(0, len(row2)), np.arange(0, len(row2)),
                                           len(row2))
        return row2, col2, data_join
    if join_type == "inner":
        idx_inner_left, idx_inner_right = get_idx(row1, col1, row2, col2, join_type="inner")
        data_join = set_and_fill_new_array(data1, data2, name,
                                           idx_inner_left, np.arange(0, len(idx_inner_left)),
                                           idx_inner_right, np.arange(0, len(idx_inner_left)),
                                           len(idx_inner_left))
        return row1[idx_inner_left], col1[idx_inner_left], data_join
    if join_type == "outer":
        idx_left, idx_left_new, idx_right, idx_right_new, row_new, col_new = get_idx_outer(row1, col1, row2, col2)
        data_join = set_and_fill_new_array(data1, data2, name,
                                           idx_left, idx_left_new, idx_right, idx_right_new,
                                           len(row_new))
        return np.array(row_new, dtype=row1.dtype), np.array(col_new, dtype=col1.dtype), data_join
    raise ValueError("Unknown join_type (must be 'left', 'right', 'inner', 'outer')")


def set_and_fill_new_array(data1, data2, name,
                           idx_left, idx_left_new, idx_right, idx_right_new,
                           length):
    """Create new structured numpy array and fill with data1 and data2.
    """
    #pylint: disable=too-many-arguments
    new_dtype = [(dname, d[0]) for dname, d in data1.dtype.fields.items()]
    if data2.dtype.names is None:
        new_dtype += [(name, data2.dtype)]
    elif (name == "") or name is None:
        new_dtype += [(f"{dname}", d[0]) for dname, d in data2.dtype.fields.items()]
    else:
        new_dtype += [(f"{name}_{dname}", d[0]) for dname, d in data2.dtype.fields.items()]
    data_join = np.zeros(shape=(length), dtype=new_dtype)

    # Add previous layers
    for dname in data1.dtype.names:
        data_join[dname][idx_left_new] = data1[dname][idx_left]
    # Add new layers
    if data2.dtype.names is None:
        data_join[name][idx_right_new] = data2[idx_right]
    #elif len(data2.dtype) == 1:
    #    data_join[name][idx_right_new] = data2[data2.dtype.names[0]][idx_right]
    else:
        for dname in data2.dtype.names:
            data_join[f"{name}_{dname}"][idx_right_new] = data2[dname][idx_right]
    return data_join


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
