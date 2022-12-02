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

    idx1 = np.lexsort((col1, row1))
    idx2 = np.lexsort((col2, row2))
    # join types
    if join_type == "left":
        idx_inner_left, idx_inner_right, _, _, _, _ = get_idx(row1, col1, row2, col2,
                                                             idx1, idx2, join_type="inner")
        data_join = set_and_fill_new_array(data1, data2, name,
                                           np.arange(0, len(row1)), np.arange(0, len(row1)),
                                           idx_inner_right, idx_inner_left,
                                           len(row1))
        return row1, col1, data_join
    if join_type == "right":
        idx_inner_left, idx_inner_right, _, _, _, _ = get_idx(row1, col1, row2, col2,
                                                             idx1, idx2, join_type="inner")
        data_join = set_and_fill_new_array(data1, data2, name,
                                           idx_inner_left, idx_inner_right,
                                           np.arange(0, len(row2)), np.arange(0, len(row2)),
                                           len(row2))
        return row2, col2, data_join
    if join_type == "inner":
        idx_inner_left, idx_inner_right, _, _, _, _ = get_idx(row1, col1, row2, col2,
                                                             idx1, idx2, join_type="inner")
        data_join = set_and_fill_new_array(data1, data2, name,
                                           idx_inner_left, np.arange(0, len(idx_inner_left)),
                                           idx_inner_right, np.arange(0, len(idx_inner_left)),
                                           len(idx_inner_left))
        return row1[idx_inner_left], col1[idx_inner_left], data_join
    if join_type == "outer":
        idx_left, idx_right, idx_left_new, idx_right_new, row_new, col_new = get_idx_outer(
            row1, col1, row2, col2,
            idx1, idx2
            )
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
def get_idx_inner(left_row, left_col, right_row, right_col,
                  idx1, idx2):
    # inner join
    idx_left = []
    idx_left_new = []
    idx_right = []
    idx_right_new = []
    row_new = []
    col_new = []
    low = 0
    counter = 0
    for i in idx1:
        for j in idx2[low:]:
            if (left_row[i] == right_row[j]) and (left_col[i] == right_col[j]):
                idx_left.append(i)
                idx_left_new.append(counter)
                idx_right.append(j)
                idx_right_new.append(counter)
                row_new.append(left_row[i])
                col_new.append(left_col[i])
                counter += 1
            if left_row[i] > right_row[j]:
                low = j
            if left_row[i] < right_row[j]:
                break
    return idx_left, idx_right, idx_left_new, idx_right_new, row_new, col_new


@numba.jit(nopython=True)
def get_idx_outer(left_row, left_col, right_row, right_col,
                  idx1, idx2):
    # outer join
    idx_left = []
    idx_left_new = []
    idx_right = []
    idx_right_new = []
    row_new = []
    col_new = []

    right_in_inner = []
    low = 0
    counter = 0
    for i in idx1:
        current_match = False
        for j in idx2[low:]:
            if (left_row[i] == right_row[j]) and (left_col[i] == right_col[j]):
                right_in_inner.append(j)
                current_match = True
            if left_row[i] > right_row[j]:
                low = j
            if left_row[i] < right_row[j]:
                break
        if current_match:
            x = right_in_inner[-1]
            idx_left.append(i)
            idx_left_new.append(counter)
            idx_right.append(x)
            idx_right_new.append(counter)
            row_new.append(left_row[i])
            col_new.append(left_col[i])
            counter += 1
        else:
            idx_left.append(i)
            idx_left_new.append(counter)
            row_new.append(left_row[i])
            col_new.append(left_col[i])
            counter += 1

    for j in set(idx2).difference(set(right_in_inner)):
        idx_right.append(j)
        idx_right_new.append(counter)
        row_new.append(right_row[j])
        col_new.append(right_col[j])
        counter += 1
    return idx_left, idx_right, idx_left_new, idx_right_new, row_new, col_new


def get_idx(left_row, left_col, right_row, right_col, idx1, idx2,
            join_type="left"):
    if join_type == "inner":
        return get_idx_inner(left_row, left_col, right_row, right_col,
                             idx1, idx2)
    if join_type == "outer":
        return get_idx_outer(left_row, left_col, right_row, right_col,
                             idx1, idx2)
    else:
        raise ValueError("Unknown join_type")
