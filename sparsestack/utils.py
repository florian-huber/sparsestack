import numpy as np
import polars as pl


def array_to_df(input_array,
                name=None):
    """Convert numpy array (including structured ones) to pandas DataFrame.
    """
    if input_array is None:
        return None
    assert isinstance(input_array, np.ndarray), "Expected numpy array"

    if input_array.ndim not in (1, 2):
        raise IndexError('Index dimension must be 1 or 2')
    if input_array.ndim == 1:
        idx_row = np.where(input_array)[0]
        idx_col = np.zeros(idx_row.shape).astype(int)
    else:
        (idx_row, idx_col) = np.where(input_array)

    df = pl.DataFrame()
    if len(input_array.dtype) > 1:  # if structured array
        if name is None:
            name = ""
        for dtype_name in input_array.dtype.names:
            col_name = f"{name}{dtype_name}"
            if input_array.ndim == 1:
                df = df.with_columns([
                    pl.Series(name=col_name, values=input_array[dtype_name][idx_row])
                    ])
            else:
                df = df.with_columns([
                    pl.Series(name=col_name, values=input_array[dtype_name][idx_row, idx_col])
                    ])
    else:
        if name is None:
            name = "data"
        if input_array.ndim == 1:
            df = df.with_columns([
                pl.Series(name=name, values=input_array[idx_row])
                ])
        else:
            df = df.with_columns([
                pl.Series(name=name, values=input_array[idx_row, idx_col])
                ])

    # df = df.set_index([idx_row, idx_col])
    df = df.with_columns([
        pl.Series(name="row", values=idx_row),
        pl.Series(name="col", values=idx_col)])
    #df = df.with_columns([
    #        pl.lit(idx_row).alias("row"),
    #        pl.lit(idx_col).alias("col"),
    #    ])

    return df


def coo_matrix_to_df(coo_matrix, name):
    """Convert sparse coo matrix to pandas DataFrame.
    """
    if coo_matrix is None:
        return None

    idx_row = coo_matrix.row
    idx_col = coo_matrix.col

    return coo_values_to_df(coo_matrix.data, idx_row, idx_col, name)


def coo_values_to_df(data, row, col, name):
    """Convert sparse coo values to pandas DataFrame.
    """
    if name is None:
        df = pl.DataFrame({"row": row,
                           "col": col,
                           "data": data})
    else:
        df = pl.DataFrame({"row": row,
                           "col": col,
                           "name": data})

    return df


def sparse_stack_to_array(input_sparse_stack):
    """Convert sparse stack to dense numpy array.
    """
    # Create structured numpy array
    n_rows, n_cols, _ = input_sparse_stack.shape
    dtypes = list(zip(input_sparse_stack.data.columns, input_sparse_stack.data.dtypes))
    array = np.zeros((n_rows, n_cols),
                     dtype=dtypes)
    array[input_sparse_stack.row, input_sparse_stack.col] = [row for row in input_sparse_stack.data.itertuples(index=False)]
    return array
