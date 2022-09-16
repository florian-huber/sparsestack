import numpy as np
import pandas as pd


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

    if name is None:
        df = pd.DataFrame(input_array[idx_row, idx_col])
    else:
        if len(input_array.dtype) > 1:  # if structured array
            column_names = []
            for dtype_name in input_array.dtype.names:
                column_names.append(f"{name}{dtype_name}")
        else:
            column_names = [name]
        if input_array.ndim == 1:
            df = pd.DataFrame(input_array[idx_row])
        else:
            df = pd.DataFrame(input_array[idx_row, idx_col])
        df.columns = column_names

    df = df.set_index([idx_row, idx_col])

    return df
