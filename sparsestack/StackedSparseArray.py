# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.sputils import get_index_dtype
from .utils import join_arrays


_slicing_not_implemented_msg = "Wrong slicing, or option not yet implemented"


class StackedSparseArray:
    """ 2.5D sparse matrix in COO-style with multiple possible entries per i-j-position.

    Add description...

    Parameters
    ----------
    n_row
        Number of rows of sparse array.
    n_cols
        Number of colums of sparse array.

    Code example:

    .. code-block:: python
        import numpy as np
        from sparsestack import StackedSparseArray

        # Create some fake data
        scores1 = np.random.random((12, 10))
        scores1[scores1 < 0.9] = 0  # make "sparse"
        scores2 = np.random.random((12, 10))
        scores2[scores2 < 0.75] = 0  # make "sparse"
        sparsestack = StackedSparseArray(12, 10)
        sparsestack.add_dense_matrix(scores1, "scores_1")

        # Add second scores and filter
        sparsestack.add_dense_matrix(scores2, "scores_2", join_type="left")

        # Scores can be accessed using (limited) slicing capabilities
        sparsestack[3, 4]  # => scores_1 and scores_2 at position row=3, col=4
        sparsestack[3, :]  # => tuple with row, col, scores for all entries in row=3
        sparsestack[:, 2]  # => tuple with row, col, scores for all entries in col=2
        sparsestack[3, :, 0]  # => tuple with row, col, scores_1 for all entries in row=3
        sparsestack[3, :, "scores_1"]  # => same as the one before

        # Scores can also be converted to a dense numpy array:
        scores2_after_merge = sparsestack.to_array("scores_2")

    """
    def __init__(self, n_row, n_col):
        self.__n_row = n_row
        self.__n_col = n_col
        self.idx_dtype = get_index_dtype(maxval=max(n_row, n_col))
        self.row = np.array([], dtype=self.idx_dtype)
        self.col = np.array([], dtype=self.idx_dtype)
        self.data = None

    def __repr__(self):
        msg = f"<{self.shape[0]}x{self.shape[1]}x{self.shape[2]} stacked sparse array" \
            f" containing scores for {self.score_names}" \
            f" with {len(self.row)} stored elements in COOrdinate format>"
        return msg

    def __str__(self):
        msg = f"StackedSparseArray array of shape {self.shape}" \
            f" containing scores for {self.score_names}."
        return msg

    def __eq__(self, other):
        if self.score_names != other.score_names:
            return False
        if self.shape != other.shape:
            return False
        if np.any(self.row != other.row):
            return False
        if np.any(self.col != other.col):
            return False
        if np.any(self.data != other.data):
            return False
        return True

    def __setitem__(self, key, d):
        # Typical COO method (e.g. below) would not be safe for stacked array.
        raise NotImplementedError

    def __getitem__(self, key):
        row, col, name = self._validate_indices(key)
        r, c, d = self._getitem_method(row, col, name)
        if isinstance(row, int) and isinstance(col, int):
            if len(r) == 0:
                return np.array([0])
            return d
        return r, c, d

    def _getitem_method(self, row, col, name):
        # e.g.: matrix[3, 7, "score_1"]
        if isinstance(row, int) and isinstance(col, int):
            idx = np.where((self.row == row) & (self.col == col))
            return self.row[idx], self.col[idx], self._slicing_data(name, idx)
        # e.g.: matrix[3, :, "score_1"]
        if isinstance(row, int) and isinstance(col, slice):
            self._is_implemented_slice(col)
            idx = np.where(self.row == row)
            return self.row[idx], self.col[idx], self._slicing_data(name, idx)
        # e.g.: matrix[:, 7, "score_1"]
        if isinstance(row, slice) and isinstance(col, int):
            self._is_implemented_slice(row)
            idx = np.where(self.col == col)
            return self.row[idx], self.col[idx], self._slicing_data(name, idx)
        # matrix[:, :, "score_1"]
        if isinstance(row, slice) and isinstance(col, slice):
            self._is_implemented_slice(row)
            self._is_implemented_slice(col)
            return self.row, self.col, self._slicing_data(name)
        if row == col is None and isinstance(name, str):
            return self.row, self.col, self._slicing_data(name)
        raise IndexError(_slicing_not_implemented_msg)

    def _is_implemented_slice(self, input_slice):
        # Currently slices like matrix[2:4, :] or not implemented
        if not input_slice.start == input_slice.stop == input_slice.step is None:
            raise IndexError(_slicing_not_implemented_msg)

    def _slicing_data(self, name, idx=None):
        if isinstance(name, slice) and len(self.score_names) == 1:
            name = self.score_names[0]
        if isinstance(name, str):
            if idx is None:
                return self.data[name]
            return self.data[name][idx]
        if isinstance(name, slice) and name.start == name.stop == name.step is None:
            if idx is None:
                return self.data
            return self.data[idx]
        raise IndexError(_slicing_not_implemented_msg)

    def _validate_indices(self, key):
        def validate_index(index, shape):
            if isinstance(index, int):
                if index < -shape or index >= shape:
                    raise IndexError(f"Index ({index}) out of range")
                if index < 0:
                    index += shape
            elif not isinstance(index, slice):
                index = self._asindices(index, shape)
            return index

        m, n, _ = self.shape
        row, col, name = _unpack_index(key)
        if row == col is None and isinstance(name, str):
            return row, col, name

        if isinstance(name, int):
            name = self.score_names[name]

        row = validate_index(row, m)
        col = validate_index(col, n)
        return row, col, name

    def _asindices(self, idx, length):
        """Convert `idx` to a valid index for an axis with a given length.
        Subclasses that need special validation can override this method.
        """
        try:
            x = np.asarray(idx)
        except (ValueError, TypeError, MemoryError) as e:
            raise IndexError('invalid index') from e

        if x.ndim not in (1, 2):
            raise IndexError('Index dimension must be 1 or 2')

        if x.size == 0:
            return x

        # Check bounds
        max_indx = x.max()
        if max_indx >= length:
            raise IndexError(f"Index ({max_indx}) out of range")

        min_indx = x.min()
        if min_indx < 0:
            if min_indx < -length:
                raise IndexError(f"Index ({min_indx}) out of range")
            if x is idx or not x.flags.owndata:
                x = x.copy()
            x[x < 0] += length
        return x

    @property
    def shape(self):
        return self.__n_row, self.__n_col, len(self.score_names)

    @property
    def score_names(self):
        if self.data is None:
            return []
        if self.data.dtype.names is None:
            return [self.data.dtype.str]
        return self.data.dtype.names

    def clone(self):
        """ Returns clone (deepcopy) of StackedSparseArray instance."""
        cloned_array = StackedSparseArray(self.__n_row, self.__n_col)
        cloned_array.row = self.row
        cloned_array.col = self.col
        cloned_array.data = self.data
        return cloned_array

    def add_dense_matrix(self, matrix: np.ndarray,
                         name: str,
                         join_type="left"):
        """Add dense array (numpy array) to stacked sparse scores.

        If the StackedSparseArray is still empty, the full dense matrix will
        be added.
        If the StackedSparseArray already contains one or more scores, than only
        those values of the input matrix will be added which have the same position
        as already existing entries!

        Parameters
        ----------
        matrix
            Input (dense) array, such as numpy array to be added to the stacked sparse
            scores.
        name
            Name of the score which is added. Will later be used to access and address
            the added scores, for instance via `sss_array.toarray("my_score_name")`.
        join_mode
            Choose from left, right, outer, inner to specify the merge type.
        """
        if matrix is None:
            self.data = np.array([])
        elif len(matrix.dtype) > 1:  # if structured array
            for dtype_name in matrix.dtype.names:
                self._add_dense_matrix(matrix[dtype_name],
                                       f"{name}_{dtype_name}",
                                       join_type)
        else:
            self._add_dense_matrix(matrix, name, join_type)

    def _add_dense_matrix(self, matrix, name, join_type):
        if matrix.dtype.type == np.void:
            input_dtype = matrix.dtype[0]
        else:
            input_dtype = matrix.dtype

        # Handle 1D arrays
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)

        if self.shape[2] == 0 or (self.shape[2] == 1 and name in self.score_names):
            # Add first (sparse) array of scores
            (idx_row, idx_col) = np.where(matrix)
            self.row = idx_row
            self.col = idx_col
            self.data = np.array(matrix[idx_row, idx_col], dtype=[(name, input_dtype)])
        else:
            # Add new stack of scores
            (idx_row, idx_col) = np.where(matrix)
            self.add_sparse_data(idx_row, idx_col, matrix[idx_row, idx_col],
                                 name=name,
                                 join_type=join_type)

    def guess_score_name(self):
        if len(self.score_names) == 1:
            return self.score_names[0]
        if len(self.score_names) == 0:
            raise KeyError("Array is empty.")
        raise KeyError("Name of score is required.")

    def add_coo_matrix(self, coo_matrix,
                       name,
                       join_type="left"):
        """Add sparse matrix (scipy COO-matrix) to stacked sparse scores.

        If the StackedSparseArray is still empty, the full sparse matrix will
        be added.
        Default join type is "left", which means that only those values
        of the input matrix will be added which have the same position
        as already existing entries.
        Other possible join_type options are "inner", "right", "outer".

        Parameters
        ----------
        matrix
            Input sparse matrix (COO-style containing .row, .col, .data) to be
            added to the stacked sparse scores.
        name
            Name of the score which is added. Will later be used to access and address
            the added scores, for instance via `sss_array.toarray("my_score_name")`.
        join_mode
            Choose from left, right, outer, inner to specify the merge type.
        """
        self.add_sparse_data(coo_matrix.row, coo_matrix.col, coo_matrix.data, name, join_type)

    def add_sparse_data(self, row, col, data: np.ndarray,
                        name: str,
                        join_type="left"):
        """Add sparse data to stacked sparse scores.

        If the StackedSparseArray is still empty, the full sparse data will
        be added.
        Default join type is "left", which means that only those values
        of the input data will be added which have the same position
        as already existing entries.
        Other possible join_type options are "inner", "right", "outer".

        Parameters
        ----------
        data
            Input data (1D array).
        name
            Name of the score which is added. Will later be used to access and address
            the added scores, for instance via `sss_array.toarray("my_score_name")`.
        join_mode
            Choose from left, right, outer, inner to specify the merge type.
        """
        # pylint: disable=too-many-arguments
        if self.shape[2] == 0 or (self.shape[2] == 1 and name in self.score_names):
            # Add first (sparse) array of scores
            self.data = np.array(data, dtype=[(name, data.dtype)])
            self.row = row.copy()
            self.col = col.copy()
            # self.__n_row, self.__n_col = coo_matrix.shape
        else:
            if join_type in ["outer", "right"]:
                assert np.max(row) <= self.shape[0], "row values have dimension larger than sparse stack"
                assert np.max(col) <= self.shape[1], "column values have dimension larger than sparse stack"
            self.row, self.col, self.data = join_arrays(self.row, self.col, self.data,
                                                        row, col,
                                                        data,
                                                        name,
                                                        join_type=join_type)

    def filter_by_range(self, name: str = None,
                        low=-np.inf, high=np.inf,
                        above_operator='>',
                        below_operator='<'):
        """Remove all scores for which the score `name` is outside the given range.

        Parameters
        ----------
        name
            Name of the score which is used for filtering. Run `.score_names` to
            see all scores scored in the sparse array.
        low
            Lower threshold below which all scores will be removed.
        high
            Upper threshold above of which all scores will be removed.
        above_operator
            Define operator to be used to compare against `low`. Default is '>'.
            Possible choices are '>', '<', '>=', '<='.
        below_operator
            Define operator to be used to compare against `high`. Default is '<'.
            Possible choices are '>', '<', '>=', '<='.
        """
        # pylint: disable=too-many-arguments
        above_operator = _get_operator(above_operator)
        below_operator = _get_operator(below_operator)
        if name is None:
            name = self.guess_score_name()
        idx = np.where(above_operator(self.data[name], low)
                       & below_operator(self.data[name], high))
        cloned_array = StackedSparseArray(self.__n_row, self.__n_col)
        cloned_array.col = self.col[idx]
        cloned_array.row = self.row[idx]
        cloned_array.data = self.data[idx]
        return cloned_array

    def to_array(self, name=None):
        """Return scores as (non-sparse) numpy array.

        Parameters
        ----------
        name
            Name of the score that should be returned (if multiple scores are stored).
            If set to None (default) a 3D array with all scores will be returned.
        """
        if self.data is None:
            return None
        if name is None and self.shape[2] == 1:
            name = self.score_names[0]
        if isinstance(name, str):
            array = np.zeros((self.__n_row, self.__n_col),
                             dtype=self.data[name].dtype)
            array[self.row, self.col] = self.data[name]
            return array
        array = np.zeros((self.__n_row, self.__n_col),
                         dtype=self.data.dtype)
        array[self.row, self.col] = self.data
        return array

    def to_coo(self, name):
        return coo_matrix((self.data[name], (self.row, self.col)),
                          shape=(self.__n_row, self.__n_col))


def _unpack_index(index):
    if isinstance(index, tuple):
        if len(index) == 3:
            row, col, name = index
            if name is None:
                name = slice(None)
        elif len(index) == 2:
            row, col, name = index[0], index[1], slice(None)
        elif len(index) == 1:
            row, col, name = index[0], slice(None), slice(None)
        else:
            raise IndexError("Invalid number of indices")
    elif isinstance(index, str):
        row, col, name = None, None, index
    elif isinstance(index, int):
        row, col, name = index, slice(None), slice(None)
    else:
        raise IndexError(_slicing_not_implemented_msg)
    return row, col, name


def _get_operator(relation: str):
    relation = relation.strip()
    ops = {'>': np.greater,
           '<': np.less,
           '>=': np.greater_equal,
           '<=': np.less_equal}
    if relation in ops:
        return ops[relation]
    raise ValueError(f"Unknown relation {relation}")
