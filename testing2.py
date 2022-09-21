import numpy as np
import pytest
from scipy.sparse import coo_matrix
from sparsestack.utils import array_to_df
#from tests.test_stacked_sparse_array import dense_array_sparse


arr = np.arange(0, 120).reshape(12, 10)
arr[arr % 2 == 1] = 0
arr[arr % 4 == 0] = 0


arr2 = np.array([[(1, 1.5), (2, 3.3)], [(3, 2.3), (4, 5.5)]],
               dtype=[("x", int), ("y", float)])
arr3 = np.array([[(11, 11.5), (12, 13.3), (0, 77)], [(13, 12.3), (14, 15.5), (5, 55)]],
               dtype=[("a", int), ("b", float)])

df = array_to_df(arr)

df2 = array_to_df(arr2, "yes")
#print(df2.loc[1, 1])

#df2 = df2.set_index([[1, 2], [1, 2]])
df2.rename(index={0:2}, inplace=True)
#print(df2.index)
df3 = array_to_df(arr3)

df_merge = df2.join(df3, how='outer')

print(df_merge.iloc[[]].values)

df_merge = df_merge.rename_axis(index=["row", "column"])
#print(df_merge)
#print(df_merge.iloc[np.array([2, 4, 5])])
#print(df_merge.iloc[2])
#print("row:", df_merge.index.get_level_values(0).to_numpy())
#print("column:", df_merge.index.get_level_values(1).to_numpy())

#from numpy.lib import recfunctions as rfn
#print(np.array(df.index.get_level_values(level=0)))
from sparsestack.StackedSparseArray import StackedSparseArray

scores1 = np.arange(0, 120).reshape(12, 10)
scores2 = np.arange(0, 120).reshape(12, 10).astype(float)
scores2[scores2 < 80] = 0
scores2[scores2 > 0] = 0.9

matrix = StackedSparseArray(12, 10)
matrix.add_dense_matrix(scores1.astype(np.int64), "scores1")
matrix = matrix.filter_by_range(low=70, high=85)
assert matrix.shape == (12, 10, 1)
print(matrix.data["scores1"])
assert np.all(matrix.data["scores1"] == np.arange(71, 85))

matrix.add_dense_matrix(scores2, "scores2", nan_to_zeros=True)
matrix = matrix.filter_by_range("scores2", low=0.5)
# Test slicing
assert matrix[8, 1, 0] == np.array([81])
assert matrix[8, 1, 1] == np.array([0.9])
print(matrix.data)
print(matrix.data.iloc[3, 0]) # keeps int
print(matrix.data.iloc[3, :]) # makes it float
print(matrix[8, 3])
assert np.all(matrix[8, 3] == np.array([(83, 0.9)],
                                        dtype=[('scores1', '<i8'), ('scores2', '<f8')]))
assert np.all(matrix[8, :][2]["scores1"] == np.array([80, 81, 82, 83, 84], dtype=np.int64))
assert np.all(matrix[8, :][2]["scores2"] == np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float64))
assert np.all(matrix[8, :, "scores2"][2] == np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float64))
assert np.all(matrix[8, :, 1][2] == np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float64))

# Test more properties
assert np.all(matrix.data["scores1"] == np.arange(80, 85))
assert np.all(matrix.col == np.arange(0, 5))
assert np.all(matrix.row == 8)
assert matrix.shape == (12, 10, 2)
assert matrix.score_names == ('scores1', 'scores2')