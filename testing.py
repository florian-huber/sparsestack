import time
from turtle import st
from sparsestack.StackedSparseArray import StackedSparseArray

from scipy.sparse import random


dim_x = 10000
dim_y = 20000
random_sparse = random(dim_x, dim_y, density=0.001, random_state=0)

tstart = time.time()
stacked_arr = StackedSparseArray(dim_x, dim_y)
stacked_arr.add_coo_matrix(random_sparse, "scores1")
print(f"Adding coo matrix took {time.time() - tstart}s to compute.")

tstart = time.time()
for i in range(100):
    slicetest = stacked_arr[i, :]
print(f"Slicing took {time.time() - tstart}s to compute.")

print(stacked_arr)
print(stacked_arr.data.shape)
print(stacked_arr[50, :])