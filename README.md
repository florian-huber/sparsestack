![GitHub](https://img.shields.io/github/license/florian-huber/sparsestack)
[![PyPI](https://img.shields.io/pypi/v/sparsestack?color=teal)](https://pypi.org/project/sparsestack/)
[![Conda](https://img.shields.io/conda/v/conda-forge/sparsestack?color=blue)](https://anaconda.org/conda-forge/sparsestack)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/florian-huber/sparsestack/CI_build.yml?branch=main)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)

![sparsestack logo](https://github.com/florian-huber/sparsestack/blob/main/images/sparsestack_logo.png)

Memory efficient stack of multiple 2D sparse arrays.

![sparsestack-overview-figure](/docs/sparsestack_overview_01.png)

## Installation

### Requirements

Python 3.8 or higher

### Pip Install
Simply install using pip: `pip install sparsestack`

## First code example

```python
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
```

## Adding data to a `sparsestack`-array
Sparsestack provides three options to add data to a new layer.
1) `.add_dense_matrix(input_array)`
Can be used to add all none-zero elements of `input_array` to the sparsestack. Depending on the chosen `join_type` either all such values will be added (`join_type="outer"` or `join_type="right"`), or only those which are already present in underlying layers ("left" or "inner" join).
2) `.add_sparse_matrix(input_coo_matrix)`
This method will expect a COO-style matrix (e.g. scipy) which has attributes .row, .col and .data. The join type can again be specified using `join_type`.
3) `.add_sparse_data(row, col, data)`
This essentially does the same as `.add_sparse_matrix(input_coo_matrix)` but might in some cases be a bit more flexible because row, col and data are separate input arguments.

## Accessing data from `sparsestack`-array
The collected sparse data can be accessed in multiple ways.

1) Slicing.
`sparsestack` allows multiple types of slicing (see also code example above).
```python
sparsestack[3, 4]  # => tuple with all scores at position row=3, col=4
sparsestack[3, :]  # => tuple with row, col, scores for all entries in row=3
sparsestack[:, 2]  # => tuple with row, col, scores for all entries in col=2
sparsestack[3, :, 0]  # => tuple with row, col, scores_1 for all entries in row=3
sparsestack[3, :, "scores_1"]  # => same as the one before
```
2) `.to_array()`
Creates and returns a dense numpy array of size `.shape`. Can also be used to create a dense numpy array of only a single layer when used like `.to_array(name="layerX")`.  
**Carefull:** Obviously by converting to a dense array, the sparse nature will be lost and all empty positions in the stack will be filled with zeros.
3) `.to_coo(name="layerX")`
Returns a scipy sparse COO-matrix of the specified layer.
