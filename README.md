![GitHub](https://img.shields.io/github/license/florian-huber/sparsestack)
[![PyPI](https://img.shields.io/pypi/v/sparsestack?color=teal)](https://pypi.org/project/sparsestack/)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/florian-huber/sparsestack/CI%20Build)

# sparsestack
Memory efficient stack of multiple 2D sparse arrays.

![sparsestack-overview-figure](/docs/sparsestack_overview_01.png)

## Installation

### Requirements

Python 3.7 or higher

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
