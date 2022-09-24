![GitHub](https://img.shields.io/github/license/florian-huber/sparsestack)
[![PyPI](https://img.shields.io/pypi/v/sparsestack)](https://pypi.org/project/sparsestack/)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/florian-huber/sparsestack/CI%20Build)

# sparsestack
Memory efficient stack of multiple 2D sparse arrays.

![sparsestack-overview-figure](/docs/sparsestack_overview_01.png)


## First code example (until better readme is done...)

```python
import numpy as np
from sparsestack import StackedSparseArray

# Create some fake data
scores1 = np.random.random((12, 10))
scores2 = np.random.random((12, 10))
scores_array = StackedSparseArray(12, 10)
scores_array.add_dense_matrix(scores1, "scores_1")
scores_array = scores_array.filter_by_range("scores_1", low=0.5)

# Add second scores and filter
scores_array.add_dense_matrix(scores2, "scores_2")
scores_array = scores_array.filter_by_range("scores_2", low=0.1, high=0.4)

# Scores can be accessed using (limited) slicing capabilities
scores_array[3, 4]  # => scores_1 and scores_2 at position row=3, col=4
scores_array[3, :]  # => tuple with row, col, scores for all entries in row=3
scores_array[:, 2]  # => tuple with row, col, scores for all entries in col=2
scores_array[3, :, 0]  # => tuple with row, col, scores_1 for all entries in row=3
scores_array[3, :, "scores_1"]  # => same as the one before

# Scores can also be converted to a dense numpy array:
scores2_after_filtering = scores_array.to_array("scores_2")
```
