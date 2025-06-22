# import cupy as cp

# x_gpu = cp.array([1,2,3])
# l2_gpu = cp.linalg.norm(x_gpu)

import numpy as np
from network_functions import softmax

test_array = np.array([[1,1,1],[2,2,2],[3,3,4]])
print(test_array[-2:])