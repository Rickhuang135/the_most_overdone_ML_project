import cupy as cp
import numpy as np
from time import sleep

x_cpu = np.array([1, 2, 3])
x_gpu = cp.asarray(x_cpu)  # move the data to the current device.
sleep(10)