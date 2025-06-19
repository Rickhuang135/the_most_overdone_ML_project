from network_functions import square_diff
from network_functions import softmax
import numpy as np

raw_into_softmax = np.array([2,1,1])
output_after_softmax, dO_dR = softmax(raw_into_softmax)
# output_after_softmax , dO_dR=output_after_softmax.round(5), dO_dR.round(5)
output_after_softmax , dO_dR=output_after_softmax, dO_dR
desired_output = np.array([0,0,1])
C, dC_dO = square_diff(output_after_softmax, desired_output)
print(dC_dO)
print(dO_dR@dC_dO)