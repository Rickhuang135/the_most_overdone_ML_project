import numpy as np

# activation functions
def ReLU(n_by_1, alpha=1):
    n = len(n_by_1)
    less_than_0 = n_by_1<0
    n_by_1[less_than_0] = 0
    neurons_wrt_raw_2d=np.zeros((n,n))
    neurons_wrt_raw_2d[np.arange(n), np.arange(n)]=1
    return n_by_1*alpha, neurons_wrt_raw_2d

def softmax(O, k=1):
    n = len(O)
    new_O = O*k- np.max(O)
    e_powered = np.e**(new_O)
    normalised = e_powered/(e_powered.sum())
    e_powered_nested = np.array([e_powered])
    neurons_wrt_raw_2d = e_powered_nested.T@e_powered_nested
    neurons_wrt_raw_2d *=-k
    neurons_wrt_raw_2d[np.arange(n), np.arange(n)] += k* (e_powered.sum()*e_powered[np.arange(n)])
    neurons_wrt_raw_2d = neurons_wrt_raw_2d / (e_powered.sum()**2)
    return normalised, neurons_wrt_raw_2d

def arctan(n_by_1, alpha=1):
    n = len(n_by_1)
    result = np.arctan(alpha*(n_by_1-0.5))/np.pi + 0.5
    neurons_wrt_raw_1d=(alpha/np.pi) / ((alpha*(n_by_1-0.5))**2 + 1)
    neurons_wrt_raw_2d=np.zeros((n,n))
    neurons_wrt_raw_2d[np.arange(n), np.arange(n)] = neurons_wrt_raw_1d[np.arange(n)]
    return result, neurons_wrt_raw_2d

# cost functions
def square_diff(O, error):
    C = ((O - error)**2).sum()
    cost_wrt_outputs = 2*( O- error)
    return C, cost_wrt_outputs

def cross_entropy(O, error):
    C = (-error*np.log(O+1e-9)).sum()
    cost_wrt_outputs = -error/(O)
    return C, cost_wrt_outputs