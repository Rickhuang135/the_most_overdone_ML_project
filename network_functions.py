import numpy as np

# activation functions
def ReLU(n_by_1, alpha=1):
    less_than_0 = n_by_1<0
    n_by_1[less_than_0] = 0
    neurons_wrt_raw_2d=np.zeros((len(n_by_1),len(n_by_1)))
    for index, columns in enumerate(neurons_wrt_raw_2d):
        if n_by_1[index]>0:
            columns[index] = alpha
    return n_by_1*alpha, neurons_wrt_raw_2d

def softmax(O, k=1):
    new_O = O*k- np.max(O)
    e_powered = np.e**(new_O)
    normalised = e_powered/(e_powered.sum())
    neurons_wrt_raw_2d= []
    for index, e_pow_R in enumerate(e_powered):
        columns = -1* (k*e_pow_R*e_powered)
        columns[index] += k* (e_powered.sum()*e_pow_R)
        columns = columns /((e_powered.sum()**2))
        neurons_wrt_raw_2d.append(columns)
    neurons_wrt_raw_2d = np.array(neurons_wrt_raw_2d)
    return normalised, neurons_wrt_raw_2d


def arctan(n_by_1, alpha=1):
    result = np.arctan(alpha*(n_by_1-0.5))/np.pi + 0.5
    neurons_wrt_raw_1d=(alpha/np.pi) / ((alpha*(n_by_1-0.5))**2 + 1)
    neurons_wrt_raw_2d=np.zeros((len(n_by_1),len(n_by_1)))
    for index, columns in enumerate(neurons_wrt_raw_2d):
        columns[index]=neurons_wrt_raw_1d[index]
    return result, neurons_wrt_raw_2d

# cost functions
def square_diff(O, error):
    C = ((O - error)**2).sum()
    cost_wrt_outputs = 2*( O- error)
    return C, cost_wrt_outputs

def cross_entropy(O, error):
    C = (-error*np.log(O)).sum()
    cost_wrt_outputs = -error/(O)
    return C, cost_wrt_outputs