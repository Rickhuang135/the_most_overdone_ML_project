import numpy as np
import pandas as pd
from extract import extract_images
from extract import extract_label

def get_layer_shape(in_index: int):
    return (Model_Info["neurons"].iloc[in_index+1], Model_Info["neurons"].iloc[in_index])

Model_Info = pd.DataFrame([784, 256, 256, 10],index=["input_layer", "first_layer", "second_layer", "outputlayer"], columns=["neurons"])
print(Model_Info.head())
# parameters_in_to_1 = np.random.randn(*get_layer_shape(0))*np.sqrt(2/Model_Info.loc["input_layer", "neurons"])
# parameters_1_to_2 = np.random.randn(*get_layer_shape(1))*np.sqrt(2/Model_Info.loc["first_layer", "neurons"])
# parameters_2_to_out = np.random.randn(*get_layer_shape(2))*np.sqrt(2/Model_Info.loc["second_layer", "neurons"])
parameters_in_to_1 = np.random.rand(*get_layer_shape(0))*np.sqrt(2/Model_Info.loc["input_layer", "neurons"])
parameters_1_to_2 = np.random.rand(*get_layer_shape(1))*np.sqrt(2/Model_Info.loc["first_layer", "neurons"])
parameters_2_to_out = np.random.rand(*get_layer_shape(2))*np.sqrt(2/Model_Info.loc["second_layer", "neurons"])

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
        columns[index] += k* (normalised.sum()*e_pow_R)
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

def process_picture(image28x28, Label:int, k=1, alpha=1, verbose=False):
    global parameters_in_to_1, parameters_1_to_2, parameters_2_to_out
    input_layer= np.array(image28x28).ravel()/255
    first_layer, partial_derivatives_in_to_1 = ReLU(parameters_in_to_1@input_layer, alpha)
    second_layer, partial_derivatives_1_to_2 = ReLU(parameters_1_to_2@first_layer, alpha)
    O, derivative_O_wrt_2R_2d = softmax(parameters_2_to_out@second_layer, k)
    error=np.array([ 1 if x==Label else 0 for x in range(10)])
    C, d_C_wrt_O = square_diff(O, error)
    print(f"scaled outputs are:\n{O}") if verbose else 0
    print(f"overall Cost is:\n{C}") if verbose else 0
    print(f"printing gradients:") if verbose else 0
    dC_dR = derivative_O_wrt_2R_2d@d_C_wrt_O
    gradient_2_to_out =[]
    for dC_dRu in dC_dR:
        gradient_2_to_out.append(dC_dRu*second_layer)
    gradient_2_to_out = np.array(gradient_2_to_out)
    print(gradient_2_to_out) if verbose else 0

    dC_db = (parameters_2_to_out@partial_derivatives_1_to_2).T@dC_dR
    gradient_1_to_2 = []
    for dC_dbv in dC_db:
        gradient_1_to_2.append(dC_dbv*first_layer)
    gradient_1_to_2 = np.array(gradient_1_to_2)
    print(gradient_1_to_2) if verbose else 0
    
    dC_dl = (parameters_1_to_2@partial_derivatives_in_to_1).T@dC_db
    gradient_in_to_1 = []
    for dC_dlp in dC_dl:
        gradient_in_to_1.append(dC_dlp*input_layer)
    gradient_in_to_1 = np.array(gradient_in_to_1)
    print(gradient_in_to_1[gradient_in_to_1!=0]) if verbose else 0

    step_size= 0.01
    parameters_in_to_1-=gradient_in_to_1*step_size
    parameters_1_to_2-=gradient_1_to_2*step_size
    parameters_2_to_out-=gradient_2_to_out*step_size
    # raise Exception("stop for a moment")
    return C

    

def loop(its: int):
    pictures= extract_images("./data/train-images-idx3-ubyte/train-images.idx3-ubyte",its)
    labels = extract_label("./data/train-labels-idx1-ubyte/train-labels.idx1-ubyte", its)
    for picture, label in zip(pictures, labels):
        print(process_picture(pictures[0], labels[0]))
        # print(process_picture(picture, label))
    print(process_picture(pictures[0], labels[0], verbose=True))

old_in_to_1 = np.copy(parameters_in_to_1)
old_1_to_2 = np.copy(parameters_1_to_2)
old_2_to_out = np.copy(parameters_2_to_out)
loop(20)
print(f"diff in to 1\n{np.sort(parameters_in_to_1-old_in_to_1)}")
print(f"diff 1 to 2\n{np.sort(parameters_1_to_2-old_1_to_2)}")
print(f"diff 2 to out\n{np.sort(parameters_2_to_out-old_2_to_out)}")

