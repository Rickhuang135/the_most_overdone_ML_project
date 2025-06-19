import numpy as np
import pandas as pd
from extract import extract_images
from extract import extract_label
from network_functions import ReLU
from network_functions import arctan
from network_functions import square_diff
from network_functions import softmax
from matplotlib import pyplot as plt

def get_layer_shape(in_index: int):
    return (Model_Info["neurons"].iloc[in_index+1], Model_Info["neurons"].iloc[in_index])
Model_Info = pd.DataFrame([784, 256, 256, 10],index=["input_layer", "first_layer", "second_layer", "outputlayer"], columns=["neurons"])
parameters_in_to_1 = np.random.randn(*get_layer_shape(0))*np.sqrt(2/Model_Info.loc["input_layer", "neurons"])
parameters_1_to_2 = np.random.randn(*get_layer_shape(1))*np.sqrt(2/Model_Info.loc["first_layer", "neurons"])
parameters_2_to_out = np.random.randn(*get_layer_shape(2))*np.sqrt(2/Model_Info.loc["second_layer", "neurons"])
# parameters_in_to_1 = np.random.rand(*get_layer_shape(0))*np.sqrt(1/Model_Info.loc["input_layer", "neurons"])
# parameters_1_to_2 = np.random.rand(*get_layer_shape(1))*np.sqrt(1/Model_Info.loc["first_layer", "neurons"])
# parameters_2_to_out = np.random.rand(*get_layer_shape(2))*np.sqrt(1/Model_Info.loc["second_layer", "neurons"])


def process_picture(image28x28, Label:int, k=1, alpha=1, back_prop=True):
    global parameters_in_to_1, parameters_1_to_2, parameters_2_to_out
    input_layer= np.array(image28x28).ravel()/255
    first_layer, partial_derivatives_in_to_1 = ReLU(parameters_in_to_1@input_layer, alpha)
    second_layer, partial_derivatives_1_to_2 = ReLU(parameters_1_to_2@first_layer, alpha)
    O, derivative_O_wrt_2R_2d = softmax(parameters_2_to_out@second_layer, k)
    error=np.array([ 1 if x==Label else 0 for x in range(10)])
    C, dC_dO = square_diff(O, error)
    if back_prop:
        # dC_dR = derivative_O_wrt_2R_2d@dC_dO
        dC_dR= dC_dO
        gradient_2_to_out =[]
        for dC_dRu in dC_dR:
            gradient_2_to_out.append(dC_dRu*second_layer)
        gradient_2_to_out = np.array(gradient_2_to_out)

        # dC_db = (parameters_2_to_out@partial_derivatives_1_to_2).T@dC_dR
        dC_db = parameters_2_to_out.T@dC_dR
        gradient_1_to_2 = []
        for dC_dbv in dC_db:
            gradient_1_to_2.append(dC_dbv*first_layer)
        gradient_1_to_2 = np.array(gradient_1_to_2)
        
        # dC_dl = (parameters_1_to_2@partial_derivatives_in_to_1).T@dC_db
        dC_dl = parameters_1_to_2.T@dC_db
        gradient_in_to_1 = []
        for dC_dlp in dC_dl:
            gradient_in_to_1.append(dC_dlp*input_layer)
        gradient_in_to_1 = np.array(gradient_in_to_1)

        step_size=0.01
        parameters_in_to_1-=gradient_in_to_1*step_size
        parameters_1_to_2-=gradient_1_to_2*step_size
        parameters_2_to_out-=gradient_2_to_out*step_size
    return O, C
    

def loop(its: int):
    pictures= extract_images("./data/train-images-idx3-ubyte/train-images.idx3-ubyte",its)
    labels = extract_label("./data/train-labels-idx1-ubyte/train-labels.idx1-ubyte", its)
    for i in range(its//2):
        O,C = process_picture(pictures[i], labels[i])
    error_record=[]
    for i in range(its//2, its):
        O,C = process_picture(pictures[i], labels[i], back_prop=False)
        print(f"function cost is:\t{C:4f}, \t{O.round(2)}")
        error_record.append(C)
    error_record = pd.Series(error_record)
    plt.scatter(error_record.index, error_record.values)
    plt.show()
    

old_in_to_1 = np.copy(parameters_in_to_1)
old_1_to_2 = np.copy(parameters_1_to_2)
old_2_to_out = np.copy(parameters_2_to_out)
loop(600)
