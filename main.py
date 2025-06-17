import numpy as np
import pandas as pd
from extract import extract_images
from extract import extract_label

def get_layer_shape(in_index: int):
    return (Model_Info["neurons"].iloc[in_index+1], Model_Info["neurons"].iloc[in_index])

Model_Info = pd.DataFrame([784, 216, 216, 10],index=["input_layer", "first_layer", "second_layer", "outputlayer"], columns=["neurons"])
print(Model_Info.head())
parameters_in_to_1 = np.random.random(get_layer_shape(0))
parameters_1_to_2 = np.random.random(get_layer_shape(1))
parameters_2_to_out = np.random.random(get_layer_shape(2))

def activation(n_by_1, alpha=100):
    return np.arctan(alpha*(n_by_1-0.5))/np.pi + 0.5

def process_picture(image28x28, Label):
    global parameters_in_to_1, parameters_1_to_2, parameters_2_to_out
    k = 1/400
    alpha = 100
    input_layer784 = np.array(image28x28).ravel()
    first_layer = activation(parameters_in_to_1@input_layer784, alpha)
    second_layer = activation(parameters_1_to_2@first_layer, alpha)
    O = parameters_2_to_out@second_layer
    e_powered = np.e**(O*k)
    error=np.array([ 1 if x==Label else 0 for x in range(10)])
    C = (e_powered/e_powered.sum()-error)**2
    print(C.sum())
    dC_dO = k*(np.e**(2*O*k)-np.e**(O*k)*e_powered.sum())/(e_powered.sum())**2 * 2*(e_powered/e_powered.sum()-error)
    gradient_2_to_out =[]
    for row, dC_dOk in zip(parameters_2_to_out, dC_dO):
        gradient_2_to_out.append(row*dC_dOk)
    gradient_2_to_out = np.array(gradient_2_to_out)

    gradient_1_to_2 = []
    for row, av in zip(parameters_1_to_2, parameters_2_to_out.T):
        dBv_dvM= (alpha/np.pi) / ((alpha*(row.sum()-0.5))**2 + 1) * (second_layer)
        gradient_1_to_2.append(dBv_dvM*(dC_dO*av).sum())
    gradient_1_to_2 = np.array(gradient_1_to_2)
    
    gradient_in_to_1 = []
    for row, mv in zip(parameters_in_to_1, parameters_1_to_2.T):
        dCh_dhL= (alpha/np.pi) / ((alpha*(row.sum()-0.5))**2 + 1) * (first_layer)
        gradient_in_to_1.append(dCh_dhL*(gradient_1_to_2*mv).sum())
    gradient_in_to_1 = np.array(gradient_in_to_1)

    step_size= 0.01
    gradient_in_to_1-=gradient_in_to_1*step_size
    parameters_1_to_2-=gradient_1_to_2*step_size
    parameters_2_to_out-=gradient_2_to_out*step_size



    

def loop(its: int):
    pictures= extract_images("./data/train-images-idx3-ubyte/train-images.idx3-ubyte",its)
    labels = extract_label("./data/train-labels-idx1-ubyte/train-labels.idx1-ubyte", its)
    for picture, label in zip(pictures, labels):
        process_picture(picture, label)

loop(500)