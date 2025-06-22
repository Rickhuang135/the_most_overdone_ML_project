from network import Network
from network_functions import *
from extract import extract_images
from extract import extract_label
from matplotlib import pyplot as plt
import numpy as np
import asyncio

def get_accuracy(predictions, correct_ones):
    values_predictions = np.argmax(predictions,1)
    count = np.sum([values_predictions==correct_ones])
    return count/len(correct_ones)

def loop(its: int):
    # network = Network([784, 16, 16, 10],[ReLU, ReLU, softmax], cross_entropy) #0.725
    # network = Network([784, 256, 256, 256, 10],[ReLU, ReLU, ReLU, softmax], cross_entropy) #0.81 without softmax gradient
    # network = Network([784, 256, 256, 10],[ReLU, ReLU, softmax], square_diff) #0.81
    # network = Network([784, 784, 784, 10],[ReLU, ReLU, softmax], cross_entropy) #0.88
    # network = Network([784, 256, 256, 10],[arctan, ReLU, softmax], cross_entropy) #0.88 without ReLU and arctan gradient *clear winner
    # network = Network([784, 256, 256, 10],[ReLU, arctan, softmax], cross_entropy) #0.85
    # network = Network([784, 256, 256, 216, 10],[ReLU, arctan, ReLU, softmax], cross_entropy) #0.84
    network = Network([784, 256, 256, 10],[ReLU, ReLU, softmax], cross_entropy)  #0.86 *classic model
    pictures= extract_images("./data/train-images-idx3-ubyte/train-images.idx3-ubyte",its)
    labels = extract_label("./data/train-labels-idx1-ubyte/train-labels.idx1-ubyte", its)
    correct_record =[]
    prediction_record = []
    error_record = []
    for i in range(its):
        error = np.zeros(10)
        error[labels[i]]=1
        C, results = network.inferance(np.array(pictures[i]).ravel()/255, error)
        O = results.latest()
        network.back(results)
        asyncio.run(evalutate(C, O, labels[i], i, correct_record, prediction_record, error_record))

    # plt.scatter(range(len(error_record)),error_record)
    # plt.show()

async def evalutate(C, O, label, i , correct_record: list, prediction_record: list, error_record: list, frequency: int = 200):
    # print(f"function cost is:\t{C:4f}, \t{O.round(2)},\t{labels[i]}")
    error_record.append(C)
    correct_record.append(label)
    prediction_record.append(O)
    if (i)%frequency == 0:
        print(get_accuracy(np.array(prediction_record[-frequency:]), np.array(correct_record[-frequency:])))

loop(3000)