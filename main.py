from network import Network
from network_functions import *
from extract import extract_images
from extract import extract_label
from matplotlib import pyplot as plt
import numpy as np
def loop(its: int):
    network = Network([784, 256, 256, 10],[ReLU, ReLU, softmax], cross_entropy)
    pictures= extract_images("./data/train-images-idx3-ubyte/train-images.idx3-ubyte",its)
    labels = extract_label("./data/train-labels-idx1-ubyte/train-labels.idx1-ubyte", its)
    for i in range(its//2):
        error=np.array([ 1 if x==labels[i] else 0 for x in range(10)])
        C, results = network.inferance(np.array(pictures[i]).ravel()/255, error)
        O = results.latest()
        network.back(results)
        print(f"function cost is:\t{C:4f}, \t{O.round(2)}")
    print(f"training completed with {its//2} examples")
    error_record=[]
    for i in range(its//2, its):
        error=np.array([ 1 if x==labels[i] else 0 for x in range(10)])
        C, results = network.inferance(np.array(pictures[i]).ravel()/255, error)
        O = results.latest()
        print(f"function cost is:\t{C:4f}, \t{O.round(2)}")
        error_record.append(C)
    plt.scatter(range(len(error_record)), error_record)
    plt.show()

loop(3000)