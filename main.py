import numpy as np
from extract import extract_images

print(*extract_images("./data/train-images-idx3-ubyte/train-images.idx3-ubyte",1))