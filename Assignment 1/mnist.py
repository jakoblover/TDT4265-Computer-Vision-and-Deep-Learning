# Code modified from: https://github.com/hsjeong5/MNIST-for-Numpy

import numpy as np
from urllib import request
import gzip
import pickle
import os
filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]
SAVE_PATH = "data"

def download_mnist():
    os.makedirs(SAVE_PATH, exist_ok=True)
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        save_path = os.path.join(SAVE_PATH, name[1])
        request.urlretrieve(base_url+name[1], save_path)
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        path = os.path.join(SAVE_PATH, name[1])
        with gzip.open(path, 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        path = os.path.join(SAVE_PATH, name[1])
        with gzip.open(path, 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    save_path = os.path.join(SAVE_PATH, "mnist.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    download_mnist()
    save_mnist()

def load():
    save_path = os.path.join(SAVE_PATH, "mnist.pkl")
    with open(save_path,'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

if __name__ == '__main__':
    init()