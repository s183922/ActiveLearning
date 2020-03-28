import os, sys
import gzip
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import urllib.request



def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def datasets(path):
    #input: Path as string, should be 'data'
    X_train, y_train = load_mnist(path, kind='train')
    X_test, y_test = load_mnist(path, kind='t10k')

    trainidx = np.arange(len(X_train),dtype=np.int)

    order = np.random.permutation(range(len(X_train)))

    poolnum = 20000
    poolset = order[:poolnum]

    #Take and define the pool from the trainingset
    Xpool = np.take(X_train,poolset,axis=0)
    ypool = np.take(y_train,poolset,axis=0)

    trainidx = np.setdiff1d(trainidx,poolset)

    return X_train, y_train, X_test, y_test, Xpool, ypool, trainidx
