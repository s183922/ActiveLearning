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

def datasets(path, poolnum = 20000):
    #input: Path as string, should be 'data'
    X_train, y_train = load_mnist(path, kind='train')
    X_test, y_test = load_mnist(path, kind='t10k')

    #used to define the idx of the trainset
    trainidx = np.arange(len(X_train),dtype=np.int)

    order = np.random.permutation(range(len(X_train)))
    #define the size of the pool, and define the part of the trainingset
    #will be in the pool from the permutation
    
    poolset = order[:poolnum]

    #Take and define the pool from the trainingset
    Xpool = np.take(X_train,poolset,axis=0)
    ypool = np.take(y_train,poolset,axis=0)
    
    #define the new trainset without the pool
    trainidx = np.setdiff1d(trainidx,poolset)
    
    return X_train[trainidx], y_train[trainidx], X_test, y_test, Xpool, ypool, np.arange(len(Xpool), dtype = np.int)


