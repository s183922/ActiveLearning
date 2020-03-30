from sklearn.linear_model import LogisticRegression as Log
from sklearn.utils import resample
from dataloader import *

import numpy as np
from collections import Counter
import random

random.seed(10)

X_train, y_train, X_test, y_test, Xpool, ypool, poolidx = datasets('data', poolnum = 59900)

model = Log(penalty = 'l2', multi_class= 'multinomial', max_iter= 100, solver='lbfgs')
addn = 5

n_model = 10

ye = np.zeros([n_model,Xpool.shape[0]])

for i in range(n_model):

    idx = resample(range(X_train.shape[0]))

    X_train = X_train[idx]
    y_train = y_train[idx]
    
    model.fit(X_train, y_train)
    ye[i] = model.predict(Xpool)

ye = ye.T

least_conf = []

for i,val in enumerate(ye):
    least_conf.append(Counter(val).most_common(1)[0][1])

x_star = least_conf.index(min(least_conf))

print(x_star, ye[x_star])