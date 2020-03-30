from sklearn.utils import resample
from collections import Counter
import numpy as np

def QBC(X_train, y_train, X_test, y_test, Xpool, ypool, model, n_model = 20):

    # Matrix for future predictions
    ye = np.zeros([n_model,Xpool.shape[0]])

    for i in range(n_model):

        # resample training data and labels and train
        idx = resample(range(X_train.shape[0]))

        X_train = X_train[idx]
        y_train = y_train[idx]
        model.fit(X_train, y_train)

        # make predictions on pool data
        ye[i] = model.predict(Xpool)

    ye = ye.T

    least_conf = []

    # find the pool id where the classifiers have the minimum number of same predictions
    for i,val in enumerate(ye):
        least_conf.append(Counter(val).most_common(1)[0][1])

    x_star = least_conf.index(min(least_conf))

    return x_star

