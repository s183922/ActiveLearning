from sklearn.utils import resample
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score

def QBC(X_train, y_train, X_test, y_test, model, Xpool, ypool, poolidx, n_iter = 100, n_model = 10):
    test_acc = []
    for i in range(n_iter):
        model.fit(X_train, y_train)
        ye = model.predict(X_test)
        test_acc.append((len(X_train), accuracy_score(y_test, ye)))
        # Matrix for future predictions

        ye = np.zeros([n_model,Xpool.shape[0]])

        for i in range(n_model):

            # resample training data and labels and train
            Xtr, ytr = resample(X_train, y_train, stratify = y_train)
        
            model.fit(Xtr, ytr)

            # make predictions on pool data
            ye[i] = model.predict(Xpool)

        ye = ye.T

        least_conf = []

        # find the pool id where the classifiers have the minimum number of same predictions
        for i,val in enumerate(ye):
            least_conf.append(Counter(val).most_common(1)[0][1])
        
        x_star = least_conf.index(min(least_conf))
        X_train = np.concatenate((X_train, Xpool[x_star].reshape(1,-1)))
        y_train = np.concatenate((y_train, ypool[x_star].reshape(-1)))
        poolidx = np.setdiff1d(poolidx, x_star)
        print("With {:} trainingpoints".format(len(X_train)))

    return test_acc

