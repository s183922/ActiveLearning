from sklearn.linear_model import LogisticRegression as Log
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.metrics import accuracy_score
from dataloader import *
from UncertaintySampling import *
from ExpectedModelChange_old import *
from QBC_func import *
import pickle
import os, sys
from sklearn.utils import resample
import  matplotlib.pyplot as plt
path = sys.path[0]
os.chdir(path)


X_train, y_train, X_test, y_test, Xpool, ypool, poolidx = datasets('data', poolnum = 1000)

# Multinomial Logistic Regression Classifier
model = Log(penalty = 'l2', multi_class= 'multinomial', max_iter= 500, solver='lbfgs')

# methods = ["Least Confident", "Entropy", "Margin", "Baseline"]
# test = [[] for _ in methods]
# for i, method in enumerate(methods):
#     np.random.seed(2020)
#     for j in range(20):
#         X_train, y_train, X_test, y_test, Xpool, ypool, poolidx = datasets('data', poolnum = 1000)
#         test_acc = Uncertainty_Sampling(X_train[:100].copy(), y_train[:100].copy(), X_test.copy(), y_test.copy(),
#                      model, Xpool.copy(), ypool.copy(), poolidx.copy(),
#                      n_iter = 100, addn = 1, method = method)
#         test[i].append(test_acc)

# plt.plot(*zip(*test_acc))

# filehandler = open("UncertaintyTest.obj","wb")
# pickle.dump(test, filehandler)
# filehandler.close()




np.random.seed(2020)
test = []
for j in range(20):
    print("Iteration {:}".format(j))
    X_train, y_train, X_test, y_test, Xpool, ypool, poolidx = datasets('data', poolnum = 1000)
    test_acc = expModelChange(X_train[:100].copy(), y_train[:100].copy(), X_test.copy(), y_test.copy(),
                          model, Xpool.copy(), ypool.copy(), poolidx.copy(), n_iter=100)

    test.append(test_acc)

filehandler = open("ExpectedGradient.obj","wb")
pickle.dump(test, filehandler)
filehandler.close()

# test_acc = QBC(X_train[:100].copy(), y_train[:100].copy(), X_test, y_test,
#          model, Xpool.copy(), ypool.copy(), poolidx.copy(), n_iter = 100, n_model = 10)
# plt.plot(*zip(*test_acc))

# # plt.plot(*zip(*test_acc))
# plt.legend(["Baseline", "QBC"])
# plt.show()




# plt.plot(*zip(*test_acc))
# # plt.legend(methods)
# plt.show()


# Testing QBC
"""
for i in range()
    fit model to train
    make predictions, store accuracy score

    for k members in committee:
        bootstrap training 
        fit model to bootstrapped data
        make predictions

    find datapoints with highest disagreement
    add to train and remove from pool
"""


# Testing Uncertainty Sampling
"""
for i in range()
    fit model to train
    make predictions, store accuracy score

    LEAST CONFIDENT:
    Get probability distribution over classes (softmax) for each data point
    choose the datapoint with the lowest max probability (least confident)

    MARGIN:
    Get probability distribution over classes (softmax) for each data point
    choose the datapoint where the two highest class probabilities are closest

    ENTROPY:
    Get probability distribution over classes (softmax) for each data point
    equation 3.15

    add to train, remove from pool
"""


# Testing Expected Model Impact

"""
for i in range()
    for each datapoint compute gradient of loss function
    query selects point with largest gradient

    add to train, remove from pool
"""




# Compare results to random baseline