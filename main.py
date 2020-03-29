from sklearn.linear_model import LogisticRegression as Log
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.metrics import accuracy_score
from dataloader import *
from UncertaintySampling import *
import os, sys
import  matplotlib.pyplot as plt
path = sys.path[0]
os.chdir(path)


X_train, y_train, X_test, y_test, Xpool, ypool, poolidx = datasets('data', poolnum = 59900)


# Multinomial Logistic Regression Classifier
model = Log(penalty = 'l2', multi_class= 'multinomial', max_iter= 500, solver='lbfgs')
addn = 5

test_acc = Uncertainty_Sampling(X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy(),
                     model, Xpool.copy(), ypool.copy(), poolidx.copy(),
                     n_iter = 200, addn = addn, method = "Entropy")

plt.plot(*zip(*test_acc))
plt.show()

    


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
    finde point with largest gradient

    add to train, remove from pool
"""




# Compare results to random baseline