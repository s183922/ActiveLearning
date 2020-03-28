from sklearn.linear_model import LogisticRegression as Log
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.metrics import accuracy_score
from dataloader import *
from UncertaintySampling import *
import os, sys
path = sys.path[0]
os.chdir(path)

def BaseLine(pool, addn):
    return np.random.choice(np.arange(len(pool), dtype = np.int), addn, replace=False)

X_train, y_train, X_test, y_test, Xpool, ypool, poolidx = datasets('data', poolnum = 59900)


# Multinomial Logistic Regression Classifier

model = Log(penalty = 'l2', multi_class= 'multinomial', max_iter= 20, solver='lbfgs', verbose=1)
addn = 10
test_acc = []
for i in range(20):

    model.fit(X_train, y_train)
    ye = model.predict(X_test)
    test_acc.append(accuracy_score(y_test, ye))

    method = UncertaintySampling(model, Xpool[poolidx], addn)
    x_star = method.Entropy()
    # x_star = BaseLine(poolidx, addn)

    
    X_train = np.concatenate((X_train, Xpool[poolidx[x_star]]))
    y_train = np.concatenate((y_train, ypool[poolidx[x_star]]))

    poolidx = np.setdiff1d(poolidx, x_star)


plt.plot(np.arange(20),test_acc)
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