from sklearn.linear_model import LogisticRegression as Log
from dataloader import *


Xtrain, ytrain = dataloader
Xpool, ypool = dataloader
Xtest, ytest = dataloader
pool_index = dataloader

# Multinomial Logistic Regression Classifier
model = Log(penlty = 'l2', multi_class= 'multinomial', max_iter= 100)

# Data Points to add each time
add_n = 50


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