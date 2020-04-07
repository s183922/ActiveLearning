import numpy as np
from numpy import linalg as LA
from sklearn.metrics import accuracy_score
import math


def expModelChange(X_train, y_train, X_test, y_test, model, Xpool, ypool, poolidx, n_iter=100):
    """
    ExpModelChange defines the change in model after the model has learned a new label
    The query strategy selects the x with the largest expected gradient length with respect to the
    posterior predictive distribution of the labels
    """
    testacc_emc = []
    for i in range(n_iter):
        # Fit model and make predicitons
        model.fit(X_train, y_train)
        ye = model.predict(X_test)
        testacc_emc.append((len(X_train), accuracy_score(y_test, ye)))

        # Get probability distribution over labels
        p_pred = model.predict_proba(Xpool[poolidx])

        # select model with largest expected gradient length
        grad = gradJ(Xpool[poolidx], model.coef_, p_pred)
        x_star = ExpGradL(grad, p_pred)

        # Add to train - remove from pool
        X_train = np.concatenate((X_train, Xpool[poolidx[x_star]].reshape(1, -1)))
        y_train = np.concatenate((y_train, ypool[poolidx[x_star]].reshape(-1)))
        poolidx = np.setdiff1d(poolidx, poolidx[x_star])
        print("Expected model with {:} trainingpoints".format(len(X_train)))

    return testacc_emc


# Define objective function - cost function (for each pool point)
def g(z):  # sigmoid function
    return 1.0 / (1.0 + np.exp(-z))


def h_logistic(pool, theta):  # Model function
    return g(np.dot(pool, theta.T))


def J(pool, theta, y):  # Cost Function
    m = y.size
    cost = -(np.sum(np.log(h_logistic(pool, theta))) + np.dot((y - 1).T, (np.dot(pool, theta)))) / m
    return cost


# EVENTUELT Regularization..?
# def J_reg(X,a,y,reg_lambda) :
#    m = y.size
#    return J(X,a,y) + reg_lambda/(2.0*m) * np.dot(a[1:],a[1:])

def gradJ(pool, theta, y):  # Gradient of Cost Function, y should be p_pool
    m = y.size
    return (np.dot(pool.T, (h_logistic(pool, theta) - y))) / m


def ExpGradL(gradd, y):
    return np.argmax(np.sum(np.dot(y, LA.norm(gradd))))