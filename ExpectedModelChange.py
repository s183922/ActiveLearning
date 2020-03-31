import numpy as np
from numpy import linalg as LA
from sklearn.metrics import accuracy_score
from dataloader import *
import random
from sklearn.linear_model import LogisticRegression as Log
# hope this works

random.seed(10)

X_train, y_train, X_test, y_test, Xpool, ypool, poolidx = datasets('data', poolnum = 59900)

model = Log(penalty = 'l2', multi_class= 'multinomial', max_iter= 500, solver='lbfgs')
addn = 5

n_model = 10

ye = np.zeros([n_model,Xpool.shape[0]])

def normgrad(x, theta, y_train, model, ypool):
    # probability of each class
    # Get probability distribution over labels, vector
    p_pred = model.predict_proba(x)

    # derivative of cost as derived for each class
    # defined in paper

    m = len(y_train)
    sumM = []
    sumK = []

    probJ = []
    eval = 1
    #np.exp(np.transpose(theta[j]) * x[i])

    for i in range(m):
        for k,k_real in enumerate(ypool):
            probJ = np.zeros(len(ypool),dtype=object)
            for j in range(len(ypool)):
                probJ[j] = np.exp(np.transpose(theta[j]) * x[i])
            probJ = np.sum(probJ)
            pred = (np.exp(np.transpose(theta[k]) * x[i]))/(probJ)
            if y_train[i] != k_real:
                eval = 0
            form = x[i] * (eval - pred)
            sumK.append(form)
    sumM.append(sumK)

    gradcost = -sumM

    # take norm and add probability
    # for alle punkter udregner den emc, i store funktion tager den det punkt med den bedste
    # emc og laver query på.
    emc = np.dot(p_pred, LA.norm(gradcost))
    # ypool_p_sort tager argmax af denne sum

    return emc

print("here")


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
        emc = normgrad(model.coef_, Xpool[poolidx], y_train, model, ypool)
        ye = model.predict(X_test)
        testacc_emc.append((len(X_train), accuracy_score(y_test, ye)))

        # for hver gang den kører dette loop, tager den for et x-pool noget den gerne vil evaluere og så evaluerer vi
        # modellen efter den har evalueret dette punkt.

        # Add to train - remove from pool
        ypool_p_sort_idx = np.argmax(emc)
        X_train = np.concatenate((X_train, Xpool[poolidx[ypool_p_sort_idx]]))
        y_train = np.concatenate((y_train, ypool[poolidx[ypool_p_sort_idx]]))
        poolidx = np.setdiff1d(poolidx, ypool_p_sort_idx)
        print("Expected model change with {:} training-points".format(len(X_train)))

    return testacc_emc
print("here2")

print(expModelChange(X_train[:100].copy(), y_train[:100].copy(), X_test.copy(), y_test.copy(),
                     model, Xpool.copy(), ypool.copy(), poolidx.copy(),
                     n_iter=100))

# def Calcost(theta, X, C=1.):


#    J = []
#    m = len(y)
#        for i in range(m):
#            for k in k:
#                a = 0
#                if y[i] == k:
#                    a = 1
#                J.append(a*math.log(math.exp ( np.transpose(theta[k])*x[i] /    ) ))


#    return cost
