import numpy as np
from sklearn.linear_model import LogisticRegression as Log
from sklearn.metrics import accuracy_score




def Uncertainty_Sampling(X_train, y_train, X_test, y_test, model, Xpool, ypool, poolidx, n_iter = 100, addn =2, method = "Least Confident"):
    """
    Adds datapoints from pool to trainset according to 3 uncertaintysampling methods: "Least Confident", "Entropy" and "Minimum Margin"
    Input:
    X_train: Trainset of observations
    y_train: Trainset of labels
    X_test: Testset of observations
    y_test: Testset of labels
    model: Model to fit - works with e.g. sklearn logregression and decision tree 
    Xpool: pool of unlabeled observations
    ypool: oracle labels to observations in Xpool
    poolidx: list of index to draw from pool
    n_iter: number of iterations
    addn: number of unlabeled observations to be labelled at each iteration
    method: Choice of Uncertainty sampling method

    Returns:
    List of n_iter tuples with accuracy score and number of training observations
    """
    test_acc = []
    for i in range(n_iter):
        # Fit model and make predicitons
        model.fit(X_train, y_train)
        ye = model.predict(X_test)
        test_acc.append((len(X_train), accuracy_score(y_test, ye)))
        
        # Choice of Active Learning method
        if method == "Least Confident":
            x_star = LeastConfident(model, Xpool[poolidx], addn)
        elif method == "Entropy":
            x_star = Entropy(model, Xpool[poolidx], addn)
        elif method == "Margin":
            x_star = Margin(model, Xpool[poolidx], addn)
        else:
            x_star = BaseLine(poolidx, addn)
       

        # Add to train - remove from pool
        X_train = np.concatenate((X_train, Xpool[poolidx[x_star]]))
        y_train = np.concatenate((y_train, ypool[poolidx[x_star]]))
        poolidx = np.setdiff1d(poolidx, x_star)
        print("Method: {:}".format(method))
        print("With {:} trainingpoints".format(len(X_train)))
    return test_acc

def LeastConfident(model, pool, addn):
    """
    Chooses addn datapoints in the unlabeled pool with the lowest prediction confidence
    """

    # Get probability distribution over labels
    pool_p = model.predict_proba(pool)

    # Find the labels with highes probabilities - Choose the addn of these with lowest score
    x_star = np.argsort(pool_p.max(1))[:addn]
    return x_star
    
def Entropy(model, pool, addn):
    """
    Chooses addn datapoints in the unlabeled pool with highest entropy over the distriution of labels
    """

    # Get probability distribution over labels
    pool_p = model.predict_proba(pool)

    # Calculate entropy and sort for each datapoint in pool
    Entropy = (pool_p * np.log(1/pool_p)).sum(axis = 1)
    Information_gain = np.argsort(Entropy)

    # Choose addn datapoints with highest entropy
    x_star = Information_gain[-addn:]
    return x_star

def Margin(model, pool, addn):
    """
    Chooses addn datapoints in the unlabeled pool where the difference between the two highest label probabilities is the lowest.
    """

    # Get probability distribution over labels and sort for each x
    pool_p = np.sort(model.predict_proba(pool), axis = 1)

    # Calulate difference between two highes label probabilities
    Margin = np.argsort(pool_p[:,-1] - pool_p[:,-2])

    # Choses the addn observations with lowest margin
    x_star = Margin[:addn]
    return x_star

def BaseLine(pool, addn):
    return np.random.choice(np.arange(len(pool), dtype = np.int), addn, replace=False)
    