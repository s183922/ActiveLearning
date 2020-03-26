import numpy as np
from sklearn.linear_model import LogisticRegression as Log
from sklearn.metrics import accuracy_score
model = Log(penlty = 'l2', multi_class= 'multinomial', max_iter= 100)


class UncertaintySampling:
    def __init__(self, model, Xtrain, ytrain, Xtest, ytest, pool):
        super().__init__()
        self.model = model
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xpool = pool
        self.Xtest = Xtest
        self.ytest = ytest
        self.acc_LC = []
        self.acc_E = []
        self.acc_MS = []

    def LeastConfident(self):
        model.fit(self.Xtrain, self.ytrain)
        y_estimate = model.predict(self.Xtest)
        self.acc_LC.append(len(self.Xtrain), accuracy_score(ytest, y_estimate))

        pool_p = model.predict_proba(self.Xpool)
        x_star = np.argsort(pool_p.max(1))[:addn]

    def Entropy(self):
        model.fit(self.Xtrain, self.ytrain)
        y_estimate = model.predict(self.Xtest)
        self.acc_LC.append(len(self.Xtrain), accuracy_score(ytest, y_estimate))

        pool_p = model.predict_proba(self.Xpool)
        Entropy = pool_p * np.log(1/pool_p)
        Information_gain = np.argsort(np.sum(Entropy, axis = 1))
        x_star = Information_gain[-addn:]

    def Margin(self):
        model.fit(self.Xtrain, self.ytrain)
        y_estimate = model.predict(self.Xtest)
        self.acc_LC.append(len(self.Xtrain), accuracy_score(ytest, y_estimate))

        pool_p = np.sort(model.predict_proba(self.Xpool), axis = 1)
        Margin = np.argsort(pool_p[:,-1] - pool_p[:,-2])

        x_star = Margin[:addn]
        

    
    