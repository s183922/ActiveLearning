import numpy as np
from sklearn.linear_model import LogisticRegression as Log
from sklearn.metrics import accuracy_score



class UncertaintySampling:
    def __init__(self, model, pool, addn):
        super().__init__()
        self.model = model
        self.pool = pool
        self.addn = addn

    def LeastConfident(self):
        pool_p = self.model.predict_proba(self.pool)
        x_star = np.argsort(pool_p.max(1))[:self.addn]
        return x_star
        
    def Entropy(self):
        pool_p = self.model.predict_proba(self.pool)
        Entropy = pool_p * np.log(1/pool_p)
        Information_gain = np.argsort(np.sum(Entropy, axis = 1))
        x_star = Information_gain[-self.addn:]
        return x_star

    def Margin(self):
        pool_p = np.sort(self.model.predict_proba(self.pool), axis = 1)
        Margin = np.argsort(pool_p[:,-1] - pool_p[:,-2])
        x_star = Margin[:self.addn]
        
        return x_star
    
    